//! MIR -> CLIF translator and the Cranelift build the bench and tests compare
//! against. The CLIF has the shape of `emit.rs`: header block parameters for
//! loop values, guards as `brif` to a cold block that stores the snapshot and
//! returns `-1 - index`, sides as blocks taking the snapshot values. The JIT
//! flags are the Optimized tier's: `opt_level=none`, `backtracking`.
#![allow(dead_code)]

use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, InstBuilder, MemFlags, Signature, UserFuncName, Value,
    condcodes::{FloatCC, IntCC},
    types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use oximond_mir::ir::{Cc, Func, NONE, Op, Ty};

fn clif_ty(t: Ty) -> cranelift_codegen::ir::Type {
    match t {
        Ty::I64 => types::I64,
        Ty::F64 => types::F64,
    }
}

fn int_cc(cc: Cc) -> IntCC {
    match cc {
        Cc::Eq => IntCC::Equal,
        Cc::Ne => IntCC::NotEqual,
        Cc::Lt => IntCC::SignedLessThan,
        Cc::Le => IntCC::SignedLessThanOrEqual,
        Cc::Gt => IntCC::SignedGreaterThan,
        Cc::Ge => IntCC::SignedGreaterThanOrEqual,
        Cc::Ult => IntCC::UnsignedLessThan,
        Cc::Ule => IntCC::UnsignedLessThanOrEqual,
        Cc::Ugt => IntCC::UnsignedGreaterThan,
        Cc::Uge => IntCC::UnsignedGreaterThanOrEqual,
        Cc::Nlt | Cc::Nle | Cc::Ngt | Cc::Nge => unreachable!("float-only condition"),
    }
}

fn float_cc(cc: Cc) -> FloatCC {
    match cc {
        Cc::Eq => FloatCC::Equal,
        Cc::Ne => FloatCC::NotEqual,
        Cc::Lt | Cc::Ult => FloatCC::LessThan,
        Cc::Le | Cc::Ule => FloatCC::LessThanOrEqual,
        Cc::Gt | Cc::Ugt => FloatCC::GreaterThan,
        Cc::Ge | Cc::Uge => FloatCC::GreaterThanOrEqual,
        Cc::Nlt => FloatCC::UnorderedOrGreaterThanOrEqual,
        Cc::Nle => FloatCC::UnorderedOrGreaterThan,
        Cc::Ngt => FloatCC::UnorderedOrLessThanOrEqual,
        Cc::Nge => FloatCC::UnorderedOrLessThan,
    }
}

/// The convention of `extern "C"` on this host, which the entry and the
/// called functions use.
pub fn host_call_conv() -> cranelift_codegen::isa::CallConv {
    if cfg!(windows) {
        cranelift_codegen::isa::CallConv::WindowsFastcall
    } else {
        cranelift_codegen::isa::CallConv::SystemV
    }
}

pub fn signature() -> Signature {
    let mut sig = Signature::new(host_call_conv());
    for _ in 0..3 {
        sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::I64));
    sig
}

/// Translates `f` into `func`. Returns the number of CLIF instructions made.
pub fn translate(
    f: &Func,
    func: &mut cranelift_codegen::ir::Function,
    fbc: &mut FunctionBuilderContext,
) {
    let mut b = FunctionBuilder::new(func, fbc);
    let mut vals: Vec<Option<Value>> = vec![None; f.num_vals()];
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    let sides: Vec<Block> = (0..f.num_sides())
        .map(|k| {
            let blk = b.create_block();
            let (s, _) = f.region(1 + k);
            let n = f.insts[s].b;
            for j in 0..n {
                b.append_block_param(blk, clif_ty(f.val_ty[(f.insts[s].a + j) as usize]));
            }
            blk
        })
        .collect();
    let mut header: Option<Block> = None;
    let mut open_ifs: Vec<Block> = Vec::new();
    let mut joins: Vec<Vec<Block>> = vec![Vec::new(); f.num_sides() + 1];
    let area = b.block_params(entry)[2];
    let control = b.block_params(entry)[1];
    let cells: Vec<_> = (0..f.cells)
        .map(|_| {
            b.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                8,
                3,
            ))
        })
        .collect();
    for r in 0..=f.num_sides() {
        let (s, e) = f.region(r);
        if r == 0 {
            b.switch_to_block(entry);
        } else {
            b.switch_to_block(sides[r - 1]);
        }
        for idx in s..e {
            let i = f.insts[idx];
            let g = |vals: &Vec<Option<Value>>, v: u32| vals[v as usize].expect("defined");
            match i.op {
                Op::Entry => {
                    let p = b.block_params(entry).to_vec();
                    vals[i.a as usize] = Some(p[0]);
                    vals[i.a as usize + 1] = Some(p[1]);
                }
                Op::SideEntry => {
                    let p = b.block_params(sides[r - 1]).to_vec();
                    for (k, v) in p.into_iter().enumerate() {
                        vals[i.a as usize + k] = Some(v);
                    }
                }
                Op::Const => {
                    let v = match i.ty {
                        Ty::I64 => b.ins().iconst(types::I64, i.imm),
                        Ty::F64 => b.ins().f64const(f64::from_bits(i.imm as u64)),
                    };
                    vals[i.dst as usize] = Some(v);
                }
                Op::Load | Op::Store | Op::AddMem => {
                    let base = g(&vals, i.a);
                    let addr = if i.b == NONE {
                        base
                    } else {
                        let idx = g(&vals, i.b);
                        let off = if i.scale == 1 {
                            idx
                        } else {
                            b.ins().imul_imm(idx, i64::from(i.scale))
                        };
                        b.ins().iadd(base, off)
                    };
                    if i.op == Op::Load {
                        let v =
                            b.ins()
                                .load(clif_ty(i.ty), MemFlags::trusted(), addr, i.imm as i32);
                        vals[i.dst as usize] = Some(v);
                    } else if i.op == Op::AddMem {
                        let old = b
                            .ins()
                            .load(types::I64, MemFlags::trusted(), addr, i.imm as i32);
                        let v = b.ins().iadd(old, g(&vals, i.c));
                        b.ins().store(MemFlags::trusted(), v, addr, i.imm as i32);
                    } else {
                        let v = g(&vals, i.c);
                        b.ins().store(MemFlags::trusted(), v, addr, i.imm as i32);
                    }
                }
                Op::Add
                | Op::Sub
                | Op::Mul
                | Op::And
                | Op::Or
                | Op::Xor
                | Op::SDiv
                | Op::SRem
                | Op::UDiv
                | Op::URem => {
                    let (x, y) = (g(&vals, i.a), g(&vals, i.b));
                    let v = match i.op {
                        Op::SDiv => b.ins().sdiv(x, y),
                        Op::SRem => b.ins().srem(x, y),
                        Op::UDiv => b.ins().udiv(x, y),
                        Op::URem => b.ins().urem(x, y),
                        Op::Add => b.ins().iadd(x, y),
                        Op::Sub => b.ins().isub(x, y),
                        Op::Mul => b.ins().imul(x, y),
                        Op::And => b.ins().band(x, y),
                        Op::Or => b.ins().bor(x, y),
                        _ => b.ins().bxor(x, y),
                    };
                    vals[i.dst as usize] = Some(v);
                }
                Op::Shl | Op::Shr | Op::Sar => {
                    let x = g(&vals, i.a);
                    let y = g(&vals, i.b);
                    let v = match i.op {
                        Op::Shl => b.ins().ishl(x, y),
                        Op::Shr => b.ins().ushr(x, y),
                        _ => b.ins().sshr(x, y),
                    };
                    vals[i.dst as usize] = Some(v);
                }
                Op::FAdd | Op::FSub | Op::FMul | Op::FDiv => {
                    let (x, y) = (g(&vals, i.a), g(&vals, i.b));
                    let v = match i.op {
                        Op::FAdd => b.ins().fadd(x, y),
                        Op::FSub => b.ins().fsub(x, y),
                        Op::FMul => b.ins().fmul(x, y),
                        _ => b.ins().fdiv(x, y),
                    };
                    vals[i.dst as usize] = Some(v);
                }
                Op::FSqrt => {
                    let x = g(&vals, i.a);
                    vals[i.dst as usize] = Some(b.ins().sqrt(x));
                }
                Op::FCopySign => {
                    let (x, y) = (g(&vals, i.a), g(&vals, i.b));
                    let abs = b.ins().fabs(x);
                    vals[i.dst as usize] = Some(b.ins().fcopysign(abs, y));
                }
                Op::FNma => {
                    let (x, y, z) = (g(&vals, i.a), g(&vals, i.b), g(&vals, i.c));
                    let negated = b.ins().fneg(x);
                    vals[i.dst as usize] = Some(b.ins().fma(negated, y, z));
                }
                Op::FNeg | Op::FAbs | Op::FFloor | Op::FCeil | Op::FTrunc | Op::F2I => {
                    let x = g(&vals, i.a);
                    vals[i.dst as usize] = Some(match i.op {
                        Op::FNeg => b.ins().fneg(x),
                        Op::FAbs => b.ins().fabs(x),
                        Op::FFloor => b.ins().floor(x),
                        Op::FCeil => b.ins().ceil(x),
                        Op::FTrunc => b.ins().trunc(x),
                        _ => b.ins().fcvt_to_sint_sat(types::I64, x),
                    });
                }
                Op::Bits | Op::FromBits => {
                    let x = g(&vals, i.a);
                    let ty = if i.op == Op::Bits {
                        types::I64
                    } else {
                        types::F64
                    };
                    vals[i.dst as usize] = Some(b.ins().bitcast(ty, MemFlags::new(), x));
                }
                Op::Load8 | Op::Load32 | Op::Store8 => {
                    let base = g(&vals, i.a);
                    let addr = if i.b == NONE {
                        base
                    } else {
                        let idx = g(&vals, i.b);
                        let off = if i.scale == 1 {
                            idx
                        } else {
                            b.ins().imul_imm(idx, i64::from(i.scale))
                        };
                        b.ins().iadd(base, off)
                    };
                    if i.op == Op::Load32 {
                        let v = b.ins().uload32(MemFlags::trusted(), addr, i.imm as i32);
                        vals[i.dst as usize] = Some(v);
                    } else if i.op == Op::Load8 {
                        let v = b
                            .ins()
                            .uload8(types::I64, MemFlags::trusted(), addr, i.imm as i32);
                        vals[i.dst as usize] = Some(v);
                    } else {
                        let v = g(&vals, i.c);
                        b.ins().istore8(MemFlags::trusted(), v, addr, i.imm as i32);
                    }
                }
                Op::I2F => {
                    let x = g(&vals, i.a);
                    vals[i.dst as usize] = Some(b.ins().fcvt_from_sint(types::F64, x));
                }
                Op::Select => {
                    let (x, y, t, e) = (g(&vals, i.a), g(&vals, i.b), g(&vals, i.c), g(&vals, i.d));
                    let c = match f.val_ty[i.a as usize] {
                        Ty::I64 => b.ins().icmp(int_cc(i.cc), x, y),
                        Ty::F64 => b.ins().fcmp(float_cc(i.cc), x, y),
                    };
                    vals[i.dst as usize] = Some(b.ins().select(c, t, e));
                }
                Op::Guard => {
                    let (x, y) = (g(&vals, i.a), g(&vals, i.b));
                    let c = match i.ty {
                        Ty::I64 => b.ins().icmp(int_cc(i.cc), x, y),
                        Ty::F64 => b.ins().fcmp(float_cc(i.cc), x, y),
                    };
                    let cont = b.create_block();
                    if i.d != NONE {
                        let args: Vec<BlockArg> = f
                            .snap_of(i.c)
                            .iter()
                            .map(|s| BlockArg::Value(g(&vals, s.val)))
                            .collect();
                        b.ins().brif(c, cont, &[], sides[i.d as usize], &args);
                    } else {
                        let exit = b.create_block();
                        b.set_cold_block(exit);
                        b.ins().brif(c, cont, &[], exit, &[]);
                        b.switch_to_block(exit);
                        emit_exit(&mut b, f, &vals, area, i.c);
                    }
                    b.switch_to_block(cont);
                }
                Op::Header => {
                    let hb = b.create_block();
                    let mut init = Vec::new();
                    for k in 0..i.b {
                        let p = b.append_block_param(hb, clif_ty(f.val_ty[(i.a + k) as usize]));
                        vals[(i.a + k) as usize] = Some(p);
                        init.push(BlockArg::Value(g(&vals, f.lists[(i.c + k) as usize])));
                    }
                    b.ins().jump(hb, &init);
                    b.switch_to_block(hb);
                    header = Some(hb);
                }
                Op::Join => {
                    let jb = b.create_block();
                    let mut init = Vec::new();
                    for k in 0..i.b {
                        let p = b.append_block_param(jb, clif_ty(f.val_ty[(i.a + k) as usize]));
                        vals[(i.a + k) as usize] = Some(p);
                        init.push(BlockArg::Value(g(&vals, f.lists[(i.c + k) as usize])));
                    }
                    b.ins().jump(jb, &init);
                    b.switch_to_block(jb);
                    joins[r].push(jb);
                }
                Op::Rejoin => {
                    let args: Vec<BlockArg> = f
                        .list(i.a, i.b)
                        .iter()
                        .map(|&v| BlockArg::Value(g(&vals, v)))
                        .collect();
                    let nth = if i.e == NONE { 0 } else { i.e as usize };
                    b.ins().jump(joins[i.d as usize][nth], &args);
                }
                Op::Back => {
                    let args: Vec<BlockArg> = f
                        .list(i.a, i.b)
                        .iter()
                        .map(|&v| BlockArg::Value(g(&vals, v)))
                        .collect();
                    b.ins().jump(header.unwrap(), &args);
                }
                Op::Exit => emit_exit(&mut b, f, &vals, area, i.c),
                // The comparison kernels are loops with the three-argument
                // entry; functions that return are tested against the
                // evaluator (tests/activation.rs).
                Op::Ret => panic!("the Cranelift reference has no Ret"),
                Op::Goto => {
                    let args: Vec<BlockArg> = f
                        .snap_of(i.c)
                        .iter()
                        .map(|s| BlockArg::Value(g(&vals, s.val)))
                        .collect();
                    b.ins().jump(sides[i.d as usize], &args);
                }
                Op::CallIf => {
                    let (x, y) = (g(&vals, i.a), g(&vals, i.b));
                    let c = match i.ty {
                        Ty::I64 => b.ins().icmp(int_cc(i.cc), x, y),
                        Ty::F64 => b.ins().fcmp(float_cc(i.cc), x, y),
                    };
                    let call_block = b.create_block();
                    let cont = b.create_block();
                    b.set_cold_block(call_block);
                    b.ins().brif(c, call_block, &[], cont, &[]);
                    b.switch_to_block(call_block);
                    let mut sig = Signature::new(host_call_conv());
                    let mut call_args = Vec::new();
                    for &a in f.list(i.c, i.d) {
                        sig.params
                            .push(AbiParam::new(clif_ty(f.val_ty[a as usize])));
                        call_args.push(g(&vals, a));
                    }
                    sig.returns.push(AbiParam::new(types::I64));
                    let sig = b.import_signature(sig);
                    let callee = if i.scale == 1 {
                        b.ins()
                            .load(types::I64, MemFlags::trusted(), control, i.imm as i32)
                    } else {
                        b.ins().iconst(types::I64, i.imm)
                    };
                    let call = b.ins().call_indirect(sig, callee, &call_args);
                    if i.e != NONE {
                        let result = b.inst_results(call)[0];
                        b.ins()
                            .store(MemFlags::trusted(), result, control, i.e as i32);
                    }
                    b.ins().jump(cont, &[]);
                    b.switch_to_block(cont);
                }
                Op::If => {
                    let (x, y) = (g(&vals, i.a), g(&vals, i.b));
                    let c = match i.ty {
                        Ty::I64 => b.ins().icmp(int_cc(i.cc), x, y),
                        Ty::F64 => b.ins().fcmp(float_cc(i.cc), x, y),
                    };
                    let end = f.insts[i.e as usize];
                    let then_block = b.create_block();
                    let join_block = b.create_block();
                    let mut skipped = Vec::new();
                    for k in 0..end.b {
                        b.append_block_param(join_block, clif_ty(f.val_ty[(end.a + k) as usize]));
                        let s = f.lists[(end.c + end.b + k) as usize];
                        skipped.push(BlockArg::Value(g(&vals, s)));
                    }
                    b.ins().brif(c, then_block, &[], join_block, &skipped);
                    b.switch_to_block(then_block);
                    open_ifs.push(join_block);
                }
                Op::EndIf => {
                    let join_block = open_ifs.pop().expect("an end-if closes an if");
                    let ran: Vec<BlockArg> = f
                        .list(i.c, i.b)
                        .iter()
                        .map(|&v| BlockArg::Value(g(&vals, v)))
                        .collect();
                    b.ins().jump(join_block, &ran);
                    b.switch_to_block(join_block);
                    let p = b.block_params(join_block).to_vec();
                    for (k, v) in p.into_iter().enumerate() {
                        vals[i.a as usize + k] = Some(v);
                    }
                }
                Op::CellStore => {
                    let v = g(&vals, i.c);
                    b.ins().stack_store(v, cells[i.imm as usize], 0);
                }
                Op::CellLoad => {
                    let v = b.ins().stack_load(clif_ty(i.ty), cells[i.imm as usize], 0);
                    vals[i.dst as usize] = Some(v);
                }
                Op::ExitArea => vals[i.dst as usize] = Some(area),
                Op::Call | Op::CallCtl => {
                    let mut sig = Signature::new(host_call_conv());
                    let mut args = Vec::new();
                    for &a in f.list(i.a, i.b) {
                        sig.params
                            .push(AbiParam::new(clif_ty(f.val_ty[a as usize])));
                        args.push(g(&vals, a));
                    }
                    if i.dst != NONE {
                        sig.returns.push(AbiParam::new(clif_ty(i.ty)));
                    }
                    let sig = b.import_signature(sig);
                    let callee = if i.op == Op::CallCtl {
                        b.ins()
                            .load(types::I64, MemFlags::trusted(), control, i.imm as i32)
                    } else {
                        b.ins().iconst(types::I64, i.imm)
                    };
                    let call = b.ins().call_indirect(sig, callee, &args);
                    if i.dst != NONE {
                        vals[i.dst as usize] = Some(b.inst_results(call)[0]);
                    }
                }
            }
        }
    }
    b.seal_all_blocks();
    b.finalize();
}

fn emit_exit(
    b: &mut FunctionBuilder<'_>,
    f: &Func,
    vals: &[Option<Value>],
    area: Value,
    snap: u32,
) {
    for s in f.snap_of(snap) {
        let v = vals[s.val as usize].unwrap();
        b.ins()
            .store(MemFlags::trusted(), v, area, 8 * s.slot as i32);
    }
    let code = b.ins().iconst(types::I64, -1 - i64::from(snap));
    b.ins().return_(&[code]);
}

/// Cranelift flags of the Optimized tier (`emit.rs`, `lower`).
pub fn jit_builder() -> JITBuilder {
    JITBuilder::with_flags(
        &[
            ("opt_level", "none"),
            ("enable_verifier", "false"),
            ("regalloc_algorithm", "backtracking"),
        ],
        default_libcall_names(),
    )
    .unwrap()
}

pub struct ClifCode {
    pub module: JITModule,
    pub entry: *const u8,
    pub code: Vec<u8>,
    pub clif_insts: usize,
}

/// The whole per-tree path of `lower`: new builder and module, translate,
/// define, finalize.
pub fn compile(f: &Func, keep_code: bool) -> ClifCode {
    let mut module = JITModule::new(jit_builder());
    let sig = signature();
    let id = module
        .declare_function("trace", Linkage::Export, &sig)
        .unwrap();
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    ctx.func.name = UserFuncName::user(0, id.as_u32());
    let mut fbc = FunctionBuilderContext::new();
    translate(f, &mut ctx.func, &mut fbc);
    let clif_insts = ctx.func.dfg.num_insts();
    module.define_function(id, &mut ctx).unwrap();
    let code = if keep_code {
        ctx.compiled_code().unwrap().code_buffer().to_vec()
    } else {
        Vec::new()
    };
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    let entry = module.get_finalized_function(id);
    ClifCode {
        module,
        entry,
        code,
        clif_insts,
    }
}
