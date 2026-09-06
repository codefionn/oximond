//! Function traces: the tree is an activation, a function that runs one
//! call of the guest function and returns, and the entry is a second
//! function placed after it that checks the entry contract and calls it
//! (`define_function_entry` and `lower_activation` of `emit.rs`).
//!
//! The activation is `(left, control, area, fuel, [count], parameters...)
//! -> (status, result)`. `left` counts the activations the run may still
//! nest below this one, `area` is this activation's part of the exit area
//! (`stride` words, the last of which names the exit the activation left
//! through when a callee left first), `count` is the tail call count of
//! the frame when the function has one. A nil parameter is not passed. The
//! status is the fuel left when the activation returned, and `-1 - index`
//! when it left through exit `index`. It travels in the integer return
//! register and the result in the second one of its class, as `Op::Ret`
//! has it.
//!
//! Differences from Cranelift's activation, none of which the run sees:
//! the exit area of the activation is an argument, where Cranelift's
//! computes it from the depth at each exit, and an exit stores `left` as the
//! depth, which the entry turns into `depth_cap - left` when the status
//! reaches it. A cell or global input of the root is read where it is used,
//! as Cranelift's are, and an exit copies its word through `control.inputs`
//! (`INDIRECT_SLOT`) instead of loading it in the code.

use super::super::super::exit::{TraceControl, TraceInputSlot, entry_tag};
use super::{Lowerer, NONE, Off, Reject, Res, Val, class, offset};
use crate::jit_ir::trace::{
    ExternalSlot, GuardId, TraceIr, TraceOp, TraceTerminator, TraceType, ValueId,
};
use oximond_mir::ir::{CONTROL_SLOT, Cc, Func, INDIRECT_SLOT, Inst, Op, RETURN_SLOT, SnapVal, Ty};
use std::mem;

/// What a function tree's lowering keeps beyond a loop tree's.
#[derive(Clone, Copy)]
pub(super) struct Activation {
    /// Bytes of exit area each activation owns.
    stride_bytes: i64,
    /// The exit-area word that names the exit an unwinding activation left
    /// through.
    header_slot: u32,
    /// The window register of the tail call count, if the frame has one.
    count: Option<u16>,
    /// The member being lowered: its depth left, its exit area, and the
    /// depth left below it once a self call checked it (`NONE` before).
    pub(super) left: Val,
    pub(super) area: Val,
    pub(super) deeper: Val,
    /// The exit area of the activations it calls, once a self call made it
    /// (`NONE` before).
    pub(super) below: Val,
}

/// The control-block word an exit stores the depth left in.
const DEPTH_SLOT: u32 = CONTROL_SLOT | (mem::offset_of!(TraceControl, depth) / 8) as u32;

fn input_at(index: usize, field: usize) -> i64 {
    offset(index * mem::size_of::<TraceInputSlot>() + field)
}

fn bits_at(index: usize) -> i64 {
    input_at(index, mem::offset_of!(TraceInputSlot, bits))
}

/// The window register of the frame's tail call count.
fn count_register(ir: &TraceIr) -> Option<u16> {
    ir.external
        .iter()
        .position(|slot| *slot == ExternalSlot::TailCalls)
        .map(|position| {
            u16::try_from(usize::from(ir.frames[0].register_count) - ir.external.len() + position)
                .expect("window fits u16")
        })
}

/// The MIR type a parameter or result of `ty` travels as, `None` for nil,
/// which does not travel.
fn passed(ty: TraceType) -> Res<Option<Ty>> {
    match ty {
        TraceType::Nil => Ok(None),
        TraceType::Any | TraceType::Userdata | TraceType::StringRef => Err(Reject::new(
            "function trace with an untyped or borrowed parameter or result",
        )),
        _ => Ok(class(ty)),
    }
}

/// The types of the activation's arguments.
pub(super) fn activation_args(ir: &TraceIr) -> Res<Vec<Ty>> {
    let function = ir.function.as_ref().expect("a function trace");
    let mut tys = vec![Ty::I64; 4];
    if count_register(ir).is_some() {
        tys.push(Ty::I64);
    }
    for ty in &function.parameters {
        if let Some(ty) = passed(*ty)? {
            tys.push(ty);
        }
    }
    Ok(tys)
}

/// Checks that the tree is one this lowering handles: every member a
/// function trace with no loop of its own, the root's sides ending in a
/// return or an exit, and the plan made for activations.
pub(super) fn check_tree(
    members: &[super::super::super::compile::LoweredMember<'_>],
    plan: &super::super::super::compile::ExitPlan,
) -> Res<()> {
    if plan.stride.is_none() {
        return Err(Reject::new("function trace without an activation stride"));
    }
    for (index, member) in members.iter().enumerate() {
        let ir = member.ir;
        if ir.function.is_none() {
            return Err(Reject::new("function tree with a loop member"));
        }
        if ir.counted {
            return Err(Reject::new("counted trace"));
        }
        if !ir.loop_params.is_empty() || member.terminal.is_some() || member.link_loop.is_some() {
            return Err(Reject::new("function trace with a loop"));
        }
        match ir.terminator {
            TraceTerminator::Return { .. } | TraceTerminator::Exit { .. } => {}
            // A side would go back to the root's start through a join that
            // carries every value the start needs.
            TraceTerminator::TailCall { .. } if index == 0 => {}
            TraceTerminator::TailCall { .. } => {
                return Err(Reject::new("side of a function trace that tail calls"));
            }
            _ => return Err(Reject::new("function trace with an unsupported end")),
        }
        for &(_, side) in &member.attached {
            if side <= index {
                return Err(Reject::new("side numbered before its parent"));
            }
        }
    }
    Ok(())
}

impl<'a> Lowerer<'a, '_> {
    /// The root of a function tree: the entry of an activation, its start
    /// (the loop header a tail call goes back to), one unit of fuel, and
    /// the root's code, which ends in returns.
    pub(super) fn activation(&mut self) -> Res<()> {
        self.begin_member(0)?;
        let ir = self.ir;
        let function = ir.function.as_ref().expect("a function trace");
        let stride = self.plan.stride.expect("check_tree saw the stride");
        let count = count_register(ir);
        let tys = activation_args(ir)?;
        self.f.indirect =
            i32::try_from(mem::offset_of!(TraceControl, inputs)).expect("control offset fits");
        let first = self.f.entry_args(&tys);
        let (left, control, area) = (first, first + 1, first + 2);
        self.control = control;
        self.function = Some(Activation {
            stride_bytes: offset(stride * mem::size_of::<i64>()),
            header_slot: u32::try_from(stride - 1).expect("an exit area fits u32"),
            count,
            left,
            area,
            deeper: NONE,
            below: NONE,
        });
        self.inputs = self.load_at(
            Ty::I64,
            control,
            Off::K(0),
            offset(mem::offset_of!(TraceControl, inputs)),
        );
        self.load_words()?;
        // The start: the fuel, the count and the parameters, which a tail
        // call passes again.
        let mut start: Vec<Val> = (first + 3..first + tys.len() as u32).collect();
        if matches!(ir.terminator, TraceTerminator::TailCall { .. }) {
            let header = self.f.header(&start);
            for (k, val) in start.iter_mut().enumerate() {
                *val = header + k as u32;
            }
        }
        let fuel = start[0];
        let first_parameter = 1 + usize::from(count.is_some());
        let mut passed_at = Vec::with_capacity(function.parameters.len());
        let mut next = first_parameter;
        for ty in &function.parameters {
            if passed(*ty)?.is_some() {
                passed_at.push(next);
                next += 1;
            } else {
                passed_at.push(usize::MAX);
            }
        }
        let guest = usize::from(ir.frames[0].register_count).saturating_sub(ir.external.len());
        for (index, input) in ir.entry.inputs.iter().enumerate() {
            let register = usize::from(input.register);
            let val = if register < function.parameters.len() {
                match passed_at[register] {
                    usize::MAX => self.iconst(0),
                    at => start[at],
                }
            } else if Some(input.register) == count {
                start[1]
            } else if register < guest {
                // A fresh activation holds nil past its parameters.
                if input.ty != TraceType::Nil {
                    return self.reject("register past the parameters that is not nil");
                }
                self.iconst(0)
            } else if matches!(input.ty, TraceType::Any | TraceType::StringRef) {
                return self.reject("cell or global input of any type");
            } else {
                // A cell or global the entry checked, read where it is used.
                self.s.lazy_in[input.value.index()] = u32::try_from(index + 1).expect("an input");
                continue;
            };
            if input.ty != TraceType::Nil && self.f.val_ty[val as usize] != self.mty(input.ty)? {
                return self.reject("parameter of another type than the entry input");
            }
            self.s.vals[input.value.index()] = val;
        }
        // Each activation spends one unit of fuel, so a long recursion still
        // comes back to the VM.
        let one = self.iconst(1);
        self.fuel = self.f.bin(Op::Sub, fuel, one);
        let poll = self.plan.members[0]
            .poll
            .ok_or(Reject::new("function trace without a poll exit"))?;
        let snap = self.fill_exit(poll)?;
        let zero = self.iconst(0);
        let mut inst = Inst::new(Op::Guard, Ty::I64);
        inst.cc = Cc::Gt;
        inst.a = self.fuel;
        inst.b = zero;
        inst.c = snap;
        self.f.push(inst);
        self.member_code()
    }

    /// The preheader, body and terminator of the member being lowered.
    pub(super) fn member_code(&mut self) -> Res<()> {
        let ir = self.ir;
        for (at, instruction) in ir.preheader.iter().chain(&ir.body).enumerate() {
            self.at = at;
            self.op(&instruction.op, instruction.result)?;
        }
        self.at = usize::MAX;
        match &ir.terminator {
            TraceTerminator::Return { value } => self.ret(*value),
            TraceTerminator::TailCall { args, .. } => self.tail_call(args),
            TraceTerminator::Exit { snapshot } => {
                let terminal = self.plan.members[self.member].terminal;
                self.ends_in_exit(self.member, *snapshot, terminal)
            }
            _ => self.reject("function trace with an unsupported end"),
        }
    }

    /// The value of the lazy entry input `value`, if it is one: loaded from
    /// its input slot here, as Cranelift's `load_input`.
    pub(super) fn lazy_input(&mut self, value: ValueId) -> Res<Option<Val>> {
        let at = self.s.lazy_in.get(value.index()).copied().unwrap_or(0);
        if at == 0 {
            return Ok(None);
        }
        let ty = self.ty(value);
        self.load_input(at as usize - 1, ty).map(Some)
    }

    /// The snapshot entry of exit slot `slot` for a lazy entry input whose
    /// word the exit copies as it is, if `value` is one: every type but a
    /// boolean, whose slot may hold any non-zero word for true.
    pub(super) fn lazy_exit(&self, value: ValueId, ty: TraceType, slot: u32) -> Option<SnapVal> {
        let at = self.s.lazy_in.get(value.index()).copied().unwrap_or(0);
        (at != 0 && ty != TraceType::Boolean).then(|| SnapVal {
            val: u32::try_from(bits_at(at as usize - 1)).expect("an input slot offset fits"),
            slot: slot | INDIRECT_SLOT,
        })
    }

    /// The entries every exit adds to its slots: the fuel, a counted tree's
    /// budget, and an activation's depth left.
    pub(super) fn exit_extras(&mut self) {
        self.s.buf.push(SnapVal {
            val: self.fuel,
            slot: CONTROL_SLOT,
        });
        if self.counted {
            self.s.buf.push(SnapVal {
                val: self.budget,
                slot: CONTROL_SLOT | super::BUDGET_WORD,
            });
        }
        if let Some(activation) = self.function {
            self.s.buf.push(SnapVal {
                val: activation.left,
                slot: DEPTH_SLOT,
            });
        }
    }

    /// For a side of a function tree entered with `snapshot`: the root's
    /// input slot of `register` when it is a cell or global of the window
    /// that the snapshot keeps as the root read it. The side loads it from
    /// the input slots, so the code before the guard need not.
    pub(super) fn window_input(
        &self,
        snapshot: &crate::jit_ir::trace::Snapshot,
        register: u16,
        ty: TraceType,
    ) -> Option<usize> {
        self.function?;
        let root = self.members[0].ir;
        let guest = usize::from(root.frames[0].register_count).saturating_sub(root.external.len());
        if usize::from(register) < guest
            || Some(register) == self.activation_state().count
            || matches!(ty, TraceType::Any | TraceType::StringRef)
            || snapshot.frames[0].registers.get(usize::from(register))
                != Some(&crate::jit_ir::trace::RegisterEntry::Unchanged)
        {
            return None;
        }
        root.entry
            .inputs
            .iter()
            .position(|input| input.register == register && input.ty == ty)
    }

    /// The snapshot of the guard that enters side `member`.
    pub(super) fn entered_snapshot(
        &self,
        member: usize,
    ) -> Option<&'a crate::jit_ir::trace::Snapshot> {
        let parent = *self.s.parent.get(member)?;
        let ir = self.members.get(parent)?.ir;
        let (guard, _) = self.members[parent]
            .attached
            .iter()
            .find(|(_, side)| *side == member)?;
        let snapshot = ir.guards.get(*guard)?.snapshot;
        ir.snapshots.get(snapshot.index())
    }

    fn activation_state(&self) -> Activation {
        self.function.expect("a function trace")
    }

    /// A [`TraceOp::SelfCall`]: the callee runs as a nested activation one
    /// area further into the exit area, unless the run is as deep as it may
    /// go, when the guard leaves before the call. A callee that left makes
    /// this activation leave too: it stores the unwind exit's slots and its
    /// index in its own area and returns the callee's status.
    pub(super) fn self_call(
        &mut self,
        args: &[ValueId],
        guard: GuardId,
        result: Option<ValueId>,
    ) -> Res<()> {
        let unwind = self.plan.members[self.member]
            .unwinds
            .iter()
            .find(|(held, _)| *held == guard.index())
            .map(|(_, exit)| *exit)
            .ok_or(Reject::new("self call without an unwind exit"))?;
        let root = self.members[0].ir;
        let function = root.function.as_ref().expect("a function trace");
        if args.len() != function.parameters.len() {
            return self.reject("self call with another number of arguments");
        }
        let mut state = self.activation_state();
        // An earlier self call of the member, which dominates this one, found
        // room for the callee already.
        if state.deeper == NONE {
            let zero = self.iconst(0);
            self.guard_cc(Cc::Gt, state.left, zero, guard)?;
            let one = self.iconst(1);
            state.deeper = self.f.bin(Op::Sub, state.left, one);
            if let Some(held) = self.function.as_mut() {
                held.deeper = state.deeper;
            }
        }
        if state.below == NONE {
            let step = self.iconst(state.stride_bytes);
            state.below = self.f.bin(Op::Add, state.area, step);
            if let Some(held) = self.function.as_mut() {
                held.below = state.below;
            }
        }
        let mut list = vec![state.deeper, self.control, state.below, self.fuel];
        // A nested activation starts with no tail call of its own.
        if state.count.is_some() {
            list.push(self.iconst(0));
        }
        for (arg, ty) in args.iter().zip(&function.parameters) {
            let Some(want) = passed(*ty)? else {
                continue;
            };
            let val = self.get(*arg)?;
            if self.f.val_ty[val as usize] != want {
                return self.reject("self call argument of another type");
            }
            list.push(val);
        }
        let ret2 = match function.result {
            Some(ty) => passed(ty)?,
            None => None,
        };
        let (status, returned) = self.f.call_rel(0, &list, ret2);
        // The unwind: this activation's slots of the exit, the exit's index
        // in the area's last word, and the callee's status.
        self.s.buf.clear();
        self.exit_slots(unwind)?;
        let index = self.iconst(i64::try_from(unwind).expect("exit index fits"));
        self.s.buf.push(SnapVal {
            val: index,
            slot: state.header_slot,
        });
        self.s.buf.push(SnapVal {
            val: status,
            slot: RETURN_SLOT,
        });
        self.f.set_snap(unwind as u32, &self.s.buf);
        let zero = self.iconst(0);
        let mut inst = Inst::new(Op::Guard, Ty::I64);
        inst.cc = Cc::Gt;
        inst.a = status;
        inst.b = zero;
        inst.c = unwind as u32;
        self.f.push(inst);
        self.fuel = status;
        // The callee may have stored into any table.
        self.forget_words();
        if let Some(result) = result {
            let val = match function.result {
                Some(TraceType::Nil) => self.iconst(0),
                Some(_) => returned,
                None => return self.reject("self call result of a function that returns none"),
            };
            self.set(Some(result), val);
        }
        Ok(())
    }

    /// Pushes the slots of exit `index`'s descriptor onto `s.buf`.
    pub(super) fn exit_slots(&mut self, index: usize) -> Res<()> {
        let descriptor = &self.plan.descriptors[index];
        for slot in &descriptor.slots {
            if matches!(slot.ty, TraceType::Nil | TraceType::Any) {
                continue;
            }
            if let Some(entry) = self.lazy_exit(slot.value, slot.ty, slot.slot) {
                self.s.buf.push(entry);
                continue;
            }
            let val = self.get(slot.value)?;
            self.s.buf.push(SnapVal {
                val,
                slot: slot.slot,
            });
        }
        Ok(())
    }

    /// [`TraceTerminator::Return`]: the fuel left and the value.
    fn ret(&mut self, value: Option<ValueId>) -> Res<()> {
        let root = self.members[0].ir;
        let wants = root.function.as_ref().and_then(|function| function.result);
        let second = match (value, wants) {
            (Some(value), Some(ty)) => match passed(ty)? {
                Some(want) => {
                    let val = self.get(value)?;
                    if self.f.val_ty[val as usize] != want {
                        return self.reject("return of another type than the function's");
                    }
                    val
                }
                None => NONE,
            },
            (None, None) => NONE,
            _ => return self.reject("return that does not match the function's result"),
        };
        self.f.ret(self.fuel, second);
        Ok(())
    }

    /// [`TraceTerminator::TailCall`] of the root: back to the start with the
    /// arguments and one tail call more.
    fn tail_call(&mut self, args: &[ValueId]) -> Res<()> {
        let root = self.members[0].ir;
        let function = root.function.as_ref().expect("a function trace");
        let Some(register) = self.activation_state().count else {
            return self.reject("tail call without a tail call count");
        };
        let Some(counted) = root
            .entry
            .inputs
            .iter()
            .find(|input| input.register == register)
            .map(|input| input.value)
        else {
            return self.reject("tail call count that is no entry input");
        };
        if args.len() != function.parameters.len() {
            return self.reject("tail call with another number of arguments");
        }
        let count = self.get(counted)?;
        let one = self.iconst(1);
        let mut next = vec![self.fuel, self.f.bin(Op::Add, count, one)];
        for (arg, ty) in args.iter().zip(&function.parameters) {
            let Some(want) = passed(*ty)? else {
                continue;
            };
            let val = self.get(*arg)?;
            if self.f.val_ty[val as usize] != want {
                return self.reject("tail call argument of another type");
            }
            next.push(val);
        }
        self.f.back(&next);
        Ok(())
    }
}

/// Scans an op of a function tree that a loop tree does not have.
pub(super) fn function_op(op: &TraceOp) -> bool {
    matches!(op, TraceOp::SelfCall { .. })
}

/// The entry of a function tree, placed `before` bytes after the start of
/// the activation: it checks the entry inputs' tags (a mismatch of input
/// `i` returns `-1 - i`), records the inputs and the exit area in the
/// control block, loads the parameters and calls the activation. A return
/// stores the fuel left and the result (in exit-area word 0, as an exit slot
/// of its type holds it) and returns `RETURNED_CODE`; an exit stores the
/// depth it left at and returns its index.
pub(in super::super) fn wrapper(ir: &TraceIr, before: i64, w: &mut Func) -> Res<()> {
    let function = ir.function.as_ref().expect("a function trace");
    w.clear();
    w.indirect = i32::try_from(mem::offset_of!(TraceControl, inputs)).expect("offset fits");
    let a = w.entry_args(&[Ty::I64; 3]);
    let (inputs, control, area) = (a, a + 1, a + 2);
    let load = |w: &mut Func, ty: Ty, base: Val, at: i64| {
        w.load(
            ty,
            base,
            NONE,
            1,
            i32::try_from(at).expect("an offset fits"),
        )
    };
    for (index, input) in ir.entry.inputs.iter().enumerate() {
        let Some(required) = entry_tag(input.ty) else {
            return Err(Reject::new("entry input without a tag"));
        };
        let tag = load(
            w,
            Ty::I64,
            inputs,
            input_at(index, mem::offset_of!(TraceInputSlot, tag)),
        );
        let wanted = w.iconst(required);
        let code = -i64::try_from(index).expect("input index fits") - 1;
        let snap = w.snap_returning(&[], code);
        let mut inst = Inst::new(Op::Guard, Ty::I64);
        inst.cc = Cc::Eq;
        inst.a = tag;
        inst.b = wanted;
        inst.c = snap;
        w.push(inst);
    }
    let field = |name: usize| i32::try_from(name).expect("control offset fits");
    let cap = load(
        w,
        Ty::I64,
        control,
        offset(mem::offset_of!(TraceControl, depth_cap)),
    );
    let fuel = load(
        w,
        Ty::I64,
        control,
        offset(mem::offset_of!(TraceControl, fuel)),
    );
    w.store(
        control,
        NONE,
        1,
        field(mem::offset_of!(TraceControl, inputs)),
        inputs,
    );
    w.store(
        control,
        NONE,
        1,
        field(mem::offset_of!(TraceControl, area)),
        area,
    );
    let mut args = vec![cap, control, area, fuel];
    let input_of = |register: usize| {
        ir.entry
            .inputs
            .iter()
            .position(|input| usize::from(input.register) == register)
    };
    if let Some(register) = count_register(ir) {
        let Some(index) = input_of(usize::from(register)) else {
            return Err(Reject::new("the tail call count is no entry input"));
        };
        args.push(load(w, Ty::I64, inputs, bits_at(index)));
    }
    for (register, ty) in function.parameters.iter().enumerate() {
        let Some(want) = passed(*ty)? else {
            continue;
        };
        let Some(index) = input_of(register) else {
            return Err(Reject::new("a parameter is no entry input"));
        };
        let bits = load(w, want, inputs, bits_at(index));
        args.push(if *ty == TraceType::Boolean {
            let (zero, one) = (w.iconst(0), w.iconst(1));
            let zero2 = w.iconst(0);
            w.select(Cc::Ne, bits, zero2, one, zero)
        } else {
            bits
        });
    }
    let result = match function.result {
        Some(ty) => passed(ty)?,
        None => None,
    };
    let (status, returned) = w.call_rel(-before, &args, result);
    let zero = w.iconst(0);
    w.guard_side(Cc::Gt, status, zero, 0, &[status, cap, control]);
    w.store(
        control,
        NONE,
        1,
        field(mem::offset_of!(TraceControl, fuel)),
        status,
    );
    if returned != NONE {
        w.store(area, NONE, 1, 0, returned);
    }
    let code = w.iconst(super::super::super::emit::RETURNED_CODE);
    w.ret(code, NONE);
    // The activation left: its exit stored the depth left, which becomes
    // the depth `cap - left`, and the status is `-1 - index`.
    let (_, side) = w.begin_side(&[Ty::I64, Ty::I64, Ty::I64]);
    let (status, cap, control) = (side, side + 1, side + 2);
    let depth_at = field(mem::offset_of!(TraceControl, depth));
    let left = w.load(Ty::I64, control, NONE, 1, depth_at);
    let depth = w.bin(Op::Sub, cap, left);
    w.store(control, NONE, 1, depth_at, depth);
    let minus_one = w.iconst(-1);
    let index = w.bin(Op::Sub, minus_one, status);
    w.ret(index, NONE);
    Ok(())
}
