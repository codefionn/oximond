//! Inlined calls in traces, driven through hand-assembled chunks.
//!
//! A trace inlines a call of a Lua closure: the callee's operations run in
//! place, behind a guard on the callee's identity, and a failed guard inside
//! the callee must rebuild the whole call stack the interpreter would have
//! had. Every case here compares with an isolated interpreter that single
//! steps the same program, frame by frame. Counters prove native code ran.
//!
//! The root program is a numeric loop that calls a closure made by its own
//! `Closure` instruction, so every run has real closure handles. Programs:
//!
//! - `branching`: `acc = acc + f(i)`, where `f` doubles `x` unless `x` is
//!   above 150, when it returns zero. The branch is the guard inside the
//!   callee.
//! - `counter`: `f` adds its argument to an upvalue, which is a captured
//!   local of the root frame.
//! - `chain`: calls nested `depth` deep.

use super::trace_native::{native_vm, stats_of};
use super::trace_record::*;
use super::*;
use crate::bytecode::{UpvalueDescriptor, UpvalueSource};
use crate::jit_ir::trace::reference::{self, ForcedExit, Outcome};
use crate::jit_ir::trace::{
    FrameId, GuardId, HelperId, MAX_TRACE_FRAME_DEPTH, SnapshotPhase, TraceIr, TraceOp,
    TraceScalar, TraceType,
};
use crate::vm::jit::trace::{TraceAbortReason, TraceExitKind};
use crate::vm::stack::CallStack;

// ---------------------------------------------------------------- numbers

/// A numeric constant in the program's profile.
fn number(integers: bool, value: i64) -> Constant {
    if integers {
        Constant::Integer(value)
    } else {
        Constant::Number(value as f64)
    }
}

// --------------------------------------------------------------- programs

/// The callee of `branching`: `x > 150 and 0 or x + x`.
fn branching_callee(integers: bool) -> Chunk {
    Chunk::with_parts(
        4,
        1,
        vec![number(integers, 150), number(integers, 0)],
        vec![
            load(1, 0),
            binary(OpCode::LessThan, 2, 1, 0),
            jump_if_false(2, 2, 5),
            load(3, 1),
            Instruction::return_(3),
            binary(OpCode::Add, 3, 0, 0),
            Instruction::return_(3),
        ],
        vec![],
        vec![],
    )
}

/// `for i = 1, limit do acc = acc + f(i) end return acc`, with `f` the
/// closure of prototype 0. Header pc 6, latch pc 8.
fn call_loop(integers: bool, limit: i64, callee: Chunk) -> Chunk {
    Chunk::with_parts(
        7,
        0,
        vec![
            number(integers, 1),
            number(integers, limit),
            number(integers, 0),
        ],
        vec![
            Instruction::closure(4, 0),
            load(5, 2),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 5, 8),
            Instruction::call_one(6, 4, 3),
            binary(OpCode::Add, 5, 5, 6),
            for_loop(0, 8, 6),
            Instruction::return_(5),
        ],
        vec![callee],
        vec![],
    )
}

fn branching(integers: bool, limit: i64) -> Chunk {
    call_loop(integers, limit, branching_callee(integers))
}

/// `f(delta)` adds `delta` to its upvalue, the root's register 6, and
/// returns the new total.
fn counter_callee() -> Chunk {
    Chunk::with_parts(
        2,
        1,
        vec![],
        vec![
            Instruction::get_upvalue(1, 0),
            binary(OpCode::Add, 1, 1, 0),
            Instruction::set_upvalue(1, 0),
            Instruction::return_(1),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: "total".into(),
            source: UpvalueSource::ParentRegister(6),
        }],
    )
}

/// `local total = 0; local f = function(d) total = total + d; return total
/// end; for i = 1, limit do f(i) end return total`. Header pc 6, latch pc 8.
fn counter(integers: bool, limit: i64) -> Chunk {
    Chunk::with_parts(
        7,
        0,
        vec![
            number(integers, 1),
            number(integers, limit),
            number(integers, 0),
        ],
        vec![
            load(6, 2),
            Instruction::closure(4, 0),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 5, 8),
            Instruction::call_one(5, 4, 3),
            Instruction::load_nil(5),
            for_loop(0, 8, 6),
            Instruction::return_(6),
        ],
        vec![counter_callee()],
        vec![],
    )
}

/// The register of closure `k` (1-based) of `chain`.
const fn chain_register(k: usize) -> u8 {
    if k == 1 { 4 } else { (7 + k - 2) as u8 }
}

/// `chain(depth)`: the loop calls `f1(i)`, which calls `f2(x)`, and so on
/// down to `f<depth>`, which returns `x + 1`. Each level adds one, so the
/// loop sums `i + depth`. The closures are all made by the root frame, which
/// keeps `f2` and the others in captured registers: `f<k>` captures the
/// register of `f<k+1>`. Header pc `depth + 3 + ...`, see the returned
/// header and latch.
fn chain(integers: bool, limit: i64, depth: usize) -> (Chunk, usize, usize) {
    assert!(depth >= 1);
    let leaf = Chunk::with_parts(
        2,
        1,
        vec![number(integers, 1)],
        vec![
            load(1, 0),
            binary(OpCode::Add, 1, 0, 1),
            Instruction::return_(1),
        ],
        vec![],
        vec![],
    );
    let mut prototypes = Vec::new();
    for k in 1..depth {
        prototypes.push(Chunk::with_parts(
            3,
            1,
            vec![],
            vec![
                Instruction::get_upvalue(1, 0),
                Instruction::call_one(2, 1, 0),
                Instruction::return_(2),
            ],
            vec![],
            vec![UpvalueDescriptor {
                name: "next".into(),
                source: UpvalueSource::ParentRegister(chain_register(k + 1)),
            }],
        ));
    }
    prototypes.push(leaf);
    let mut code = Vec::new();
    // Innermost first, so every capture finds its closure.
    for k in (1..=depth).rev() {
        code.push(Instruction::closure(
            chain_register(k),
            u16::try_from(k - 1).unwrap(),
        ));
    }
    code.extend([load(5, 2), load(0, 0), load(1, 1), load(2, 0)]);
    let prep = code.len();
    let header = prep + 1;
    let latch = header + 2;
    code.extend([
        for_prep(0, prep, latch),
        Instruction::call_one(6, 4, 3),
        binary(OpCode::Add, 5, 5, 6),
        for_loop(0, latch, header),
        Instruction::return_(5),
    ]);
    let chunk = Chunk::with_parts(
        u8::try_from(7 + depth).unwrap(),
        0,
        vec![
            number(integers, 1),
            number(integers, limit),
            number(integers, 0),
        ],
        code,
        prototypes,
        vec![],
    );
    (chunk, header, latch)
}

// ---------------------------------------------------------------- harness

/// A VM that traces natively and the chunk it runs.
struct World {
    vm: Vm,
    chunk: Arc<Chunk>,
}

impl World {
    /// A world whose traces leave a branch guard inside a call through a
    /// snapshot of the callee's frames, which is what these tests restore
    /// and compare. [`World::restarting`] has the default exits.
    fn new(profile: CompatibilityProfile, chunk: Chunk) -> Self {
        let mut world = Self::restarting(profile, chunk);
        world.vm.jit.trace.set_rebuild_callee_frames(true);
        world
    }

    /// A world with the default exits: a branch guard inside a call that has
    /// stored nothing leaves before the call, and the interpreter makes it.
    fn restarting(profile: CompatibilityProfile, chunk: Chunk) -> Self {
        Self {
            vm: native_vm(profile),
            chunk: Arc::new(chunk),
        }
    }

    fn fresh_frames(&self) -> CallStack {
        frames_for(&self.vm, &self.chunk, None)
    }

    /// Runs the program in this VM and returns its results.
    fn execute(&mut self) -> Vec<Value> {
        let chunk = Arc::clone(&self.chunk);
        self.vm.execute(&chunk).unwrap()
    }
}

/// What one frame holds, with closures reduced to their chunk's shape so two
/// VMs can compare.
#[derive(Clone, Debug, PartialEq)]
struct FrameShot {
    chunk: usize,
    pc: usize,
    last: Option<usize>,
    registers: Vec<Shot>,
    caller: (u8, u8),
}

#[derive(Clone, Debug, PartialEq)]
enum Shot {
    Nil,
    Boolean(bool),
    Integer(i64),
    Float(u64),
    /// A closure, by the number of instructions of its chunk.
    Closure(usize),
}

fn shot_of(vm: &Vm, value: &Value) -> Shot {
    match value {
        Value::Nil => Shot::Nil,
        Value::Boolean(flag) => Shot::Boolean(*flag),
        Value::Integer(integer) => Shot::Integer(*integer),
        Value::Number(number) => Shot::Float(number.to_bits()),
        Value::Closure(handle) => Shot::Closure(
            vm.runtime()
                .closure_call_parts(*handle)
                .map_or(0, |parts| parts.chunk.code.len()),
        ),
        other => panic!("not a program value: {other:?}"),
    }
}

/// Every frame of the stack, bottom first.
fn capture_stack(vm: &Vm, frames: &CallStack) -> Vec<FrameShot> {
    (0..frames.len())
        .map(|index| {
            let frame = frames.at(index);
            FrameShot {
                chunk: frame.chunk.address() as usize,
                pc: frame.pc,
                last: frame.last_interpreted_pc.get(),
                registers: frame
                    .registers
                    .iter()
                    .map(|value| shot_of(vm, value))
                    .collect(),
                caller: match &frame.caller {
                    Some(crate::vm::ReturnTarget::Direct(call)) => (call.base, call.results),
                    _ => (u8::MAX, u8::MAX),
                },
            }
        })
        .collect()
}

/// Whether two stacks are the same to the interpreter. A register nothing
/// reads from the frame's pc on may differ: a trace leaves a placeholder in a
/// dead one. So may a register the call the root frame is stopped at writes
/// when it returns, if nothing read it at the call: the callee's return
/// overwrites it before anything can see it.
fn same_stack(chunk: &Chunk, a: &[FrameShot], b: &[FrameShot]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let root_live = crate::jit_ir::observable_registers(chunk);
    let pending = |shot: &FrameShot, register: usize| {
        let Some(call) = shot
            .last
            .and_then(|pc| chunk.code.get(pc).map(|word| (pc, *word)))
        else {
            return false;
        };
        let (pc, call) = call;
        let (base, results) = match call.opcode() {
            Ok(OpCode::CallOne) => (usize::from(call.a()), 1),
            Ok(OpCode::Call | OpCode::CallOpen) => (usize::from(call.a()), usize::from(call.c())),
            _ => return false,
        };
        (base..base + results).contains(&register)
            && root_live.get(pc).is_some_and(|set| {
                u8::try_from(register)
                    .is_ok_and(|register| !crate::jit_ir::register_set_contains(set, register))
            })
    };
    a.iter().zip(b).enumerate().all(|(depth, (left, right))| {
        if left.chunk != right.chunk
            || left.pc != right.pc
            || left.last != right.last
            || left.caller != right.caller
            || left.registers.len() != right.registers.len()
        {
            return false;
        }
        left.registers
            .iter()
            .zip(&right.registers)
            .enumerate()
            .all(|(register, (x, y))| {
                x == y
                    || (depth == 0
                        && root_live.get(left.pc).is_some_and(|set| {
                            u8::try_from(register).is_ok_and(|register| {
                                !crate::jit_ir::register_set_contains(set, register)
                            })
                        }))
                    || (depth == 0 && a.len() > 1 && pending(left, register))
            })
    })
}

/// Steps `vm` until the root frame is at `header` with `latch` as its last
/// instruction and register 3 (the loop variable) holds `variable`. Returns
/// the stack there.
fn step_to_header(
    vm: &mut Vm,
    frames: &mut CallStack,
    header: usize,
    latch: usize,
    variable: &Value,
) -> bool {
    for _ in 0..2_000_000 {
        if frames.len() == 1 {
            let frame = frames.last().unwrap();
            if frame.pc == header
                && frame.last_interpreted_pc.get() == Some(latch)
                && frame.registers[3] == *variable
            {
                return true;
            }
        }
        match vm.step(frames) {
            Ok(None) => {}
            Ok(Some(_)) | Err(_) => return false,
        }
    }
    false
}

/// A trace installed at a loop header, and a stack to enter it with.
struct Entered {
    index: u32,
    /// The root frame at a header visit, as a fresh stack of the traced VM.
    start: CallStack,
    /// The loop variable of that visit.
    variable: Value,
    /// The IR the installed code was lowered from, as it was at install.
    lowered: TraceIr,
}

impl Entered {
    /// A run writes the root frame's captured locals; each restart of the
    /// root puts the cells back over a stack holding the registers of the
    /// header visit, so there is nothing to reset.
    #[allow(clippy::unused_self)]
    fn reset_cells(&self, _vm: &mut Vm) {}
}

/// Runs the program under the tracer until a trace is installed at `header`,
/// then to the next header visit.
fn install_at(world: &mut World, header: usize, latch: usize) -> Entered {
    let mut frames = world.fresh_frames();
    let mut index = None;
    for _ in 0..2_000_000 {
        match index {
            None => index = world.vm.jit.trace.installed_index_at_pc(header),
            Some(index) if frames.len() == 1 => {
                let frame = frames.last().unwrap();
                if frame.pc == header && frame.last_interpreted_pc.get() == Some(latch) {
                    let lowered = world
                        .vm
                        .jit
                        .trace
                        .installed_lowered_ir(index)
                        .expect("installed")
                        .clone();
                    return Entered {
                        index,
                        variable: frame.registers[3].clone(),
                        start: restart_root(&mut world.vm, &world.chunk, &frames),
                        lowered,
                    };
                }
            }
            Some(_) => {}
        }
        if world.vm.step(&mut frames).unwrap().is_some() {
            break;
        }
    }
    panic!("no trace at pc {header}: {:?}", stats_of(&world.vm));
}

/// The program's stack at the header visit with loop variable `variable`, in
/// an isolated interpreter-only VM. Returns that VM and the stack.
fn isolated_at(
    profile: CompatibilityProfile,
    chunk: &Arc<Chunk>,
    header: usize,
    latch: usize,
    variable: &Value,
) -> (Vm, CallStack) {
    let mut vm = interpreter_vm_with(profile);
    let mut frames = frames_for(&vm, chunk, None);
    assert!(
        step_to_header(&mut vm, &mut frames, header, latch, variable),
        "the isolated interpreter never reaches the header with {variable:?}"
    );
    (vm, frames)
}

/// Copies the root frame of `from` onto a fresh stack of the same VM.
#[allow(unsafe_code)]
fn restart_root(vm: &mut Vm, chunk: &Arc<Chunk>, from: &CallStack) -> CallStack {
    let mut frames = frames_for(vm, chunk, None);
    let source = from.at(0);
    let mut frame = frames.at_mut(0);
    frame.pc = source.pc;
    frame.last_interpreted_pc = source.last_interpreted_pc;
    for (register, value) in frame.registers.iter_mut().zip(source.registers) {
        *register = value.clone();
    }
    // The cells of captured locals stand for the same registers of the new
    // stack, so a trace that shares one with a callee finds the same cell
    // the live run had.
    let anchor = frames.anchor();
    for open in from.open_upvalues() {
        // SAFETY: the slot is a register of the new stack's root frame, and
        // the anchor names its buffer. The tests leave the stack open when
        // they drop it, which leaks the buffer.
        unsafe { vm.runtime_mut().reopen_cell(open.cell, anchor, open.slot) };
        frames.push_open(open.slot, open.cell);
    }
    frames
}

// ---------------------------------------------------------------- recording

/// Records `chunk` under a VM that queues closed traces instead of compiling
/// them, and returns the first trace it recorded.
fn recorded(profile: CompatibilityProfile, chunk: &Chunk) -> TraceIr {
    recorded_with(profile, chunk, true)
}

/// The trace recorded for `chunk`. `rebuild` makes its branch guards inside
/// calls leave through snapshots of the callee's frames.
fn recorded_with(profile: CompatibilityProfile, chunk: &Chunk, rebuild: bool) -> TraceIr {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    vm.jit.trace.set_native_compilation(false);
    vm.jit.trace.set_rebuild_callee_frames(rebuild);
    vm.execute(chunk).unwrap();
    vm.jit
        .trace
        .queued()
        .front()
        .unwrap_or_else(|| panic!("nothing was recorded: {:?}", vm.jit_statistics().trace))
        .ir
        .clone()
}

#[test]
fn a_recorded_call_is_a_pinned_closure_behind_an_identity_guard() {
    for integers in [false, true] {
        let profile = if integers {
            CompatibilityProfile::Lua54
        } else {
            CompatibilityProfile::Lua51
        };
        let ir = recorded(profile, &branching(integers, 300));
        ir.verify().unwrap();
        // The root frame and one inlined call of the pinned closure.
        assert_eq!(ir.frames.len(), 2, "{profile:?}");
        assert_eq!(ir.pinned_closures.len(), 1);
        let call = ir.frames[1].inlined.expect("an inlined frame");
        assert_eq!(call.parent, FrameId(0));
        assert_eq!(call.closure, 0);
        assert_eq!(call.call_pc, 6);
        assert_eq!((call.result_base, call.results), (6, 1));
        // The callee is compared with its pinned lane, and the guard on that
        // comparison comes before anything of the callee runs.
        let compare = ir
            .body
            .iter()
            .position(|instruction| {
                matches!(&instruction.op, TraceOp::Compare { left, .. }
                    if ir.value_type(*left) == Some(TraceType::Closure))
            })
            .expect("a closure comparison");
        assert!(
            ir.body[..compare]
                .iter()
                .all(|instruction| instruction.site.frame == FrameId(0))
        );
        assert!(ir.body.iter().any(|instruction| {
            matches!(instruction.op, TraceOp::Constant(TraceScalar::Closure(0)))
        }));
        let callee_guard = &ir.guards[0];
        assert_eq!(callee_guard.site.pc, 6);
        assert_eq!(callee_guard.site.frame, FrameId(0));
        let snapshot = &ir.snapshots[callee_guard.snapshot.index()];
        assert_eq!(snapshot.frames.len(), 1);
        // A guard inside the callee describes both frames, the root stopped
        // at its call.
        let inner = ir
            .snapshots
            .iter()
            .find(|snapshot| snapshot.frames.len() == 2)
            .expect("a snapshot inside the callee");
        assert_eq!(inner.frames[0].frame, FrameId(0));
        assert_eq!(inner.frames[0].last_interpreted_pc, Some(6));
        assert_eq!(inner.frames[1].frame, FrameId(1));
        assert_eq!(
            inner.frames[1].registers.len(),
            usize::from(ir.frames[1].register_count)
        );
    }
}

#[test]
fn a_branch_guard_inside_a_callee_that_stored_nothing_leaves_before_the_call() {
    for integers in [false, true] {
        let profile = if integers {
            CompatibilityProfile::Lua54
        } else {
            CompatibilityProfile::Lua51
        };
        let ir = recorded_with(profile, &branching(integers, 300), false);
        ir.verify().unwrap();
        // The callee's branch guard names the root frame at the call, like
        // the guard on the callee itself. No snapshot has two frames.
        let branch = ir
            .guards
            .iter()
            .find(|guard| guard.site.frame == FrameId(1))
            .expect("a guard inside the callee");
        let snapshot = &ir.snapshots[branch.snapshot.index()];
        assert_eq!(snapshot.frames.len(), 1, "{profile:?}");
        assert_eq!(snapshot.phase, SnapshotPhase::Before { pc: 6 });
        assert!(
            ir.snapshots
                .iter()
                .all(|snapshot| snapshot.frames.len() == 1)
        );
    }
}

#[test]
fn a_branch_guard_inside_a_callee_starts_the_call_over_in_the_interpreter() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let integers = profile.is_lua53_or_later();
        let mut world = World::restarting(profile, branching(integers, 400));
        let entered = install_at(&mut world, 6, 8);
        entered.reset_cells(&mut world.vm);
        let mut frames = restart_root(&mut world.vm, &world.chunk, &entered.start);
        let ran = world
            .vm
            .enter_native_trace_with_fuel(&mut frames, 0, entered.index, 1_000_000)
            .unwrap();
        assert_eq!(ran, Some(false), "{profile:?}");
        // The trace ran until the callee's branch went the other way, at the
        // call with the argument 151. The root frame is stopped before that
        // call, and the callee has no frame yet.
        let actual = capture_stack(&world.vm, &frames);
        assert_eq!(actual.len(), 1, "{profile:?}: the root alone");
        assert_eq!(actual[0].pc, 6, "{profile:?}");
        assert_eq!(
            actual[0].registers[3],
            if integers {
                Shot::Integer(151)
            } else {
                Shot::Float(151.0_f64.to_bits())
            }
        );
        let (mut isolated, mut isolated_frames) =
            isolated_at(profile, &world.chunk, 6, 8, &entered.variable);
        let mut reached = false;
        for _ in 0..4_000 {
            let shot = capture_stack(&isolated, &isolated_frames);
            if same_stack(&world.chunk, &shot, &actual) {
                reached = true;
                break;
            }
            if isolated.step(&mut isolated_frames).unwrap().is_some() {
                break;
            }
        }
        assert!(
            reached,
            "{profile:?}: the interpreter never passes through {actual:#?}"
        );
        // The interpreter makes the call and finishes the loop as it would
        // have.
        let mut results = None;
        for _ in 0..100_000 {
            if let Some(done) = world.vm.step(&mut frames).unwrap() {
                results = Some(done);
                break;
            }
        }
        assert_eq!(
            results.expect("the loop finishes"),
            interpreted(profile, &branching(integers, 400))
        );
    }
}

#[test]
fn native_inlined_traces_compute_what_the_interpreter_computes() {
    for (integers, profile) in [
        (false, CompatibilityProfile::Lua51),
        (false, CompatibilityProfile::LuaJit21),
        (true, CompatibilityProfile::Lua53),
        (true, CompatibilityProfile::Lua54),
        (true, CompatibilityProfile::Lua55),
    ] {
        for (name, chunk) in [
            ("branching", branching(integers, 600)),
            ("counter", counter(integers, 600)),
            ("chain 3", chain(integers, 600, 3).0),
        ] {
            let expected = interpreted(profile, &chunk);
            let mut world = World::new(profile, chunk);
            let actual = world.execute();
            assert_eq!(actual, expected, "{name} {profile:?}");
            let stats = stats_of(&world.vm);
            assert!(stats.inlined_calls >= 1, "{name} {profile:?}: {stats:?}");
            assert!(stats.iterations > 100, "{name} {profile:?}: {stats:?}");
        }
    }
}

#[test]
fn a_hot_call_loop_runs_almost_entirely_in_native_code() {
    let mut world = World::new(
        CompatibilityProfile::Lua51,
        call_loop(
            false,
            200_000,
            Chunk::with_parts(
                2,
                1,
                vec![Constant::Number(1.0)],
                vec![
                    load(1, 0),
                    binary(OpCode::Add, 1, 0, 1),
                    Instruction::return_(1),
                ],
                vec![],
                vec![],
            ),
        ),
    );
    // sum of i + 1 for i in 1..=200000
    let results = world.execute();
    assert_eq!(results, vec![Value::Number(20_000_300_000.0)]);
    let stats = stats_of(&world.vm);
    assert_eq!(stats.native_entries, 1, "{stats:?}");
    assert!(stats.iterations > 199_900, "{stats:?}");
    assert_eq!(stats.exits_of(TraceExitKind::LoopComplete), 1, "{stats:?}");
    assert_eq!(stats.inlined_calls, 1, "{stats:?}");
}

// --------------------------------------------------------------- recovery

/// Enters the trace at `index` on a copy of the root frame of `state` with
/// every guard forced in turn, and compares the stack each exit leaves with
/// what the reference evaluator restores and with a stack the isolated
/// interpreter passes through.
fn every_guard_restores_the_interpreter_stack(
    profile: CompatibilityProfile,
    chunk: Chunk,
    header: usize,
    latch: usize,
    promote: bool,
) -> usize {
    let mut world = World::new(profile, chunk);
    if !promote {
        world
            .vm
            .jit
            .trace
            .set_compile_tier(crate::jit::OptimizationTier::Baseline);
    }
    let entered = install_at(&mut world, header, latch);
    let Entered {
        index,
        start,
        variable,
        lowered,
        ..
    } = &entered;
    let (index, variable) = (*index, variable.clone());
    let mut lowered = lowered.clone();
    if promote {
        // Run long enough to ask for optimized code, and let it install. The
        // optimizer hoists the callee guards into a preheader, where a
        // failure restores the entry state.
        entered.reset_cells(&mut world.vm);
        let mut frames = restart_root(&mut world.vm, &world.chunk, start);
        world
            .vm
            .enter_native_trace_with_fuel(&mut frames, 0, index, 100)
            .unwrap();
        world.vm.jit.trace.forgive(index);
        world.vm.service_trace_compilations();
        assert_eq!(
            world.vm.jit.trace.installed_tier(index),
            Some(crate::jit::OptimizationTier::Optimized),
            "{:?}",
            stats_of(&world.vm)
        );
        lowered = world
            .vm
            .jit
            .trace
            .installed_lowered_ir(index)
            .expect("installed")
            .clone();
    }
    let guards = lowered.guards.len();
    let mut multi_frame_exits = 0;
    for guard in 0..guards {
        // Recompile the trace so this guard always exits. Only branch guards
        // can be forced.
        if !world.vm.jit.trace.force_guard(index, guard) {
            continue;
        }
        entered.reset_cells(&mut world.vm);
        let mut frames = restart_root(&mut world.vm, &world.chunk, start);
        let ran = world
            .vm
            .enter_native_trace_with_fuel(&mut frames, 0, index, 1_000)
            .unwrap();
        assert!(ran.is_some(), "guard {guard}: the forced trace did not run");
        world.vm.jit.trace.forgive(index);
        let actual = capture_stack(&world.vm, &frames);
        // What the reference evaluator restores for the same forced exit.
        let site = lowered.guards[guard].site;
        let registers: Vec<TraceScalar> = {
            let root = restart_root(&mut world.vm, &world.chunk, start);
            let frame = root.at(0);
            frame
                .registers
                .iter()
                .map(|value| match value {
                    Value::Nil => TraceScalar::Nil,
                    Value::Boolean(flag) => TraceScalar::Boolean(*flag),
                    Value::Integer(integer) => TraceScalar::Integer(*integer),
                    Value::Number(number) => TraceScalar::Float(*number),
                    // A pinned closure is the lane of its place in the list.
                    Value::Closure(handle) => TraceScalar::Closure(
                        lowered
                            .pinned_closures
                            .iter()
                            .position(|identity| *identity == handle.native_identity())
                            .map_or(u64::MAX, |lane| lane as u64),
                    ),
                    other => panic!("{other:?}"),
                })
                .collect()
        };
        let evaluation = reference::evaluate_forced(
            &lowered,
            &registers,
            &[],
            1_000,
            Some(ForcedExit {
                frame: site.frame,
                pc: site.pc,
                guard: Some(GuardId(u32::try_from(guard).unwrap())),
                iteration: 0,
            }),
            |_, _| None,
        );
        let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
            panic!(
                "guard {guard}: the reference did not fail it: {:?}",
                evaluation.outcome
            );
        };
        let top = actual.last().unwrap();
        assert_eq!(top.pc, restored.pc, "guard {guard}");
        assert_eq!(top.last, restored.last_interpreted_pc, "guard {guard}");
        assert_eq!(
            actual.len(),
            restored.outer.len() + 1,
            "guard {guard}: frame count; snapshot {:?}; lowered guards {:?}",
            lowered.snapshots[lowered.guards[guard].snapshot.index()]
                .frames
                .iter()
                .map(|frame| frame.frame)
                .collect::<Vec<_>>(),
            lowered.guards
        );
        for (frame, outer) in actual.iter().zip(&restored.outer) {
            assert_eq!(frame.last, outer.last_interpreted_pc, "guard {guard}");
            assert_eq!(frame.pc, outer.last_interpreted_pc.unwrap() + 1);
        }
        if actual.len() > 1 {
            multi_frame_exits += 1;
        }
        // A `Before` snapshot describes a stack the isolated interpreter
        // passes through whatever the data. An `After` snapshot of a branch
        // describes the other successor, which only a guard that fails
        // naturally leaves in a consistent state, so those are compared with
        // the reference only (`natural_exits_*` takes them on real data).
        let phase = lowered.snapshots[lowered.guards[guard].snapshot.index()].phase;
        if !matches!(phase, SnapshotPhase::Before { .. }) {
            continue;
        }
        let (mut isolated, mut isolated_frames) =
            isolated_at(profile, &world.chunk, header, latch, &variable);
        let mut reached = false;
        for _ in 0..200 {
            let shot = capture_stack(&isolated, &isolated_frames);
            if same_stack(&world.chunk, &shot, &actual) {
                reached = true;
                break;
            }
            if isolated.step(&mut isolated_frames).unwrap().is_some() {
                break;
            }
        }
        assert!(
            reached,
            "guard {guard}: the interpreter never passes through {actual:#?}"
        );
    }
    multi_frame_exits
}

#[test]
fn every_forced_guard_restores_a_stack_the_interpreter_passes_through() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let integers = profile.is_lua53_or_later();
        for promote in [false, true] {
            let (chain_chunk, chain_header, chain_latch) = chain(integers, 400, 3);
            for (name, chunk, header, latch) in [
                ("branching", branching(integers, 400), 6, 8),
                ("counter", counter(integers, 400), 6, 8),
                ("chain", chain_chunk, chain_header, chain_latch),
            ] {
                let multi = every_guard_restores_the_interpreter_stack(
                    profile, chunk, header, latch, promote,
                );
                // The guards inside the callees leave two or more frames. The
                // counter's callee has no guard of its own, and the optimizer
                // hoists the callee guards of the chain out of the loop.
                if name == "branching" || (name == "chain" && !promote) {
                    assert!(
                        multi >= 1,
                        "{name} {profile:?} promoted {promote}: no exit inside a callee"
                    );
                }
            }
        }
    }
}

/// Programs the call tests install a trace for: name, chunk, header, latch.
fn call_programs(integers: bool) -> Vec<(&'static str, Chunk, usize, usize)> {
    let (chain_chunk, chain_header, chain_latch) = chain(integers, 400, 3);
    vec![
        ("branching", branching(integers, 400), 6, 8),
        ("counter", counter(integers, 400), 6, 8),
        ("chain", chain_chunk, chain_header, chain_latch),
    ]
}

#[test]
fn the_fast_restore_plan_matches_the_generic_restore_of_every_inlined_exit() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        for (name, chunk, header, latch) in call_programs(profile.is_lua53_or_later()) {
            let mut world = World::new(profile, chunk);
            // Optimized code hoists the callee guards out of the callees.
            world
                .vm
                .jit
                .trace
                .set_compile_tier(crate::jit::OptimizationTier::Baseline);
            let entered = install_at(&mut world, header, latch);
            let index = entered.index;
            let pinned: Vec<Value> = entered
                .lowered
                .pinned_closures
                .iter()
                .map(|identity| {
                    Value::Closure(crate::runtime::ClosureHandle::from_native_identity(
                        *identity,
                    ))
                })
                .collect();
            let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
            let mut inner_exits = 0;
            for exit in 0..world.vm.jit.trace.exit_count(index) {
                let descriptor = world
                    .vm
                    .jit
                    .trace
                    .exit_descriptor(index, exit)
                    .unwrap()
                    .clone();
                for round in 0..4 {
                    // Arbitrary bits in every slot, except lanes: a table or
                    // closure slot holds an index of the bank.
                    let mut area: Vec<i64> = (0..64)
                        .map(|_| {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            if round == 0 { 0 } else { seed.cast_signed() }
                        })
                        .collect();
                    for slot in &descriptor.slots {
                        if matches!(slot.ty, TraceType::Closure | TraceType::Table) {
                            area[slot.slot as usize] = (round % pinned.len().max(1)) as i64;
                        }
                    }
                    *world.vm.jit.trace.bank_mut() = pinned.clone();
                    entered.reset_cells(&mut world.vm);
                    let mut planned = restart_root(&mut world.vm, &world.chunk, &entered.start);
                    let mut generic = restart_root(&mut world.vm, &world.chunk, &entered.start);
                    world.vm.jit.trace.plan_restore_stack(
                        index,
                        exit,
                        &area,
                        &mut planned,
                        &mut world.vm.runtime,
                    );
                    let plan_writes = world.vm.jit.trace.plan_upvalues(index, exit, &area);
                    let generic_writes = world
                        .vm
                        .jit
                        .trace
                        .generic_restore_stack(
                            index,
                            exit,
                            &area,
                            &mut generic,
                            &mut world.vm.runtime,
                        )
                        .expect("the generic restore succeeds");
                    assert_eq!(
                        format!("{plan_writes:?}"),
                        format!("{generic_writes:?}"),
                        "{name} {profile:?} exit {exit} round {round}: cell writes"
                    );
                    assert_eq!(
                        capture_bits_stack(&world.vm, &planned),
                        capture_bits_stack(&world.vm, &generic),
                        "{name} {profile:?} exit {exit} round {round}"
                    );
                    if planned.len() > 1 {
                        inner_exits += 1;
                    }
                }
            }
            world.vm.jit.trace.bank_mut().clear();
            if name != "counter" {
                assert!(
                    inner_exits > 0,
                    "{name} {profile:?}: no exit inside a callee"
                );
            }
        }
    }
}

/// [`capture_stack`] with `{:?}` of every register, which keeps NaN bits.
fn capture_bits_stack(vm: &Vm, frames: &CallStack) -> Vec<String> {
    capture_stack(vm, frames)
        .into_iter()
        .map(|frame| format!("{frame:?}"))
        .collect()
}

#[test]
fn a_guard_that_fails_inside_the_callee_leaves_the_stack_the_interpreter_has() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let integers = profile.is_lua53_or_later();
        let mut world = World::new(profile, branching(integers, 400));
        let entered = install_at(&mut world, 6, 8);
        entered.reset_cells(&mut world.vm);
        let mut frames = restart_root(&mut world.vm, &world.chunk, &entered.start);
        let before = stats_of(&world.vm);
        let ran = world
            .vm
            .enter_native_trace_with_fuel(&mut frames, 0, entered.index, 1_000_000)
            .unwrap();
        assert_eq!(ran, Some(false), "{profile:?}");
        let stats = stats_of(&world.vm);
        // The trace ran until the callee's branch went the other way, at the
        // call with the argument 151.
        assert_eq!(
            stats.exits_of(TraceExitKind::GuardFail),
            before.exits_of(TraceExitKind::GuardFail) + 1,
            "{profile:?}"
        );
        let actual = capture_stack(&world.vm, &frames);
        assert_eq!(actual.len(), 2, "{profile:?}: root and callee");
        assert_eq!(
            actual[1].registers[0],
            if integers {
                Shot::Integer(151)
            } else {
                Shot::Float(151.0_f64.to_bits())
            }
        );
        let (mut isolated, mut isolated_frames) =
            isolated_at(profile, &world.chunk, 6, 8, &entered.variable);
        let mut reached = false;
        for _ in 0..4_000 {
            let shot = capture_stack(&isolated, &isolated_frames);
            if same_stack(&world.chunk, &shot, &actual) {
                reached = true;
                break;
            }
            if isolated.step(&mut isolated_frames).unwrap().is_some() {
                break;
            }
        }
        assert!(
            reached,
            "{profile:?}: the interpreter never passes through {actual:#?}"
        );
        // The interpreter finishes the call and the loop as it would have.
        let mut results = None;
        for _ in 0..100_000 {
            if let Some(done) = world.vm.step(&mut frames).unwrap() {
                results = Some(done);
                break;
            }
        }
        assert_eq!(
            results.expect("the loop finishes"),
            interpreted(profile, &branching(integers, 400))
        );
    }
}

// ------------------------------------------------- results and arguments

/// `f(a, b)` returns `a` and `b`, called with one argument and keeping three
/// results, so `b` is nil and the third result is missing. The loop adds the
/// first result and, for each nil result, one thousand.
/// Header pc 7, latch pc 17.
fn truncation(integers: bool, limit: i64) -> Chunk {
    let callee = Chunk::with_parts(
        2,
        2,
        vec![],
        vec![Instruction::return_values(0, 2)],
        vec![],
        vec![],
    );
    Chunk::with_parts(
        11,
        0,
        vec![
            number(integers, 1),
            number(integers, limit),
            number(integers, 0),
            number(integers, 1000),
        ],
        vec![
            Instruction::closure(4, 0),
            load(5, 2),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(9, 3),
            for_prep(0, 6, 17),
            // header
            Instruction::move_(6, 4),
            Instruction::move_(7, 3),
            Instruction::call(6, 1, 3),
            binary(OpCode::Add, 5, 5, 6),
            Instruction::unary(OpCode::Not, 10, 7),
            jump_if_false(10, 11, 13),
            binary(OpCode::Add, 5, 5, 9),
            Instruction::unary(OpCode::Not, 10, 8),
            jump_if_false(10, 14, 17),
            binary(OpCode::Add, 5, 5, 9),
            for_loop(0, 17, 7),
            Instruction::return_(5),
        ],
        vec![callee],
        vec![],
    )
}

#[test]
fn missing_arguments_and_extra_results_normalise_as_the_interpreter_does() {
    for (integers, profile) in [
        (false, CompatibilityProfile::Lua51),
        (true, CompatibilityProfile::Lua54),
    ] {
        let chunk = truncation(integers, 500);
        let expected = interpreted(profile, &chunk);
        let mut world = World::new(profile, chunk);
        let actual = world.execute();
        assert_eq!(actual, expected, "{profile:?}");
        let stats = stats_of(&world.vm);
        assert!(
            stats.inlined_calls >= 1 && stats.iterations > 300,
            "{profile:?}: {stats:?}"
        );
    }
}

// --------------------------------------------------------------- rebinding

/// `if i > 200 then f = g end; acc = acc + f(i)`: the loop rebinds its callee
/// from `a` (adds `i`) to `b` (adds `2 i`) once `i` passes 200. Header pc 6,
/// latch pc 12. The branch is the first guard of the trace and the callee guard
/// the second.
fn rebinding(integers: bool, limit: i64) -> Chunk {
    let adds_once = Chunk::with_parts(
        2,
        1,
        vec![],
        vec![binary(OpCode::Add, 1, 0, 0), Instruction::return_(1)],
        vec![],
        vec![],
    );
    let adds_twice = Chunk::with_parts(
        2,
        1,
        vec![],
        vec![
            binary(OpCode::Add, 1, 0, 0),
            binary(OpCode::Add, 1, 1, 0),
            Instruction::return_(1),
        ],
        vec![],
        vec![],
    );
    Chunk::with_parts(
        10,
        0,
        vec![
            number(integers, 1),
            number(integers, limit),
            number(integers, 0),
            number(integers, 200),
        ],
        vec![
            Instruction::closure(4, 0),
            Instruction::closure(7, 1),
            load(5, 2),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            // `for_prep` at pc 6, so the header is pc 7.
            for_prep(0, 6, 13),
            load(9, 3),
            binary(OpCode::LessThan, 8, 9, 3),
            jump_if_false(8, 9, 11),
            Instruction::move_(4, 7),
            Instruction::call_one(6, 4, 3),
            binary(OpCode::Add, 5, 5, 6),
            for_loop(0, 13, 7),
            Instruction::return_(5),
        ],
        vec![adds_once, adds_twice],
        vec![],
    )
}

#[test]
fn a_rebound_callee_leaves_the_trace_and_the_new_callee_runs_in_the_interpreter() {
    for (integers, profile) in [
        (false, CompatibilityProfile::Lua51),
        (true, CompatibilityProfile::Lua54),
    ] {
        let chunk = rebinding(integers, 3_000);
        let expected = interpreted(profile, &chunk);
        let mut world = World::new(profile, chunk);
        let actual = world.execute();
        assert_eq!(actual, expected, "{profile:?}");
        let stats = stats_of(&world.vm);
        assert!(stats.inlined_calls >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations >= 100, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_trace_pinned_to_a_callee_the_site_no_longer_calls_retires_after_a_few_entries() {
    // The callee guard here is hoisted into the preheader, so no side trace
    // can continue from it: with none to wait for, the misses retire the
    // trace (see `a_call_site_that_takes_turns_between_callees_gets_a_side_trace`
    // for the site that does get one).
    let profile = CompatibilityProfile::Lua51;
    let mut world = World::new(profile, rebinding(false, 3_000));
    let entered = install_at(&mut world, 7, 13);
    assert!(world.vm.jit.trace.installed_index_at_pc(7).is_some());
    for entry in 0..crate::vm::jit::trace::native::TRACE_CALLEE_STREAK_LIMIT {
        assert!(
            world.vm.jit.trace.installed_index_at_pc(7).is_some(),
            "retired early, at entry {entry}"
        );
        let mut frames = restart_root(&mut world.vm, &world.chunk, &entered.start);
        {
            let frame = frames.at_mut(0);
            let other = frame.registers[7].clone();
            frame.registers[4] = other;
        }
        world
            .vm
            .enter_native_trace_with_fuel(&mut frames, 0, entered.index, 1_000)
            .unwrap();
    }
    assert!(
        world.vm.jit.trace.installed_index_at_pc(7).is_none(),
        "the trace stayed installed"
    );
}

#[test]
fn a_callee_guard_failure_is_counted_as_one() {
    // The loop below enters the trace with `f` rebound: the hoisted callee
    // guard fails before the first iteration.
    let profile = CompatibilityProfile::Lua51;
    let mut world = World::new(profile, rebinding(false, 3_000));
    world
        .vm
        .jit
        .trace
        .set_compile_tier(crate::jit::OptimizationTier::Baseline);
    let entered = install_at(&mut world, 7, 13);
    // Rebind the closure register to the other closure of the program, then
    // enter. `Move 4 <- 7` is what the program does at i = 201.
    let mut frames = restart_root(&mut world.vm, &world.chunk, &entered.start);
    {
        let frame = frames.at_mut(0);
        let other = frame.registers[7].clone();
        assert!(matches!(other, Value::Closure(_)));
        frame.registers[4] = other;
    }
    let before = stats_of(&world.vm);
    let ran = world
        .vm
        .enter_native_trace_with_fuel(&mut frames, 0, entered.index, 1_000)
        .unwrap();
    assert_eq!(ran, Some(false));
    let after = stats_of(&world.vm);
    assert_eq!(
        after.side_exits_of(crate::vm::jit::trace::TraceGuardKind::Callee),
        before.side_exits_of(crate::vm::jit::trace::TraceGuardKind::Callee) + 1,
        "{after:?}"
    );
    // It left before completing an iteration.
    assert_eq!(after.iterations, before.iterations);
    // The baseline trace has nothing hoisted, so the guard fails at the call
    // after the iteration's earlier instructions ran.
    let frame = frames.last().unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frame.pc, 11);
    assert_eq!(frame.last_interpreted_pc.get(), Some(9));
}

// ------------------------------------------------------- recording limits

/// Runs `chunk` under the tracer and returns the statistics, requiring that
/// no trace was recorded.
fn refused(profile: CompatibilityProfile, chunk: &Chunk) -> crate::vm::jit::trace::TraceStatistics {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    vm.jit.trace.set_native_compilation(false);
    let expected = interpreted(profile, chunk);
    assert_eq!(vm.execute(chunk).unwrap(), expected);
    let stats = vm.jit_statistics().trace;
    assert_eq!(stats.traces_recorded, 0, "{stats:?}");
    stats
}

#[test]
fn calls_the_recorder_cannot_follow_abort_the_recording() {
    let profile = CompatibilityProfile::Lua51;
    // Nested calls deeper than a snapshot can describe.
    let (deep, _, _) = chain(false, 300, MAX_TRACE_FRAME_DEPTH + 1);
    let stats = refused(profile, &deep);
    assert!(
        stats.aborts(TraceAbortReason::LimitExceeded) >= 1,
        "{stats:?}"
    );
    // The deepest chain that fits records.
    let (fits, _, _) = chain(false, 300, MAX_TRACE_FRAME_DEPTH - 1);
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    vm.jit.trace.set_native_compilation(false);
    vm.execute(&fits).unwrap();
    let ir = vm.jit.trace.queued().front().expect("recorded").ir.clone();
    assert_eq!(ir.frames.len(), MAX_TRACE_FRAME_DEPTH);

    // A callee that calls itself is recursion: `f(x)` is `x > 0 and f(x - 1)
    // or 0`, and the loop calls `f(3)`.
    let recursive = Chunk::with_parts(
        5,
        1,
        vec![number(false, 0), number(false, 1)],
        vec![
            load(1, 0),
            binary(OpCode::LessThan, 2, 1, 0),
            jump_if_false(2, 2, 8),
            load(3, 1),
            binary(OpCode::Subtract, 3, 0, 3),
            Instruction::get_upvalue(4, 0),
            Instruction::call_one(2, 4, 3),
            Instruction::return_(2),
            Instruction::return_(1),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: "f".into(),
            source: UpvalueSource::ParentRegister(4),
        }],
    );
    let root = Chunk::with_parts(
        9,
        0,
        vec![
            number(false, 1),
            number(false, 300),
            number(false, 0),
            number(false, 3),
        ],
        vec![
            Instruction::closure(4, 0),
            load(5, 2),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 5, 9),
            load(8, 3),
            Instruction::call_one(6, 4, 8),
            binary(OpCode::Add, 5, 5, 6),
            for_loop(0, 9, 6),
            Instruction::return_(5),
        ],
        vec![recursive],
        vec![],
    );
    // The loop's recording refuses the recursion. `f`'s own self call makes
    // its entry a function-entry site, whose trace may record: that is the
    // only trace there is.
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    vm.jit.trace.set_native_compilation(false);
    let expected = interpreted(profile, &root);
    assert_eq!(vm.execute(&root).unwrap(), expected);
    let stats = vm.jit_statistics().trace;
    assert!(
        vm.jit
            .trace
            .queued()
            .iter()
            .all(|queued| queued.ir.function.is_some()),
        "{stats:?}"
    );
    assert!(
        stats.aborts(TraceAbortReason::UnsupportedOperand) >= 1,
        "{stats:?}"
    );
    // Recursion is refused for what it is, not by running into the depth
    // bound.
    assert_eq!(
        stats.aborts(TraceAbortReason::LimitExceeded),
        0,
        "{stats:?}"
    );
}

// ------------------------------------------------- rotating call sites

/// `for i = 1, limit do acc = acc + f(i); f, g = g, f end return acc`, with
/// `f` the closure of prototype 0 (`x + x`) and `g` that of prototype 1
/// (`x + x + x`), so the call site takes turns between two callees. Header
/// pc 7, latch pc 12.
fn two_callees(integers: bool, limit: i64) -> Chunk {
    let twice = Chunk::with_parts(
        2,
        1,
        vec![],
        vec![binary(OpCode::Add, 1, 0, 0), Instruction::return_(1)],
        vec![],
        vec![],
    );
    let thrice = Chunk::with_parts(
        2,
        1,
        vec![],
        vec![
            binary(OpCode::Add, 1, 0, 0),
            binary(OpCode::Add, 1, 1, 0),
            Instruction::return_(1),
        ],
        vec![],
        vec![],
    );
    Chunk::with_parts(
        10,
        0,
        vec![
            number(integers, 1),
            number(integers, limit),
            number(integers, 0),
        ],
        vec![
            Instruction::closure(4, 0),
            Instruction::closure(7, 1),
            load(5, 2),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 6, 12),
            Instruction::call_one(6, 4, 3),
            binary(OpCode::Add, 5, 5, 6),
            Instruction::move_(8, 4),
            Instruction::move_(4, 7),
            Instruction::move_(7, 8),
            for_loop(0, 12, 7),
            Instruction::return_(5),
        ],
        vec![twice, thrice],
        vec![],
    )
}

#[test]
fn a_call_site_that_takes_turns_between_callees_gets_a_side_trace() {
    // The trace pins the callee it recorded, so the other callee fails the
    // identity guard every second iteration. Those misses count toward the
    // side threshold, and the trace is not retired while a side can still be
    // recorded: the side pins the other callee and the loop stays native.
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let integers = profile.is_lua53_or_later();
        let chunk = two_callees(integers, 60_000);
        let expected = interpreted(profile, &chunk);
        let mut vm = native_vm(profile);
        assert_eq!(vm.execute(&chunk).unwrap(), expected, "{profile:?}");
        let stats = stats_of(&vm);
        assert!(stats.side_recordings >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_installs, 1, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 20_000, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 59_000, "{profile:?}: {stats:?}");
        assert!(
            vm.jit.trace.installed_index_at_pc(7).is_some(),
            "{profile:?}: the tree was retired"
        );
    }
}

/// `make(k)` returns `function(x) count = count + 1; return x + k + count
/// end` over its own `count`, and the loop calls two such closures in turn
/// (they swap places every iteration): one prototype, two sets of cells.
/// Header pc 10, latch pc 15.
fn family(integers: bool, limit: i64) -> Chunk {
    let callee = Chunk::with_parts(
        4,
        1,
        vec![number(integers, 1)],
        vec![
            Instruction::get_upvalue(1, 0),
            load(2, 0),
            binary(OpCode::Add, 1, 1, 2),
            Instruction::set_upvalue(1, 0),
            Instruction::get_upvalue(2, 1),
            binary(OpCode::Add, 2, 0, 2),
            binary(OpCode::Add, 2, 2, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![
            UpvalueDescriptor {
                name: "count".into(),
                source: UpvalueSource::ParentRegister(1),
            },
            UpvalueDescriptor {
                name: "k".into(),
                source: UpvalueSource::ParentRegister(0),
            },
        ],
    );
    let make = Chunk::with_parts(
        3,
        1,
        vec![number(integers, 0)],
        vec![
            load(1, 0),
            Instruction::closure(2, 0),
            Instruction::return_(2),
        ],
        vec![callee],
        vec![],
    );
    Chunk::with_parts(
        11,
        0,
        vec![
            number(integers, 1),
            number(integers, limit),
            number(integers, 0),
            number(integers, 2),
        ],
        vec![
            Instruction::closure(9, 0),
            load(10, 0),
            Instruction::call_one(4, 9, 10),
            load(10, 3),
            Instruction::call_one(7, 9, 10),
            load(5, 2),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 9, 15),
            Instruction::call_one(6, 4, 3),
            binary(OpCode::Add, 5, 5, 6),
            Instruction::move_(8, 4),
            Instruction::move_(4, 7),
            Instruction::move_(7, 8),
            for_loop(0, 15, 10),
            Instruction::return_(5),
        ],
        vec![make],
        vec![],
    )
}

#[test]
fn closures_of_one_prototype_share_one_trace_with_a_prototype_guard() {
    // The callee takes turns between two closures of one prototype, each with
    // cells of its own. The call site is guarded on the prototype, and the
    // callee's cells are read and written through the live closure, never as
    // window slots (a cell as a window slot would be one closure's).
    for integers in [false, true] {
        let profile = if integers {
            CompatibilityProfile::Lua54
        } else {
            CompatibilityProfile::Lua51
        };
        let ir = recorded(profile, &family(integers, 4_000));
        ir.verify().unwrap();
        assert!(ir.external.is_empty(), "{profile:?}: {:?}", ir.external);
        let call = ir.frames[1].inlined.expect("an inlined frame");
        assert!(call.dynamic, "{profile:?}");
        // The callee's registers and, after them, the closure of this run.
        assert_eq!(ir.frames[1].register_count, 5, "{profile:?}");
        let helpers: Vec<HelperId> = ir
            .preheader
            .iter()
            .chain(&ir.body)
            .filter_map(|instruction| match &instruction.op {
                TraceOp::CallHelper { helper, .. } => Some(*helper),
                _ => None,
            })
            .collect();
        let number_read = if integers {
            HelperId::CELL_GET_INTEGER
        } else {
            HelperId::CELL_GET_FLOAT
        };
        assert_eq!(
            helpers
                .iter()
                .filter(|helper| **helper == HelperId::CLOSURE_SAME_PROTO)
                .count(),
            1,
            "{profile:?}: {helpers:?}"
        );
        assert_eq!(
            helpers
                .iter()
                .filter(|helper| **helper == number_read)
                .count(),
            2,
            "{profile:?}: {helpers:?}"
        );
        assert_eq!(
            helpers
                .iter()
                .filter(|helper| **helper == HelperId::CELL_SET)
                .count(),
            1,
            "{profile:?}: {helpers:?}"
        );
        // No guard compares closures by identity: a rotating callee would
        // fail it every iteration.
        assert!(!ir.body.iter().any(|instruction| {
            matches!(&instruction.op, TraceOp::Compare { left, .. }
                if ir.value_type(*left) == Some(TraceType::Closure))
        }));
        // A snapshot inside the callee names the closure of the run last.
        let inner = ir
            .snapshots
            .iter()
            .find(|snapshot| snapshot.frames.len() == 2)
            .expect("a snapshot inside the callee");
        let registers = &inner.frames[1].registers;
        assert_eq!(registers.len(), 5);
        let crate::jit_ir::trace::RegisterEntry::Value(closure) = registers[4] else {
            panic!("the closure of the run is a value: {registers:?}");
        };
        assert_eq!(ir.value_type(closure), Some(TraceType::Closure));
    }
}

#[test]
fn closures_of_one_prototype_run_in_one_native_trace_and_match_the_interpreter() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let integers = profile.is_lua53_or_later();
        let chunk = family(integers, 60_000);
        let expected = interpreted(profile, &chunk);
        let mut vm = native_vm(profile);
        assert_eq!(vm.execute(&chunk).unwrap(), expected, "{profile:?}");
        let stats = stats_of(&vm);
        // One trace, no side: the prototype guard never fails, so the call
        // site needs no turn-taking.
        assert_eq!(stats.traces_compiled, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_installs, 0, "{profile:?}: {stats:?}");
        assert_eq!(
            stats.aborts(TraceAbortReason::UnsupportedOperand),
            0,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.iterations > 59_000, "{profile:?}: {stats:?}");
        // The rows of the closures answer every prototype check and cell
        // access: nothing reached the VM's helpers.
        assert_eq!(stats.closure_helper_calls, 0, "{profile:?}: {stats:?}");
        assert!(
            vm.jit.trace.installed_index_at_pc(10).is_some(),
            "{profile:?}: the trace was retired"
        );
    }
}

/// Three callees of three prototypes, each adding to a counter of its own
/// that is a captured local of the root frame (an open cell), taking turns at
/// one call site. After the loop the root adds the counters to its result, so
/// a trace that left a write to an open cell undone shows. Header pc 11,
/// latch pc 17.
fn three_counters(integers: bool, limit: i64) -> Chunk {
    let callee = |extra: usize, register: u8| {
        let mut code = vec![
            Instruction::get_upvalue(1, 0),
            load(2, 0),
            binary(OpCode::Add, 1, 1, 2),
            Instruction::set_upvalue(1, 0),
            binary(OpCode::Add, 1, 1, 0),
        ];
        for _ in 0..extra {
            code.push(binary(OpCode::Add, 1, 1, 0));
        }
        code.push(Instruction::return_(1));
        Chunk::with_parts(
            3,
            1,
            vec![number(integers, 1)],
            code,
            vec![],
            vec![UpvalueDescriptor {
                name: "count".into(),
                source: UpvalueSource::ParentRegister(register),
            }],
        )
    };
    Chunk::with_parts(
        13,
        0,
        vec![
            number(integers, 1),
            number(integers, limit),
            number(integers, 0),
        ],
        vec![
            load(10, 2),
            load(11, 2),
            load(12, 2),
            Instruction::closure(4, 0),
            Instruction::closure(7, 1),
            Instruction::closure(9, 2),
            load(5, 2),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 10, 17),
            Instruction::call_one(6, 4, 3),
            binary(OpCode::Add, 5, 5, 6),
            Instruction::move_(8, 4),
            Instruction::move_(4, 7),
            Instruction::move_(7, 9),
            Instruction::move_(9, 8),
            for_loop(0, 17, 11),
            binary(OpCode::Add, 5, 5, 10),
            binary(OpCode::Add, 5, 5, 11),
            binary(OpCode::Add, 5, 5, 12),
            Instruction::return_(5),
        ],
        vec![callee(0, 10), callee(1, 11), callee(2, 12)],
        vec![],
    )
}

#[test]
fn callees_that_write_their_own_cells_get_a_side_trace_each() {
    // Three prototypes take turns at one call site, and each writes an
    // upvalue (SetUpvalue). The root's trace and the sides it grows are
    // guarded on the prototype, and the cells are written through the live
    // closures, so a side may write a cell too: no recording aborts at the
    // `SetUpvalue`, and the counters the root frame reads after the loop
    // (its own captured locals, open cells) hold what the loop wrote.
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let integers = profile.is_lua53_or_later();
        let chunk = three_counters(integers, 60_000);
        let expected = interpreted(profile, &chunk);
        let mut vm = native_vm(profile);
        assert_eq!(vm.execute(&chunk).unwrap(), expected, "{profile:?}");
        let stats = stats_of(&vm);
        assert_eq!(stats.side_installs, 2, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_aborts, 0, "{profile:?}: {stats:?}");
        assert_eq!(
            stats.aborts(TraceAbortReason::UnsupportedOperand),
            0,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.iterations > 59_000, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 30_000, "{profile:?}: {stats:?}");
        assert_eq!(stats.closure_helper_calls, 0, "{profile:?}: {stats:?}");
        assert!(
            vm.jit.trace.installed_index_at_pc(11).is_some(),
            "{profile:?}: the tree was retired"
        );
    }
}

/// `f(x)` is `x > 0 and f(x - 1) + 1 or 0` through the upvalue `f`, a local
/// of the root that the loop calls as `f(depth)`. The callee has registers
/// past its parameter that its first instructions write before they read.
fn recursive_sum(limit: i64, depth: i64) -> Chunk {
    let recursive = Chunk::with_parts(
        6,
        1,
        vec![number(false, 0), number(false, 1)],
        vec![
            load(1, 0),
            binary(OpCode::LessThan, 2, 1, 0),
            jump_if_false(2, 2, 10),
            load(3, 1),
            binary(OpCode::Subtract, 3, 0, 3),
            Instruction::get_upvalue(4, 0),
            Instruction::call_one(2, 4, 3),
            load(5, 1),
            binary(OpCode::Add, 2, 2, 5),
            Instruction::return_(2),
            Instruction::return_(1),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: "f".into(),
            source: UpvalueSource::ParentRegister(4),
        }],
    );
    Chunk::with_parts(
        9,
        0,
        vec![
            number(false, 1),
            number(false, limit),
            number(false, 0),
            number(false, depth),
        ],
        vec![
            Instruction::closure(4, 0),
            load(5, 2),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 5, 9),
            load(8, 3),
            Instruction::call_one(6, 4, 8),
            binary(OpCode::Add, 5, 5, 6),
            for_loop(0, 9, 6),
            Instruction::return_(5),
        ],
        vec![recursive],
        vec![],
    )
}

#[test]
fn tracing_mode_offers_the_entry_of_a_plain_self_call_to_the_tracer() {
    // The calls the fast loop pushes inline still land at a function-entry
    // site: the recursion records, compiles and runs natively, and the
    // frames it enters hold nil past their parameters as the recorder expects.
    let profile = CompatibilityProfile::Lua51;
    let chunk = recursive_sum(2_000, 6);
    let expected = interpreted(profile, &chunk);
    let mut vm = native_vm(profile);
    assert_eq!(vm.execute(&chunk).unwrap(), expected);
    let stats = stats_of(&vm);
    assert!(stats.compiled_function_traces >= 1, "{stats:?}");
    assert!(stats.native_entries >= 1, "{stats:?}");
    assert!(stats.native_self_calls >= 1_000, "{stats:?}");
}

#[test]
fn tracing_mode_pushes_plain_calls_inline_while_nothing_records() {
    // Below the hotness threshold no trace exists, so every call of the loop
    // is the fast loop's own. None may reach the out-of-line routine that
    // tracing mode used before it took the interpreter's inline push.
    let profile = CompatibilityProfile::Lua51;
    let chunk = counter(false, 40);
    let expected = interpreted(profile, &chunk);
    let mut vm = native_vm(profile);
    assert_eq!(vm.execute(&chunk).unwrap(), expected);
    assert_eq!(stats_of(&vm).native_entries, 0);
    assert_eq!(vm.tracing_out_of_line_calls, 0);
}

// ------------------------------------------------------ depth and lifetime

/// A stack of `below` filler frames with the root frame of `chunk` on top,
/// set to `state`'s registers.
fn deep_stack(world: &mut World, below: usize, from: &CallStack) -> CallStack {
    let mut frames = CallStack::new();
    for _ in 0..below {
        frames.push(Frame::new(
            Arc::clone(&world.chunk),
            Vec::new(),
            Vec::new(),
            None,
            None,
            world.vm.runtime().global_environment(),
            None,
        ));
    }
    let top = restart_root(&mut world.vm, &world.chunk, from);
    let mut owned = top.into_owned_frames();
    frames.push(owned.remove(0));
    frames
}

#[test]
fn a_trace_enters_only_where_an_exit_inside_its_callees_has_room() {
    let profile = CompatibilityProfile::Lua51;
    let mut world = World::new(profile, branching(false, 400));
    let entered = install_at(&mut world, 6, 8);
    let limit = crate::vm::MAX_LUA_CALL_FRAMES;
    // One callee frame on top of the root needs the root at depth `limit - 1`
    // at the deepest.
    let mut frames = deep_stack(&mut world, limit - 2, &entered.start);
    assert_eq!(frames.len(), limit - 1);
    let top = frames.len() - 1;
    entered.reset_cells(&mut world.vm);
    let ran = world
        .vm
        .enter_native_trace_with_fuel(&mut frames, top, entered.index, 1_000_000)
        .unwrap();
    assert!(ran.is_some(), "room for the callee frame");
    // The exit rebuilt the callee: the stack is at the limit and valid.
    assert_eq!(frames.len(), limit);

    let mut frames = deep_stack(&mut world, limit - 1, &entered.start);
    assert_eq!(frames.len(), limit);
    let top = frames.len() - 1;
    let before = stats_of(&world.vm);
    entered.reset_cells(&mut world.vm);
    let ran = world
        .vm
        .enter_native_trace_with_fuel(&mut frames, top, entered.index, 1_000_000)
        .unwrap();
    assert!(
        ran.is_none(),
        "no room: the interpreter raises its own error"
    );
    assert_eq!(frames.len(), limit);
    assert_eq!(stats_of(&world.vm).iterations, before.iterations);
}

#[test]
fn a_collected_callee_never_matches_a_trace_pinned_to_it() {
    let profile = CompatibilityProfile::Lua51;
    let mut world = World::new(profile, branching(false, 400));
    let entered = install_at(&mut world, 6, 8);
    let pinned = entered.lowered.pinned_closures[0];
    let handle = crate::runtime::ClosureHandle::from_native_identity(pinned);
    let start_registers: Vec<Value> = entered.start.at(0).registers.to_vec();
    // Drop every reference to the closure and collect it.
    drop(entered.start);
    let mut nothing = CallStack::new();
    world.vm.collect_garbage(&mut nothing, &[]).unwrap();
    assert!(
        world.vm.runtime().closure_call_parts(handle).is_none(),
        "nothing refers to the closure any more"
    );
    // A new closure of the same prototype may reuse the slot. It is another
    // closure, and a stale handle names none.
    let root = Arc::clone(&world.chunk);
    let mut again = frames_for(&world.vm, &root, None);
    let mut fresh = None;
    for _ in 0..4 {
        world.vm.step(&mut again).unwrap();
        if let Value::Closure(new) = again.last().unwrap().registers[4] {
            fresh = Some(new);
            break;
        }
    }
    let fresh = fresh.expect("the program makes its closure first");
    assert_ne!(fresh.native_identity(), pinned);
    // Entering the trace at the loop header with the new closure must not run
    // it: the trace is specialised on the old one.
    let mut frames = frames_for(&world.vm, &root, None);
    {
        let mut frame = frames.at_mut(0);
        frame.pc = 6;
        frame.last_interpreted_pc = Some(8).into();
        for (register, value) in frame.registers.iter_mut().zip(&start_registers) {
            *register = value.clone();
        }
        frame.registers[4] = Value::Closure(fresh);
    }
    let before = stats_of(&world.vm);
    let ran = world
        .vm
        .enter_native_trace_with_fuel(&mut frames, 0, entered.index, 1_000)
        .unwrap();
    assert!(ran.is_none());
    assert_eq!(stats_of(&world.vm).iterations, before.iterations);
}

/// The cell reads a call of a closure of `body` records, by helper, for a
/// loop whose callee takes turns between two closures of one prototype.
fn family_cell_reads(profile: CompatibilityProfile, body: &str) -> Vec<HelperId> {
    let source = format!(
        "local function make(seed)
           local object = {{ v = seed }}
           local n = seed
           return function() {body} end
         end
         local a, b = make(1), make(2)
         local sum = 0
         for i = 1, 4000 do
           local f = a
           a = b
           b = f
           sum = (sum + f()) % 1000003
         end
         return sum"
    );
    let chunk = crate::compiler::compile(source).unwrap();
    let ir = recorded(profile, &chunk);
    ir.verify().unwrap();
    assert!(
        ir.frames[1].inlined.expect("an inlined frame").dynamic,
        "{profile:?}"
    );
    ir.preheader
        .iter()
        .chain(&ir.body)
        .filter_map(|instruction| match &instruction.op {
            // A number read counts as one kind, whichever the chunk's
            // numbers are.
            TraceOp::CallHelper { helper, .. } if *helper == HelperId::CELL_GET_INTEGER => {
                Some(HelperId::CELL_GET_FLOAT)
            }
            TraceOp::CallHelper { helper, .. } if helper.is_cell_access() => Some(*helper),
            _ => None,
        })
        .collect()
}

#[test]
fn a_cell_read_twice_in_one_call_is_one_helper_call() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let number_read = HelperId::CELL_GET_FLOAT;
        // Two reads of the table with a table store between them, which
        // writes no cell.
        let reads = family_cell_reads(profile, "object.v = (object.v + 1) % 1009; return object.v");
        assert_eq!(reads, [HelperId::CELL_GET_TABLE], "{profile:?}");
        // Three reads of the number and nothing between.
        let reads = family_cell_reads(profile, "return (n + n * 2 + n) % 7");
        assert_eq!(reads, [number_read], "{profile:?}");
        // A read of each cell: neither answers for the other.
        let reads = family_cell_reads(profile, "return object.v + n + object.v + n");
        assert_eq!(reads.len(), 2, "{profile:?}: {reads:?}");
        assert!(reads.contains(&HelperId::CELL_GET_TABLE), "{profile:?}");
        assert!(reads.contains(&number_read), "{profile:?}");
    }
}

#[test]
fn a_cell_read_after_a_write_of_a_cell_is_a_second_helper_call() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let number_read = HelperId::CELL_GET_FLOAT;
        // The closure's own store to the cell comes between its two reads.
        let reads = family_cell_reads(
            profile,
            "local before = n; n = (before + 3) % 1013; return before + n",
        );
        assert_eq!(
            reads,
            [number_read, HelperId::CELL_SET, number_read],
            "{profile:?}"
        );
        // A store to another cell also ends the reuse: the search does not
        // tell cells apart.
        let reads = family_cell_reads(
            profile,
            "local held = object; n = (n + 1) % 1013; return held.v + object.v",
        );
        assert_eq!(
            reads
                .iter()
                .filter(|helper| **helper == HelperId::CELL_GET_TABLE)
                .count(),
            2,
            "{profile:?}: {reads:?}"
        );
    }
}
