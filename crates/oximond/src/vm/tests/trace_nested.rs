//! Nested loops: the root trace of an inner loop continues, through a side
//! trace from its exit, into the rest of the loops around it, and links back
//! to its own header. The whole loop nest then runs as one native function.
//!
//! The programs are hand-assembled loops with work before and after each
//! inner loop. Every case compares with an isolated interpreter, and counters
//! prove the nest ran natively: the number of entries stays small while the
//! outer loop runs hundreds of iterations, and `side_entries` counts the
//! jumps through the outer iterations directly.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::bool_assert_comparison,
    clippy::len_zero
)]

use super::trace_native::{native_vm, stats_of};
use super::trace_record::*;
use super::*;
use crate::vm::jit::trace::nest;
use crate::vm::jit::trace::policy::SiteState;
use crate::vm::jit::trace::{TraceExitKind, TraceRun};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// A number in the profile's number type.
fn number(integers: bool, value: i64) -> Constant {
    if integers {
        Constant::Integer(value)
    } else {
        Constant::Number(value as f64)
    }
}

/// Where each loop of a program starts: the pc its backward branch lands on.
struct Nest {
    chunk: Chunk,
    /// Loop headers, innermost first.
    headers: Vec<usize>,
}

/// A small assembler for loop nests: constants are shared, a loop is opened
/// with its prep and closed with its latch.
struct Asm {
    integers: bool,
    constants: Vec<Constant>,
    code: Vec<Instruction>,
    open: Vec<(u8, usize, usize)>,
    headers: Vec<usize>,
}

impl Asm {
    fn new(integers: bool) -> Self {
        Self {
            integers,
            constants: Vec::new(),
            code: Vec::new(),
            open: Vec::new(),
            headers: Vec::new(),
        }
    }

    fn constant(&mut self, value: i64) -> u16 {
        let wanted = number(self.integers, value);
        if let Some(at) = self.constants.iter().position(|known| *known == wanted) {
            return u16::try_from(at).unwrap();
        }
        self.constants.push(wanted);
        u16::try_from(self.constants.len() - 1).unwrap()
    }

    /// `register = value`.
    fn set(&mut self, register: u8, value: i64) {
        let constant = self.constant(value);
        self.code.push(load(register, constant));
    }

    fn pc(&self) -> usize {
        self.code.len()
    }

    /// Opens `for v = 1, limit` with its control registers at `base`
    /// (`base + 3` is the visible variable). `limit` is a register.
    fn open_for(&mut self, base: u8, limit: u8) {
        self.set(base, 1);
        self.code.push(Instruction::move_(base + 1, limit));
        self.set(base + 2, 1);
        let prep = self.pc();
        self.code.push(for_prep(base, prep, 0));
        self.open.push((base, prep, prep + 1));
    }

    fn close_for(&mut self) {
        let (base, prep, header) = self.open.pop().expect("an open loop");
        let latch = self.pc();
        self.code[prep] = for_prep(base, prep, latch);
        self.code.push(for_loop(base, latch, header));
        self.headers.push(header);
    }

    fn add(&mut self, destination: u8, left: u8, right: u8) {
        self.code
            .push(binary(OpCode::Add, destination, left, right));
    }

    fn finish(mut self, registers: u8, results: (u8, u8)) -> Nest {
        assert_eq!(self.open, [] as [(u8, usize, usize); 0]);
        self.code
            .push(Instruction::return_values(results.0, results.1));
        // Innermost loop first: it has the highest pc.
        self.headers.sort_unstable_by(|a, b| b.cmp(a));
        Nest {
            chunk: Chunk::new(registers, self.constants, self.code),
            headers: self.headers,
        }
    }
}

/// `for i = 1, outer do for j = 1, inner do acc = acc + i + j end
/// count = count + 1 end return acc, count`. Registers: r8 acc, r9 count, r10
/// the outer limit, r11 the inner limit, r12 one.
fn nest2(integers: bool, outer: i64, inner: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.set(8, 0);
    asm.set(9, 0);
    asm.set(10, outer);
    asm.set(11, inner);
    asm.set(12, 1);
    asm.open_for(0, 10);
    asm.open_for(4, 11);
    asm.add(8, 8, 3);
    asm.add(8, 8, 7);
    asm.close_for();
    asm.add(9, 9, 12);
    asm.close_for();
    asm.finish(13, (8, 2))
}

/// `nest` as the body of a function that a loop calls `calls` times, so the
/// nest runs again and again and the loop around its inner loop is always
/// nearly done when the inner loop turns hot. The headers are the nest's.
fn called_repeatedly(integers: bool, nest: Nest, calls: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.code.push(Instruction::closure(8, 0));
    asm.set(10, calls);
    asm.open_for(0, 10);
    asm.code.push(Instruction::move_(12, 8));
    asm.code.push(Instruction::call(12, 0, 1));
    asm.close_for();
    let driver = asm.finish(14, (12, 1));
    Nest {
        chunk: Chunk::with_parts(
            14,
            0,
            driver.chunk.constants.clone(),
            driver.chunk.code.clone(),
            vec![nest.chunk],
            vec![],
        ),
        headers: nest.headers,
    }
}

/// `nest2` whose outer part also reads the table in upvalue 0 into r10:
/// `for i for j acc = acc + j end count = count + 1; t = upvalue end return
/// acc, count, t`. The table is an opaque lane that crosses every link.
fn nest_with_cell_table(integers: bool, outer: i64, inner: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.set(8, 0);
    asm.set(9, 0);
    asm.set(11, outer);
    asm.set(12, inner);
    asm.set(13, 1);
    asm.open_for(0, 11);
    asm.open_for(4, 12);
    asm.add(8, 8, 7);
    asm.close_for();
    asm.add(9, 9, 13);
    asm.code.push(Instruction::get_upvalue(10, 0));
    asm.close_for();
    asm.finish(14, (8, 3))
}

/// Three levels: `for i for j for k acc = acc + k end mid = mid + j end
/// top = top + i end`. Registers: r12 acc, r13 mid, r14 top.
fn nest3(integers: bool, a: i64, b: i64, c: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.set(12, 0);
    asm.set(13, 0);
    asm.set(14, 0);
    asm.set(15, a);
    asm.set(16, b);
    asm.set(17, c);
    asm.set(18, 1);
    asm.open_for(0, 15);
    asm.open_for(4, 16);
    asm.open_for(8, 17);
    asm.add(12, 12, 11);
    asm.close_for();
    asm.add(13, 13, 18);
    asm.close_for();
    asm.add(14, 14, 18);
    asm.close_for();
    asm.finish(19, (12, 3))
}

/// `nest2` whose outer part branches on the outer index: `if i % 3 < 1 then
/// a = a + 1 else b = b + 1 end`. Registers: r8 acc, r9 a, r10 b.
fn nest_branchy(integers: bool, outer: i64, inner: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.set(8, 0);
    asm.set(9, 0);
    asm.set(10, 0);
    asm.set(11, outer);
    asm.set(12, inner);
    asm.set(13, 1);
    asm.set(14, 3);
    asm.open_for(0, 11);
    asm.open_for(4, 12);
    asm.add(8, 8, 7);
    asm.close_for();
    asm.code.push(binary(OpCode::Modulo, 15, 3, 14));
    asm.code.push(binary(OpCode::LessThan, 16, 15, 13));
    let test = asm.pc();
    asm.code.push(jump_if_false(16, test, test + 3));
    asm.add(9, 9, 13);
    asm.code.push(jump_to(test + 2, test + 4));
    asm.add(10, 10, 13);
    asm.close_for();
    asm.finish(17, (8, 3))
}

/// An inner loop whose trip count is the outer index, so the inner loop is
/// short and varies: `for i = 1, outer do for j = 1, i do acc = acc + j end
/// end`. Registers: r8 acc, r9 outer limit.
fn triangle(integers: bool, outer: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.set(8, 0);
    asm.set(9, outer);
    asm.open_for(0, 9);
    asm.open_for(4, 3);
    asm.add(8, 8, 7);
    asm.close_for();
    asm.close_for();
    asm.finish(10, (8, 1))
}

/// An inner `while` loop: `for i = 1, outer do j = 0; while j < inner do acc
/// = acc + j; j = j + 1 end end`. Registers: r4 j, r5 inner, r6 test, r7
/// one, r8 acc, r9 outer limit.
fn nest_while(integers: bool, outer: i64, inner: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.set(8, 0);
    asm.set(9, outer);
    asm.set(5, inner);
    asm.set(7, 1);
    asm.open_for(0, 9);
    asm.set(4, 0);
    let header = asm.pc();
    asm.code.push(binary(OpCode::LessThan, 6, 4, 5));
    let test = asm.pc();
    asm.code.push(jump_if_false(6, test, 0));
    asm.add(8, 8, 4);
    asm.add(4, 4, 7);
    let back = asm.pc();
    asm.code.push(jump_to(back, header));
    asm.code[test] = jump_if_false(6, test, back + 1);
    asm.headers.push(header);
    asm.close_for();
    asm.finish(10, (8, 1))
}

fn run_traced(profile: CompatibilityProfile, chunk: &Chunk) -> (Vm, Vec<Value>) {
    let mut vm = native_vm(profile);
    let results = vm.execute(chunk).unwrap();
    (vm, results)
}

fn has_integers(profile: CompatibilityProfile) -> bool {
    profile.is_lua53_or_later()
}

#[test]
fn an_inner_loop_trace_runs_the_loops_around_it_in_every_profile() {
    for profile in PROFILES {
        let integers = has_integers(profile);
        for (name, nest) in [
            ("nest2", nest2(integers, 400, 7)),
            ("triangle", triangle(integers, 150)),
            ("while", nest_while(integers, 400, 6)),
        ] {
            let expected = interpreted(profile, &nest.chunk);
            let (vm, results) = run_traced(profile, &nest.chunk);
            assert_eq!(results, expected, "{name} under {profile:?}");
            let stats = stats_of(&vm);
            assert!(
                stats.outer_sides >= 1,
                "{name} under {profile:?}: {stats:?}"
            );
            assert!(
                stats.covered_loops >= 1,
                "{name} under {profile:?}: {stats:?}"
            );
            // The outer loop ran inside the tree: links, not entries.
            assert!(
                stats.side_entries > 100,
                "{name} under {profile:?}: {stats:?}"
            );
            assert!(
                stats.native_entries < 60,
                "{name} under {profile:?}: {stats:?}"
            );
        }
    }
}

#[test]
fn the_root_of_a_short_inner_loop_is_not_compiled_alone_its_tree_is_recorded_ahead() {
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = called_repeatedly(integers, nest2(integers, 28, 3), 60);
        let expected = interpreted(profile, &nest.chunk);
        let (vm, results) = run_traced(profile, &nest.chunk);
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        // Three trips are fewer than a trace needs per entry to pay, and the
        // outer loop is one a tree carries and is nearly done: the root only
        // leads to the tree, whose outer side is recorded in the interpreter,
        // so the compile is the tree's alone and no stepping stone is
        // compiled.
        assert_eq!(stats.baseline_roots, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.compile_submissions, 1, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 0, "{profile:?}: {stats:?}");
        let index = vm
            .jit
            .trace
            .installed_index_at_pc(nest.headers[0])
            .unwrap_or_else(|| panic!("{profile:?}: the inner loop has a trace: {stats:?}"));
        assert_eq!(
            vm.jit.trace.installed_tier(index),
            Some(OptimizationTier::Optimized),
            "{profile:?}: the tree is the code that runs: {stats:?}"
        );
    }
}

#[test]
fn the_root_of_a_short_inner_loop_compiles_at_the_baseline_tier_and_its_tree_at_the_optimized() {
    // The stepping stone: what a nest deeper than two levels does, and what
    // the policy falls back on when the sides cannot be recorded ahead.
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = nest2(integers, 400, 3);
        let expected = interpreted(profile, &nest.chunk);
        let mut vm = native_vm(profile);
        vm.jit.trace.set_stage_roots(false);
        let results = vm.execute(&nest.chunk).unwrap();
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        // Three trips are fewer than a trace needs per entry to pay, and the
        // outer loop is one a tree carries: the root only leads to the tree.
        assert_eq!(stats.baseline_roots, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.compile_submissions, 2, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        let index = vm
            .jit
            .trace
            .installed_index_at_pc(nest.headers[0])
            .unwrap_or_else(|| panic!("{profile:?}: the inner loop has a trace: {stats:?}"));
        assert_eq!(
            vm.jit.trace.installed_tier(index),
            Some(OptimizationTier::Optimized),
            "{profile:?}: the tree is the code that runs: {stats:?}"
        );
    }
}

#[test]
fn a_background_compile_waits_for_the_stepping_stone_and_its_tree() {
    // With a worker thread the interpreter would run the nest at full price
    // until the root and then the tree land. They are waited for, so the
    // tree runs almost every outer iteration, as it does with compiles
    // inline.
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = nest2(integers, 400, 3);
        let expected = interpreted(profile, &nest.chunk);
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        vm.jit.trace.set_stage_roots(false);
        vm.jit.broker = Some(crate::jit::JitBroker::new(
            crate::vm::jit::state::JIT_QUEUE_CAPACITY,
        ));
        vm.count_trace_links();
        let results = vm.execute(&nest.chunk).unwrap();
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        assert_eq!(stats.baseline_roots, 1, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 300, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_background_compile_waits_for_the_tree_of_a_staged_root() {
    // The root is a placeholder while its sides are recorded, and the tree
    // is the only compile: the interpreter waits for it, so the tree runs
    // almost every outer iteration, as it does with compiles inline.
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = called_repeatedly(integers, nest2(integers, 28, 3), 60);
        let expected = interpreted(profile, &nest.chunk);
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        vm.jit.broker = Some(crate::jit::JitBroker::new(
            crate::vm::jit::state::JIT_QUEUE_CAPACITY,
        ));
        vm.count_trace_links();
        let results = vm.execute(&nest.chunk).unwrap();
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        assert_eq!(stats.baseline_roots, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.compile_submissions, 1, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn the_root_of_a_long_inner_loop_compiles_once_at_the_optimized_tier() {
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = nest2(integers, 400, 200);
        let expected = interpreted(profile, &nest.chunk);
        let (vm, results) = run_traced(profile, &nest.chunk);
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        // Two hundred trips (more than an outer side is worth carrying a loop
        // through, see `OUTER_SIDE_MAX_ITERATIONS_PER_ENTRY`) pay for an
        // entry on their own: no root is a stepping stone, though the outer
        // side still comes.
        assert_eq!(stats.baseline_roots, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn three_levels_run_as_one_tree_with_a_side_of_a_side() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let integers = has_integers(profile);
        let nest = nest3(integers, 60, 40, 5);
        let expected = interpreted(profile, &nest.chunk);
        let (vm, results) = run_traced(profile, &nest.chunk);
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        // The exit of the innermost loop continues through the middle loop,
        // and the exit of the middle loop through the outer loop: at least
        // two sides in a chain, one of which runs through a loop.
        assert!(stats.side_installs >= 2, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.covered_loops >= 2, "{profile:?}: {stats:?}");
        assert!(stats.native_entries < 40, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 1_500, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_branch_in_the_outer_part_gets_its_own_side() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua53] {
        let integers = has_integers(profile);
        let nest = nest_branchy(integers, 600, 5);
        let expected = interpreted(profile, &nest.chunk);
        // Both arms are pure: a value select would run the branch with no
        // guard at all.
        let mut vm = super::trace_native::guarded_vm(profile);
        let results = vm.execute(&nest.chunk).unwrap();
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        // The outer iteration is recorded along one arm of the branch. The
        // other arm is a guard of that side, which takes a second side.
        assert!(stats.side_installs >= 2, "{profile:?}: {stats:?}");
        assert!(stats.native_entries < 80, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_outer_loop_costs_no_native_entries_and_no_frame_rebuilds() {
    // Every entry that ran code ends in exactly one exit that rebuilds the
    // frame. With the outer loop inside the tree, 2000 outer iterations cost
    // a handful of entries, while the links they took are counted apart.
    let profile = CompatibilityProfile::Lua51;
    let nest = nest2(false, 2_000, 6);
    let (vm, _) = run_traced(profile, &nest.chunk);
    let stats = stats_of(&vm);
    let exits: u64 = stats.exits.iter().sum();
    assert_eq!(exits, stats.native_entries, "{stats:?}");
    assert!(stats.native_entries < 60, "{stats:?}");
    assert!(stats.side_entries > 1_500, "{stats:?}");
    assert!(stats.iterations > 10_000, "{stats:?}");
}

#[test]
fn every_state_the_nest_publishes_is_a_state_the_interpreter_passes_through() {
    for (name, chunk) in [
        ("nest2", nest2(false, 300, 6).chunk),
        ("nest3", nest3(false, 30, 12, 4).chunk),
        ("branchy", nest_branchy(false, 300, 4).chunk),
        ("triangle", triangle(false, 120).chunk),
    ] {
        let chunk = Arc::new(chunk);
        let reference = drive(&chunk, None);
        let live_in = crate::jit_ir::observable_registers(&chunk);
        let mut vm = native_vm(CompatibilityProfile::Lua51);
        let mut frames = frames_for(&vm, &chunk, None);
        let mut seen = 0_usize;
        let results = loop {
            if let Some(results) = vm.step(&mut frames).unwrap() {
                break scalars(&results);
            }
            let state = capture(&frames);
            let found = reference.states[seen..]
                .iter()
                .position(|candidate| candidate.same_as_with(&state, &live_in))
                .unwrap_or_else(|| panic!("{name}: {state:?} is off the path"));
            seen += found;
        };
        assert_eq!(results, reference.results, "{name}");
        assert!(stats_of(&vm).outer_sides >= 1, "{name}");
    }
}

/// Runs `chunk` until the inner loop's tree has `linked` sides and returns
/// the VM, its frames and the trace's index.
fn until_tree(
    chunk: &Arc<Chunk>,
    header: usize,
    linked: usize,
) -> (Vm, crate::vm::stack::CallStack, u32) {
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let mut frames = frames_for(&vm, chunk, None);
    for _ in 0..2_000_000 {
        if let Some(index) = vm.jit.trace.installed_index_at_pc(header)
            && vm
                .jit
                .trace
                .installed_sides(index)
                .is_some_and(|(_, count)| count == linked)
        {
            return (vm, frames, index);
        }
        assert!(vm.step(&mut frames).unwrap().is_none());
    }
    panic!("the tree never got {linked} sides: {:?}", stats_of(&vm));
}

#[test]
fn a_fuel_poll_inside_an_outer_iteration_returns_to_the_interpreter_in_step() {
    // Poll every few loop-backs and links through the nest, from many header
    // visits, and check that every state the trace publishes lies on the
    // interpreter's own path. The fuel crosses the link of the outer side at
    // every phase, so polls land in the inner loop, in the outer part and at
    // the link itself.
    let nest = nest2(false, 400, 4);
    let header = nest.headers[0];
    let chunk = Arc::new(nest.chunk);
    let (mut vm, _, index) = until_tree(&chunk, header, 1);
    let reference = drive(&chunk, None);
    let visits = header_visits(&reference, header);
    let live_in = crate::jit_ir::observable_registers(&chunk);
    let mut polls = 0;
    for fuel in [1_i64, 2, 3, 4, 5, 6, 7, 11] {
        let mut cursor = visits[100];
        let start = &reference.states[cursor];
        let mut frames = frames_for(&vm, &chunk, Some(start));
        loop {
            let run = {
                let jit = &mut vm.jit;
                jit.trace.enter(
                    &mut jit.statistics.trace,
                    index,
                    &mut frames,
                    0,
                    fuel,
                    &mut vm.runtime,
                )
            };
            let TraceRun::Ran { poll, .. } = run else {
                panic!("the trace enters at the header");
            };
            let state = capture(&frames);
            let found = reference.states[cursor..]
                .iter()
                .position(|candidate| candidate.same_as_with(&state, &live_in))
                .unwrap_or_else(|| panic!("fuel {fuel}: {state:?} is off the path"));
            cursor += found;
            if !poll {
                break;
            }
            polls += 1;
            assert!(polls < 100_000);
            vm.jit.trace.forgive(index);
        }
    }
    assert!(polls > 200, "{polls}");
}

#[test]
fn a_budget_carries_across_every_link_of_the_nest() {
    // The fuel is one budget for the whole tree: loop-backs and links spend
    // it alike, so a run with fuel F ends in a poll after exactly F of them.
    let nest = nest2(false, 400, 4);
    let header = nest.headers[0];
    let chunk = Arc::new(nest.chunk);
    let (mut vm, _, index) = until_tree(&chunk, header, 1);
    let reference = drive(&chunk, None);
    let visits = header_visits(&reference, header);
    for fuel in [3_i64, 9, 40, 200] {
        let start = &reference.states[visits[50]];
        let mut frames = frames_for(&vm, &chunk, Some(start));
        let before = stats_of(&vm);
        let run = {
            let jit = &mut vm.jit;
            jit.trace.enter(
                &mut jit.statistics.trace,
                index,
                &mut frames,
                0,
                fuel,
                &mut vm.runtime,
            )
        };
        let TraceRun::Ran { poll, .. } = run else {
            panic!("the trace enters at the header");
        };
        assert!(poll, "fuel {fuel} runs out inside a 400 x 4 nest");
        let after = stats_of(&vm);
        assert_eq!(
            after.iterations - before.iterations,
            u64::try_from(fuel).unwrap(),
            "fuel {fuel}"
        );
        assert_eq!(
            after.exits_of(TraceExitKind::Poll) - before.exits_of(TraceExitKind::Poll),
            1
        );
    }
}

#[test]
fn the_restore_plan_of_every_exit_of_a_nest_matches_the_generic_restore() {
    let nest = nest3(false, 40, 16, 4);
    let header = nest.headers[0];
    let chunk = Arc::new(nest.chunk);
    let (mut vm, _, index) = until_tree(&chunk, header, 2);
    let reference = drive(&chunk, None);
    let visits = header_visits(&reference, header);
    let base = &reference.states[visits[200]];
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let exits = vm.jit.trace.exit_count(index);
    assert!(exits > 8, "{exits}");
    for exit in 0..exits {
        for round in 0..4 {
            let area: Vec<i64> = (0..96)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    if round == 0 { 0 } else { seed.cast_signed() }
                })
                .collect();
            let mut planned = frames_for(&vm, &chunk, Some(base));
            let mut generic = frames_for(&vm, &chunk, Some(base));
            vm.jit
                .trace
                .plan_restore(index, exit, &area, &mut planned.at_mut(0), &mut vm.runtime);
            vm.jit
                .trace
                .generic_restore(index, exit, &area, &mut generic.at_mut(0), &mut vm.runtime)
                .expect("the generic restore succeeds");
            assert_eq!(
                format!("{:?}", capture(&planned)),
                format!("{:?}", capture(&generic)),
                "exit {exit} round {round}"
            );
        }
    }
}

#[test]
fn a_flush_at_any_moment_leaves_the_nest_correct() {
    // The compile results install after some number of polls, as a worker's
    // would after some time, and the flush lands before the side recording,
    // during it, while the tree compiles and after it landed. Every run must
    // finish with the interpreter's answer, and a flush drops the tree and
    // the loops it covered together.
    let nest = nest3(false, 50, 30, 4);
    let chunk = Arc::new(nest.chunk);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut stale = 0;
    let mut rebuilt = 0;
    for delay in [0, 1, 3, 8, 30] {
        for flush_at in [300, 700, 1_500, 3_000, 6_000, 12_000, 30_000] {
            let mut vm = Vm::with_profile_and_execution_mode(
                CompatibilityProfile::Lua51,
                ExecutionMode::Tracing,
            );
            vm.use_delayed_synchronous_jit_compilation(delay);
            let mut frames = frames_for(&vm, &chunk, None);
            let mut steps = 0;
            let mut covered_at_flush = 0;
            let results = loop {
                steps += 1;
                if steps == flush_at {
                    covered_at_flush = stats_of(&vm).covered_loops;
                    vm.runtime_mut().request_jit_flush();
                }
                if let Some(results) = vm.step(&mut frames).unwrap() {
                    break results;
                }
            };
            assert_eq!(results, expected, "delay {delay}, flush at {flush_at}");
            let stats = stats_of(&vm);
            stale += stats.stale_completions;
            if stats.covered_loops > covered_at_flush && steps > flush_at {
                rebuilt += 1;
            }
        }
    }
    assert!(stale > 0, "some flush landed on an outstanding compile");
    assert!(rebuilt > 0, "some nest was covered again after a flush");
}

#[test]
fn a_flush_gives_the_covered_loops_back_to_the_counters() {
    let nest = nest2(false, 400, 6);
    let (header, outer_header) = (nest.headers[0], nest.headers[1]);
    let chunk = Arc::new(nest.chunk);
    let (mut vm, mut frames, _) = until_tree(&chunk, header, 1);
    assert_eq!(
        vm.jit.trace.site_state_any(outer_header),
        Some(SiteState::Covered),
        "{:?}",
        stats_of(&vm)
    );
    vm.runtime_mut().request_jit_flush();
    for _ in 0..10_000 {
        if vm.jit.trace.installed_count() == 0 && !vm.runtime().jit_flush_requested() {
            break;
        }
        assert!(vm.step(&mut frames).unwrap().is_none());
    }
    assert_eq!(vm.jit.trace.installed_count(), 0);
    assert_eq!(vm.jit.trace.site_state_any(outer_header), None);
}

#[test]
fn a_table_that_crosses_the_links_survives_a_collection_at_every_poll() {
    // The outer part reads a table from an upvalue every iteration, so the
    // table is a lane value that the side trace carries across its link. A
    // poll lands inside the loops, in the outer part and at the link, and a
    // full collection runs at each. The table lives in a cell and in the
    // restored registers, never only in the bank, and the answer is the
    // interpreter's.
    let nest = nest_with_cell_table(false, 300, 4);
    let header = nest.headers[0];
    let chunk = Arc::new(nest.chunk);
    let frames_with = |vm: &mut Vm, cell| {
        crate::vm::stack::CallStack::from(vec![Frame::new(
            Arc::clone(&chunk),
            Vec::new(),
            vec![cell],
            None,
            None,
            vm.runtime().global_environment(),
            None,
        )])
    };
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::from("name"), Value::from("kept"))
        .unwrap();
    let cell = vm.runtime_mut().create_cell(Value::Table(table)).unwrap();
    let mut frames = frames_with(&mut vm, cell);
    let index = loop {
        if let Some(index) = vm.jit.trace.installed_index_at_pc(header)
            && vm
                .jit
                .trace
                .installed_sides(index)
                .is_some_and(|(_, linked)| linked == 1)
        {
            break index;
        }
        assert!(
            vm.step(&mut frames).unwrap().is_none(),
            "{:?}",
            stats_of(&vm)
        );
    };
    let mut polls = 0;
    let mut finished = None;
    for _ in 0..50_000 {
        let frame = frames.last().unwrap();
        let at_header =
            frame.pc == header && frame.last_interpreted_pc.get().is_some_and(|l| l >= header);
        if at_header {
            let ran = vm
                .enter_native_trace_with_fuel(&mut frames, 0, index, 3)
                .unwrap();
            if ran == Some(true) {
                polls += 1;
                vm.jit.trace.forgive(index);
                let roots = vm.gc_roots_with_coroutines(&mut frames, &[]);
                vm.runtime_mut().collect_full(roots);
                assert_eq!(
                    vm.runtime().table_get(table, &Value::from("name")).unwrap(),
                    Value::from("kept"),
                    "after poll {polls}"
                );
                assert_eq!(vm.jit.trace.root_values(), [] as [Value; 0]);
                continue;
            }
        }
        if let Some(results) = vm.step(&mut frames).unwrap() {
            finished = Some(results);
            break;
        }
    }
    let results = finished.expect("the program finishes");
    assert!(polls > 100, "{polls}");
    assert_eq!(results[2], Value::Table(table));

    // The interpreter's answer for the same program and cell.
    let mut plain = interpreter_vm_with(CompatibilityProfile::Lua51);
    let plain_table = plain.runtime_mut().create_table().unwrap();
    let plain_cell = plain
        .runtime_mut()
        .create_cell(Value::Table(plain_table))
        .unwrap();
    let mut plain_frames = frames_with(&mut plain, plain_cell);
    let expected = loop {
        if let Some(results) = plain.step(&mut plain_frames).unwrap() {
            break results;
        }
    };
    assert_eq!(results[..2], expected[..2]);
}

// ------------------------------------------------------------------ tables

/// What the interpreter can see of a frame: pc, `last_interpreted_pc`, every
/// live register (a table as the order it first appears in), and the
/// contents of those tables at keys 1 to `keys`.
fn observe(vm: &Vm, chunk: &Chunk, frames: &crate::vm::stack::CallStack, keys: i64) -> String {
    use std::fmt::Write as _;
    let frame = frames.last().unwrap();
    let live_in = crate::jit_ir::observable_registers(chunk);
    let live = live_in.get(frame.pc);
    let mut tables = Vec::new();
    let mut out = format!("{}:{:?}:", frame.pc, frame.last_interpreted_pc);
    for (register, value) in frame.registers.iter().enumerate() {
        let is_live = live.is_none_or(|set| {
            u8::try_from(register)
                .is_ok_and(|register| crate::jit_ir::register_set_contains(set, register))
        });
        match value {
            _ if !is_live => out.push_str("dead,"),
            Value::Closure(_) => out.push_str("closure,"),
            Value::Table(handle) => {
                let at = tables.iter().position(|known| known == handle);
                let at = at.unwrap_or_else(|| {
                    tables.push(*handle);
                    tables.len() - 1
                });
                let _ = write!(out, "T{at},");
            }
            other => {
                let _ = write!(out, "{other:?},");
            }
        }
    }
    for handle in tables {
        for key in 1..=keys {
            let value = vm
                .runtime()
                .table_get(handle, &Value::Integer(key))
                .or_else(|_| vm.runtime().table_get(handle, &Value::Number(key as f64)))
                .unwrap();
            let _ = write!(out, "{value:?};");
        }
    }
    out
}

/// `for i = 1, limit do x = u[i]; if x > 2 then t[i] = x + 1; pos = pos + x
/// else w[i] = x * 2; neg = neg + 1 end end return pos, neg`. Registers: r4
/// pos, r5 neg, r6 x, r7 temporary, r8 test, r9 two, r10 one, and the tables
/// u, t and w in r11, r12 and r13. The loop header is pc 8.
fn array_arms(integers: bool, limit: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.set(4, 0);
    asm.set(5, 0);
    asm.set(9, 2);
    asm.set(10, 1);
    asm.set(14, limit);
    asm.open_for(0, 14);
    asm.code.push(Instruction::get_table(6, 11, 3));
    asm.code.push(binary(OpCode::LessThan, 8, 9, 6));
    let test = asm.pc();
    asm.code.push(jump_if_false(8, test, test + 5));
    asm.code.push(binary(OpCode::Add, 7, 6, 10));
    asm.code.push(Instruction::set_table(12, 3, 7));
    asm.add(4, 4, 6);
    asm.code.push(jump_to(test + 4, test + 8));
    asm.code.push(binary(OpCode::Multiply, 7, 6, 9));
    asm.code.push(Instruction::set_table(13, 3, 7));
    asm.add(5, 5, 10);
    asm.close_for();
    let mut nest = asm.finish(15, (4, 2));
    nest.headers = vec![test - 2];
    nest
}

/// Three arrays of `length` numbers of the profile's type: u holds `i % 5`,
/// t and w hold zeros.
fn arrays(vm: &mut Vm, integers: bool, length: i64) -> [crate::runtime::TableHandle; 3] {
    let value = |n: i64| {
        if integers {
            Value::Integer(n)
        } else {
            Value::Number(n as f64)
        }
    };
    let mut made = [None; 3];
    for (which, slot) in made.iter_mut().enumerate() {
        let table = vm.runtime_mut().create_table().unwrap();
        for index in 1..=length {
            let held = if which == 0 { index % 5 } else { 0 };
            vm.runtime_mut()
                .table_set(table, &value(index), value(held))
                .unwrap();
        }
        *slot = Some(table);
    }
    made.map(Option::unwrap)
}

fn table_frames(
    vm: &Vm,
    chunk: &Arc<Chunk>,
    tables: &[crate::runtime::TableHandle; 3],
) -> crate::vm::stack::CallStack {
    let mut frames = frames_for(vm, chunk, None);
    {
        let frame = frames.last_mut().unwrap();
        for (register, table) in [11, 12, 13].into_iter().zip(tables) {
            frame.registers[register] = Value::Table(*table);
        }
    }
    frames
}

#[test]
fn a_side_that_stores_into_a_table_the_root_never_bound_matches_the_interpreter() {
    for profile in PROFILES {
        let integers = has_integers(profile);
        let limit = 3_000;
        let nest = array_arms(integers, limit);
        let chunk = Arc::new(nest.chunk);

        let mut plain = interpreter_vm_with(profile);
        let plain_tables = arrays(&mut plain, integers, limit);
        let mut plain_frames = table_frames(&plain, &chunk, &plain_tables);
        let expected = loop {
            if let Some(results) = plain.step(&mut plain_frames).unwrap() {
                break results;
            }
        };

        let mut vm = native_vm(profile);
        let tables = arrays(&mut vm, integers, limit);
        let mut frames = table_frames(&vm, &chunk, &tables);
        let results = loop {
            if let Some(results) = vm.step(&mut frames).unwrap() {
                break results;
            }
        };
        assert_eq!(results, expected, "{profile:?}");
        for (index, (native, reference)) in tables.iter().zip(&plain_tables).enumerate() {
            for key in 1..=limit {
                let key = if integers {
                    Value::Integer(key)
                } else {
                    Value::Number(key as f64)
                };
                assert_eq!(
                    vm.runtime().table_get(*native, &key).unwrap(),
                    plain.runtime().table_get(*reference, &key).unwrap(),
                    "{profile:?} table {index} at {key:?}"
                );
            }
        }
        let stats = stats_of(&vm);
        // Both arms run natively: the root has one, a side the other, and
        // the side binds a table the root does not.
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 500, "{profile:?}: {stats:?}");
        assert!(stats.native_entries < 200, "{profile:?}: {stats:?}");
    }
}

#[test]
fn every_state_a_tree_over_tables_publishes_is_a_state_the_interpreter_passes_through() {
    // Every step of the traced run lands on the interpreter's own path,
    // tables included, so no store is replayed or lost across a guard exit,
    // a link or a poll.
    let profile = CompatibilityProfile::Lua51;
    let limit = 140;
    let nest = array_arms(false, limit);
    let chunk = Arc::new(nest.chunk);

    let mut plain = interpreter_vm_with(profile);
    let plain_tables = arrays(&mut plain, false, limit);
    let mut plain_frames = table_frames(&plain, &chunk, &plain_tables);
    let mut path = Vec::new();
    loop {
        path.push(observe(&plain, &chunk, &plain_frames, limit));
        if plain.step(&mut plain_frames).unwrap().is_some() {
            break;
        }
    }

    let mut vm = native_vm(profile);
    let tables = arrays(&mut vm, false, limit);
    let mut frames = table_frames(&vm, &chunk, &tables);
    let mut seen = 0;
    loop {
        if vm.step(&mut frames).unwrap().is_some() {
            break;
        }
        let state = observe(&vm, &chunk, &frames, limit);
        let found = path[seen..]
            .iter()
            .position(|candidate| *candidate == state)
            .unwrap_or_else(|| panic!("{state} is off the interpreter's path"));
        seen += found;
    }
    assert!(stats_of(&vm).side_installs >= 1, "{:?}", stats_of(&vm));
}

// ------------------------------------------------------------------- calls

/// `for i = 1, outer do for j = 1, inner do acc = acc + f(j) end count =
/// count + 1 end return acc, count` with `f(x) = x + 1` the closure of
/// prototype 0. Registers: r8 `f`, r9 acc, r10 count, r14 the call's result.
/// The inner header is pc 11 (after the closure and the outer prep).
fn call_nest(integers: bool, outer: i64, inner: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.code.push(Instruction::closure(8, 0));
    asm.set(9, 0);
    asm.set(10, 0);
    asm.set(11, outer);
    asm.set(12, inner);
    asm.set(13, 1);
    asm.open_for(0, 11);
    asm.open_for(4, 12);
    asm.code.push(Instruction::call_one(14, 8, 7));
    asm.add(9, 9, 14);
    asm.close_for();
    asm.add(10, 10, 13);
    asm.close_for();
    let mut nest = asm.finish(16, (9, 2));
    let callee = Chunk::with_parts(
        3,
        1,
        vec![number(integers, 1)],
        vec![
            load(1, 0),
            binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    nest.chunk = Chunk::with_parts(
        16,
        0,
        nest.chunk.constants.clone(),
        nest.chunk.code.clone(),
        vec![callee],
        vec![],
    );
    nest
}

#[test]
fn a_nest_whose_inner_loop_inlines_a_call_runs_inside_one_tree() {
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = call_nest(integers, 400, 8);
        let expected = interpreted(profile, &nest.chunk);
        let (vm, results) = run_traced(profile, &nest.chunk);
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        assert!(stats.inlined_calls >= 1, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries < 60, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 200, "{profile:?}: {stats:?}");
    }
}

#[test]
fn every_state_a_nest_with_an_inlined_call_publishes_is_on_the_interpreters_path() {
    let profile = CompatibilityProfile::Lua51;
    let nest = call_nest(false, 120, 6);
    let chunk = Arc::new(nest.chunk);

    let mut plain = interpreter_vm_with(profile);
    let mut plain_frames = frames_for(&plain, &chunk, None);
    let mut path = Vec::new();
    loop {
        path.push(observe(&plain, &chunk, &plain_frames, 0));
        if plain.step(&mut plain_frames).unwrap().is_some() {
            break;
        }
    }

    let mut vm = native_vm(profile);
    let mut frames = frames_for(&vm, &chunk, None);
    let mut seen = 0;
    loop {
        if vm.step(&mut frames).unwrap().is_some() {
            break;
        }
        // An exit inside the callee pushes its frame: compare the frame the
        // interpreter would be running, the innermost.
        let state = observe(&vm, &chunk, &frames, 0);
        if frames.len() == 1 {
            let found = path[seen..]
                .iter()
                .position(|candidate| *candidate == state)
                .unwrap_or_else(|| panic!("{state} is off the interpreter's path"));
            seen += found;
        }
    }
    let stats = stats_of(&vm);
    assert!(stats.outer_sides >= 1, "{stats:?}");
}

/// `for i = 1, outer do for j = 1, inner do a = a + j end for k = 1, inner do
/// b = b + k end end return a, b`: two inner loops side by side. Registers:
/// r8 a, r9 b, r10 outer limit, r11 inner limit.
fn two_inner_loops(integers: bool, outer: i64, inner: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.set(8, 0);
    asm.set(9, 0);
    asm.set(10, outer);
    asm.set(11, inner);
    asm.open_for(0, 10);
    asm.open_for(4, 11);
    asm.add(8, 8, 7);
    asm.close_for();
    asm.open_for(12, 11);
    asm.add(9, 9, 15);
    asm.close_for();
    asm.close_for();
    asm.finish(16, (8, 2))
}

/// The span of the loop whose header is `header` in `chunk`: the
/// `NumericForLoop` (or backward jump) that lands on it.
fn span_of(chunk: &Chunk, header: usize) -> nest::LoopSpan {
    let latch = (0..chunk.code.len())
        .rev()
        .find(|pc| {
            let instruction = chunk.code[*pc];
            let target = match instruction.opcode() {
                Ok(OpCode::NumericForLoop | OpCode::JumpIfFalse) => {
                    i64::from(i16::from_ne_bytes(instruction.d().to_ne_bytes()))
                }
                Ok(OpCode::Jump) => i64::from(instruction.jump_offset()),
                _ => return false,
            };
            *pc as i64 + 1 + target == header as i64
        })
        .expect("a backward branch lands on the header");
    nest::LoopSpan { header, latch }
}

#[test]
fn the_loop_structure_of_a_chunk_is_read_from_its_bytecode() {
    // Three levels: each loop encloses the next, none is beside another.
    let nest = nest3(false, 30, 20, 5);
    let [inner, middle, outer] = [nest.headers[0], nest.headers[1], nest.headers[2]]
        .map(|header| span_of(&nest.chunk, header));
    assert_eq!(
        nest::enclosing_loops(&nest.chunk, inner),
        vec![middle, outer],
        "innermost first"
    );
    assert_eq!(nest::inner_loops(&nest.chunk, middle), vec![inner]);
    assert!(!nest::has_siblings(&nest::inner_loops(&nest.chunk, outer)));
    assert!(nest::can_cover(&nest.chunk, inner));
    assert!(nest::can_cover(&nest.chunk, middle));
    assert!(!nest::can_cover(&nest.chunk, outer), "nothing encloses it");

    // Two inner loops side by side.
    let two = two_inner_loops(false, 10, 3);
    let outer = span_of(&two.chunk, two.headers.iter().copied().min().unwrap());
    assert!(nest::has_siblings(&nest::inner_loops(&two.chunk, outer)));
    for header in &two.headers {
        let span = span_of(&two.chunk, *header);
        if span != outer {
            assert!(!nest::can_cover(&two.chunk, span), "{span:?}");
        }
    }
}

/// `nest2` with a call of `f` (the closure of prototype 0, `x + 1`) in the
/// outer part, run only when `conditional` says so: `if i % 3 < 1 then
/// f(i) end` or `f(i)` in every iteration.
fn call_in_outer_part(integers: bool, outer: i64, conditional: bool) -> Nest {
    let mut asm = Asm::new(integers);
    asm.code.push(Instruction::closure(8, 0));
    asm.set(9, 0);
    asm.set(10, 0);
    asm.set(11, outer);
    asm.set(12, 6);
    asm.set(13, 1);
    asm.set(15, 3);
    asm.open_for(0, 11);
    asm.open_for(4, 12);
    asm.add(9, 9, 7);
    asm.close_for();
    if conditional {
        asm.code.push(binary(OpCode::Modulo, 16, 3, 15));
        asm.code.push(binary(OpCode::LessThan, 17, 16, 13));
        let test = asm.pc();
        asm.code.push(jump_if_false(17, test, test + 2));
        asm.code.push(Instruction::call_one(14, 8, 3));
    } else {
        asm.code.push(Instruction::call_one(14, 8, 3));
    }
    asm.add(10, 10, 13);
    asm.close_for();
    let mut nest = asm.finish(18, (9, 2));
    let callee = Chunk::with_parts(
        3,
        1,
        vec![number(integers, 1)],
        vec![
            load(1, 0),
            binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    nest.chunk = Chunk::with_parts(
        18,
        0,
        nest.chunk.constants.clone(),
        nest.chunk.code.clone(),
        vec![callee],
        vec![],
    );
    nest
}

#[test]
fn a_call_in_the_outer_part_does_not_rule_out_a_tree_whether_or_not_a_branch_skips_it() {
    // A side inlines a call of a Lua closure like the root does, so a call
    // that runs in every outer iteration leaves the loop coverable, and so
    // does one in a branch (the side records one arm and the other stays a
    // guard of it).
    for conditional in [false, true] {
        let nest = call_in_outer_part(false, 200, conditional);
        let inner = span_of(&nest.chunk, nest.headers[0]);
        assert!(nest::can_cover(&nest.chunk, inner), "{conditional}");
        let expected = interpreted(CompatibilityProfile::Lua51, &nest.chunk);
        let (vm, results) = run_traced(CompatibilityProfile::Lua51, &nest.chunk);
        assert_eq!(results, expected, "{conditional}");
        let stats = stats_of(&vm);
        assert!(stats.side_recordings >= 1, "{conditional}: {stats:?}");
    }
}

/// `for i = 1, outer do for j = 1, i do a = a + j end f(i) end` where `f`
/// makes a table, which no side recording can follow: every recording of the
/// outer part aborts, though the opcodes of the loop say it can be covered.
/// The inner loop is short while `i` is small and long later.
fn unrecordable_callee_in_outer_part(integers: bool, outer: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.code.push(Instruction::closure(8, 0));
    asm.set(9, 0);
    asm.set(11, outer);
    asm.open_for(0, 11);
    asm.open_for(4, 3);
    asm.add(9, 9, 7);
    asm.close_for();
    asm.code.push(Instruction::call_one(14, 8, 3));
    asm.close_for();
    let mut nest = asm.finish(18, (9, 1));
    let callee = Chunk::with_parts(
        4,
        1,
        vec![number(integers, 1)],
        vec![
            load(1, 0),
            Instruction::new_table(3, 0, 0),
            Instruction::return_(1),
        ],
        vec![],
        vec![],
    );
    nest.chunk = Chunk::with_parts(
        18,
        0,
        nest.chunk.constants.clone(),
        nest.chunk.code.clone(),
        vec![callee],
        vec![],
    );
    nest
}

/// `for i = 1, outer do for j = 1, 3 do a = a + j end f(i) end` where `f` makes
/// a table, which no side recording can follow: the inner loop is short and
/// the loop around it is one a tree carries, but every recording of the outer
/// part aborts.
fn unrecordable_callee_after_a_short_loop(integers: bool, outer: i64) -> Nest {
    let mut asm = Asm::new(integers);
    asm.code.push(Instruction::closure(8, 0));
    asm.set(9, 0);
    asm.set(11, outer);
    asm.set(12, 3);
    asm.open_for(0, 11);
    asm.open_for(4, 12);
    asm.add(9, 9, 7);
    asm.close_for();
    asm.code.push(Instruction::call_one(14, 8, 3));
    asm.close_for();
    let mut nest = asm.finish(18, (9, 1));
    let callee = Chunk::with_parts(
        4,
        1,
        vec![number(integers, 1)],
        vec![
            load(1, 0),
            Instruction::new_table(3, 0, 0),
            Instruction::return_(1),
        ],
        vec![],
        vec![],
    );
    nest.chunk = Chunk::with_parts(
        18,
        0,
        nest.chunk.constants.clone(),
        nest.chunk.code.clone(),
        vec![callee],
        vec![],
    );
    nest
}

#[test]
fn a_baseline_root_whose_outer_side_never_records_is_compiled_again_at_the_policy_tier() {
    // The inner loop is short, so its root compiles at the baseline tier to
    // lead to a tree. The outer part cannot be recorded, so no tree comes:
    // the root is what runs, and it gets the optimized tier.
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = unrecordable_callee_in_outer_part(integers, 400);
        let inner = span_of(&nest.chunk, nest.headers[0]);
        assert!(nest::can_cover(&nest.chunk, inner), "{profile:?}");
        let expected = interpreted(profile, &nest.chunk);
        let mut vm = native_vm(profile);
        vm.jit.trace.set_stage_roots(false);
        let results = vm.execute(&nest.chunk).unwrap();
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        assert_eq!(stats.baseline_roots, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.outer_sides, 0, "{profile:?}: {stats:?}");
        let index = vm
            .jit
            .trace
            .installed_index_at_pc(nest.headers[0])
            .unwrap_or_else(|| panic!("{profile:?}: the inner loop has a trace: {stats:?}"));
        assert_eq!(
            vm.jit.trace.installed_tier(index),
            Some(OptimizationTier::Optimized),
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_staged_root_whose_outer_side_never_records_is_compiled_at_the_policy_tier() {
    // The outer part cannot be recorded ahead, so no tree comes: the
    // placeholder gives way to the root alone, at the policy's tier, and no
    // baseline code is compiled on the way.
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = called_repeatedly(
            integers,
            unrecordable_callee_after_a_short_loop(integers, 28),
            60,
        );
        let header = nest.headers[0];
        let chunk = Arc::new(nest.chunk);
        let mut vm = native_vm(profile);
        let mut frames = frames_for(&vm, &chunk, None);
        // The root is installed with no code while its side is recorded.
        let mut index = None;
        for _ in 0..2_000_000 {
            if let Some(found) = vm.jit.trace.installed_index_at_pc(header) {
                index = Some(found);
                break;
            }
            assert!(vm.step(&mut frames).unwrap().is_none());
        }
        let index = index.unwrap_or_else(|| panic!("{profile:?}: no root"));
        let stats = stats_of(&vm);
        assert_eq!(stats.compile_submissions, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.native_entries, 0, "{profile:?}: {stats:?}");
        // The recording of the outer part fails; the root compiles alone at
        // the policy's tier, replacing the placeholder.
        let mut tier = None;
        for _ in 0..2_000_000 {
            tier = vm.jit.trace.installed_tier(index);
            if tier == Some(OptimizationTier::Optimized) || vm.step(&mut frames).unwrap().is_some()
            {
                break;
            }
        }
        let stats = stats_of(&vm);
        assert_eq!(
            tier,
            Some(OptimizationTier::Optimized),
            "{profile:?}: {stats:?}"
        );
        assert_eq!(stats.compile_submissions, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.baseline_roots, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.outer_sides, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn under_a_baseline_policy_no_root_is_a_stepping_stone() {
    // Every root is baseline code and no tier is better: a root whose
    // outer side never records is not compiled again at the same tier.
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = unrecordable_callee_in_outer_part(integers, 400);
        let expected = interpreted(profile, &nest.chunk);
        let mut vm = native_vm(profile);
        vm.jit.trace.set_compile_tier(OptimizationTier::Baseline);
        let results = vm.execute(&nest.chunk).unwrap();
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        assert_eq!(stats.baseline_roots, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.compile_submissions, 1, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_baseline_root_whose_outer_side_is_refused_by_the_tree_is_compiled_again_at_the_policy_tier() {
    // The refusal happens inside the tree build, after a side recorded, so
    // neither an abort nor a failed compile follows it: the check for a stone
    // that has run out of sides has to run there too.
    for profile in PROFILES {
        let nest = nest2(has_integers(profile), 400, 3);
        let header = nest.headers[0];
        let chunk = Arc::new(nest.chunk);
        let mut vm = native_vm(profile);
        vm.jit.trace.set_stage_roots(false);
        let mut frames = frames_for(&vm, &chunk, None);
        let mut index = None;
        for _ in 0..2_000_000 {
            if let Some(found) = vm.jit.trace.installed_index_at_pc(header) {
                index = Some(found);
                break;
            }
            assert!(vm.step(&mut frames).unwrap().is_none());
        }
        let index = index.unwrap_or_else(|| panic!("{profile:?}: no root"));
        assert_eq!(
            vm.jit.trace.installed_tier(index),
            Some(OptimizationTier::Baseline),
            "{profile:?}: the root is a stepping stone"
        );
        {
            let jit = &mut vm.jit;
            jit.trace
                .queue_sides_the_tree_refuses(&mut jit.statistics.trace, index);
        }
        // The recompile lands while the nest still runs.
        let mut tier = None;
        for _ in 0..2_000_000 {
            tier = vm.jit.trace.installed_tier(index);
            if tier == Some(OptimizationTier::Optimized) || vm.step(&mut frames).unwrap().is_some()
            {
                break;
            }
        }
        let stats = stats_of(&vm);
        assert!(stats.side_rejections >= 1, "{profile:?}: {stats:?}");
        assert_eq!(
            tier,
            Some(OptimizationTier::Optimized),
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_stone_recompile_refused_for_lack_of_room_is_asked_for_again() {
    for profile in PROFILES {
        let nest = nest2(has_integers(profile), 400, 3);
        let header = nest.headers[0];
        let chunk = Arc::new(nest.chunk);
        let mut vm = native_vm(profile);
        vm.jit.trace.set_stage_roots(false);
        let mut frames = frames_for(&vm, &chunk, None);
        let mut index = None;
        for _ in 0..2_000_000 {
            if let Some(found) = vm.jit.trace.installed_index_at_pc(header) {
                index = Some(found);
                break;
            }
            assert!(vm.step(&mut frames).unwrap().is_none());
        }
        let index = index.unwrap_or_else(|| panic!("{profile:?}: no root"));
        let jit = &mut vm.jit;
        jit.trace
            .queue_sides_the_tree_refuses(&mut jit.statistics.trace, index);
        let requests = jit.trace.take_outbox();
        assert_eq!(requests.len(), 1, "{profile:?}: the stone recompile");
        assert!(!jit.trace.wants_stone_recompile(index), "{profile:?}");
        for request in requests {
            jit.trace.note_submit_failed(
                &mut jit.statistics.trace,
                request,
                crate::jit::SubmitError::QueueFull,
            );
        }
        // Not right away, which the full broker would refuse again, but
        // the next side abort or tree failure asks again.
        assert!(jit.trace.take_outbox().is_empty(), "{profile:?}");
        assert!(jit.trace.wants_stone_recompile(index), "{profile:?}");
    }
}

/// `nest2`, then `for q = 1, 3000 do sum = sum + q end`: a second hot loop
/// that wants a trace once the nest's tree holds the cache's only slot.
/// Registers: r8 acc, r9 count, r18 sum.
fn nest_then_loop(integers: bool) -> Nest {
    let mut asm = Asm::new(integers);
    asm.set(8, 0);
    asm.set(9, 0);
    asm.set(18, 0);
    asm.set(10, 300);
    asm.set(11, 6);
    asm.set(12, 1);
    asm.set(13, 3_000);
    asm.open_for(0, 10);
    asm.open_for(4, 11);
    asm.add(8, 8, 7);
    asm.close_for();
    asm.add(9, 9, 12);
    asm.close_for();
    asm.open_for(14, 13);
    asm.add(18, 18, 17);
    asm.close_for();
    asm.finish(19, (8, 1))
}

#[test]
fn a_tree_evicted_for_another_trace_hands_its_covered_loops_back() {
    // The cache holds one trace, so the second loop's trace evicts the nest's
    // tree. The loop the tree covered must not stay claimed by a tree that is
    // gone: it is rejected, and the program's answer does not change.
    use crate::vm::jit::trace::cache::CacheLimits;
    let nest = nest_then_loop(false);
    let outer = nest.headers.iter().copied().min().unwrap();
    let chunk = Arc::new(nest.chunk);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    vm.jit.trace.cache_mut().set_limits(CacheLimits {
        traces: 1,
        ..CacheLimits::default()
    });
    let mut frames = frames_for(&vm, &chunk, None);
    let results = loop {
        if let Some(results) = vm.step(&mut frames).unwrap() {
            break results;
        }
    };
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert!(stats.covered_loops >= 1, "{stats:?}");
    assert_eq!(stats.cache_evictions, 1, "{stats:?}");
    assert_eq!(stats.uncovered_loops, stats.covered_loops, "{stats:?}");
    assert_eq!(
        vm.jit.trace.site_state_any(outer),
        Some(SiteState::Rejected),
        "{stats:?}"
    );
}

#[test]
fn a_short_inner_loop_gives_way_to_a_long_outer_loop_of_a_top_level_chunk() {
    // Three trips a run, the same twice in a row, inside a loop with enough
    // iterations left to repay the compile of the nest: the outer loop is the
    // root, with the inner loop unrolled in it, and the inner loop has no
    // trace and no outer side.
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = nest2(integers, 30000, 3);
        let expected = interpreted(profile, &nest.chunk);
        let (vm, results) = run_traced(profile, &nest.chunk);
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        // Its two closings turned the inner root away.
        assert_eq!(
            stats.aborts(TraceAbortReason::ShortLoop),
            2,
            "{profile:?}: {stats:?}"
        );
        assert_eq!(stats.outer_sides, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.baseline_roots, 0, "{profile:?}: {stats:?}");
        assert_eq!(
            vm.jit.trace.installed_index_at_pc(nest.headers[0]),
            None,
            "{profile:?}: the inner loop has no trace: {stats:?}"
        );
        assert!(
            vm.jit
                .trace
                .installed_index_at_pc(nest.headers[1])
                .is_some(),
            "{profile:?}: the outer loop has the trace: {stats:?}"
        );
        // One entry runs the rest of the outer loop.
        assert!(stats.native_entries <= 4, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 29000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_short_inner_loop_keeps_its_tree_when_the_outer_loop_is_short() {
    // 400 iterations left do not repay the compile of the unrolled nest, so
    // the inner loop is no different from before: a root with an outer side.
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = nest2(integers, 400, 3);
        let (vm, results) = run_traced(profile, &nest.chunk);
        assert_eq!(results, interpreted(profile, &nest.chunk), "{profile:?}");
        let stats = stats_of(&vm);
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(
            vm.jit
                .trace
                .installed_index_at_pc(nest.headers[0])
                .is_some(),
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_short_inner_loop_past_the_trips_it_may_give_way_with_keeps_its_tree() {
    // Twelve trips: unrolled, the nest costs more to get started than the
    // tree does for the first thousands of iterations (see
    // `policy::PREFER_UNROLL_TRIPS`).
    for profile in PROFILES {
        let integers = has_integers(profile);
        let nest = nest2(integers, 30000, 12);
        let (vm, results) = run_traced(profile, &nest.chunk);
        assert_eq!(results, interpreted(profile, &nest.chunk), "{profile:?}");
        let stats = stats_of(&vm);
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert_eq!(
            stats.aborts(TraceAbortReason::ShortLoop),
            0,
            "{profile:?}: {stats:?}"
        );
    }
}
