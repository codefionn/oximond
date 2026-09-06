# The MIR backend behind the trace trees (stages 2 to 7: lowering, calls, rejoining sides, loop sides, range-proven loops, the forward skip, the rejoin, exit-only values and the float remainder, string lengths and the prototype check)

Hashes below are those after the rebase onto `main` at `6847d447`. The measurements ran on the commits before it (the same code), and the hashes that name `flow/mir-skip` are that branch's own. Stage 7, the residue (the length of a lane string in line, a cheaper closure prototype check) is measured on a release build of the code of `2055a887` of `flow/mir-gaps`, against `15b7f761`, 2026-11-04; see "The residue of the gap list (stage 7, the residue)". Stage 7, exit-only values (values only an exit reads are computed in its stub, the float remainder of Lua 5.3 and later and `NumberModulo` in line) is measured on a release build of the code of `9390bee9` of `flow/mir-gaps`, against `066251ba`, 2026-11-03; see "Exit-only values and the float remainder (stage 7, exit-only values)". Stage 7, the rejoin (a rejoining side goes to a join below the entry, and the string-index probe is on by default) is measured on a release build of the code of `5fee3244` of `flow/mir-gaps`, against `411888bf` of `flow/mir-skip`, 2026-11-03; see "Rejoin straight to the loop header (stage 7, the rejoin)". Stage 6 (the string-index probe on the forward skip) is measured on a release build of `62dafd8e` (the code of `bf50f088`) of `flow/mir-skip`, 2026-11-03; see "String-index probe". Stage 4 (range-proven loop versions) is measured on a release build of `5bfcd50c` (the code, on `275f56a6`; rebased onto `ef374807` as `ffaf171d`), 2026-11-02; the rest of this document is as stage 3 left it.

Measured on commit `0b2db938` (a fix to the previous measured commit `b7003199`, see "Inline fast paths that Cranelift has") of `flow/mir-research`, 2026-11-01 (the suite, coverage, tests and conformance; the nbody and per-benchmark counts were rerun on it), except the loop kernels, the cycle study and the profile, which ran on `9d47fb9d` (the allocator and the kernels are the same on both, see "Measurements"). The code is `crates/oximond/src/jit/trace/mir/` and `crates/oximond-mir`. A claim with a number, a command or a file after it is a measurement or a fact about the code. A paragraph or bullet that starts with "Inference." is my own reasoning from those facts. Background reading is `docs/impl/native-backend/research.md`, `census.md`, `latency.md` and `prototype.md` in this directory.

## Summary

- The MIR prototype compiles real trace trees. `OXIMOND_JIT_BACKEND=mir` makes the Baseline and Optimized tiers try it first for every tree, and Cranelift takes what it refuses, tree by tree, with the reason logged under `OXIMOND_JIT_TRACE` and counted (`OXIMOND_JIT_STATS` prints the counts). Unset or `cranelift` runs no MIR code. The Peak tier and sides compiled apart stay Cranelift.
- This unit built the call op fully (stack arguments, a call through the control block, a conditional call off the straight line, frame cells), loop sides with the sides nested in them, and the lowering of the helper-backed operations the suite uses (`CallHelper`, `HelperResult`, `HostCall`, `ReleaseStrings`, `HashLoad`, `ArraySet`, table-valued array loads, `NumberLoad`). A tree that holds an operation Cranelift answers in line (see "Inline fast paths that Cranelift has") stays with Cranelift for now, so coverage on the 89 suite benchmarks at scale 3 is 83 of 128 native compiles accepted (64.8%) and 2714 of 4355 trace IR instructions (62.3%), against 64 of 128 (50.0%, 46.9% of the IR) after the previous round and 46 of 127 (28.9%) at stage 1. An earlier build of this unit accepted 110 compiles (85.9%) by calling the VM for those operations, and the suite cost +28.6% for it; that build is gone. 61 benchmarks have every compile accepted. What is left is listed with its reason under "Remaining fallbacks".
- Stage 7, the residue: the six benchmarks of the gap list that had no named cause are explained, and three of them were one cause. `HostCall` of `StringLength` went to the VM for every string a helper made, about 130 instructions a call, where Cranelift reads a word of the run's length table; it is in line now. `string/format` goes from +7.83% to -0.38% against Cranelift in the same binary, `string/pattern_run` from +6.58% to -2.55% and `string/utf8_pack` from +1.25% to -2.53%. The closure prototype check is cheaper too (`opt/closure_family` from -0.04% to -4.93%). Every benchmark of the suite is now within 0.1% of Cranelift or below it, and the suite total under MIR is 20,314.79 M against 20,616.07 M before (-1.46%) and 21,010.67 M under Cranelift. See "The residue of the gap list (stage 7, the residue)".
- Stage 7, exit-only values: a float operation that only snapshots read is not computed in the loop. The allocator computes it in each exit stub that holds it (`defer_cold_values` of Cranelift, as `COLD` operations), and the float remainder of the 5.3 and later profiles and `NumberModulo` are lowered in line. Against Cranelift in the same binary: `opt/cold_recovery` from +5.55% to -1.37%, `opt/mod_float` from +4.04% to -4.13%, `opt/mod_int` from +0.95% to -0.55%, `opt/mod_mixed` from 0.00% to -0.38%. Suite total under MIR: 20.616 G instructions against 20.818 G before (-0.97%). See "Exit-only values and the float remainder (stage 7, exit-only values)".
- Stage 7, the rejoin: a side that rejoins the root's loop goes to a `Join` below the entry loads (before the preheader when Cranelift's outer header runs it per link, after it otherwise), with what the loop reads kept in frame cells, instead of storing into the entry input slots and running the entry and the preheader again. `table/hash_read_indexed` is now 210.50 M instructions under MIR with the probe on against 214.06 M under Cranelift (-1.7%; it was +1.4%), so the probe is on by default (`OXIMOND_JIT_MIR_PROBE=0` turns it off). `opt/closure_protos` went from +11.3% to -0.5% against Cranelift, `control/loop_in_call` from +2.6% to -2.2%, `control/short_inner`, `control/branching` and `control/nested_calls` fall 3 to 8%. Suite total under MIR: 20.821 G instructions against 21.023 G before (-0.96%; Cranelift in the same binary 21.113 G); coverage 83.6% of compiles and 88.0% of trace IR instructions. See "Rejoin straight to the loop header (stage 7, the rejoin)".
- Stage 6: the string-index probe walks the probe sequence in line with nested forward skips, as Cranelift's loop does, and calls the VM only where Cranelift does. It was off by default through stage 6 because one benchmark missed the goals, which stage 7 fixed: with the probe on, `table/hash_read_indexed` runs 217.09 M instructions under MIR against 214.21 M on main (+1.35%) and 214.13 M under Cranelift (+1.4%), because of the cost of a side that rejoins. With the probe on the suite total under MIR is 20.864 G instructions against 21.058 G on main (-0.92%); seven of the eight benchmarks it brings back are below Cranelift (-6.5% to -32.7%); the six hash kernels are at most 5.0% above Cranelift and two are 18 to 20% below; coverage 82.0% of compiles and 86.5% of trace IR instructions. By default the figures are those of the closures commit (71.9% and 81.1%). See "String-index probe".
- Behaviour matches Cranelift's. All 89 benchmarks print the same under both backends. The tiered tracing conformance run has identical logs under the two (20, 7, 7, 5, 4 and 27 passed, none failed). `cargo test -p oximond -p oximond-cli -p oximond-mir --no-fail-fast` gives 4504 passed and 0 failed with the variable unset, and 8 failed with `OXIMOND_JIT_BACKEND=mir`. Each of those 8 asserts a fact only Cranelift's code can show (a CLIF count, or the native link of a side compiled apart), listed with the reason under "Test suites under both backends".
- nbody is on MIR, whole tree. Whole-process instructions at scale 3: Cranelift 33.66M, MIR 28.00M (-16.8%). Its loop costs 231 cycles an iteration against 230 (+0.2%) in the loop kernel (measured on `f18df2d9`, the lowering of nbody is unchanged). See "nbody".
- End to end, whole-process instructions with MIR are 0.39% higher over the whole suite (21.350G to 21.432G). 49 benchmarks are more than 1% lower (255.6M instructions in all), from -46.1% (`string/plain_find`) to -16.8% (nbody) and beyond, and none is more than twice Cranelift's count. The largest losses at that commit were `table/array_read` +16.8% and `opt/range_offset` +16.7%; stage 4 lowers their range-proven loop version and both are now below Cranelift (see "Range-proven loop versions"). The per-benchmark table says which is which, and "Gaps" says what each family needs.
- The stage 1 cycle gap on spill-heavy kernels is explained and fixed. Profiling the generated code by instruction (`benchmarks/mir_profile.py`) showed the cause on `pressure`: the allocator evicted the loop's float accumulator, a header parameter, so the loop stored it at its header and reloaded it before a chain of 18 dependent additions, and the store-to-load forwarding sat on the path each iteration waits for. A header parameter now costs a store on every iteration in the eviction score, as any value the loop defines does. `pressure` went from 62.9 cycles an iteration (+16.7% against Cranelift) to 54.1 (-0.1%), `guard_restore` is -13.4%.
- Compile cost is as stage 1 measured it (22.0 times fewer instructions per accepted tree than Cranelift, no allocation in a steady-state compile); this unit did not re-measure it.

Decisions that differ from the brief, and why:

- The hook is at the top of `emit::lower`, not in `compile.rs`. Both the tree path and the path of a side compiled apart reach it with the plan, the tier and the options in hand, and `emit.rs` already owns `TraceCode`, whose only backend-specific part is the code.
- An exit returns its index and a tag mismatch returns `-1`, as `emit.rs` does, not `-(i + 1)` as the brief's header summary says: the run finds the input.
- The brief's conformance command (`--execution tiered --jit-engine tracing`) does not exist in this tree. The run is `--execution tracing`.
- A root that Cranelift links natively to sides compiled apart stays with Cranelift unless the tree has a side of its own that rejoins: without the native link MIR code is correct but slower (the run loop calls each side), and several existing tests assert the link is native. A tree that has its own rejoining sides is compiled here.
- The Cranelift-built PreserveAll stubs (`admit_table`, `table_set`, `hash_get`, the helper stubs) are not called. They exist so a call on a cold path of Cranelift's code keeps every register. MIR has its own cold-call construct (`Op::CallIf`, see "Conditional calls") that saves what is live in a stub of its own, and the operations the stubs serve in Cranelift use it (the lane admission of an array load, the store into an array) or are plain calls the allocator already keeps the loop's values across (`HashLoad`, helper calls: callee-saved registers, or one store at the definition). Calling a stub would add a call level and a fixed signature per arity for nothing.
- A rejoining side goes to a `Join` that sits where Cranelift's outer header is, not to the root's entry (stage 7, see "Sides that rejoin the root loop"). The preheader runs again after a link exactly when Cranelift's does, so a root whose preheader calls into the VM is compiled here too.

## What was built

The MIR prototype of stage 1 is now a second native backend for trace trees. `OXIMOND_JIT_BACKEND=mir` makes the Baseline and Optimized tiers try it first, tree by tree, and hand any tree it cannot compile to Cranelift. Unset, empty, `cranelift` or any other value keeps Cranelift for everything and runs no MIR code. The Peak tier and the sides compiled apart always use Cranelift.

The pieces:

| Where | What |
|---|---|
| `crates/oximond/src/jit/debug_env.rs` | `backend_mir()`: reads `OXIMOND_JIT_BACKEND` once per process. `backend_mir_only()`: `OXIMOND_JIT_BACKEND=mir-only` sends nothing to Cranelift (declined trees and sides compiled apart are rejected, no Peak tier), for `oximond-difftest`'s `tracing-mir-only` mode. `mir_probe()`: reads `OXIMOND_JIT_MIR_PROBE`; the string-index probe is on unless it is `0` or empty (see "String-index probe"). Under test a thread-local can override each. |
| `crates/oximond/src/jit/trace/emit.rs` | The hook, at the top of `lower`: a tree that is not a side compiled apart, not at the Peak tier and not a forced-guard test asks `mir::lower` first. `TraceCode::from_mir` wraps the result (no `JITModule`, the entry is the address of the block). A few dozen lines changed in this file. |
| `crates/oximond/src/jit/trace/mir/lower.rs` | Trace IR to MIR, one forward pass. |
| `crates/oximond/src/jit/trace/mir/lower/calls.rs` | The operations that call into the VM: helpers, host calls, hash loads, table stores, the release of strings. |
| `crates/oximond/src/jit/trace/mir/support.rs` | The thread-local arenas, the executable region, counters, log lines, the panic guard. |
| `crates/oximond/src/jit/trace/mir/tests.rs` | Cases that compile with both backends and compare every observable (see "Tests"). |
| `crates/oximond/src/jit/icount.rs` | `OXIMOND_JIT_ICOUNT`: the instructions a thread retired in a compile, for the measurements below. |
| `crates/oximond-mir` | New ops, operand forms and a code region (see "Changes to oximond-mir"). Still free of oximond types. |
| `crates/oximond-cli/tests/mir_backend.rs` | Lua snippets under both backends. |
| `benchmarks/mir_backend.py` | Suite runner: output identity, coverage, instruction counts. |

`crates/oximond` depends on `oximond-mir` only for Linux and Windows on x86-64 (`[target.'cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))'.dependencies]`). Other targets compile `mir/unsupported.rs`, which never accepts a tree. `oximond-mir` is publishable (it no longer carries `publish = false`), so `scripts/ci.sh package` packages it ahead of `oximond`. `GuardType` is the one lowered op no Lua program reaches, because the recorder does not emit it; a hand-built tree in `mir/tests.rs` covers it.

## The hook and the fallback rule

`mir::lower(members, plan, tier, forced_guard, options)` takes exactly what `emit::lower` takes, and the exit plan `plan_exits` made, unchanged. It returns the compiled code or `None`. When it returns `None` the reason has been logged and counted already:

- Accepted: `jit-trace: mir backend compiled tree: N IR instructions, M MIR instructions, B bytes`.
- Refused: `jit-trace: mir backend fallback to cranelift (N IR instructions): instruction I: reason`, with the reason a fixed string.
- `OXIMOND_JIT_STATS` prints a `mir:` line: accepted and refused trees, their IR instructions, and the refusals by reason. The same numbers come from `oximond::jit::mir_counters()`.

A panic inside the MIR compiler (the allocator asserts on states it does not handle) is caught in `support.rs`, the thread's arena is dropped, and the tree goes to Cranelift with the reason "the MIR backend panicked". It does not reject the trace, which a panic that reached the broker's `catch_unwind` would.

What the hook refuses before it lowers anything:

- the Peak tier, a forced guard (a test device), a CPU without SSE4.1 (`roundsd`), no executable region;
- function traces and counted traces;
- a root with loop parameters its loop passes on unchanged and link slots, which Cranelift's code links to sides compiled apart natively through the exit cells (`native_link`), when the tree has no side of its own that rejoins. MIR code returns the exit index and the run loop calls the side, which is correct but slower, and the tests of the native link assert it, so those roots stay with Cranelift;
- a loop side whose parameters do not all have a register, and a nested side that rejoins a loop that is not around it.

Sides that end in an exit of their own are compiled, nested ones too, and so are sides that rejoin the root's loop (`Link`), nested ones too, loop sides (a member's exit that another member continues) and the sides nested in them that rejoin their loop (`LinkLoop`).

## The lowering

The ABI is the header of `emit.rs`, honoured bit for bit. Two points where the code decided:

- An exit returns its index, not `-1 - index` (that is the function-trace ABI), and an entry tag mismatch returns `-1` before anything runs, whichever input it was. `CompiledTrace::run_checked` finds the input.
- The exit slots, the fuel and nothing else are written at an exit. The tails Cranelift shares between exits (`ExitPlan::tails`) are a code-size device and store the same words, so MIR stubs store `descriptor.slots` whole.

How the pieces map:

| Contract | MIR |
|---|---|
| `entry(inputs, control, exit_area)` | `Entry` defines `inputs` and `control`. The prologue puts the exit-area pointer in frame slot 0 and the control pointer in frame slot 1 (two stores per entry), so stubs need no register. |
| Entry checks | One `Guard Eq` per input on the tag at `16 * i`, against `entry_tag(ty)`. All of them share a snapshot with no values that returns `-1`. |
| Entry values | `Load` of the bits at `16 * i + 8`. A boolean becomes 0 or 1 by a select. Nil is the constant 0. |
| Fuel | `control.fuel` is loaded once and is a header parameter. The back edge is `fuel' = fuel - 1`, a `Guard Gt fuel' 0` that leaves through the poll exit, and `Back`. |
| Guard exit | A MIR guard whose snapshot is the exit descriptor: the slots of the descriptor go to the exit area, the fuel goes to `control.fuel` (a snapshot slot with the `CONTROL_SLOT` bit set), the stub returns the index. Snapshot `k` of the MIR function is descriptor `k`, so nothing maps indexes. |
| Booleans | 64-bit integers holding 0 or 1. A `Compare` stays a pending condition that a guard or a select consumes. It becomes a select of 1 and 0 only if something else reads it. |
| What a guard knows | After `GuardBranch c expected` the value `c` is the constant `expected`, and in the guard's own exit it is the other one, so a snapshot that keeps `c` stores a constant and the compare is never made. |
| Sides | Each side is a region. A guard that has a side passes the side's entry inputs and frame inputs, then the fuel, the inputs pointer and the control pointer. The side reloads table words and rows from those pointers (a side cannot read the root's values). |
| Table words | Binding words (array base and length, field slot, length) are loaded in the prologue of the region that reads them, before the loop header, so the allocator treats them as loop invariants. |
| Rows | `control.rows` plus `lane * row_stride`. The field word is nonzero or the guard leaves. `MetaGuard` compares a row word with the pinned metatable identity. The group check is the same as Cranelift's (`slot_group`): at a group's first access one load of the presence mask and one compare cover every field the straight-line code after it reads or writes, and the tree leaves through the first access's guard (a group with a field of 64 or more ands the slot tests instead). |
| Array access | Bounds by one unsigned compare for integer keys. For float keys the three checks of `index_within` (`>= 1`, `<= len`, integral by adding and subtracting 2^52), then `cvttsd2si` for the index. `StringByte` has a shorter form of its own, see "Byte groups" under "String-index probe". A tag check by a byte load, then the payload load or store. A second access to the same slot with the same key skips the bounds check and, for the same type, the tag check. |
| Pass-through parameters | A loop parameter the back edge passes on unchanged is its first value in every iteration. It is not a header parameter: it is a parameter of the root's `Join` (see "Sides that rejoin the root loop"), which a link gives a new value. The preheader reads every parameter as its first value, as Cranelift's outer header gives it. |
| Rejoining side | `Link`: one unit of fuel is spent, and `Rejoin` jumps to the root's `Join` with the new value of every loop parameter and the fuel that is left. |
| `count_links` | A load, add and store on `control.links` at the start of each side. |

## What it lowers, and what it hands to Cranelift

Lowered: `Constant`, `Move`, `GuardType`, `Negate`, `Arithmetic`, `IntToFloat`, `ToWide`, `Numeric` (`Floor`, `Ceil` and `Abs` of floats, `Abs` of integers, `FloatToInteger`, `TrickBits`), `NativeUnary` (`math.sqrt` inline, the other functions by a call), `NativeBinary` (a call), `Compare`, `Select`, `Truthy`, `Not`, `BoolAnd`, `GuardBranch`, `ArrayLoad` (floats, integers, booleans, and tables, closures, threads, strings and holes), `ArrayStore` of floats, integers and booleans, `ArraySet`, `NumberLoad`, `NumberTag`, `HashLoad`, `CallHelper`, `HelperResult`, `HostCall`, `ReleaseStrings`, `FieldLoad`, `FieldStore`, `TableFieldLoad`, `TableFieldStore`, `MetaGuard`, `TableLength`, `StringLength`, `StringByte`, `PhantomTable`.

`Arithmetic` in detail:

| Operand type | Lowered | Refused |
|---|---|---|
| Float | add, subtract, multiply, divide, `Modulo` and `ModuloWith(Floor)` (`a - floor(a / b) * b`), `FloorDivide`, `PowerSquare` (`a * a` for the constant 2, a call of the `pow` helper and a select otherwise), `Power` (a call), `ModuloWith(Fmod53)` and `ModuloWith(Fmod54)` (a call of the compat function) | |
| Integer | add, subtract, multiply, and, or, xor, floor division and modulo (a mask or shift for a constant power of two, the sequences of `emit.rs` otherwise, with `idiv` guarded against 0 and -1), shifts by a constant, shifts by a value (BMI2 targets, stage 8) | shifts by a value on a target without BMI2 |
| `Int64`, `UInt64` | add, subtract, multiply, and, or, xor, divide, modulo (signed and unsigned, never trapping), shifts and rotations by a constant, comparison (signed or unsigned), negate, conversion to float (an unsigned one by halving with a sticky bit, then doubling), `ToWide` from a float by truncation, `ToWide` of the `Bits32` kind (a call) | shifts and rotations by a value |

Refused with a logged reason: `SelfCall` and function traces, `ArrayNext`, an array load past the end of the array, a pattern helper that has a prepared twin, an integer compared with a float (exact comparison calls a helper), counted traces, `Read` and `Effect`. The counts, and why each is still refused, are in "Remaining fallbacks".

## Changes to oximond-mir

The crate stays free of oximond types. It grew what the lowering needed:

- Ops: `FNeg`, `FAbs`, `FFloor`, `FCeil` (the last two `roundsd`), `F2I` (`cvttsd2si`), `Load8`, `Store8`, `SDiv`, `SRem`, `UDiv`, `URem`. A function with a division does not allocate rdx (`Target::division_regs`), so `idiv` and `div` need no save and the allocator needs no fixed registers. The lowering never gives them a divisor of 0 or -1.
- `Select` with float arms, float compares, or both (`select_any`). Float arms on a float compare are `vcmpsd` and `vblendvpd` on AVX. The other mixes are built in the scratch registers through integer `cmov`, with the ordered-compare rules for NaN.
- A guard of a value against zero can use the flags of the `Add` or `Sub` before it (`Guard` with `scale` 1). The lowering uses it for the fuel, so the back edge is `sub`, `jle`.
- A snapshot-only value takes a register when the register file has more than three to spare (it is the first to be evicted, so it never costs another value its register), where stage 1 always gave it a slot written on every pass of the loop.
- Conditions `Nlt`, `Nle`, `Ngt`, `Nge`: the negations of the ordered float compares, true on NaN. A guard that continues when `!(a < b)` needs them, and `Cc::negate` makes them.
- Snapshots: a value can go to the control block (`CONTROL_SLOT`), a snapshot can return a code of its own, and `reserve_snaps` and `set_snap` let a lowering fill snapshots whose index is fixed by an outside plan.
- Frame slot 1 holds the control pointer (frame slot 0 still holds the exit-area pointer).
- `exec::CodeRegion`: one read-write-execute mapping of 128 MiB per process, reserved with `MAP_NORESERVE`, carved per compile by `carve(len, fill)`. A first-fit free list with merging takes blocks back when a `CodeBlock` drops. Inference: write protection per compile would make the pages the VM thread runs in unreadable for execution while a compile thread writes the next block into the same page, so the region is mapped RWX once. A system that refuses RWX mappings gets the fallback reason "no executable code region".
- `Compiler::write_code`: writes the image into the block without an intermediate vector.
- Stage 2b: `Op::Call` (with `Target::arg_regs`, `ret_reg`, `call_frame` and `call`), `Op::Join` and `Op::Rejoin` with `Tgt::Outer`, and an eviction score that charges a value the loop defines for the store it would make on every iteration (`LOOP_STORE_PENALTY`). See "Calls and rejoining sides (stage 2b)". The evaluator calls the same C functions through a transmute, the Cranelift comparator of the tests emits `call_indirect`, and the random program generator emits calls.
- Stage 2c: stack arguments for `Op::Call`, `Op::CallCtl` (the function pointer in the control block), `Op::CallIf` (a conditional call out of line), `Op::CellStore` and `Op::CellLoad` (frame cells), `Op::Goto` (enter a side from the end of a region), `Op::Rejoin` naming the region it jumps to (a `Join` in a side as well as in the root), `Op::Load32`, `Op::Bits` and `Op::FromBits` (float bits to and from an integer register), per-region loop state in the allocator (header, invariants, join, label), `Target::call_ctl`, `Target::load32`, and the frame layout: stack arguments, cells, the save slots of `CallIf`, then the spill slots. `ExecMem::address`, and `examples/bench.rs` can write the machine code of a run (`BENCH_DUMP`) for `benchmarks/mir_profile.py`. The CLIF translator of the tests and the evaluator know the new ops, and the random programs call a nine-argument function and keep values in cells.
- Stage 6: `Op::If` and `Op::EndIf`, a forward skip with join values (see "The forward skip").
- Stage 7, exit-only values: `COLD` (the `scale` of a float operation only snapshots read, computed in the exit stubs, see `ra/cold.rs`), `FTrunc` (`roundsd` with 11), `FCopySign`, `FNma` (`c - a * b` fused, `vfnmadd213sd`, only with `X64::fma`) and `CallIf` on a float compare. `Target` gained `fused`, `fnma` and `copysign`. The SSE2 form of `FNeg` and `FAbs` was wrong for a result that is computed into the scratch register (a value only a slot holds), and is fixed.
- Tests: `tests/ops.rs` (every new op on both encodings against Rust), `tests/region.rs` (carving, reuse, merging, running code in a block), `tests/calls.rs`, `tests/rejoin.rs`, `tests/loopside.rs`.

## Calls, conditional calls, rejoining sides and loop sides (stages 2b and 2c)

Stage 2b added a call op and sides that jump back into the root's loop. The next round (2c) added what the helper-backed operations and the nbody tree needed: stack arguments, a call through the control block, a conditional call off the straight line, frame cells, and a loop inside a side with the sides nested in it. All of it is checked against the reference evaluator in `crates/oximond-mir/tests/` and against Cranelift in `crates/oximond/src/jit/trace/mir/tests.rs`.

### The call op

`Op::Call` names a C function by address (`imm`), takes `lists[a..a+b]` as arguments and gives an optional result of type `ty`. Integer arguments go to rdi, rsi, rdx, rcx, r8, r9 in order and floats to xmm0 to xmm7 (`Target::arg_regs`); the result comes back in rax or xmm0 (`Target::ret_reg`). The call itself is `mov rax, imm64; call rax`.

How the backward allocator handles it (`ra.rs`, `do_call`):

- The prescan marks every value that is live across a call: one used after a call that came after its definition, and every loop invariant of a loop that holds a call. An integer value so marked prefers a callee-saved register (rbx, rbp, r12 to r15), so a loop that calls keeps its integers where the callee leaves them.
- All of xmm0 to xmm14 are caller-saved in the System V ABI, so a float live across a call lives in a frame slot. The allocator does not special-case that. At the call it evicts every value still bound in a caller-saved register, the way it evicts under pressure: the value is stored once at its definition and read back by a load right after the call. A value that was already evicted for pressure costs nothing more.
- The arguments are one parallel move into the ABI registers (the same planner as the back edge, with cycle breaking). Each argument is hinted to its own ABI register, so most moves vanish. The result moves out of rax or xmm0 before the reloads, because a reload may use either register.
- Arguments past the sixth integer or the eighth float go to the stack, in order, as the ABI says. The frame starts with an outgoing area of as many words as the largest call needs (`Compiler::out_slots`), and every slot number is a word offset from the stack pointer, so a stack argument is a destination of the same parallel move (a slot to slot copy goes through the second scratch register).
- The stack is 16-byte aligned at the call: the prologue pushes the callee-saved registers the function uses, and `Target::call_frame` pads the frame by 8 bytes when that count is even. A test (`the_stack_is_aligned_at_the_call`) calls a function written in assembly that reports the alignment it sees, with 1 to 14 live values (so every parity of pushed registers).
- The checking shim of `exec.rs` puts a distinct value in every callee-saved register before each test run, so a call path that clobbers one fails the test.

`Op::CallCtl` is the same call with the function read from a byte offset of the control block (`mov rax, [frame slot of control]; call [rax + offset]`). The host entry (`HostCall`, `ReleaseStrings`) and the VM's table-store entry (`ArraySet`) are such pointers: they are run-time values, not addresses the lowering knows.

### Conditional calls

`Op::CallIf` is a call made when `a cc b` holds (integers) and skipped otherwise, with no result. This is the cold-call construct, and it exists for one purpose: a call that most iterations do not make must not make the loop keep its values on the stack.

- The call may leave its result in the control block (`CallIf`'s `e` field, a byte offset): the stub stores rax there after the call, and the code after the instruction reads that word. It holds what it held before when the call is not made, so a caller that needs the answer either way reads it only on the path that made the call, or selects between it and a value it computed in line.
- The straight line pays one compare and a branch. The branch goes to a stub that is part of the exit stubs (after the body, so the hot code stays dense): it saves every caller-saved register that holds a value into a dedicated frame slot (32 slots after the cells, reserved only when the function has a `CallIf`), places the arguments with a parallel move, calls, restores the registers and jumps back to the code after the instruction.
- Registers that hold values are known exactly at that point of the backward pass (`reg_val`), so the stub saves what is there and nothing else. The arguments are weak uses (snapshot-like): they take no register on the straight line.
- Callee-saved registers need no save, the callee preserves them. The shared stubs of `SharedStubs` (`admit_table`, `table_set`, `hash_get`, the helper stubs) do the same job in Cranelift by preserving every register inside a compiled function. The MIR stub does it with the target's own move instructions and the allocator's knowledge of what is live, so no Cranelift-built stub is called.

Tests: `a_conditional_call_keeps_every_live_value` (2 to 13 integers and 1 to 13 floats live across a `CallIf` that is taken and one that is not, both by address and through the control block, SSE2 and AVX, the callee's effect counted).

Where it is used: `ReleaseStrings` (the check of the pending string weight is the condition, the VM call is the stub), the lane admission of a table loaded from an array (the lane index probe is in line, the dispatcher's admission is the stub) and the store into an array slot (the overwrite of a slot that holds a value of the stored type is in line, the VM's store entry is the stub). `HashLoad` of a string key has an inline form only behind `OXIMOND_JIT_MIR_PROBE` (see "String-index probe"); by default a tree that Cranelift probes is declined, and any other lookup calls the VM every time.

### Helper-backed operations

The lowering is `crates/oximond/src/jit/trace/mir/lower/calls.rs`. Each operation calls the function Cranelift's code reaches on its slow path, with the arguments it builds:

| Operation | What the code does |
|---|---|
| `CallHelper`, `HelperResult` | The encoding of `call_helper_call`: the host context if the helper wants it, the address of the control block's `helper_out`, the address of the binding slots for a helper that rewrites binding words, the count of a variadic call, then each parameter (a value is two words, tag and bits; a float passes in an xmm register; a boolean as 0 or 1; a missing parameter as two zero words). The helper address is the one of `helpers::helper_address`. A helper that can miss returns a status and a guard on it leaves through the call's guard. The result words are loaded right away from `helper_out`, so `HelperResult` reads a value. After a helper that returns a table, `trace_ensure_row` makes the row of its lane when `emit.rs` does (the handler helpers, a helper that rewrites arrays, a tree that lists table addresses, and since stage 8 the lookup `TABLE_GET_TABLE` in a loop tree with row fields, see "The lookup row fill"), and after one that changes a metatable the row of the complement lane. A helper that has a prepared twin (a pattern the trace holds as a constant) is refused, with its reason. |
| `HostCall` | The arguments as tag and bits in the control block's `host_args`, the call through `control.host_entry` (`CallCtl`), a guard on the tag of `host_result`, then its bits. `math.max` and `math.min` of two integers are a select. `StringLength` of a string held by lane reads the run's length table in a skip, and a root lane goes to the VM in a second skip (stage 7, the residue, `string_length`). |
| `ReleaseStrings` | See the conditional calls. |
| `HashLoad` | A borrowed string key with a scalar expected type in a tree that has `native_probe`: with `OXIMOND_JIT_MIR_PROBE` set, the probe of the run's string index in line (see "String-index probe"); without it the tree is declined with `HashLoad without the inline probe`. Any other key: a call of `trace_hash_get` (the function the `hash_get` stub forwards to) with the key as tag and bits (an owned string, an integer or a float), a guard on the tag of `hash_result`, then its bits. |
| `ArraySet` | For a float, integer or boolean stored under a number key: the slot's address is computed from the key and the array words, and replaced by a word of the control block when the key is outside the array or not whole (so nothing outside the array is read or written), the tag is compared with the stored type, and the payload is stored there, all without a branch. Where the store did not happen a `CallIf` calls the VM's store entry through `control.set_entry` with the key and the value as tag and bits, and leaves its status in `set_status`, which a guard reads. Any other value (a string, a table, nil) goes straight to the VM's store. The array words are read again afterwards. No in-place append: the VM does it. |
| `ArrayLoad` of a table, closure, thread or hole | A tag check, then the lane: the first place of the run's lane index when it names the table and its row is filled (a hash of the table's handle word, loads and compares, no branch), the dispatcher's admission (`trace_admit_table`, a `CallIf` that leaves its answer in `admit_lane`) otherwise, and a guard on the lane. The probe is the one `admit_table_probe` makes in `emit.rs`, used under the same condition (a tree with no function trace, no table made and no coroutine resumed). Of a string: the address of the slot, as a borrow. |
| `NumberLoad`, `NumberTag` | The slot is read as a float and as an integer, a guard that the tag is a float, or an integer of magnitude at most 2^53, and a select. |

Inference. Without the two fast paths above, a first measurement of the suite had `table/array_write` at 9 times Cranelift's instructions and `algorithm/sieve` at 7.4 times (an array store through the VM costs about 430 instructions, an in-place one about 10), and nbody had gained only 2% over Cranelift, because every `bodies[j]` asked the dispatcher for a lane. The remaining calls that Cranelift answers in line (the hash probe and the stores through it, the closure rows, the in-place field fill of a made table, the native stepping of `pairs`) are slower here by the same kind of factor, which the per-benchmark table shows. `Op::CallIf` is the mechanism to put a fast path in front of each of them.

A borrowed string (the address of the array slot that holds it) is a plain integer value here: it goes to a helper or a host call as a borrowed tag, to a hash lookup as a borrowed key, and into a side only as a borrowed input.

Binding words. A member that has a call that may move storage (`ArraySet`, a helper that `moves_storage` or `takes_words`) does not load the words of its array and length bindings once at entry. It loads each where it first uses it after the last such call, forgets the cached words, slot facts and row addresses after the call, and forgets them again at its loop header (`words_of`, `forget_words`). A member without such a call keeps the entry loads, so the code of the trees that did not change is the same.

The gate `trace that sets table entries` is gone for this reason: the words are read again, as `emit.rs` does after the same calls (`forget_array_words`).

### Sides that rejoin the root loop

A side that ends in `TraceTerminator::Link` hands the root new values for its loop parameters and continues the root's loop. A back edge into the root's loop header does not work in a model where each region allocates on its own: a side cannot see the values the root defined before its header (table words, the preheader's work), and so cannot put them where the header expects them. Cranelift avoids this with an outer header: entry and every link jump to it with a value for every loop parameter, and every value defined above it dominates the side.

MIR does the same with one more header, the root's `Join`, placed where Cranelift's outer header is (stage 7; stages 2 to 6 put it above the entry loads, see "Gaps" for the cost that had):

- Cranelift runs the preheader at the outer header when the loop passes some parameter on unchanged (`pass_through_params`, a parameter only a side's link changes), so each link redoes it with the new values; otherwise the preheader runs once at the entry and a link jumps to the loop header. `Lowerer::enter_loop` puts the `Join` before the preheader in the first case and after it in the second.
- `Op::Join` takes the start of every loop parameter and the fuel. A link (`Lowerer::link`) spends one unit of fuel (the poll exit when it is gone) and `Op::Rejoin` is one parallel move of the new value of every loop parameter and the fuel into the join's parameters. Nothing goes through the entry input slots any more, so a parameter that starts from a constant, or from a value that is no entry input, rejoins like any other.
- What the code above the join made and the code below reads (entry values, the two pointers, the binding words, the rows pointer, and with the join below the preheader what the preheader computed, its pending comparisons and the row and field addresses it reached) is stored once in frame cells and loaded again right after the join (`keep_in_cells`, `load_from_cells`; everything the lowering holds is renamed to the loaded values). Constants are made again where they are read. The allocator drops a store into a cell none of whose loads is kept (`cell_read` in `ra.rs`), and a load nobody uses, so only what the loop reads costs anything: a store at the entry and a load per link. The binding words are the exception when any member of the tree may move array storage (`sets_tables`, or a call that `moves_words`): a side can insert into or re-border an array before it links, so the words held above the join are stale. `join_root` then keeps no words, row or field addresses in cells, and reads the words again below the join, as Cranelift forgets its array words at the header every link jumps to. The first version of this unit carried them and read a freed array after a `table.insert` in a side (`a side that rejoins after moving the array the root reads` in `mir_backend.rs`, which failed with the carry on).
- The join's parameters arrive in the ABI argument registers in order, and in frame slots past those (`plan_join`), decided before any side is compiled, so a side's `Rejoin` is a parallel move into known locations. They are kept apart from the loop header's locations (`jlin` and `lin`) because a join parameter is also a loop invariant of the inner loop.
- Nested sides rejoin like any other: the side of a side jumps to the same join and receives the pointers through its guard's snapshot.
- A preheader that calls into the VM is no reason to refuse a root any more: it runs again per link under the same condition as Cranelift's, and once otherwise. A loop side that sides rejoin puts its join where Cranelift has its header: below its preheader when the loop passes no parameter on unchanged (stage 8, links), before it otherwise. Only the second kind keeps the fallback `loop side whose preheader calls into the VM under a side that rejoins its loop`.
- The guarded version of a versioned loop (region 1) is entered with every value the lowering held when a check failed (stage 8, links: see "Direct links to the loop header"); before that it took the loop parameters as they were, after the fuel and the two pointers, and loaded the rest again.

A link of a versioned root that steps every induction as the root does goes straight to the fast loop since stage 8, and a link out of a side entered from the guarded loop that changes only what that loop carries goes straight to the guarded loop (see "Direct links to the loop header").

### Range-proven loop versions

Stage 4. A counted loop whose array keys `RangePlan` (`jit_ir/trace/range.rs`) proves runs in a second version without the per-access bounds checks, as Cranelift's `lower_versions` does. MIR takes the same decision: `RangePlan::of` at a tier other than Baseline, with `RANGE_PROOFS` set, a tree of at most 2000 IR instructions and `versioning_pays` (the rule and its two constants moved from `emit.rs` to `range.rs`, with `FLOAT_ROUND_MAGIC`; Cranelift's output did not change).

- Region 0 runs the entry, the preheader and the checks, then the fast loop. The checks are the ones Cranelift makes in the same order (the start inside `[first, last]` of each array, the bound, every parameter of an induction equal to the root, the step sign and size, the shift size, integral float starts and the integer copy of each float induction). Each is a `Guard` that enters a side when it fails.
- That side is region 1, the guarded loop: the loop as before range proofs. Stages 4 to 7: it took the fuel and the two pointers, loaded the entry inputs and the binding words, ran the preheader again and the loop; since stage 8 it continues from the values held at the checks, with no entry load and no preheader. Its exits have snapshots of their own (`Lowerer::slow`, `fill_exit`: same exit codes, other values). Region 1 comes before the sides of the tree, because the allocator makes the code of a region after the regions it enters (a guard of region 1 can enter side 0 of the tree), so the sides move one region on (`side_of`).
- The fast loop carries one integer copy of each float induction as extra header parameters after the fuel, stepped at the back edge. A proven access addresses the slot as `base + scaled + disp`, with `scaled` the index times 16 computed once per index and `disp` the key's offset less one slot (`Off::VK`); Cranelift adds the displacement to a row address, so this saves an add. The tag guard of the access stays.
- A rejoining side jumps to the root's `Join`, which sits before the checks, so the checks run again for each link and it enters the fast or the guarded version by them. Since stage 8 a side that steps every induction as the root does (Cranelift's `fast_link`) jumps to a second `Join` below the checks instead (see "Direct links to the loop header").
- The guarded version takes the loop parameters as the checks saw them among the values of its entry when sides rejoin the loop, since a link no longer rewrites the entry input slots.
- Declined to Cranelift, with a logged reason: a root whose preheader calls into the VM (kept from the time region 1 ran it again; the guarded loop no longer does, but no test shows the combination yet) and a root with a call that moves array words. `RangePlan::of` is `None` for any tree that resizes an array, so the second is rare.
- `CompiledTrace::versioned_loop()` reports the MIR version too, and the compile log says `range-proven loop version`.

Tests: `range_proven_loops_run_through_mir_like_cranelift` (the 18 loops of the range-proof tests, integer and float keys, starts inside and outside the array so the proof holds or fails at entry, limits that end the loop in the middle, offset and negative keys, shifted keys, fuel that runs out) and `range_proven_loops_with_sides_run_through_mir_like_cranelift` (a side on the first read that leaves, and one that rejoins the loop) in `jit/trace/table_tests.rs`, both against Cranelift on the same data with the same loops versioned; `counted_loops_with_range_proofs_run_versioned_through_mir` in `crates/oximond-cli/tests/mir_backend.rs` (integer, offset, float and ending-past-the-array loops from Lua, and the log line). The steady-state allocation test compiles with `RANGE_PROOFS` off: a versioned compile allocates, in `RangePlan::of` and in the backend's versioned lowering (checks, shifts, stepped copies).

### Loop sides and the sides nested in them

A member whose exit another member continues (`LoweredMember::terminal`) ends in a jump into a loop side, the side with its own loop. The nbody tree is that shape: the pair loop's exit leads into the code of the outer loop, which runs in place around it. A side nested in a loop side can go back to the loop's header (`TraceTerminator::LinkLoop`).

- `Op::Goto` ends the member that enters the loop side: the values of the exit's snapshot go to the entry locations of the side, as for a guard that has a side, and the code jumps. A side whose entry needs a number subtype check (`snapshot.number_tag`) is refused, as for a guard.
- A loop side has its own `Header` and `Back`. The allocator state that was per tree is per region now: the header, the loop invariants, the join, and the label.
- A loop side that sides nested in it rejoin gets a `Join` of its own after its entry, and `Op::Rejoin` names the region it jumps to. The same difficulty as at the root comes back: a nested side cannot see what the loop side defined above its loop (its entry values, the pointers). The root solves it by running its entry again from memory. A side has no such memory, so the loop side keeps those values in frame cells (`Op::CellStore`, `Op::CellLoad`: words of the frame, below the spill slots, that no allocation touches), loads them again after its `Join`, and then runs its preheader and loop. The join's parameters are the loop parameters and the fuel only, and a `LinkLoop` is a parallel move of the new loop parameters, one unit of fuel spent, and the jump.
- The locations of a join's parameters are fixed before any region is compiled, for every region that has one (`plan_join` runs for each), and the frame slots they may need are reserved in every region's frame, so a nested side's `Rejoin` is a parallel move into known locations.

Inference (stages 5 to 7). A loop side with a `LinkLoop` below it ran its preheader again at each link, where Cranelift jumps straight to the header and keeps the preheader's values in SSA. Since stage 8 that holds only for a loop side that passes a parameter on unchanged, which is also where Cranelift runs the preheader at the header.

Tests: `tests/loopside.rs` in `oximond-mir` (a root that enters a loop side with `Goto`, a nested side that rejoins its `Join`, 0 to 14 values in cells, with and without a call, against the evaluator), `a_loop_side_entered_by_an_exit_runs_through_mir_like_cranelift` and `a_nested_side_that_rejoins_the_loop_side_runs_through_mir_like_cranelift` in the lib tests (the trees of `tree_tests::loop_side`, both tiers, many start states and fuels, outcome, control block and exit area against Cranelift), and the nbody benchmark.

### The forward skip (stage 6)

`Op::If` and `Op::EndIf` bracket a block that the straight line jumps over when `a cc b` is false (integers or floats, as for `Guard`). The block may hold any instruction except one that ends a region or opens a loop: guards (with or without a side), calls, `CallIf`, cells, and further `If` regions. `EndIf` defines the join values: join value `k` is `lists[c + k]` when the block ran and `lists[c + b + k]` when it was skipped. The first is any value in scope at `EndIf` (the block's own included), the second a value defined before the `If`. Nothing else the block defines can be read after it. The builder is `Func::begin_if` and `Func::end_if`; the validator checks the pairing, the scopes, the types, that no terminator or loop sits in a block, and that a flags-only guard does not follow an `If` or `EndIf`.

Design choice. A forward skip with join values, not a loop construct for the probe walk. Inference: the probe walk is bounded and the first steps decide almost every lookup, so the walk is a chain of nested skips, one per step, with the native walk through `CallIf` only after the last one. A loop would need a second header in a region, and the allocator's loop model (invariants pinned at the back edge before the body is seen, one header per region) has no room for it. A skip needs only a snapshot of the registers at one point. Join values are block parameters (the else-less form of a phi) because a probe step has one thing to hand on, the bucket or the value found, and the skipped side's answer is a value that was already computed.

How the backward pass handles it (`ra.rs`, `do_end_if` and `do_if`):

- The pass meets `EndIf` first. It records the join point (a position in the body, which later code can branch to without a fixup), unbinds the join values and notes the register each had, and saves which register every value live after the join is bound to (`if_saved`). Then it ends the block: one parallel move of the block's values into the join values' registers and slots, with the same cycle breaking as a back edge. A block value that is not in a register gets one, hinted to its join value's register, so the instruction that makes it usually writes it there.
- The block is compiled with the state the join expects, so everything inside it (eviction, calls, `CallIf` stubs, guards) works as before. A value the block evicts is reloaded after the evicting instruction, on the path that ran the block only.
- At `If` the pass reads the locations the values have at the top of the block and compares them with the saved state. A value that is where the join wants it needs nothing. A value the block moved or evicted is moved back, or loaded from its slot (every value that was bound has one once it is evicted). The skipped side's join values are read like a snapshot (`weak_loc`) and moved into the join values. With no move the branch goes to the join directly. Otherwise it goes to a stub (in the stub region, so the hot code stays dense) that does the parallel move and jumps to the join. The skipped values are hinted to their join value's register before the compare, so a value nothing else reads is made there.
- The prescan counts the block's values and the then-side join values as uses, and the skipped side's join values as snapshot reads at the `If`, so liveness across calls and loop invariants see the block like any other code.
- A block in a loop works as anywhere: the back edge was processed first, so invariants are bound across the block, and the skip path reloads the ones the block evicted.

- The registers above the `If` (`shuffle_entry`, added with the probe). A block that needs more registers than the join leaves free evicts values that live across it, and the first version of the `If` loaded them back on the skipped path, the path most blocks are skipped on. Now a value the block evicted is bound, above the `If`, to the register the join expects (its slot is valid, since an evicted value is stored where it is defined), and what the block had in that register at its top moves to a free register that the join does not use, with a move at the start of the block (`emit_plan`, so the block path pays and the skipped path does not). A value the compare reads, a register the join expects and the compare holds, and the case with no free register keep the old stub. Counted in `Stats::if_rebound` and `Stats::if_entry_moves`. Measured against the same build with the shuffle off (whole-process instructions, before the `imul` and `lea` change, scale 3): `hash_read` 123.0 M against 128.9 M, `hash_read_small` 208.9 M against 216.0 M, `metamethod_function` 237.4 M against 249.2 M, `metamethod_newindex` 201.6 M against 214.9 M, `hash_write` and `hash_churn` 0.6 and 1.2% lower, `hash_read_shaped` 0.6% lower and `hash_read_indexed` 0.8% higher. In the six hash kernels (instructions an iteration, same two builds) it is 1.4 to 4.1 lower for five and 2.5 higher for equal strings in two allocations (105.3 against 107.7). Inference: it pays where the block sits in a loop with many invariants, which is every loop of the suite that probes.

Cost. The straight line pays the compare and the branch the guard pays, and moves only for what the block disturbed. Compile time is one more match arm per instruction and three reused vectors (`if_frames`, `if_saved`, `if_dst`), so the steady-state compile still allocates nothing (`kernels::SKIP`, two nested skips, is in the allocation test). Limits: a block cannot end a region (use a guard to leave), and a flags-only guard cannot follow an `If` or `EndIf` because the two paths arrive with different flags.

Measured against main 81460ee6 (`benchmarks/mir_backend.py`, release binaries at equal-length paths, one run each under `scripts/heavy.sh -x`, pinned, load average 4.9 6.9 7.1 at the end): no lowering used the construct at that commit, so no tree changed. The string-index probe is its first user (see "String-index probe"). Coverage: every benchmark has the same trees, the same MIR instruction counts and the same bytes of code before and after. Suite total under MIR 21,057.5 M instructions before and 21,035.5 M after (-0.10%, the per-benchmark spread is -0.79% to +0.37%, which is the path-length noise the other sections describe). Compile thread under MIR, 92 trees: 153,252 instructions per tree before, 153,067 after (-0.12%).

Tests: `tests/skip.rs` in `oximond-mir` (values live across a block at 7 pressure levels with 1 to 16 temporaries in it, both encodings, taken and skipped; a block that disturbs no register branches straight to the join, `if_stubs` is 0; join values of 1 to 20 integers and up to 15 floats, constants and block values on each side, so some live in slots; a join value built in its own register needs no stub; float comparisons with NaN under 8 conditions; a block with a call, a `CallIf` that stores to the control block and a skip nested in it; three nested skips with join values passed outward; a block in a loop with 2 to 13 integers and floats live across it and around the back edge; the validator's rejections), the random programs (up to three nested skips with join values, 3,200 skips in 1,500 seeds, about half with a stub; 300,000 seeds on both encodings and against the CLIF translation passed), and the steady-state allocation test.

## Compile path

- Lowering is one forward pass over the optimized `TraceIr`: preheader, then the loop body, then each side. Constants make no instruction (the allocator turns each use into an immediate or a pool operand).
- The MIR function, the compiler's buffers, the lowerer's tables and the exit snapshots live in one thread-local arena, and `clear` keeps capacity. A test (`a_steady_state_compile_allocates_nothing_in_the_backend`) installs a counting allocator for the lib tests and requires zero allocations inside the backend on the compiles after the first few. Debug builds validate the MIR function (`Func::validate`), which allocates, and that is outside the counted span.
- The image goes from the compiler's two buffers into the carved block directly. There is one mapping for the process, and no `JITModule` per tree.
- Exit metadata is the plan the runtime already consumes. Nothing in `CompiledTrace` changed: `code` is the only backend-specific field. `clif_insts` is 0 for MIR code and `code_size` the bytes.

## Tests

| Where | Tests | What they prove |
|---|---|---|
| `crates/oximond/src/jit/trace/mir/tests.rs` | 45 | A hand-built trace is compiled with each backend (a thread-local picks the backend), the test requires that MIR made its code (not a fallback), and it compares the outcome, the whole control block (fuel, links and the elided counters), and every word of the exit area, from many start states, at both tiers. |
| `crates/oximond-cli/tests/mir_backend.rs` | 10 | 52 Lua snippets under no variable, `cranelift` and `mir`: identical output; no mention of the MIR backend unless asked; under `mir`, the optimized IR holds the operations the snippet names and MIR compiled a tree; any fallback has a reason the snippet lists. |
| `crates/oximond-mir/tests/calls.rs` | 12 | The call op against the evaluator: values live across a call in every register class and at every pressure, six integer and eight float arguments, float and integer results, a constant argument and one value passed twice, floats spilled across four calls in a row (reloads required), a call in a loop with carried and invariant values, a division and a call in one function, and the stack alignment the callee sees with 1 to 14 live values (so every parity of pushed registers). The checking shim fails a call path that clobbers a callee-saved register. Also: integer arguments past the sixth (10 arguments, 10 to 14 values live), nine integers with ten floats (stack arguments of both classes, against the function run by Rust), a call through the control block, frame cells across a call, and `CallIf` taken and not taken with 2 to 13 integers and 1 to 13 floats live, by address and through the control block, and with its result stored in the control block. |
| `crates/oximond-mir/tests/loopside.rs` | 2 | A root that enters a loop side with `Goto`, a nested side that rejoins the loop side's own `Join` (`Rejoin` to a region other than the root), 0 to 14 values kept in frame cells, with and without a call, fuel from 1 to 500, against the evaluator on both encodings. |
| `crates/oximond-mir/tests/rejoin.rs` | 8 | Sides that rejoin the outer header: a swap on the back edge (a two-register cycle of the parallel move), nested sides (a second side rejoins with a three-value rotation), a call beside the sides, a join next to an inner loop, join parameters that are loop invariants of the inner loop under pressure (up to 15, so some arrive in frame slots), and a join below code that made values the loop reads (they go through cells; a store into a cell nobody loads is dropped)., a cell value that has no register that stays in its cell, and a load above the only store of a cell inside a loop (the value must not live in the cell) |
| `crates/oximond-mir/tests/random.rs` | 4 | As before, with a join of the counter and a budget under cells for every value the loads made, and sides that rejoin it, in a third of the looped programs (60,000 seeds on both encodings passed), and the random programs now call functions (over 200 calls in the default 1,500 seeds, among them a call with nine integer arguments and a value kept in a cell across a call; 20,000 seeds on both encodings passed), next to guards, sides, loops and pressure. |
| `crates/oximond-mir/tests/skip.rs` | 10 | The forward skip against the evaluator on both encodings: pressure inside and across a block, join values, nesting with calls and `CallIf`, a block in a loop, float conditions, the validator. See "The forward skip". |
| `crates/oximond-mir/tests/ops.rs` | 13 | Every op and operand form added for the lowering, native against Rust, SSE2 and AVX encodings, and the sum and product with a constant (`lea`, three-operand `imul`) whether the operand is read again or not. |
| `crates/oximond-mir/tests/region.rs` | 2 | Carving, first fit, merging of holes, the region goes back to zero, code written into a block runs. |

How the acceptance criteria are covered:

| Criterion | Test |
|---|---|
| Selection: `mir` selects it, unset and `cranelift` do not, Peak stays Cranelift, each fallback has a reason | `the_peak_tier_and_unsupported_operations_use_cranelift`, `the_default_and_cranelift_runs_compile_the_same_trees`, `every_snippet_prints_the_same_under_both_backends` (stderr has no `mir backend` line without the variable) |
| Entry checks and `-1` on a mismatch | `entry_mismatches_run_nothing_and_leave_the_buffers_alone` |
| Loop back edge, fuel, poll exit | `a_loop_spends_fuel_and_leaves_through_its_poll_exit_like_cranelift` with fuel 1, 2, 3, 5, 11, 12, 64 and 1000 |
| Exit indices, exit area, writebacks | every case above compares `Observed { outcome, control, area }` and the exit descriptors |
| Sides that rejoin the loop (nested ones, a swap on the back edge) | `a_side_that_rejoins_the_loop_runs_through_mir_like_cranelift` (fuel 1 to 10,000, both tiers, outcome, control block and exit area against Cranelift), `a_side_that_rejoins_a_loop_with_a_constant_start_runs_through_mir_like_cranelift` (a parameter from a constant, a hoisted value, a join below the preheader), `a_hoisted_check_on_a_register_only_a_side_writes_runs_through_mir_like_cranelift` (a parameter only the side changes, the check in the preheader that runs again per link, a join above it), `range_proven_loops_with_sides_run_through_mir_like_cranelift` (a versioned root with a rejoining side), the CLI snippets "a branch that merges back into the loop" and "a hoisted call into the VM under a side that rejoins" (each asserts a `Link` tree compiled by MIR, no fallback), `tests/rejoin.rs`, the random programs |
| Helper-backed operations (`CallHelper` with two results and a guard on its status, `HostCall`, `HashLoad`, `ArraySet`, table-valued loads, `ReleaseStrings`) | `a_helper_call_with_two_results_and_a_guard_matches_cranelift` (every pair of 14 integers, the helper misses for a zero divisor and for the overflowing quotient, both tiers, outcome, control block and exit area against Cranelift), the CLI snippets "table stores that grow an array", "helper calls, host calls and the release of strings", "hash lookups by a computed string key", "strings from a table of strings" (each asserts the optimized IR holds the operation and that MIR compiled the tree), and the suite |
| Loop sides, nested sides that rejoin them | `a_loop_side_entered_by_an_exit_runs_through_mir_like_cranelift`, `a_nested_side_that_rejoins_the_loop_side_runs_through_mir_like_cranelift`, `tests/loopside.rs`, the CLI snippet "nested loops over a table of tables", nbody |
| Calls and native math | `calls_to_helper_functions_match_cranelift` (`Power`, `PowerSquare`, `Fmod53`, `Fmod54`, the 32-bit number conversion, and a loop with a live float accumulator across a call of `pow`), the CLI snippets "power and math library calls" and "float modulo that calls the compat function", `tests/calls.rs` |
| Sides that end in an exit, and `count_links` | `a_side_that_ends_in_an_exit_of_its_own_runs_through_mir_like_cranelift`, `the_options_count_links_without_changing_the_exits` |
| Arithmetic, compare, select, conversions | `float_arithmetic_…`, `integer_arithmetic_…`, `integer_division_and_modulo_…`, `shifts_by_a_constant_…`, `sixty_four_bit_integers_…`, `conversions_to_a_64_bit_integer_…`, `unary_operations_and_conversions_…`, `compares_and_guard_branches_…`, `selects_…`, `a_float_comparison_stored_as_a_value_…` |
| Field loads and stores through rows, `MetaGuard` | `rows::field_loads_stores_and_the_metatable_guard_match_cranelift` (rows built by a test dispatcher, memory compared afterwards) |
| Array loads and stores, `FieldLoad`, string bytes | the CLI snippets, and the existing `table_tests` and `string_tests`, which ran with the variable set (below) |
| No allocation in a steady-state compile | `a_steady_state_compile_allocates_nothing_in_the_backend`: a counting global allocator for the lib tests, and the backend reports what it allocated between the start of lowering and the end of the carve, validation excluded |
| One code region per process | `code_lives_in_the_one_region_of_the_process` (20 trees, every address inside the one region, no two the same), `tests/region.rs` |

The existing suites run unchanged under the variable too (next section), which compiles the trees of their hand-built traces with MIR wherever it can and checks them against the reference evaluator. `unsupported_and_invalid_traces_are_rejected` and the forced-guard tests need Cranelift's lowering (a forced guard is a reason to refuse a tree here).

## Test suites under both backends

`scripts/heavy.sh -m 24G cargo test -p oximond -p oximond-cli -p oximond-mir --no-fail-fast`, once with the variable unset and once with `OXIMOND_JIT_BACKEND=mir`, on commit `b7003199`:

| Run | Test binaries | Passed | Failed |
|---|---:|---:|---:|
| `OXIMOND_JIT_BACKEND` unset | 201 | 4504 | 0 |
| `OXIMOND_JIT_BACKEND=mir` | 199 (no doc tests) | 4495 | 8 |

Unset: all pass. The `mir` run was made with `--tests`, so without doc tests, and before the snippet test of `oximond-cli` was updated for the hash lookup that now stays with Cranelift; that one test failed in the run (9 failures, 4494 passed) and passes alone since (`cargo test -p oximond-cli --test mir_backend`), which gives the 4495 and 8 above.

Under `mir`, 8 tests fail. All 8 assert a fact only Cranelift's code can show. Each one first checks that the results equal the interpreter's, and then asserts a CLIF count or the native link of a side compiled apart. None is a difference in program output, exit values or writebacks.

- CLIF instruction counts and calls in the CLIF (5):
  - `jit::trace::tests::float_fmod_uses_fma_only_when_the_isa_has_it`
  - `jit::trace::tests::float_modulo_forms_call_a_helper_only_off_the_inline_path`
  - `jit::trace::tree_tests::clif_instructions_of_a_side_compiled_apart_are_fewer_than_the_tree_that_holds_it`
  - `vm::tests::trace_detached::a_side_that_links_back_is_compiled_alone_after_its_tree_runs`
  - `vm::tests::trace_detached::compiling_a_side_costs_less_than_compiling_the_tree_with_it`
- the native link of a side compiled apart (3):
  - `vm::tests::trace_detached::a_hot_nest_native_code_links_in_still_stops_for_a_count_hook`
  - `vm::tests::trace_detached::native_code_calls_the_side_that_returns_into_the_calling_loop`
  - `vm::tests::trace_detached::the_loop_variables_after_a_run_that_native_links_left_are_the_interpreters`

The CLIF counts and the native link of a side compiled apart are as stage 2b described them: MIR code has no CLIF, and its root returns the exit index for the run loop to call the side. The 26 tests that the previous build of this unit also listed (the native probe, the in-place store, the closure rows, the stepping of `pairs`, the hole fill) pass now, because the trees they run stay with Cranelift: those were contracts on native behaviour, not Cranelift-specific facts, and the fix was to keep the trees there until the inline paths exist.

Known flaky tests, rerun alone before blaming a change: the native-directory test of `oximond-jit`, the string-cache test, and `compiler::pipeline::tests::syntax_failure_stops_parser_before_eof`.

## Measurements

Machine: AMD Ryzen 9 8945HS, 16 logical CPUs with SMT siblings N and N+8, Linux 7.2.9, shared with other sessions. Measured commit `b7003199` on `flow/mir-research` for the tests, the suite diff, coverage, the end-to-end instruction counts and conformance. The loop-kernel and cycle figures ("Steady-state loop cost", "The spill-kernel cycle gap") are from `f18df2d9`: the later commits changed only the field group check and the strict header refusal, which those kernels do not exercise, and they were not measured again. Built with `scripts/heavy.sh -m 16G cargo build --locked --release -p oximond-cli -p oximond-conformance` (thin LTO, line tables). Every figure below that counts instructions is `instructions:u` from `perf stat`, or from the per-thread counter of `OXIMOND_JIT_ICOUNT`, so load does not move it. Instruction counts ran through `scripts/heavy.sh -x`, pinned with `taskset -c 3` (sibling CPU 11), under `env -i PATH=/usr/bin:/bin`, with the same binary at the same path for both backends (only `OXIMOND_JIT_BACKEND` differs). `benchmarks/mir_backend.py` picked CPU 3 by sampling `/proc/stat` for the least busy pair. Load average (1, 5, 15 minutes) when the instruction counts started: `46.05, 35.64, 31.14`. Scale 3 for the suite, as the census used. The loop kernels, the cycle study and the profile ran on commit `9d47fb9d`: the next commit changed the lowering (an inline lane probe and array store) and gave the conditional call a result store, which no kernel of `oximond-mir` uses, so the allocator and the machine code of the kernels are the same. The commits after the measured one changed documentation only.

### Output identity

`benchmarks/mir_backend.py diff` runs all 89 entries of `benchmarks/suite.json` with `--execution=tracing` under `cranelift` and under `mir` and compares stdout and exit status: 89 of 89 identical, none timed out. The tiered tracing conformance run (`oximond-conformance run --execution tracing --dir .oximond/conformance`; this CLI has no `--jit-engine`, and tracing is its only JIT mode) gives these ledger summaries:

```
lua51: 20 passed, 0 failed, 0 expected failures, 0 unexpected passes, 0 infrastructure errors
lua52: 7 passed, 0 failed, 16 expected failures, 0 unexpected passes, 0 infrastructure errors
lua53: 7 passed, 0 failed, 18 expected failures, 0 unexpected passes, 0 infrastructure errors
lua54: 5 passed, 0 failed, 23 expected failures, 0 unexpected passes, 0 infrastructure errors
lua55: 4 passed, 0 failed, 24 expected failures, 0 unexpected passes, 0 infrastructure errors
luajit: 27 passed, 0 failed, 0 expected failures, 0 unexpected passes, 0 infrastructure errors
```

The two runs print byte-identical logs (rerun on `b7003199`: 20, 7, 7, 5, 4 and 27 passed, none failed).

### Coverage on the suite

`benchmarks/mir_backend.py coverage` runs the suite under `mir` with `OXIMOND_JIT_SYNC=1 OXIMOND_JIT_TRACE=1` and counts the log lines. A compile is one native compile of a tree (a tier upgrade or a new side compiles the tree again, so a root can count twice).

128 compiles: 92 accepted (71.9%), 36 refused. 4211 trace IR instructions: 3416 accepted (81.1%), 795 refused. The accepted ones made 12723 MIR instructions and 156455 bytes. Of the 89 benchmarks, 3 compile nothing at scale 3, 63 have every compile accepted and 17 have only refusals. The run is `benchmarks/mir_backend.py coverage` through the gate, on the stage 5 closure-rows build. The same run on main `0f8f8258` gave 91 accepted compiles (3392 IR instructions) and 37 refused (819), so the closure rows brought back one compile of 24 IR instructions. The table of the previous unit showed 9 compiles and 846 IR instructions for the closure reason. 8 of them (822 IR instructions) were already gone on main `0f8f8258` when this unit started; I did not trace which change did it, and `opt/closure_protos` compiles 8 trees through MIR on main. The ninth, the tree of `opt/closure_family` (24 IR instructions), is the one this stage brings back. These are the default figures, and they stand: the string-index probe of stage 5 is off by default. With `OXIMOND_JIT_MIR_PROBE=1` it brings the accepted compiles to 105 (see "String-index probe").

### Closure rows

Stage 5 lowers Cranelift's inline closure rows (`closure_cell` in `emit.rs`). `closure_use` in `lower/calls.rs` asks the same `closure_cell_use` as Cranelift, with the same table-cell flag (`closure_row_words` of all members), and `call_helper` then emits no call:

- `CLOSURE_SAME_PROTO` loads the prototype word of the live lane's row and of the pinned lane's row, and is 1 when they are equal and non-zero.
- `CELL_GET_*` guards that the cell word of the row is non-zero and that the cell's tag is the wanted one, then loads the payload. `CELL_SET` does the same checks and stores the payload. Neither has a write barrier or a call, as in Cranelift; both leave through the helper's guard where the helper would miss (no such cell, another type), and a missing row reads as a zero word.
- A helper that returns a closure ensures the lane's row after the call when the tree has closure rows (`fills_closure`), so a stale prototype word cannot pass the check.
- The rows pointer is loaded for a tree whose only row use is one of these helpers (`reads_rows`), and the stride is the one from `closure_row_words`.

`CELL_GET_TABLE` in a tree with `CLOSURE_ROW_TABLE_WORDS` is lowered as `closure_table_cell` is. It loads the cell word and the named-lane word of the closure's row and `TraceControl::bank`, and guards that all three are non-zero, that the cell's tag is a table, and that the cell's payload equals the payload of the bank entry at `(named - 1) << 4`. The result is `named - 1`, with no call and no `trace_ensure_row`, as in Cranelift.

Tests: `jit::trace::mir::tests::closures` runs the Cranelift closure-row trace under both backends and compares outcome, control block and exit area for a float cell, an integer cell, another prototype, a lane that is no Lua closure, a cell of another type and a closure without the cell, a table cell (hit, changed cell, no lane, empty bank, not a table, no cell), and for repeated runs that read what the earlier run wrote. `tests/mir_backend.rs` has five closure snippets whose trace log proves the trees compiled through MIR with no fallback: the cells and prototype checks in line, a float cell one closure writes and another reads, a call site that meets a native function with no closure row, one that meets a closure of another prototype, and the table held by a closure cell (`CELL_GET_TABLE`, from `75dcf5ae`).

Measured with `benchmarks/mir_backend.py icount` through `scripts/heavy.sh -x`, pinned to CPU 0 (sibling 8 idle), binaries at equal-length paths, equal environments. Load average at the start of the run: 6.73 (another session's jobs were finishing). Before is main `0f8f8258`, after is this commit.

| Benchmark | Before MIR | After MIR | Cranelift (after) |
|---|---:|---:|---:|
| opt/closure_family | 343,421,146 | 345,312,813 (+0.55%) | 343,383,167 |
| opt/closure_protos | 485,158,298 | 485,082,928 | 435,952,747 |
| suite total | 21,025,978,169 | 21,029,449,188 (+0.02%) | 21,038,630,628 |

The suite total is flat: only `opt/closure_family` has a tree that moved (one compile of 24 IR instructions), and it costs 1.9 M instructions (0.55%) more than Cranelift's code for it, so the goal of at or below Cranelift is missed there by that margin. I have not found the cause; it is a follow-up (see "Gaps"). No benchmark is more than 1% slower than on main; the largest changes besides it are `allocation/gc_cycles` (-0.48%) and `stdlib/tonumber_parse` (-0.41%), which no closure tree reaches (noise). `opt/closure_protos` is 11% above Cranelift under MIR on main as well; its eight trees were already accepted, so the gap is code quality elsewhere (the call path), not closure rows. The main suite total under MIR here differs from the 21.283 G of the stage 4 note because other work landed in between.

Final checks ran on commit `75dcf5ae`, the one that adds the in-line `CELL_GET_TABLE` lowering, with release and debug binaries built from it. Through `scripts/heavy.sh -m 24G`, `cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast` reports 0 failed with `OXIMOND_JIT_BACKEND` unset and 0 failed with `OXIMOND_JIT_BACKEND=cranelift`, and `cargo clippy --locked --workspace --all-targets` passes.

The difftest slice (`run-jit --seeds 0..300 --profiles all --variants tracing,tracing-mir,tracing-mir-profile`, release): 1800 cases (300 seeds, 6 profiles, 3 variants each), 0 mismatches. The 100-seed debug slice had 3 timeouts at the 20 s default (lua53 seed 15 under `tracing-mir-profile`, luajit21 seed 15 under `tracing-mir-profile`, luajit21 seed 32 under `tracing`, which the harness flags as flaky). The interpreter oracle alone takes 11.6 s on lua53 seed 15 with the debug binary on this loaded machine. All three pass under all three variants when rerun alone with `--timeout 120`.

The coverage and icount numbers above were measured on the commit before the table-cell lowering. No benchmark reaches `CELL_GET_TABLE`, so they do not change with it.

### String-index probe

Stage 5 lowered Cranelift's inline string-index forms (`hash_probe`, `hash_store` and `hash_overwrite` in `emit.rs`) in `lower/probe.rs` and left them off: with no forward branch in MIR the lowering was a straight line of selects plus a 60-instruction conditional call for everything past the first bucket, and the eight benchmarks it brought back ran 4 to 69% more instructions than Cranelift's code (the stage 5 numbers are at the end of this section). Stage 6 gave `oximond-mir` the forward skip (see "The forward skip") and rewrote the probe on it. The probe stayed off through stage 6, because one benchmark missed the goals (see "Goals" below); stage 7 fixed that benchmark and made the probe the default. It is on unless `OXIMOND_JIT_MIR_PROBE` is `0` or empty (`debug_env::mir_probe`, read once per process; `mir_probe_for_this_test` and `mir_probe_off_for_this_test` set it for a unit test's thread). With it off MIR declines the trees Cranelift probes, with the two reasons it gave before, `HashLoad without the inline probe` (in `hash_load`) and `hash store or delete without the inline probe` (in `inline_in_cranelift`), and Cranelift compiles them. The condition for a tree to probe is Cranelift's: `native_probe` (a member reads or stores by string index and none moves hash storage), a `StringRef` key, a scalar expected type for a lookup, and for a store a root member with an entry-input table whose binding is `NoMetatable` (or `Hash` for `SET_PRESENT` and `DELETE_STRING`). Everything else keeps the VM call it had. The figures below are those of stage 6, with the probe on (set to `1` then, as it was off by default) unless a sentence says otherwise.

Design choice. The task allowed an `If` region or a small loop construct for the walk. The walk is a chain of skips, one per bucket, and the loop construct was not built. Reason: the allocator's loop model (invariants pinned at the back edge before the body is seen, one header per region) has no room for a second loop, and the walk is bounded, decided by its first buckets in almost every lookup. A chain of `WALK_STEPS` (8) skips is the loop unrolled; the native walk behind it covers any sequence longer than that.

How the lowering walks:

- The first bucket is read in a straight line: the key's storage and hash, the same multiplicative mix the runtime places by, the bucket's key word. One skip, run when that word is not the key's own string, holds everything else. A key that meets the table's own allocation (the common case) pays one compare and one branch and takes the bucket as the answer.
- In that block: a key with no hash gives up; an empty bucket is a miss; a bucket with the key's hash is compared (two inline strings are equal when their three words are, a long key gives up, as in `probe_bucket`); any other occupied bucket goes on to the next, which is a skip nested in the block of the step before and runs while the state is zero. A binding with no index is given two buckets that hold no hash and a mask of zero (`load_binding`), so its first bucket is read in line and gives up.
- The state is one word, passed outward as a join value: the address of a bucket when the walk has its answer, `KEEP_WALKING` (0) or `GIVE_UP` (1). A lookup and a store share it. A missing key is a static bucket: its value is a nil for a lookup, and has a tag no store takes for a store. A lookup then loads the entry's value from the bucket and guards its tag, as Cranelift does at `hit`.
- If no step answered (`GIVE_UP`, or the sequence is longer than the steps made in line), a conditional call walks the rest of the sequence in native code (`locate`, the Rust copy of `probe_bucket`'s loop) and asks the VM only when even that cannot tell (no index, no hash, a long key against another allocation). That is the only way the VM is called, as in Cranelift's code. The call sits in a skip of its own inside the block of the first one, so the hit path never tests it. A lookup's call answers with the address of a value, which the lowering reads as the value word of a bucket 16 bytes before the control-block word the call stored it in.
- A store is a skip that runs when the table takes it in line (`bad` is zero: the entry has the type, the mask's flags allow it, and for a delete or a revive the key is inline and does not start with `__`), and the VM's helper is a skip that runs when it does not, with the guard on `set_status` inside it. Exits are Cranelift's. A guard inside a skip must read a snapshot made before the skip opens: a pending comparison first materialized in a block would not be defined on the other path (`Lowerer::guard_target`, `prepare_helpers`).
- `DELETE_STRING` followed by `SET_REVIVE` of the same key (`fused_delete_stores`' rule) is one probe: a live number or boolean takes the value where it stands, a nil takes it and the counter rises (in a skip of its own), and anything else calls both helpers in order, each with its own guard. The pair no longer copies its argument lists, so a fused compile allocates nothing.
- The VM is called for what it was called for under Cranelift: the helper counters (`hash_helper_calls`, `delete_string_helper_calls`, `set_revive_helper_calls`, `set_present_helper_calls`) are equal under both backends on the snippets, including a 3000-key table whose probe sequences pass the steps made in line. No write barrier is involved, as in Cranelift: the stores are scalars over scalars.

Two changes outside the probe came out of measuring it. The entry shuffle of the `If` in the allocator (see "The forward skip") removed the loads a skip path made for values the block had evicted. A product with a constant is now the three-operand `imul` and a sum with one is `lea` when the operand is read again, which saves a copy at each use (a probe step has two). A sum that a flags-only guard reads stays a two-address `add`, because `lea` sets no flags (`5d2789ff`, test `a_flags_only_guard_reads_the_flags_of_a_sum_with_a_constant`). Both are in `oximond-mir`, and the second changes the code of every tree that has such an operation (the suite total below includes it).

Tests: `jit::trace::hash_tests` turns the probe on for its own thread and runs the lookups against Cranelift on every layout (hashed or not, long chains, keys shared with the table or not, a key without a hash, long keys, near-equal keys, a deleted key, an index rebuilt between runs) and compares the guard, the registers, the iterations and the number of calls to the VM; `mir_walks_dense_indexes_past_its_steps_in_line_as_cranelift_does` adds an index of 4096 buckets with 3300 keys and 40 absent keys whose chains are long. It also passed with `WALK_STEPS` set to 1, 2 and 3. `a_steady_state_compile_of_the_string_probe_allocates_nothing_in_the_backend` (in `mir/tests.rs`, with the probe on for its thread) compiles the four lookup traces at both tiers and requires zero allocations in the backend. Stores need the VM's helpers, which a unit test does not have, so `tests/mir_backend.rs` has nine snippets that run with `OXIMOND_JIT_MIR_PROBE=1` (a hit, a miss, a probe after a rehash, a store over a present key, a delete and a revive fused, the pair a guard apart, a key that may name a metamethod, a dense index, a tree that moves hash storage), a check that the helper counters equal Cranelift's, and a check that seven programs that change the table under the loop take the same exits the same number of times under both. `without_the_variable_mir_declines_the_trees_that_probe` runs a lookup snippet and a store snippet without the variable and checks the two reasons and the same output. In `oximond-mir`: `a_chain_of_skips_ends_in_a_conditional_call` (chains of 1 to 12 skips nested in each other, with a conditional call behind them, in and out of a loop, with up to 13 integers and 13 floats live), a rebound-and-moved assertion in the loop test of `skip.rs` and in the random programs (200,000 seeds on both encodings pass), and `a_sum_or_product_with_a_constant_keeps_or_reuses_its_operand`.

Measured with `scripts/heavy.sh -x`, `benchmarks/mir_backend.py`, one binary per column at paths of equal length, `env -i` environments of equal length (the probe switch is set to `1` under both backends so the lengths match), pinned. Load average at the start: 3.49 5.72 7.44 for the suite counts (CPU 3 with sibling 11 for main, CPU 6 with sibling 14 for this build) and 4.55 6.93 9.52 for the stage 5 columns. Main is `81460ee6`, built from a `git archive` export. The loop kernels first (instructions per iteration, the difference of runs of 300k and 1.5M iterations; the kernels are new in this unit, so the stage 5 column is main's binary with the probe on, run on them):

| Kernel | Cranelift | MIR, stage 5 | MIR, probe on | Probe on vs Cranelift |
|---|---:|---:|---:|---:|
| lookup, key and table share the string (200 keys) | 87.4 | 97.2 | 71.4 | -18.3% |
| lookup, equal strings in two allocations (200 keys) | 102.4 | 152.2 | 105.0 | +2.5% |
| lookup of an absent key (200 keys) | 80.6 | 95.6 | 84.7 | +5.0% |
| lookup in 2000 keys | 86.6 | 90.7 | 69.6 | -19.6% |
| store over a present key (2000 keys) | 86.8 | 115.5 | 86.3 | -0.6% |
| delete and revive (fused, 2000 keys) | 95.8 | 159.9 | 97.1 | +1.4% |

The benchmarks whose trees moved, whole process, scale 3. "Stage 5" is main with `OXIMOND_JIT_MIR_PROBE=1`:

| Benchmark | Main MIR (M) | Stage 5 (M) | MIR, probe on (M) | Probe on vs main | Cranelift, same binary (M) | Probe on vs Cranelift |
|---|---:|---:|---:|---:|---:|---:|
| table/hash_read | 136.72 | 171.65 | 118.20 | -13.5% | 136.66 | -13.5% |
| table/hash_read_small | 223.71 | 261.05 | 203.63 | -9.0% | 223.65 | -9.0% |
| table/hash_read_shaped | 139.24 | 146.21 | 93.70 | -32.7% | 139.20 | -32.7% |
| table/hash_read_indexed | 214.21 | 362.84 | 217.09 | +1.3% | 214.13 | +1.4% |
| table/hash_write | 159.67 | 202.52 | 145.31 | -9.0% | 159.63 | -9.0% |
| table/hash_churn | 94.43 | 143.27 | 88.28 | -6.5% | 94.39 | -6.5% |
| table/metamethod_function | 250.55 | 259.86 | 225.89 | -9.8% | 250.49 | -9.8% |
| table/metamethod_newindex | 226.06 | 267.78 | 193.42 | -14.4% | 228.51 | -15.4% |
| the eight | 1444.59 | 1815.17 | 1285.51 | -11.0% | 1446.66 | -11.1% |

Suite total under MIR with the probe on: 20,864,306,869 instructions against 21,057,629,845 on main (-0.92%). Cranelift on the same binary totals 21,050,774,293, on main 21,066,756,865. The largest per-benchmark changes against main outside the eight are `string/utf8_pack` (+0.47%), `mineclonia/captured_call` (+0.22%), `mineclonia/module_calls` (+0.07%). Goals (the rule of the task: the probe becomes the default if they hold, and stays off if not):

- The eight benchmarks at or below Cranelift: seven are, from -6.5% to -32.7%; `table/hash_read_indexed` is +1.3% above main and +1.4% above Cranelift (217.1 M against 214.2 M). It is a rejoin cost, not the probe: that loop records its common path in the root and sends a missing key (one lookup in eight) through a side that rejoins the root, and MIR's rejoin runs the root's entry again, about 110 instructions, 82,000 times a scale unit; a scale unit adds 66.4 M instructions under MIR against 57.4 M under Cranelift at scales 1 to 3, and the two are equal from scale 6 to 12, where the Peak tier (Cranelift in both) runs the loop. See "Gaps".
- The hash kernels within 10% of Cranelift: all six, from -19.6% to +5.0%.
- No suite benchmark more than 1% above main: `table/hash_read_indexed` (+1.35%) misses it, and it is the only one; the next is `string/utf8_pack` (+0.47%), which has no tree that moved, so it is layout noise of the new binary.
- Suite total under MIR below main's: yes, -0.92%.

Two goals were missed at stage 6, both by `table/hash_read_indexed`: it was over the 1% limit against main and above Cranelift. They are not hard gates, but the switch to default-on was conditional on them, so the probe stayed off by default then. What would let it become the default was the rejoin: a side that rejoins the root should not run the entry again. Stage 7 did that (see "Rejoin straight to the loop header (stage 7)"): `hash_read_indexed` is 1.7% below Cranelift's count with the probe on, and the default changed with `debug_env::mir_probe` and the tests that set the variable.

Compile cost. The compile thread (`OXIMOND_JIT_ICOUNT`) over the suite at scale 3: 92 trees under MIR on main, 153,322 instructions a tree; 105 with the probe on, 241,723 a tree. The 92 trees that were MIR's already cost 14.14 M together now against 14.11 M (+0.2%), so the per-tree figure moves because 13 trees come in that Cranelift compiled before: they cost 864,702 instructions each under MIR (11.2 M in all) and 6.4 M each under Cranelift in the main run (83.1 M for the 13), 7.4 times fewer. A probe is about 60,000 instructions a step in line and about 190,000 for the rest of a small tree: `table/hash_read`'s 9-instruction tree costs 684,000 with 8 steps and 437,000 with 4 (the figure of 1,000 instructions per MIR instruction of the stage 1 measurement holds: 696 MIR instructions). The sum over every compile of the suite, either backend: 307.9 M on main, 236.0 M with the probe on. `WALK_STEPS` is the trade: 4 steps run `table/hash_read` at -1.7% and `table/hash_churn` at +4.4% against Cranelift's count, 6 at -7.5% and +1.5%, 8 at -10.0% and -0.1%, 16 at -10.3% and -0.6% (the same build, before the later changes), so 8 is where the gain stops paying for the code.

The tables of stage 5, for reference. Its kernels (instructions per iteration, other loops than the ones above): lookup with the key and table sharing the string 75 (Cranelift) and 76 (MIR); equal strings in two allocations 88 and 140; an absent key 74 and 101; 200 keys 84 and 89; 2000 keys (`table/hash_read`) 66 and 83; a store over a present key 106 and 162; delete and revive 101 and 158. Its benchmarks against Cranelift on the same binary: `hash_read` +25.6%, `hash_read_small` +16.7%, `hash_read_shaped` +5.0%, `hash_read_indexed` +69.4%, `hash_write` +26.8%, `hash_churn` +51.8%, `metamethod_function` +3.7%, `metamethod_newindex` +17.2%; suite total +1.71% above Cranelift.

Coverage with the probe on (`OXIMOND_JIT_MIR_PROBE=1`), `benchmarks/mir_backend.py coverage` through the gate on the final build: 128 compiles, 105 accepted (82.0%) and 23 refused. 4211 trace IR instructions, 3644 accepted (86.5%) and 567 refused. The accepted ones made 23,798 MIR instructions and 253,772 bytes (stage 5 with the probe on: 15,513 and 194,806, the unrolled walk is the difference). Of the 89 benchmarks, 3 compile nothing at scale 3, 71 have every compile accepted and 10 have only refusals. With the probe off, the default, the figures are those of the closures commit `8aef9d0c`: 92 accepted (71.9%), 36 refused, 3416 IR accepted (81.1%), 795 refused (the probe-off path was not measured again: no tree lowers a skip without the probe, and the tests of the default build run it). The eight benchmarks are the ones that moved, 13 compiles with 228 trace IR instructions, as in stage 5.

Final checks of stage 6, on commit `5d2789ff`, in the state that stage left (the probe off by default, so `OXIMOND_JIT_MIR_PROBE=1` is what the probe tests and the benchmark runs above set; stage 7's checks are in its own section). Through `scripts/heavy.sh -m 24G`, `cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast` gave 4691 passed and 0 failed with `OXIMOND_JIT_BACKEND` unset, and 4691 passed and 0 failed with `OXIMOND_JIT_BACKEND=cranelift`. `cargo clippy --locked --workspace --all-targets` exits 0. A first version of this unit turned the probe on by default and passed the same suites; it was turned back off because the goals above are missed, and the measurements of this section were taken on that build with the switch set to `1`, which is the same code path.

The difftest slice (`run-jit --seeds 0..300 --profiles all --variants tracing,tracing-mir,tracing-mir-profile`, release binary, through the gate, default state): 1800 cases, 0 mismatches. Under `tracing-mir`, 8120 trees compiled through MIR and 7438 were handed back; under `tracing-mir-profile`, 13,686 and 15,720. The 100-seed slice with the debug binary (600 cases, `--reduce`, another session's exclusive job running at times) had 3 timeouts at the 20 s default and no mismatch of output: luajit21 and lua53 seed 15 under `tracing-mir-profile`, luajit21 seed 32 under `tracing` (flagged flaky by the harness; the same seeds timed out in the stage 5 slice). Run alone with `--timeout 120` they pass under all three variants, so nothing went to the known-bug corpus. Because the harness does not set the probe switch, the same release slice also ran with `OXIMOND_JIT_MIR_PROBE=1`: 0 mismatches, 8590 trees compiled through MIR under `tracing-mir` and 6968 handed back, 14,118 and 15,288 under `tracing-mir-profile`. Note for `run-jit`: `--out` has to be an absolute path, or the trace-profile files of `tracing-mir-profile` cannot be written from the case directory and every case of that variant reports `unexpected_error`.

**The short first bucket and the handed probe (2026-11-13).** Four changes after the stages above, measured with `benchmarks/icount.sh -r tracing` at scale 3 through `scripts/heavy.sh -x` (M instructions, before and after): `table/hash_read` 77.6 and 57.2, `table/hash_read_small` 126.3 and 86.6, `table/hash_read_shaped` 84.3 and 61.3, `table/hash_read_indexed` 163.0 and 137.2, `table/metamethod_function` 194.5 and 154.6, `table/hash_write` 119.8 and 104.2, `table/hash_churn` 75.2 and 68.2, `table/metamethod_newindex` 162.1 and 152.7.

- The mix that spread a string's hash before the probe took its start (a fold and a multiply at every probe) is now the last step of the hash itself (`finish_hash` in `string/mod.rs`), so the start is the top bits of the cached hash and the probe reads its high half with one 32-bit load (`NATIVE_STRING_PROBE_START_OFFSET`).
- A bucket is 32 bytes and has a fourth word, `owner`: what a value that holds the key string has in its payload. The lowering masks the high half of the hash with the mask shifted by five (`Words::lenf` of a string index, made once where the binding words are loaded), which is the bucket's offset, and compares the owner word with the key's payload as loaded. From the key's value to the branch the straight line is seven instructions; with 24-byte buckets and the key word it was ten. `native_probe_start` is `hash >> 37` so that the masked half is the same bucket, and an index has at most 2^27 buckets (`NATIVE_PROBE_MAX_SLOTS`; a table with more entries than that allows has no index).
- The address of the bucket's value is loaded before the compare and is the skipped value of the join, and the walk in the block carries no flag: `step` ends at the first bucket whose hash word is the key's or zero, and `settle` reads that bucket's owner (none: a miss; the key's: the entry; another: the word compare, then the conditional call). The allocator binds an operand of an `If` compare that nothing holds yet away from the registers the join expects (`do_if` in `ra.rs`), so the skipped path has no moves.
- A side recorded from the type guard of a lookup starts by making the same lookup again (and, for a read that went through a proxy's `__index` function, the proxy's miss before it). When the side runs once per entry and its lookups pair up with the parent's (`takes_probe`), the guard passes the address its probe found as one more side argument and the side reads that. `table/metamethod_function` and the miss of `table/hash_read_indexed` are this case.

What is left on `table/hash_read_indexed` (137.2 M against LuaJIT's 65.3 M): its keys are other allocations than the table's, so every lookup runs the block and the three-word compare (about 29 instructions), and one lookup in eight is a miss that leaves through the side and re-enters the loop.

**Adopted keys (2026-11-14).** The compare of an inline key with an equal string in another allocation is no longer paid at every lookup. When `settle` finds the words equal, a conditional call (`trace_adopt_key` in `lower/probe.rs`) gives the array slot the key was borrowed from the table's own string: it clones the handle the bucket's owner word names into the slot. The next lookup of that slot in that table ends at the first compare. Lua cannot tell two equal strings apart, so the swap is invisible to the program; a long key is never swapped (its words differ and the VM compares it), and neither is a miss.

- What keeps it safe. A borrow (`TraceType::StringRef`) names the slot, so every later reader gets the new string. The only code that holds the old string by address is the probe itself: the word `key_hash` loaded, which a store reads again for the `__` check and which later probes of the same key reuse. The call therefore parks the string it takes out of a slot in a ring of 8 (`ADOPT_RETIRED`) per thread, and `key_hash` reuses a loaded word only for the 7 probes after the one that read it. Each probe adopts at most once, so a word in use always names a live string. Walking with the old word is still correct: it is an equal string with the same hash.
- What keeps it from thrashing. One slot read in two tables that hold equal keys in two allocations would be swapped at every lookup. The call counts an adoption that replaces a string an adoption installed (a 64-entry table of recent owner words) as a strike. After 16 strikes it sets `TraceControl::adopt_off`, which the code tests before calling, and the thread's next 256 runs set it at their first call. A calm run of 8 adoptions forgives a strike. Such a loop then runs the old path plus four instructions.
- The call is not in a skip of its own. An `If` around it cost every path of the probe: `table/hash_read`, whose keys are the table's own, went from 50.0 M to 53.5 M. As a bare `CallIf` on `different | adopt_off` the other benchmarks are unchanged.
- An early miss on an empty first bucket was tried and dropped for the same reason. Testing the owner word the first compare already loaded for zero, in a skip around the walk, saves a lookup of an absent key about 8 instructions (`table/metamethod_function` 153.8 M to 148.4 M, `table/metamethod_newindex` 150.2 M to 146.7 M). The skip cost the hits more: `table/hash_read` 50.1 M to 53.5 M, `table/hash_read_small` 78.8 M to 84.5 M, `table/hash_read_shaped` 61.4 M to 66.4 M. It needs a block shape whose joins the register allocator does not settle on the hit path.
- Cranelift's probe does not adopt. It reads the slots MIR already swapped, and the tests of `hash_tests.rs` that compare the backends now give each one a world of fresh strings.

Measured with `benchmarks/icount.sh -r tracing` at scale 3 through `scripts/heavy.sh -x` (M instructions, main `8325c9ee` and this change, load average 22 to 26): `table/hash_read_indexed` 134.1 and 81.1 (LuaJIT 65.3), `table/hash_read` 50.0 and 50.1, `table/hash_read_small` 78.7 and 78.8, `table/hash_read_shaped` 61.3 and 61.4, `table/hash_churn` 68.1 and 68.2, `table/hash_write` 104.0 and 104.1, `table/metamethod_function` 153.7 and 153.8, `table/metamethod_newindex` 150.2 and 150.2. A scale unit of `hash_read_indexed` now costs 21.5 M against LuaJIT's 21.3 M (the difference of runs at scales 1 and 3); the rest of the gap is fixed cost, 16.5 M against 3.7 M, spent before the loop runs compiled. A loop that reads two tables with equal keys of their own through one array (500 keys) went from 108 to 92 instructions an iteration: one of the two lookups keeps the short path after the thread stops swapping.

What a table "filled by a compiled trace" was. The warmup unit saw `hash_read_indexed` 84% slower when the fill loop compiled earlier. The table is the same either way. The earlier compile moved the iteration at which the read loop's root is recorded onto one of the misses (one lookup in eight), so the root holds the miss and seven lookups in eight leave through the side and rejoin. Moving the miss to another residue of `i % 8` in the benchmark gives the same two figures with no change to the fill: 64.1 M a scale unit when the root records a miss, 39.0 M when it records a hit, on main; 46.2 M and 21.1 M with adopted keys. The fix is not in the probe: a root whose side takes most iterations should be recorded again, or the rejoin should cost less than the 22 cell loads it runs now.

**Byte groups (2026-11-14).** `StringByte` has its own index check (`lower/bytes.rs`). A float key is truncated, converted back and compared with itself, then one unsigned compare of the zero-based index with the byte count stands for both bounds: 8 instructions where `index_within` has 12. A key that is a number plus a small whole constant (followed back through up to four sums) is checked through that number, the root: the first read of a group converts the root once, each later read adds its constant to the integer and makes the one compare, and a read whose constant lies between two already checked for the same string makes none. A byte read twice is loaded once. The facts live as long as the probe's key words (they are dropped where `keys` is, except after a table write) and a guard with a side attached is always made. Checking the root can leave where the key alone would pass (a root just below a whole number whose sum rounds up); the exit runs the call in the interpreter. `string/byte_sub` (four bytes of a substring the trace never makes, then the first again): 61.9 M and 37.5 M, LuaJIT 32.6 M. Tests: `tests/trace_string_bytes.rs` runs groups on both sides of a position, the bytes of a substring, float positions and two strings of different lengths over 20 positions that have to leave, in every profile; with the proven-range test replaced by "any read was checked" four of its five tests fail.

### Rejoin straight to the loop header (stage 7, the rejoin)

Stage 6 left the probe off because a side that rejoined the root ran the root's entry again: the tag checks stayed above its `Join`, but each link stored the new loop parameters into the entry input slots, and the root loaded every input, the binding words and the rows pointer, ran the preheader and the range checks. Cranelift's link passes the parameters in registers to its outer header and everything above it dominates. The unit rebuilt the `Join` after that model (see "Sides that rejoin the root loop"): it sits below the entry loads and, when the loop passes no parameter on unchanged, below the preheader too; it takes every loop parameter and the fuel; the values the code below reads are kept in frame cells.

Measured with `scripts/heavy.sh -x`, `benchmarks/mir_backend.py icount`, one binary per column (before: `411888bf` built from the branch; after: the code of `5fee3244`) at paths of equal length (`.../b/oximond` and `.../a/oximond`), `env -i` environments of equal length (`--probe` is passed in every run, and `PAD` equalizes the backends), pinned to CPU 6 (before) and 7 (after), whose SMT siblings were idle. Load average at the start of the four suite runs: 4.54 6.85 10.63, 4.12 6.64 10.50, 3.72 6.43 10.37, 3.34 6.21 10.24 (the gate held every slot; the load is other sessions' ungated work). "Before" is each build's default: the probe off before, on after. The Cranelift column is the same binary as the "after" column: the whole-process count of a benchmark moves by up to 1.4% between the two binaries on both backends (`opt/retained_slots` Cranelift 3031.77 M before and 3073.66 M after, MIR 3031.84 M and 3073.74 M; Cranelift's suite total is 21.051 G before and 21.113 G after), so the comparison that counts is MIR against Cranelift within one binary.

The benchmarks that have a tree with a side that rejoins (found with `OXIMOND_JIT_TRACE_IR_OPT` and `terminator: Link {`), whole process, scale 3:

| Benchmark | Before, MIR (M) | After, MIR (M) | After vs before | Cranelift, same binary (M) | Before vs Cranelift | After vs Cranelift |
|---|---:|---:|---:|---:|---:|---:|
| control/branching | 105.26 | 98.07 | -6.83% | 108.14 | -2.67% | -9.31% |
| control/nested_loops | 26.79 | 26.77 | -0.08% | 33.41 | -19.81% | -19.87% |
| control/nested_calls | 241.95 | 233.58 | -3.46% | 262.57 | -7.85% | -11.04% |
| control/loop_in_call | 46.35 | 44.20 | -4.65% | 45.18 | +2.59% | -2.18% |
| control/short_inner | 30.21 | 27.71 | -8.25% | 34.31 | -11.97% | -19.22% |
| calls/protected | 16.13 | 16.13 | +0.01% | 21.02 | -23.24% | -23.22% |
| table/array_read | 88.45 | 88.45 | -0.01% | 92.31 | -4.18% | -4.18% |
| table/hash_read | 136.70 | 118.16 | -13.57% | 136.61 | +0.03% | -13.51% |
| table/hash_iteration | 52.75 | 52.70 | -0.09% | 52.66 | +0.07% | +0.07% |
| table/metamethod_function | 250.54 | 218.24 | -12.89% | 250.49 | +0.02% | -12.87% |
| string/naive_concat | 285.29 | 292.45 | +2.51% | 292.60 | -0.47% | -0.05% |
| string/utf8_pack | 680.21 | 680.97 | +0.11% | 669.89 | +2.61% | +1.65% |
| algorithm/sieve | 108.57 | 108.37 | -0.19% | 111.67 | -2.81% | -2.96% |
| algorithm/nbody | 28.11 | 27.99 | -0.43% | 33.74 | -16.87% | -17.04% |
| algorithm/mandelbrot | 68.46 | 66.89 | -2.30% | 84.73 | -19.21% | -21.06% |
| algorithm/spectral_norm | 107.82 | 107.83 | +0.00% | 109.78 | -1.78% | -1.78% |
| table/hash_read_small | 223.70 | 200.16 | -10.52% | 223.64 | +0.02% | -10.50% |
| table/hash_read_indexed | 214.18 | 210.50 | -1.72% | 214.06 | +0.03% | -1.66% |
| opt/range_float_index | 452.34 | 452.33 | -0.00% | 458.61 | -1.37% | -1.37% |
| opt/range_fail | 512.76 | 512.77 | +0.00% | 514.81 | -0.40% | -0.40% |
| opt/arith_mixed | 554.81 | 545.10 | -1.75% | 556.55 | -0.31% | -2.06% |
| opt/mod_mixed | 678.13 | 678.13 | -0.00% | 678.10 | +0.00% | +0.00% |
| opt/cold_recovery | 257.80 | 257.80 | -0.00% | 244.25 | +5.55% | +5.55% |
| opt/closure_protos | 485.12 | 433.78 | -10.58% | 436.00 | +11.28% | -0.51% |

Rows whose before and after are equal to 0.1% either have no rejoining tree MIR compiles (`hash_iteration`, `mod_mixed` and `range_fail`'s second tree are refused for other reasons, see "Remaining fallbacks") or a link too rare to show. The three probe rows `hash_read`, `hash_read_small` and `metamethod_function` gain from the probe (on by default now), not from the rejoin; the rejoin is what `hash_read_indexed`, `closure_protos`, `loop_in_call`, `short_inner`, `branching`, `nested_calls`, `mandelbrot` and `arith_mixed` gain.

The eight probe benchmarks, with the probe on before (`--probe 1` on the before build, the stage 6 column) and after, whole process:

| Benchmark | Before, probe on (M) | After, probe on (M) | Cranelift, same binary (M) | After vs Cranelift |
|---|---:|---:|---:|---:|
| table/hash_read | 118.21 | 118.16 | 136.61 | -13.51% |
| table/hash_read_small | 203.63 | 200.16 | 223.64 | -10.50% |
| table/hash_read_shaped | 93.70 | 93.70 | 139.20 | -32.69% |
| table/hash_read_indexed | 217.10 | 210.50 | 214.06 | -1.66% |
| table/hash_write | 145.31 | 145.26 | 159.58 | -8.97% |
| table/hash_churn | 88.28 | 88.23 | 94.34 | -6.48% |
| table/metamethod_function | 225.89 | 218.24 | 250.49 | -12.87% |
| table/metamethod_newindex | 193.42 | 193.40 | 228.49 | -15.36% |

`hash_read_indexed` is the case the stage was for: its tree has a dozen entry inputs and a side for the missing key that rejoins one iteration in eight (82,000 times a scale unit, 246,000 links at scale 3). The 6.6 M instructions it saves are about 27 a link. The hash kernels (`benchmarks/mir_backend.py kernels --probe 1`) are as before to the second decimal (no kernel has a rejoining side): -19.6% to +5.0% against Cranelift, as in the stage 6 table.

Suite totals, whole process (`icount`, 89 benchmarks):

| Build, probe | MIR (G) | Cranelift, same binary (G) |
|---|---:|---:|
| before, off (its default) | 21.0234 | 21.0508 |
| before, on | 20.8646 | 21.0508 |
| after, off (`OXIMOND_JIT_MIR_PROBE=0`) | 20.9977 | 21.1131 |
| after, on (its default) | 20.8206 | 21.1132 |

The default after is 0.96% below the default before, and 1.4% below Cranelift in its own binary.

Goals of the unit:

- Each targeted benchmark at or below Cranelift: `opt/closure_protos` (+11.3% to -0.5%) and `control/loop_in_call` (+2.6% to -2.2%) are. The rest of the gap list belongs to other units and did not move: `string/format` +7.5%, `string/pattern_run` +6.6%, `opt/cold_recovery` +5.5%, `opt/mod_float` +4.0%, `string/utf8_pack` +1.7% (from +2.6%), `opt/mod_int` +0.95%, `opt/closure_family` -0.05% (at Cranelift's count; no tree of the last one has a rejoining side).
- No benchmark more than 1% above its count before: `string/naive_concat` (+2.51%), `string/format` (+1.30%) and `opt/retained_slots` (+1.38%) are, and so is Cranelift on the same two (`naive_concat` +2.08%, `format` +1.40%, `retained_slots` +1.38%): the binary moved their whole-process count, not the backend. `naive_concat` also has its second tree, the one with the rejoining side, compiled by MIR now (it was a fallback under the VM-call gate), and MIR's count equals Cranelift's within 0.05%.
- Suite total under MIR lower than before: yes, -0.96%.
- Output identity: `benchmarks/mir_backend.py diff` on the 89 benchmarks, all the same.

Coverage (`benchmarks/mir_backend.py coverage`, scale 3): 128 compiles, with the probe on 107 accepted (83.6%) and 21 refused; 4211 trace IR instructions, 3704 accepted (88.0%) and 507 refused; 24,571 MIR instructions and 268,288 bytes (before, probe on: 105 compiles, 86.5% of IR, 23,798 and 253,772; the cells and the larger join are the difference). With the probe off: 94 accepted compiles (73.4%) and 3476 IR instructions (82.5%), against 92 and 3416 (81.1%) before; the two compiles and 60 IR instructions gained were refused for `side that rejoins a loop whose parameters do not start from entry inputs`. Of the 89 benchmarks, 3 compile nothing, 73 have every compile accepted and 10 have only refusals.

Compile cost: the join keeps a cell for every value the lowering holds above it, and the allocator drops those no load reads, so the work is in the lowering (one pass over the value tables) and not in the code. `a_steady_state_compile_allocates_nothing_in_the_backend` and its probe variant still pass, with the cell and remap tables in the scratch.

Tests added: `a_side_that_rejoins_a_loop_with_a_constant_start_runs_through_mir_like_cranelift` and `a_hoisted_check_on_a_register_only_a_side_writes_runs_through_mir_like_cranelift` in `mir/tests.rs`, the join below a preheader with cells in `oximond-mir/tests/rejoin.rs`, a join with cells and rejoining sides in the random programs (60,000 seeds on both encodings), `with_the_probe_off_mir_leaves_the_lookup_to_cranelift` in `hash_tests.rs`, and the CLI snippets: "a branch that merges back into the loop" and "a hoisted call into the VM under a side that rejoins" now compile their tree with the side and allow no fallback; `with_the_probe_off_mir_declines_the_trees_that_probe` sets `OXIMOND_JIT_MIR_PROBE=0`. The range-versioned root with a rejoining side is `range_proven_loops_with_sides_run_through_mir_like_cranelift`, unchanged and passing. Review fixes: `a_call_hoisted_over_a_parameter_a_side_changes_runs_again_after_each_link_like_cranelift` and `a_call_hoisted_over_an_entry_input_runs_once_however_often_a_side_links_like_cranelift` in `mir/tests.rs` (a hoisted `HostCall` with the join before and after the preheader, a host that counts its calls: the calls, the outcome, the control block and the exit area equal Cranelift's), and `a_side_that_rejoins_the_loop_leaves_where_cranelift_leaves` in `mir_backend.rs` (exits, side exits, iterations and output of the xpcall snippet and of three programs whose side inserts into the array the root reads). The unit harness cannot run a call that moves array storage, and MIR declines an `ArraySet` of a number, so the words case is covered end to end only.

Final checks, on commit `5fee3244` (the code; the commit after it changes this file only): through `scripts/heavy.sh -m 24G`, `cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast` gave 4696 passed and 0 failed with `OXIMOND_JIT_BACKEND` unset, and 4696 passed and 0 failed with `OXIMOND_JIT_BACKEND=cranelift`. `cargo clippy --locked --workspace --all-targets` exits 0.

Final checks after the review fixes, on commit `e59b2743` (the code; the commit after it changes this file only): through `scripts/heavy.sh -m 24G`, `cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast` gave 4699 passed and 0 failed with `OXIMOND_JIT_BACKEND` unset and 4699 passed and 0 failed with `OXIMOND_JIT_BACKEND=cranelift`. `cargo clippy --locked --workspace --all-targets` exits 0. The difftest slice in the default state (probe on): `run-jit --seeds 0..300 --profiles all --variants tracing,tracing-mir,tracing-mir-profile` with the release binary, 1800 cases, 0 mismatches; the same with `--seeds 0..100` and the debug binary, 600 cases, three timeouts under load (lua53 and luajit21 seed 15, luajit21 seed 32, the last in the plain `tracing` variant) that passed rerun alone with `--timeout 120`. The campaign needed an absolute `--out` then: with the default relative one the profile variant wrote its profile to a doubled path and every case failed with `cannot write ... trace-profile`. Since `443c93e0` the harness makes `--out` absolute itself.

The difftest slice (`run-jit --seeds 0..300 --profiles all --variants tracing,tracing-mir,tracing-mir-profile`, release binary, through the gate, the default state, `--out` absolute): 1800 cases, 0 mismatches. Under `tracing-mir`, 9,892 trees compiled through MIR and 5,666 were handed back; under `tracing-mir-profile`, 15,852 and 13,554 (stage 6 with the probe on: 8,590 and 6,968, 14,118 and 15,288). The 100-seed slice with the debug binary (600 cases, `--reduce`): 3 timeouts at the 20 s default, the same three as in stage 5 and stage 6 (luajit21 seed 15 and lua53 seed 15 under `tracing-mir-profile`, luajit21 seed 32 under `tracing`, which the harness flags flaky), and no mismatch of output; run alone with `--timeout 120` the seeds pass under all three variants, so nothing went to the known-bug corpus. The harness sets no probe switch, so the slices ran with the probe on.

### Exit-only values and the float remainder (stage 7, exit-only values)

Three gaps of the list the rejoin left, each a few instructions an iteration: `opt/cold_recovery` (+5.55% against Cranelift), `opt/mod_float` (+4.04%) and `opt/mod_int` (+0.95%). `NumberModulo` was refused, so `opt/mod_mixed` ran on Cranelift.

**Values only an exit reads.** Cranelift's `defer_cold_values` marks the float arithmetic that only exits, sides and links name (no hot instruction, back edge, tail call or return takes it, and neither does another value that stays), skips it in the loop and computes it again where a cold use needs it. `lower.rs` (`mark_cold`) makes the same pass over the member's body, with one difference: it leaves out what has no stub form here (a power, the compat remainders, which call). What it marks it lowers with `Func::cold` set, so the MIR operations carry `COLD` in `scale`. In `oximond-mir` such an operation is not emitted where it stands:

- The prescan reads its operands as weak uses, like a snapshot read (`weak_snap`), so loop parameters and invariants they name keep a place.
- An exit stub (a guard's, and the `Exit` that ends a region) that holds a cold value computes it (`ra/cold.rs`, `exit_stub_for`): the value and the cold values its operands are, in order, with the operands read where the snapshot would read them (a register, a slot, a constant). The stub leaves the function, so any float register no source of the stub is in is free for the results (`assign_temps`; a result takes the register of its first operand when that dies there). The ordinary `exit_stub` then stores the registers.
- A cold value that something else reads after all (a guard that has a side, a `Goto`, the join values of a skip, a conditional call's arguments, a back edge, a cell the `Join` keeps) gets a register or a slot like any value, and the definition is emitted where it stands: the allocator meets those uses before it meets the definition, and a definition nobody holds is skipped as a dead one already is. A stub that finds the value held reads it. A stub with no float register to spare for its temporaries does the same (`Stats::cold_hot`).

The cost of an exit is a few instructions in its stub and nothing in the loop, and an operand the stub reads that is in no register is stored where it is defined, as a snapshot value that does not fit is. Counted once per stub that computes: `Stats::cold_stubs` and `cold_ops`, and the log line of a compiled tree says `N operations in exit stubs`. Differences from Cranelift: a cold value a guard passes to a side as a parameter is computed in place here (Cranelift computes it again at the guard), and function traces defer nothing, in both.

**The float remainder.** `ModuloWith(Fmod53)` and `ModuloWith(Fmod54)` were one call of the compat function for every operand pair, which is as dear as the call (the loop of `opt/mod_float` was 4 instructions an iteration above Cranelift's in-line form). On a CPU with AVX and FMA3 (`X64::fma`, the host check of Cranelift's `isa_has_fma`) `lower/modulo.rs` emits the form Cranelift emits there (`float_fmod_fma`): `q = trunc(a / b)`, `r = fnma(q, b, a)` (one rounding), the sign of `a` onto `r` (`FCopySign`), the Lua sign fix as selects, and the compat function behind two conditional calls, one when the quotient is not below 2^53 (ordered: a NaN fails, and so do an infinite or zero divisor and an infinite dividend), one when the signed remainder is not the remainder (a rounded quotient one too high, or a NaN remainder from an infinite divisor). The fast result is stored in a word of the control block (`TraceControl::scratch`, a new field that tests zero before they compare control blocks), the calls store their result over it, and the code loads it: no select, and a call made twice when both tests fail (rare, same value). The 5.4 form of the fix is four float selects on the signs of `b` and `m`, the 5.3 form selects on `m * b < 0`, as Cranelift's does. It covers every finite operand pair the quotient allows, not only integral ones. The straight line is 33 instructions from the divide to the load (Cranelift's is about 36), measured on the disassembly of `opt/mod_float`. Without FMA (a CPU older than Haswell) the lowering is Cranelift's integral form (`float_fmod`): both operands integral (`F2I` and `I2F` round trips), below 2^53 in magnitude and a divisor that is not zero take an exact `SRem` with the floor sign fix and `FCopySign` for a zero result, in a forward skip (`SRem` traps on a zero divisor); the five tests fold into one integer flag with selects, and one `CallIf` on that flag calls the compat function for every other pair. The lowering takes the fused flag from the target it compiles for (the arena's `X64`, which a test may set with `with_target`), not from the CPU, so the tests run both forms.

**`NumberModulo`.** `number_modulo` is two skips: the integer floor modulo (`F2I`, `SRem`, the divisor added when the signs differ and the remainder is not zero, `I2F`) when the flag is set, the float modulo of the profile (`float_arithmetic`) when it is not, each skip joining with the other's value. Of the two compiles of `opt/mod_mixed` the first (28 IR instructions) is MIR's now. The second, the tree with the side its `if` makes (42), is refused for `register that is a float or an integer at run time` (a snapshot that passes a number of run-time subtype to a side, which Cranelift checks behind the side's entry), so the benchmark still runs mostly on Cranelift, and the -0.38% is the first compile's.

**The integer floor modulo.** The general path of `integer_division` built a 0 or 1 from each condition, negated it into a mask and anded the divisor. It now selects the divisor itself (`select_any(Lt, signs, 0, divisor, 0)`, then `select_any(Ne, remainder, 0, ...)`) and adds it: the same value in about five instructions fewer. This is the whole of the `opt/mod_int` change.

Measured with `scripts/heavy.sh -x`, `benchmarks/mir_backend.py icount` (scale 3, `env -i` environments of equal length, `PAD` equalizing the backends), one binary per column at paths of equal length (`bin-a/oximond` built from `066251ba`, `bin-b/oximond` from `9390bee9`, the final code), pinned to CPU 7 (sibling 15 idle), load average at the start of the runs `4.11, 3.28, 3.08` (before) and `4.74, 5.27, 6.38` (after, rerun on the final code; other sessions were running, which does not move a user-mode instruction count). Whole process, the compile thread included:

| Benchmark | Cranelift before (M) | MIR before (M) | MIR vs Cranelift before | Cranelift after (M) | MIR after (M) | MIR vs Cranelift after | MIR after vs before |
|---|---:|---:|---:|---:|---:|---:|---:|
| opt/cold_recovery | 244.25 | 257.80 | +5.55% | 244.24 | 240.90 | -1.37% | -6.55% |
| opt/mod_float | 872.69 | 907.96 | +4.04% | 872.69 | 836.64 | -4.13% | -7.85% |
| opt/mod_int | 697.87 | 704.49 | +0.95% | 697.87 | 694.02 | -0.55% | -1.49% |
| opt/mod_mixed | 678.11 | 678.13 | 0.00% | 678.10 | 675.53 | -0.38% | -0.38% |

Suite total (89 benchmarks): MIR 20,818.17 M before and 20,615.73 M after (-0.97%); Cranelift in the same binaries 21,110.78 M and 21,010.35 M (the two Cranelift columns differ by -0.47%, which is the tier-up noise of the long benchmarks: `opt/retained_slots` alone moves 75 M in both columns). MIR against Cranelift in the same binary: -1.39% before, -1.88% after. No benchmark is more than 1% above its MIR count before: the largest are `string/table_concat` +0.52% and `stdlib/tonumber_parse` +0.42%, and Cranelift moves by +0.48% and +0.41% on the same two (the binary, not the backend). The three targets and `opt/mod_mixed` are at or below Cranelift, the goals of the unit. The after columns were measured again on `9390bee9` after the review: the four targets are unchanged to the 10 K instructions of the table, the suite total moved by 0.4 M. The 7.2 M iterations of `opt/cold_recovery` save 2.3 instructions each, the 9 M of `opt/mod_float` 7.9, the 9 M of `opt/mod_int` 0.7.

Coverage (`benchmarks/mir_backend.py coverage`, scale 3): 128 compiles, 108 accepted (84.4%) and 20 refused; 4211 trace IR instructions, 3732 accepted (88.6%) and 479 refused; 24,735 MIR instructions and 271,281 bytes (the rejoin: 107 accepted, 3704 IR, 24,571 and 268,288). `NumberModulo` is gone from the refusals (2 compiles, 70 IR), and `register that is a float or an integer at run time` (1 compile, 42 IR: the second compile of `opt/mod_mixed`) is a row, where the table of the rejoin counted that compile under `NumberModulo`. Of the 89 benchmarks 3 compile nothing, 73 have every compile accepted, 9 have only refusals and 4 both. With `OXIMOND_JIT_MIR_PROBE=0`: 95 accepted of 128 (74.2%), 3504 IR instructions accepted (83.2%).

Tests: `oximond-mir/tests/cold.rs` (a loop whose guard reads a chain of cold values, one something else reads too, one a side takes, a stub with no register to spare, constants and cold operands; stats say what the stubs computed), `FTrunc`, `FCopySign`, `FNma` and the float `CallIf` in `tests/ops.rs` (every special value against Rust, both encodings, `FNma` where the CPU has FMA), cold, `FTrunc` and `FCopySign` operations in the random programs (375 stubs that compute in the 1500 seeds of the default list; 20,000 seeds pass on both encodings), and `FNma` (generated where the target is fused, `generate_for`) and `CallIf` on float and integer compares, the unordered conditions among them, storing into the control block (over 100 of each kind in the default list, asserted; 100,000 seeds pass on SSE2, AVX and AVX with FMA, and 60 seeds against the CLIF translation), and in `mir/tests.rs` `values_only_exits_read_match_cranelift` (the stats say seven operations a stub; the control block, the exit area and the fuel equal Cranelift's), `float_remainders_call_the_helper_only_off_the_fast_path` (a counter in the helper, on each target the lowering has a form for: fused, AVX without FMA and SSE2. With FMA no call for finite operands with a quotient below 2^53, without it no call for integral operands below 2^53 with a divisor that is not zero, and a call for a zero or infinite divisor, NaN, an infinite dividend and a large quotient, in both profiles), `float_remainders_match_cranelift_on_every_target` and `number_modulo_matches_cranelift` (integers up to 2^53 with the flag set, every float pair with it clear, four float forms, each target). In `oximond-cli/tests/mir_backend.rs`: snippets for the remainder (divisors of both signs, zero, infinities, a dividend that is too large), `NumberModulo` and the exit-only values (the log must say the stub computes), and `modulo_and_exit_only_snippets_print_what_the_interpreter_prints` (lua51, lua53, lua54 and lua55, interpreter, MIR and Cranelift). `trace_mixed_modulo` and `float_arith` pass unchanged.

A bug the review found in the cold values, with no tree that reaches it today (the lowering marks only values of the loop body, which come after the header): a cold value defined before a loop header and read only by an exit snapshot inside the loop. When its stub had to read it where it stands (a chain over 64 operations, or no register to spare), `weak_snap` had only followed its operands, so the value was never an invariant of the loop, got no register at the back edge and its definition was skipped as dead; the exit stored garbage. `weak_snap` now treats such a value as a plain weak use, which makes it an invariant computed once before the loop. `a_cold_chain_before_the_header_is_an_invariant_of_the_loop` in `oximond-mir/tests/cold.rs` covers 3 and 70 operations (the second fails without the change), and the random programs put cold chains before the header in a fifth of the programs with a loop (20,000 seeds pass on SSE2, AVX and AVX without FMA; 12 of them failed without the change).

A second bug the full test run found, in this unit's own code: the prescan followed the operands of a cold value once for each path to it, and a run of statements on one variable (`x = (x * 3 + 7) % 1009; x = ...`, 60 of them, as `trace_limits` has) reads each value twice, so the compile of the tree with that side was exponential and the test hung. The prescan now stamps each value with the instruction that read it, and a stub that would compute more than 64 operations (`MAX_COLD_OPS`) leaves the chain where it stands, since every exit after it would repeat the chain. `a_long_chain_of_shared_cold_values_compiles_at_once_and_runs_right` in `oximond-mir/tests/cold.rs` covers 10, 40 and 200 operations.

The random programs with `FNma` found no allocator bug, but the evaluator did not model the NaN that the instruction passes on: it negated `a` first, flipping the sign of a NaN `a`, which a later `FCopySign` shows. `FNma` now passes on the first NaN operand quieted with its sign, as the hardware does (see `Op::FNma`). The test helper of the conditional call counts every NaN as one, since an operation of two NaNs of different signs gives the machine's pick (the allocator may swap the operands of a commutative operation).

A bug in `oximond-mir` the random programs found: the SSE2 form of `FNeg` and `FAbs` built the result in the destination, so a result that only a slot holds (computed into the scratch register, which also held the mask) came out as zero. It cannot be seen on an AVX host that runs only the VEX forms, and the random programs found it when they began to generate `FNeg`. `FCopySign` needed the same care for the same register.

Final checks, on commit `9390bee9` (the code; the commit after it changes this file only): through `scripts/heavy.sh -m 24G`, `cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast` gave 4715 passed and 0 failed with `OXIMOND_JIT_BACKEND` unset and 4715 passed and 0 failed with `OXIMOND_JIT_BACKEND=cranelift`. `cargo clippy --locked --workspace --all-targets` exits 0 (its warnings are the ones main has, none in a line this unit added). The difftest slice, `run-jit --seeds 0..300 --profiles all --variants tracing,tracing-mir,tracing-mir-profile` with the release binary and an absolute `--out`: 1800 cases, 0 mismatches. The same with `--seeds 0..100` and the debug binary: 600 cases, the three timeouts the previous stages had (lua53 and luajit21 seed 15 under `tracing-mir-profile`, luajit21 seed 32 under `tracing`, flagged flaky), which pass run alone with `--timeout 120`; no mismatch of output, so nothing went to the known-bug corpus. The icount and coverage numbers above are from a release binary of the same commit. A first full test run before the chain fix hung in `trace_limits` (see above), and the review found the cold value before a header (see above): those are the only failures the checks found in this unit's code.

### The residue of the gap list (stage 7, the residue)

Six benchmarks were on the gap list with no named cause: `opt/closure_protos`, `string/format`, `string/pattern_run`, `control/loop_in_call`, `string/utf8_pack` and `opt/closure_family`. The whole-process count hides where the instructions go, because MIR's compile thread is about 20 times cheaper than Cranelift's and that saving offsets a loss in the run. Each benchmark was split into the two with `OXIMOND_JIT_ICOUNT=1` (the `jit-icount` lines, summed over the compile jobs) and, for the run, a slope: two scales run with `OXIMOND_JIT_SYNC=1` under `perf stat -e instructions:u`, both small enough that the loop stays in the Optimized tier (under 2 M iterations), so the difference is the cost of the iterations and nothing else. The trees were read with `OXIMOND_JIT_TRACE_IR_OPT` and `OXIMOND_JIT_DISASM` (`benchmarks/mir_disasm.py` for the MIR code, the machine-code listing for Cranelift's).

Where the extra instructions were, on `15b7f761`, whole process at scale 3 (the compile column is MIR's jobs minus Cranelift's):

| Benchmark | Total (M) | Compile thread (M) | Run (M) | Iterations | Run per iteration | Cause |
|---|---:|---:|---:|---:|---:|---|
| string/format | +32.02 | -1.90 | +33.92 | 240,000 | +141 | `#s` of the string `string.format` made was a call into the VM |
| string/pattern_run | +121.50 | -1.69 | +123.19 | 1,200,000 | +103 | `#string.match(...)`, the same call, once an iteration |
| string/utf8_pack | +8.37 | -14.62 | +22.99 | 90,000 | +255 | `#record` and `#shifted`, two of the same call |
| opt/closure_family | -0.14 | -2.24 | +2.10 | about 2,000,000 | about +1 | the prototype check (see below), set against what MIR saves in the cells |
| opt/closure_protos | -1.66 | -76.58 | +74.92 | about 2,000,000 | +36 | the rejoin of the sides, see below |
| control/loop_in_call | -0.95 | -0.86 | -0.09 | 36,000 calls | none | no gap left since stage 7 (the rejoin) |

**The length of a lane string.** `HostCall` of `StringLength` on a string the trace holds by lane (a string a helper made) is no call in Cranelift: `host_call` of `emit.rs` reads a 32-bit word of the run's length table (`TraceControl::lengths`), and calls the VM only for a root lane (a constant or loop-carried string from `ROOT_STRING_LANE_BASE` on, which the table does not hold). The MIR lowering sent every one to the VM through `host_entry`: the arguments stored as tagged words, the call, a guard on the answer's tag, the answer loaded, about 130 instructions with the VM's own work. All three string benchmarks had one or two such calls in each iteration of their tree and nothing else MIR does worse. `string_length` in `lower/calls.rs` makes the same split: a skip that loads `lengths[lane]` (an indexed `Load32`, converted to a float when the trace's number type is float) for a lane below the root base, and a second skip with the VM call for the rest, joined in the length. A string constant is a root lane, so it keeps the VM call, as Cranelift's. The two skips cost an extra compare against Cranelift's one branch.

**The prototype check.** `closure_family` has a `CallHelper` that checks the closure's prototype against the pinned one once an iteration, then four cell accesses through the same row. MIR built the row address of the pinned closure with a multiplication of two constants (`mov ecx, 0; imul rcx, rcx, 0x48; add`), made a 0 or 1 with two selects (`cmp`, `cmove` for the equal prototypes, `cmp`, `cmovne` for the non-zero one), and the guard compared the result with zero again: 27 instructions from the end of the previous iteration to the first cell read, against Cranelift's 19, which MIR made up elsewhere by reusing the row address for the four cell reads (Cranelift multiplies the lane by the stride for each read). `pinned_row_word` now reads the pinned row at a constant offset of the rows pointer (one load), and the answer is `select(Eq, live, pinned, live, 0)`, which is non-zero exactly when the prototype matches and is not zero, held as a pending comparison (`CellResult::NonZero`) that the guard takes as one compare. A test with both prototype words zero (a lane that is no closure and a pinned row never filled) checks that equal zeros do not match.

**The rejoin of the sides (closure_protos).** The `closure_protos` row of the table of the rejoin says `-0.51%` against Cranelift, and that was true of the whole process, but only because the compile thread saves 76.6 M instructions: the iterations cost MIR 104.3 instructions each in the Optimized tier against Cranelift's 68.5 (+52%). The eight prototypes rotate through one call site, so the tree is a root and seven sides, and seven iterations in eight leave the root through a guard into a side and rejoin. The disassembly shows the cost of a link: the values the loop reads of the entry are loaded from frame cells after the `Join` (32 cells), and those that have no register were copied from their cell into a spill slot of their own on every link (`mov rax, [cell]; mov [slot], rax`, 17 instructions). `Compiler::cell_home` in `ra.rs` gives such a value its cell as its slot, when the function has one `CellStore` of that cell (it is stored at the region's entry, before any load, and nothing writes it while the value is read), and `do_cell_load` emits nothing for a value that has no register. The iterations now cost 86.8 instructions each, and the whole process 397.65 M against 436.04 M (-8.8%). `Stats::cell_homes` counts the values.

**Measured.** `scripts/heavy.sh -x`, `benchmarks/mir_backend.py icount`, one binary per column at paths of equal length (`a/oximond` built from `15b7f761`, `d/oximond` from the code of `2055a887`), `env -i` environments of equal length, pinned to CPU 7 (sibling 15 idle), scale 3, whole process. Load average at the start of the two suite runs: `5.61 3.85 4.16` (before) and `5.92 9.78 8.45` (after; other sessions' ungated work, which does not move a user-mode instruction count). Cranelift is the same binary as the "after" column (the two Cranelift columns differ by under 0.001%).

| Benchmark | Cranelift (M) | MIR before (M) | MIR after (M) | Before vs Cranelift | After vs Cranelift | After vs before |
|---|---:|---:|---:|---:|---:|---:|
| opt/closure_protos | 436.04 | 434.37 | 397.65 | -0.38% | -8.80% | -8.45% |
| opt/closure_family | 343.39 | 343.24 | 326.46 | -0.04% | -4.93% | -4.89% |
| string/format | 409.03 | 441.04 | 407.46 | +7.83% | -0.38% | -7.61% |
| string/pattern_run | 1847.86 | 1969.36 | 1800.71 | +6.58% | -2.55% | -8.56% |
| control/loop_in_call | 45.18 | 44.23 | 43.94 | -2.10% | -2.75% | -0.65% |
| string/utf8_pack | 667.93 | 676.30 | 651.05 | +1.25% | -2.53% | -3.73% |

Suite total (89 benchmarks): MIR 20,616.07 M before and 20,314.79 M after (-1.46%); Cranelift 21,010.77 M and 21,010.67 M. MIR against Cranelift in the same binary: -1.88% before, -3.31% after. No benchmark is slower than before by more than 0.16% (`stdlib/native_calls` +0.15%, then `string/plain_find` +0.04%: binary layout, neither has a tree that moved), and none is above Cranelift by more than 0.08% (`table/hash_iteration`, which MIR leaves to Cranelift). The others that moved: `table/metamethod_function` -2.47%, `opt/arith_mixed` -1.99%, `control/short_inner` -1.29%, `table/hash_read_indexed` -1.28%, `control/loop_in_call` -0.65%, all through the cell homes (they link through rejoining sides).

The run, by the slope, per iteration in the Optimized tier (`OXIMOND_JIT_SYNC=1`, two scales under the Peak threshold), MIR against Cranelift in the same binary:

| Benchmark | Before | After |
|---|---:|---:|
| string/format | +8.3% | +0.1% |
| string/pattern_run | +6.9% | -2.5% |
| string/utf8_pack | +3.5% | -0.4% |
| opt/closure_family | +1.4% | -9.7% |
| control/loop_in_call | -2.4% | -3.6% |
| opt/closure_protos | +52.2% | +26.7% |

`opt/closure_protos` is the one benchmark whose Optimized-tier code is still slower than Cranelift's, by 18 instructions an iteration, and it stays below Cranelift in the whole process only through the compile saving and the Peak recompile that both backends share. Its remaining causes are under "Gaps" below.

Coverage (`benchmarks/mir_backend.py coverage`, scale 3): 128 compiles, 108 accepted (84.4%) and 20 refused; 4211 trace IR instructions, 3732 accepted (88.6%) and 479 refused, the same refusals as after the exit-only values; 24,781 MIR instructions and 271,875 bytes (then: 24,735 and 271,281). `string/utf8_pack` has three accepted compiles and no refusal (the stage 5 table has two accepted and one refused; the reason was never recorded, and none of the three is refused now).

Tests: `the_length_of_a_lane_string_is_read_in_line_like_cranelift` and `..._in_a_loop_matches_cranelift` in `mir/tests.rs` (a host with a length table and a VM answer for root lanes: the outcome, the control block, the exit area and the number of host calls equal Cranelift's for lanes in the table, a root lane, both result types, a wrong tag that leaves through the guard, and a loop that asks every round), `a_failed_prototype_check_...` with the case of two zero prototypes, `a_word_load_with_a_scaled_index_zero_extends_and_a_skip_guards_the_index` in `oximond-mir/tests/ops.rs`, `a_cell_value_without_a_register_stays_in_its_cell` in `oximond-mir/tests/rejoin.rs` (24 kept values, a rejoining side, `Stats::cell_homes` above zero), and the random programs: a joined program takes up to 18 extra kept values, and the first guard of its loop reads each, so the programs reach the cell homes (105,000 in 100,000 seeds, all equal to the evaluator on SSE2, AVX and AVX without FMA; the test asserts the default list reaches at least 20). `oximond-cli/tests/mir_backend.rs` has three new snippets that run under the interpreter, Cranelift and MIR with identical output: a length of a string `string.format` made, `string.find` and `string.match` over short strings, and `string.pack`, `string.unpack` and the `utf8` functions (no snippet covered the pattern helpers, the pack helpers or the utf8 library before).

Final checks, on commit `2055a887` (the code; the commit after it changes this file only): through `scripts/heavy.sh -m 24G`, `cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast` gave 4719 passed and 0 failed with `OXIMOND_JIT_BACKEND` unset and 4719 passed and 0 failed with `OXIMOND_JIT_BACKEND=cranelift`. `cargo clippy --locked --workspace --all-targets` exits 0 (its warnings are the ones the branch had, none in a line this unit added). The difftest slice, `run-jit --seeds 0..300 --profiles all --variants tracing,tracing-mir,tracing-mir-profile` with the release binary and an absolute `--out`: 1800 cases, 0 mismatches; under `tracing-mir` 4,946 trees compiled through MIR and 2,833 were handed back, under `tracing-mir-profile` 7,926 and 6,777. The same with `--seeds 0..100` and the debug binary: 600 cases, the three timeouts of the earlier stages (lua53 and luajit21 seed 15 under `tracing-mir-profile`, luajit21 seed 32 under `tracing`, which the harness flags flaky), and no mismatch of output; each of the three seeds passes under all three variants run alone with `--timeout 120`, so nothing went to the known-bug corpus. The 100,000-seed random-program run of `oximond-mir` passes on SSE2, AVX and AVX without FMA.

Review fixes after the final checks above, each tested in one crate and none measured:

- `cell_home` gave a value its cell as its slot whenever the function stored the cell once. A load above that store in a loop reads the store of the iteration before, and by the time the value was read the cell held this iteration's store. The scan at the start of `compile` (`find_cell_homes`) now gives a cell a home only when its one store comes before every load of it in program order and not after a `Header` or `Join` of its region. `a_load_above_the_only_store_reads_the_store_of_the_iteration_before` in `oximond-mir/tests/rejoin.rs` fails without it (24 cells, a skip that reads the old values after the stores). `Op::CellStore` documents the ordering the allocator relies on, and a debug assertion fires when a kept load comes before a store the allocator dropped.
- `cold_cv` built the whole chain of a stub and gave its leaf operands registers or slots before `exit_stub_for` found it longer than 64 operations, and looked each value up in a list. A first pass (`cold_fits`) counts the distinct cold values and stops at the 65th, before any leaf is placed, and both passes mark a value with a per-value stamp. Stubs that run out of float registers for their temporaries still place the leaves of the failed attempt.
- `FLOAT_EXACT_LIMIT` is in `jit_ir/trace/range.rs` next to `FLOAT_ROUND_MAGIC`, and the walk that finds the cold values is `TraceIr::mark_cold_values`, called by `defer_cold_values` (every float operator) and `mark_cold` (the operators the stubs compute). The Cranelift output does not change.
- `the_prototype_check_reads_the_pinned_row_of_a_lane_that_is_not_the_first` in `mir/tests.rs` pins lane 2 with the callee in lane 1, and compares MIR with Cranelift over a matching prototype, a mismatch at the pinned lane only, another prototype and zero prototypes. It fails when `pinned_row_word` drops its `lane * row_bytes` displacement.
- The preheader fallback reason reads `loop side whose preheader calls into the VM under a side that rejoins its loop`, since the root no longer needs it.

Final checks of the whole stage, after the rebase onto `main` at `6847d447` and the review fixes, on commit `443c93e0` (the oximond code of `938322a4`). Through `scripts/heavy.sh -m 24G`, `cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast` gave 4779 passed and 0 failed with `OXIMOND_JIT_BACKEND` unset and 4779 passed and 0 failed with `OXIMOND_JIT_BACKEND=cranelift`; the difftest fix of `443c93e0` was tested with `cargo test -p oximond-difftest` (89 passed). `cargo clippy --locked --workspace --all-targets` exits 0. The difftest slice with the release binary: 1800 cases, 0 mismatches. With the debug binary and `--seeds 0..100`: 600 cases, the same three timeouts as before, each passing alone with `--timeout 120`.

The suite, `benchmarks/mir_backend.py icount` through `scripts/heavy.sh -x`, scale 3, a release binary of `main` at `6847d447` and one of `938322a4` at paths of equal length, pinned to CPU 7 and CPU 3 (siblings idle), load average `2.18 5.19 7.32` and `2.96 5.21 7.29` at the start:

| | main `6847d447` | `938322a4` |
|---|---|---|
| suite under MIR | 21,077.1 M | 20,338.4 M (-3.50%) |
| suite under Cranelift | 21,086.2 M | 21,033.9 M |
| MIR against Cranelift, same binary | -0.04% | -3.31% |
| benchmarks more than 1% below Cranelift | 53 | 68 |
| benchmarks more than 1% above Cranelift | 8 | 0 |

The largest gains against main are `table/hash_read_shaped` -32.6%, `opt/closure_protos` -18.0%, `table/metamethod_function` -15.0%, `table/metamethod_newindex` -14.4%, `table/hash_read` -13.5%, `table/hash_read_small` -10.9%, `string/format` -9.9%, `control/short_inner` -9.3%, `string/pattern_run` -9.1% and `opt/mod_float` -8.1%. The only benchmark more than 1% above main is `string/naive_concat` (+2.26%), which moved +1.85% under Cranelift between the same two binaries and is equal to Cranelift (-0.06%), so it is the layout of the binary. No benchmark is above Cranelift by more than 0.09%. Coverage of the same binary: 108 of 128 compiles (84.4%) and 3,732 of 4,211 trace IR instructions (88.6%) through MIR. The 20 fallbacks: function traces 6, `ArraySet` under a strict header 3, `ArraySet` without the inline append and hole fill 3, the native `next` step 2, table field fill without the in-place store 2, counted traces 2, an array load past the end 1, a register that is a float or an integer at run time 1.

### The lookup row fill (stage 8, rowfill)

A loop tree with row fields that calls `TABLE_GET_TABLE` (Mineclonia's `starts[hash].starts` at structures.lua:856 and :1328, `ordinal_sturdy[shape]` at nodeprops.lua:705) reads the fields of the table the lookup gave through the row of its lane. Rows are not cleared between runs, so the row holds nothing or the words another run left there, and Cranelift fills only that lookup's row after the call (`call_helper_call` in `emit.rs`, stage 9 of the Mineclonia work). Until stage 8 the MIR lowering made that fill only for a tree with lane array views, so `emit.rs::lower` kept every such tree off MIR with a `fills_lookup_rows` test that logged nothing: those trees appeared in no fallback count.

`fills_table` in `mir/lower/calls.rs` now has the clause Cranelift has: `helper == TABLE_GET_TABLE`, a root that is not a function trace and a non-empty field list, with a table result. The fill is the same `trace_ensure_row(dispatcher, lane)` call as for the other table-giving helpers. Only the lookup: a table a constructor or a cell read gives keeps the row it had (filling it broke the trees that store a fresh table into a cell). A tree that lists table addresses already filled in MIR (`reads_table_addresses`). `fills_lookup_rows` is gone from `emit.rs`.

Tests. `a_lookup_row_is_filled_where_cranelift_fills_it` in `crates/oximond-cli/tests/mir_backend.rs` runs a row hit, a miss (the key is absent on some entries), a record replaced by another shape under the compiled loop and a lookup of a lookup under both backends and the interpreter: same output, same exits, same side exits and same iterations, and at least 250,000 of 300,000 iterations (a row left unfilled leaves at the first field read of every entry; with the clause switched off the test fails on the exits). Two snippets pin that MIR compiles the hit and the nested shape with no fallback. `a_lookup_row_is_this_runs_whether_the_key_hits_misses_or_is_replaced` in `trace_closure_callees.rs` runs the four shapes in all six profiles under the default backend and under `OXIMOND_JIT_BACKEND=cranelift`. No test pinned the old gate.

Measured on `fb82922e` against main `8de8e068`, load average at the start 7.5 to 9. Whole-process `instructions:u`, pinned to CPU 3, `env -i`, binaries at equal-length paths, 1,500,000 iterations of a repro, three runs each (the three agree to 0.001%):

| Repro | main, MIR (the tree went to Cranelift) | stage 8, MIR | stage 8, Cranelift |
|---|---:|---:|---:|
| record read by two fields (`hit`) | 836.541 M | 833.043 M (-0.42%) | 836.541 M |
| key absent on some entries (`miss`) | 1,265.128 M | 1,193.369 M (-5.67%) | 1,266.246 M |
| record replaced every 20,000 entries (`change`) | 851.001 M | 851.781 M (+0.09%) | 851.001 M |
| lookup of a lookup (`nested`) | 2,281.322 M | 2,275.918 M (-0.24%) | 2,281.322 M |

The loops are bound by the lookup helper (about 550 instructions an iteration for `hit`), so MIR's gain is small, 0.2% to 0.4% where the code around the call is all there is. The miss shape gains 5.7%. The `change` shape is 0.09% above Cranelift's count; the cause was not examined.

The suite is unchanged. No benchmark of the suite records a `TABLE_GET_TABLE` (a census of the optimized IR of all 89 at scale 3 finds none), so the coverage run is the same on both binaries: 108 of 128 compiles through MIR, 3,732 of 4,211 trace IR instructions, the same 20 fallbacks. `benchmarks/mir_backend.py icount`, load average 8.97, CPU 3 (sibling 11 idle):

| | main `8de8e068` | `fb82922e` |
|---|---:|---:|
| suite under MIR | 20,338.5 M | 20,338.4 M (-0.001%) |
| suite under Cranelift | 21,033.8 M | 21,033.9 M (+0.001%) |
| MIR against Cranelift, same binary | -3.31% | -3.31% |

No benchmark moved by more than 0.12% (`string/table_concat`, +0.11%, a binary layout shift; its Cranelift count moved the same way).

The Luanti emerge (`mineclonia-emerge`, 512 blocks, `OXIMOND_JIT_TRACE=1` on the Luanti oximond variant built in the worktree for each commit, one run each, so the counts have run-to-run spread from compile timing). MIR compiled 138 trees before and 182 after:

| Reason | main | stage 8 |
|---|---:|---:|
| `LaneArrayLoad` | 221 | 195 |
| numeric operation that needs a conversion | 162 | 184 |
| `ArraySet` under a strict header | 70 | 71 |
| load of a table, closure or string field | 11 | 23 |
| `ArraySet` without the inline append and hole fill | 11 | 10 |
| function trace | 9 | 9 |
| `ArrayNext` | 6 | 6 |
| array load past the end of the array | 2 | 3 |
| table field fill without the in-place store | 2 | 2 |
| operation | 2 | 2 |
| total handed back | 496 | 505 |

The fallback total does not fall, and the compare is not clean: the gated trees were invisible before and the two runs differ in how many trees they compile (timing). What the gate removal shows is where the lookup trees stop now. The two loops of the spec's shape (structures.lua:855 and :1327, the `for z` loops around `starts[hash].starts`) log `numeric operation that needs a conversion` at IR instruction 15 and 20: a `Numeric` operation (`floor`, `ceil` or `abs`) on a source that is not a float, which is another unit's. They were silent before. So the fill is in MIR and tested on repros of the shape, but those two emerge loops stay with Cranelift until that reason is lowered, and the emerge gain from this unit is zero for them. Whether any other tree of the emerge went to MIR through the clause was not counted: the log does not say which trees hold the call.

Emerge instructions (`bench --variant oximond --workload mineclonia-emerge`, `instructions:u` in G, interleaved, through `scripts/heavy.sh -x`, load average 5.9 to 8.1 at the start of the passes):

| Run | main, MIR | stage 8, MIR | stage 8, Cranelift |
|---|---:|---:|---:|
| 1 | 260.27 | 269.70 | crashed (see below) |
| 2 | 257.21 | 261.28 | 277.15 |
| 3 | 257.39 | 265.01 | 252.44 |
| 4 | 251.04 | 261.97 | 283.77 |
| 5 | 253.24 | 277.05 | 270.63 |
| 6 | 280.64 | 264.47 | 249.00 |
| 7 | | | 257.48 |
| mean (spread) | 260.0 (11.4%) | 266.6 (5.9%) | 265.1 (13.1%) |

Main's build under Cranelift, four runs: 290.86, 253.17, 260.80, 275.82 (mean 270.2, spread 13.9%). The runs of one build spread by 6% to 14%, and the ranges of main's MIR (251.0 to 280.6) and stage 8's MIR (261.3 to 277.1) overlap, so the 2.5% between their means is inside the spread and is not credited or blamed. Stage 8's MIR mean is level with the Cranelift mean of the same build (265.1). The two loops that hold the lookup in the emerge are still handed back (above), and the compile counts of the two traced runs differ by 44 trees, so a difference between the builds is not expected from this unit on this workload.

One Cranelift run of the stage 8 build (`OXIMOND_JIT_BACKEND=cranelift`, run 1) died with SIGSEGV 17 seconds in, in the main thread during `on_mods_loaded`, in Cranelift code that loads from a lane row pointer (`mov 0x8(%rsi),%rax` with `rsi` a non-null pointer out of the rows table, a stale row) while the compile thread was optimizing another tree. Twelve more Cranelift emerges of the two builds and twenty Cranelift startups of the stage 8 build did not repeat it. The Cranelift output is unchanged by this unit, so it is a Cranelift-side race or stale row that is not MIR-specific; the core file showed no more than that and was deleted.

Difftest at `fb82922e` (`run-jit --profiles all --variants tracing,tracing-mir,tracing-mir-profile`, `--reduce`): the 300-seed release slice, 1,800 cases, 0 mismatches (MIR compiled 5,003 trees and handed back 2,877 under `tracing-mir`). The 100-seed debug slice, 600 cases: 0 mismatches besides the three known timeouts (lua53 seed 15, luajit21 seed 15 and seed 32), which pass alone with `--timeout 120`.

### Lane array loads, array stores and the array step (stage 8, arrays)

Stage 8 took the array operations off the list of reasons that keep a tree with Cranelift. The first part (`9a9d1db7` to `e7a92f7a`) lowered the lane array views and the stores into an array part; this unit reviewed it, lowered the step along an array part (`ArrayNext`) and the native step of a boxed `next` loop, and measured all of it.

**Lane array loads and stores** (`lane_array_words`, `lane_array_access` in `lower.rs`). A `LaneArrayLoad` or `LaneArrayStore` reads the array of a table met by lane through the table's row: the word `field` of the row of the lane holds the table's address, and the array pointer and length come from the table (`lane_array_words` of `emit.rs`). Cranelift branches on a zero address; here the guard on the address leaves through the access's own exit, which is the same exit. The words are kept per lane for the straight-line code after and dropped where the words of the bindings are, and also where a helper only takes words (`forget_words`), which is stricter than Cranelift's cache, so a store a helper made can only cost a reload. A load checks the slot's tag and reads the payload, a table element asks the VM for its lane (`element_ref`), a store overwrites the payload of a slot whose tag checked out. The optimizer makes lane stores for numbers and booleans only (the verifier requires it), so the reject for a nil, table or string value is unreachable and stays as a guard against a later optimizer.

**Sets into an array part** (`lower/sets.rs`). A table whose register has a header binding takes an `ArraySet` in place, under the conditions of `set_in_place` and `set_in_holes` of `emit.rs`: the append at the end of a dense list (the array grows through `native_table_reserve` when it is full), the pop, the overwrite of a slot that holds no reference, the string or table stored through the VM's store entry (which owns the count and the write barrier, so there is no barrier code here), and the boolean a few slots past the end with the nil holes between. A strict header leaves through the set's guard when nothing was stored; otherwise the general path is a skip that calls the VM. The cases are exclusive, so each skip joins the answer of the one before it (`end_if(open, &[(solid, dense_done)])`) instead of an `Or` of two words, and a slot is tested against the string tag by the skip's own compare. That removed instructions from every set: `table/array_write` and `algorithm/sieve` count about a tenth fewer (see below). The nil read past the end (`array_nil_read`) loads the header's flags to tell a solid table from one with a hash part.

**Sets into an array part, the short way** (2026-11-14, on `fbca0db9`). The chain of skips above tested the flags first and joined a word after every case, so a store over a slot that already held a value of its type ran about 55 instructions. `array_set` is now a tree on where the key lies and what its slot holds:

- A key inside the array reads the table's address, its length and array pointer, and the slot's tag. A slot of the value's type takes the payload, whatever the table is, as `ArrayStore` does: the tag stays, so the border and every GC edge stay. The flags are loaded only when the tag differs (a hole to fill, a scalar of another type), and the array binding's words are not loaded at all.
- A key outside it loads the flags and tries the growth with holes (booleans in a holey table), then the append (a dense list).
- Each case that can refuse ends in its own `rest_when`: the guard under a strict header, otherwise a skip around a `CallIf` of the VM's store. The cases that cannot refuse join nothing, so a skip they jump over costs a compare and a branch.
- A float key under a lax header is not tested for being whole. The key less its truncation is a float that is zero for a whole key alone, and its bits are or-ed into the slot's index: they are at least 2^61 for a key of 1 or more, carry the sign for a negative key, and a key between 0 and 1 has the index of slot 0, all ones, already. An index like that is neither inside the array nor near its end. The cases compare the index wherever the key may not be whole, and read the key only after the index passed. A strict header keeps a guard, which is two instructions.
- The growth with holes tests its conditions with skips, writes the first hole in line and calls `fill_nil_slots` through a `CallIf` for more than one. A plain `Call` there had every value of the loop stored at its definition. It no longer writes the array pointer to the bindings, which a growth inside the capacity does not move.

One difference from Cranelift follows: under a strict header MIR overwrites a slot of the value's type in a table whose flags are zero, or in the slot after the border, where Cranelift leaves through the guard. The overwrite moves no storage. `holds_type_of` in `mir/tests/arrays.rs` names the case, and the differential tests compare it with Cranelift under a lax header.

Instructions, tracing, scale 3, CPU 4, load average 12.6: `algorithm/sieve` 126.3 M to 75.7 M, `table/array_write` 144.3 M to 94.1 M, `stdlib/sort` 91.8 M to 88.4 M, `string/table_concat` 178.9 M to 172.7 M, `opt/retained_slots` 1896.8 M to 1863.2 M. Suite total 15.816 G to 15.672 G, and no benchmark is more than 1% worse. The overwrite of a slot of its own type now runs 26 instructions an iteration in the sieve's inner loop with its float key (the loop's own compare, add and poll are 5 of them), the fill of a hole 38, the growth by one hole about 50. What is left needs the loop: the table's address is loaded from the inputs every iteration (1), and a float key is truncated and checked every iteration (5) where a range proof of the induction would give the integer.

**The array step** (`lower/steps.rs`, `array_next`). `ArrayNext` gives the number of the first slot with a value after the slot the control key names. The guard leaves when the key is not a whole number inside the array naming a slot that holds a value, and when no value follows. MIR has no loop inside a region, so the slot after the key is read in line (it holds a value in nearly every step) and the scan over a run of holes is a call of `scan_value_slots` made by a skip when that slot is a hole. The slot found stays as a slot fact for the number the step gave, so the `ArrayLoad` that follows reads it with no bounds check (`next_slot` of `emit.rs`). A loop whose back edge passes the step's result for the key, and whose sides link back with such a result, takes `trusted_array_key` from `emit.rs` (made visible to both backends): the entry checks the key it starts with and declines the run with `-input - 1` as Cranelift's does, and the steps take the key as it is.

**The native step of a boxed `next` loop** (`call_helper_stepping`). A `pairs` loop over a table with a hash part keeps its control value in a lane and calls a `next` helper for each step; where the trace says so (`BoxedSteps`) Cranelift steps along the array part and along the snapshot of the hash part in line before the call. Until this unit `inline_in_cranelift` refused every such call of a tree with `boxed_steps`, whether Cranelift would step natively or not. `box_step` now has Cranelift's own condition (`box_step_of`): two arguments, a key of the type the array part's keys have, a value that is a number or a boolean, and an array binding or a snapshot that no operation of the tree can move. A call that fails it, for example the `pairs` over a table of tables of `gwp_compute_fixed_ground_height`, is the helper's call in both backends. A call that meets it is lowered as Cranelift's code is: each attempt is a chain of skips, one for each condition Cranelift branches to its next attempt on, and the stores of a taken step come after every check, so a step that gives up changes nothing. Two details differ from Cranelift because the skip has no block parameters. The helper writes its results to the words of the control block (`helper_out`), so a taken step writes the same words, and the results are loaded once after both: the attempts join nothing, the helper's call is a skip on a word that says whether a step was taken, and the guard on its status is inside that skip (its exit is made before the skip opens). And the entry right after the position (the next slot in the array part, the next entry of the snapshot) is committed first, with the scan over deleted entries behind it in `scan_cursor_slots`, so a step that meets no hole pays no join and no reload.

**Shared with the Cranelift lowering.** `header_binding_of` (the header binding of a table's register) and `trusted_array_key` are free functions of `emit.rs` that both lowerings call. The Cranelift output does not change (the CLIF count tests pass).

**Tests.**

- `mir/tests/arrays_tests.rs`: `array_steps_match_cranelift_for_every_key_and_pattern_of_holes` compiles an `ArrayNext` (with and without the `ArrayLoad` of its key) with both backends, at both tiers, for 14 integer and 14 float keys, integer and float step results, nine patterns of holes (a run of 38 nil slots among them) and the start of a traversal, and compares the outcome, the control block and the exit area. A mutation of the final guard (`Ult` to `Ule`) fails it.
- `crates/oximond-cli/tests/mir_backend.rs`, block "Stage 8, arrays": `array_steps_run_through_mir_as_they_run_through_cranelift` (four snippets that must compile through MIR with no `ArrayNext` fallback, among them the autogroup digtime loop and the list of tables of the pathfinding code over lists of different lengths, which reuse lanes and rows), and `a_native_next_step_gives_up_where_cranelift_gives_up` (eight programs under Lua 5.1 and 5.4 whose exits, side exits and iterations under MIR equal Cranelift's and whose output equals the interpreter's: a list with holes and both variables read, the key alone, integer keys of the hash part, a string key among numbers, a value of another type now and then, a slot cleared during the traversal, nested loops, float keys).
- `crates/oximond/tests/trace_lane_arrays.rs`: four tests in six profiles (lists of every length and pattern of holes, a list of tables walked with `pairs`, `next` over mixed values, a traversal started at any key including cleared slots, one past the end and an invalid key). `a_table_the_next_step_gives_reads_its_own_array` (the stale-row regression) passes and its tree is compiled by MIR with no fallback.

#### Measured

Whole-process `instructions:u`, pinned to CPU 3 (its sibling 11 idle), `env -i` with the same variables under both backends (`PAD` equalizes the lengths), binaries at paths of equal length (`bin-old/oximond` built from `e7a92f7a`, `bin-new/oximond` from the code of `d07c8839`), through `scripts/heavy.sh -x`. Load average at the start of the repro runs 16.9 (the gate held every slot; the load is other sessions' ungated work). Repros of one loop shape, 40000 passes over one table (`next_array` 100 numbers, `next_holes` 100 slots with a nil at every fourth, `next_hash` 60 integer keys of the hash part, `next_mixed` 30 numbers and two string keys, `next_tables` 10 tables with `box[5] + box[1]` read through the lane's row). Before, MIR handed the first four back (`ArrayNext`, `native next step without the inline form`), so its column is Cranelift's code; `next_tables` was MIR's before too (the value is a table, so the call is the helper's):

| Repro | Cranelift (M) | MIR before (M) | MIR after (M) | After vs Cranelift |
|---|---:|---:|---:|---:|
| next_array (`ArrayNext`, key unread) | 317.07 | 317.04 | 327.89 | +3.4% |
| next_holes (native step, array part, a hole in four) | 253.60 | 253.64 | 281.13 | +10.9% |
| next_hash (native step, hash snapshot) | 186.75 | 186.79 | 197.12 | +5.6% |
| next_mixed (native step, string keys fall to the helper) | 800.99 | 801.00 | 807.42 | +0.8% |
| next_tables (pairs over tables, lane reads) | 559.52 | 558.72 | 559.12 | -0.1% |

MIR stays above Cranelift on the shapes it now takes. `next_array` pays 2.7 instructions an iteration: the results go through the control block's words and back, and the step flag costs a store, a load and a compare. `next_hash` pays 4.3. `next_holes` pays 9 on average because the scan over a hole is a call of `scan_value_slots` (Cranelift loops in line), and one step in four meets one; a second slot looked at in line before the call would take most of that. The first version of the native step joined three values through six skips and cost 18 to 26 more instructions an iteration (`next_holes` +31%, `next_hash` +22%); results through the control block, the entry right after the position committed first and the key taken as it is brought it to the table.

The suite, `benchmarks/mir_backend.py icount`, scale 3, the two binaries above, load average at the start 7.5 (the first of the four runs; the gate was exclusive for each), pinned to CPU 3:

| | before | after |
|---|---:|---:|
| suite under MIR | 20,392.6 M | 20,372.9 M (-0.10%) |
| suite under Cranelift, same binary | 21,036.5 M | 21,047.9 M (+0.05%) |
| MIR against Cranelift, same binary | -3.06% | -3.21% |

No benchmark under MIR is more than 1.1% above its count before: `string/utf8_pack` +1.09% moves +1.06% under Cranelift too, so it is the layout of the binary, as are `allocation/gc_cycles` (+0.83%, Cranelift +0.82%), `opt/retained_slots` (+0.56%, +0.56%) and `string/format` (-2.47%, -2.46%). The benchmarks this unit moves:

| Benchmark | MIR before (M) | MIR after (M) | Change | Cranelift, same binary (M) | After vs Cranelift |
|---|---:|---:|---:|---:|---:|
| table/array_write | 161.47 | 144.84 | -10.30% | 129.61 | +11.75% |
| algorithm/sieve | 149.59 | 135.32 | -9.54% | 111.78 | +21.06% |
| table/hash_iteration | 52.73 | 53.14 | +0.76% | 52.70 | +0.84% |

The first two are the set refactor above (a join value instead of an `Or`, a compare in the skip instead of a select). They are still above Cranelift; the cause was not examined. `table/hash_iteration` is the one suite benchmark whose tree the native step moves to MIR (its two compiles were refused before): the loop is bound by the step, MIR is 0.84% above Cranelift's code, inside the 1% that was set as a goal.

Suite coverage, `benchmarks/mir_backend.py coverage` at `e7a92f7a` and at the code of this unit: MIR compiled 115 of 128 trees (3876 of 4211 trace IR instructions) before and 117 (3912) after. The refusals left are `counted trace` (2 compiles, 127 IR instructions), `table field fill without the in-place store` (2, 72), `function trace` (6, 58) and `register that is a float or an integer at run time` (1, 42): none of them is an array reason.

The Luanti emerge, `mineclonia-emerge`, 512 blocks, `OXIMOND_JIT_TRACE=1`, one run of each build (the compile counts vary with timing). The oximond variant of Luanti was built in this worktree from `e7a92f7a` and from the code of `d07c8839`:

| Reason | `fb82922e` (rowfill) | `e7a92f7a` | after |
|---|---:|---:|---:|
| `LaneArrayLoad` | 195 | 0 | 0 |
| numeric operation that needs a conversion | 184 | 311 | 286 |
| `ArraySet` under a strict header | 71 | 0 | 0 |
| `ArraySet` without the inline append and hole fill | 10 | 0 | 0 |
| array load past the end of the array | 3 | 0 | 0 |
| `ArrayNext` | 6 | 7 | 0 |
| function trace | 9 | 7 | 7 |
| table field fill without the in-place store | 2 | 2 | 2 |
| compiled by MIR | 182 | 284 | 304 |
| handed back | 505 | 327 | 295 |

The column of `fb82922e` is the table of "The lookup row fill" and the other two are this unit's runs. The numeric reason is another unit's. The seven `ArrayNext` trees include the autogroup digtime loop (`_mcl_autogroup/init.lua:223`) and the `next` loop of `serialize.lua:153`, the two the log names.

Emerge instructions (`bench --variant oximond --workload mineclonia-emerge`, `instructions:u` in G, interleaved, `scripts/heavy.sh -x`, load average 5 to 12 at the start of the runs), the same builds, MIR (no variable) and `OXIMOND_JIT_BACKEND=cranelift`:

| Run | `e7a92f7a`, MIR | `e7a92f7a`, Cranelift | after, MIR | after, Cranelift |
|---|---:|---:|---:|---:|
| 1 | 281.19 | 260.49 | 257.47 | 257.23 |
| 2 | 246.95 | 265.00 | 274.05 | 252.27 |
| 3 | 250.23 | 263.70 | 259.34 | 261.18 |
| mean (spread) | 259.5 (13.2%) | 263.1 (1.7%) | 263.6 (6.3%) | 256.9 (3.5%) |

The spread of one build's MIR runs (6.3% and 13.2%) is larger than the difference of the means (1.6% above Cranelift after, 1.4% below before), so no change is credited, and the goal of MIR at or below Cranelift is neither met nor missed by this measurement. `mineclonia-startup` ran 10 times under MIR on the same build: 10 exits with status 0, no `ERROR[` line, 78.06 to 78.63 G instructions. Before the fix of the stale row about one run in fifteen crashed.

Random programs, as a check beyond the tests: 1,350 generated programs (lists with holes, integer keys of the hash part, string keys, numbers, booleans, strings and tables as values, the key and the value read in every combination, a slot cleared or a key added between runs, Lua 5.1, 5.4 and LuaJIT) print the same under the interpreter, MIR and Cranelift.

Difftest (`oximond-difftest run-jit --profiles all --variants tracing,tracing-mir,tracing-mir-profile --reduce`, release binaries built from the code of `d07c8839`): seeds 0 to 299, 1,800 cases, no mismatch (under `tracing-mir` MIR compiled 5,756 trees and handed back 2,124; the slice at `fb82922e` had 5,003 and 2,877). The 100-seed debug slice, 600 cases, timed out in four: lua53 seed 15, luajit21 seed 15 and 32 (the three known ones) and lua51 seed 51 (`tracing-mir-profile`, a flaky timeout in a run that shared the machine). All four pass alone with `--timeout 120`.

Final checks on `dd70aab4`, which differs from the code the binaries above were built from only in a test, a comment and this file: `scripts/heavy.sh -m 24G cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast` reports 4,798 passed and 0 failed with `OXIMOND_JIT_BACKEND` unset and 4,798 passed and 0 failed with `OXIMOND_JIT_BACKEND=cranelift`. `scripts/heavy.sh -m 24G cargo clippy --locked --workspace --all-targets` exits 0, with no warning in a line this unit added.

### Direct links to the loop header (stage 8, links)

MIR's links used to go to the root's `Join`, which sits where Cranelift's outer header does. For a root whose loop is range-versioned that means the checks run on every link; for a loop side that sides rejoin, the preheader ran on every link although Cranelift runs it once. This unit removes both, and the allocator costs of the rejoin path that `opt/closure_protos` showed.

What changed:

- `oximond-mir`: a region may have several `Join`s. `Op::Rejoin` names one by its position in the region (`Func::rejoin_nth`, the `e` field of the instruction; `rejoin_to` is the first). The allocator labels a join by its index among the function's joins (`joins`, `join_start`), the evaluator, the validator and the CLIF translation of the tests know the form, and the random programs build a second join under the first for half of the joined programs and have some sides rejoin it (`second_rejoins` in `tests/random.rs`, `a_root_with_two_joins_reaches_the_one_each_side_names` in `tests/rejoin.rs`).
- Shared predicates. `carried_for_links`, `side_origins`, `fast_link` and `slow_link` moved from `emit.rs` to `jit_ir/trace/range.rs` as functions of the IR; Cranelift calls them and its output did not change. MIR uses the first three.
- A versioned root that sides rejoin has a second `Join` (`join_fast`) right below the range checks and above the fast loop. It takes the parameters the fast loop carries (`Scratch::changing`: the ones the loop-back changes plus `carried_for_links`, in Cranelift's order), the fuel and the integer copy of each float induction. What the checks and the code above them made and the loop reads (steps, shifts, entry values, pointers, binding words, rows) goes through frame cells again, as at the first join, so a link that jumps here runs no check, no entry load and no preheader.
- A side whose link `fast_link` proves (it steps every induction as the root does, passes the root's bound, and gives back every other parameter unchanged) is lowered twice. The guards of the fast loop's body enter a copy whose link is a `Rejoin` of the second join (`link_fast`); the guards of the guarded loop and of the preheader enter the other copy, which links to the first join as before (`side_entered`). The two copies have snapshots of their own, as the guarded loop has. The integer copy of a float induction for the link is the conversion of the stepped parameter, exact because the side's bound guard and the root's checks keep it inside the array. The regions are numbered by `Scratch::region_of` and `fast_region`; a copy comes right after its side so a region only enters later ones.
- A loop side that passes no parameter on unchanged puts its `Join` below its preheader (`join_root` serves it too), so a rejoin runs the preheader zero times. The fallback `loop side whose preheader calls into the VM under a side that rejoins its loop` now applies only to a loop side that passes a parameter on unchanged.
- Allocator, for the `opt/closure_protos` rejoin path: a header parameter that starts as a join parameter is hinted to the join's register, so the join's move into the header is empty. A first version also gave a side parameter that every guard and goto fills from the same homed cell that cell as its slot, so entering the side copied nothing for it. It made `closure_protos` slower (85.4 M instructions in the slope of two scales against 82.5 M without it: a parameter with a preset slot is stored at the side's entry and evicted more readily), so it was taken back.

- The guarded loop continues from the state of the checks. `guard_entry` (called by `range_checks`) makes the snapshot a failing check enters region 1 with: the fuel and every value `visit_held` lists (the one list of what the lowering holds, which `keep_in_cells` and `load_from_cells` use too), and saves the tables (`Saved`). `guarded_loop` starts region 1 with those as its entry values, makes the constants again, restores the tables with the new names and runs the loop, so a failed check runs neither the entry loads nor the preheader again. The moves happen in the trampoline of the failing guard, off the straight path.
- A side whose link `slow_link` accepts (every parameter the guarded loop does not carry comes back as the root had it) gets a copy for the guards of the guarded loop, which rejoins a `Join` of region 1 (`join_slow`, `link_slow`) placed right above the guarded loop. Cranelift's `link_target` jumps to the slow header there. Copies are numbered `fast_region` and `slow_region` after the side.

Not done: the first join (before the checks) stays even when every side that rejoins has both copies, so its entry-time cell stores are still paid; dropping it needs the own copies of those sides, which only a guard of the preheader enters, to go too. A root whose preheader calls into the VM stays declined. The float reloads after each join of `closure_protos` are unchanged.

Tests: `link_tests.rs` (declared in `mir/mod.rs`): `a_link_that_steps_like_the_root_goes_to_the_fast_loop_like_cranelift` (integer and float loops, a step of 1 and 2 and a downward loop, starts inside, at the edges and outside the arrays so the checks pass and fail, arrays with holes so the side is taken from the fast and the guarded loop, six fuels, both tiers; it counts the links lowered straight to the loop through `FAST_LINKS`), `a_link_without_the_bound_guard_runs_the_checks_again_like_cranelift`, and `a_loop_side_runs_its_hoisted_call_once_however_often_it_is_rejoined` (a loop side with a host call in its preheader and a nested side that rejoins it, host calls counted against Cranelift's; it was a fallback before). `links_that_step_like_the_loop_print_what_the_other_backends_print` in `mir_backend.rs`.

Measured (`benchmarks/mir_backend.py icount`, one pass under `scripts/heavy.sh -x`, binaries at `.../b/oximond` (before: e7a92f7a) and `.../a/oximond` (after: 5134571e), equal-length `env -i` environments, pinned to CPUs 6 and 7; the load average was 13 to 19 because other sessions run ungated work, so only instruction counts are used). Slope of two scales (0.5 and 1.2, `OXIMOND_JIT_SYNC=1`, compile on the VM thread), instructions for 0.7 scale units:

| Benchmark | MIR before (M) | MIR after (M) | Cranelift (M) |
|---|---:|---:|---:|
| opt/closure_protos | 97.20 | 82.50 | 76.78 |
| opt/range_dense | 103.11 | 103.11 | 103.11 |
| opt/range_fail | 110.73 | 110.73 | 110.73 |
| opt/range_float_index | 103.11 | 103.11 | 103.11 |
| opt/range_offset | 109.82 | 109.84 | 109.77 |
| table/array_read | 16.83 | 16.83 | 17.95 |

`closure_protos` is the rejoin path (seven iterations in eight link): its gap to Cranelift went from +26.6% to +7.4% in the steady state. The range benchmarks have no rejoining side in their tree (their sides compile apart), so the direct link does not show in them; the focused tests are what cover it. Whole-process suite total (89 benchmarks): MIR 20,392.6 M before, 20,395.3 M after (+0.01%); Cranelift 21,036.5 M in the before binary and 21,073.2 M in the after binary (the binary moves counts by up to 1.4%). `opt/closure_protos` whole process 398.3 M to 371.1 M (-6.8%, Cranelift 436.9 M). Benchmarks that moved by more than 1% (whole process, small programs where the compile of a larger tree counts): `mineclonia/captured_call` +2.5%, `mineclonia/ull_kernel` +1.3%, `string/utf8_pack` +1.3%, `mineclonia/module_calls` +1.1%, `algorithm/sieve` -1.2%, `opt/range_fail` -0.9%, `opt/retained_slots` +0.9% (Cranelift +0.9% in its own binary).

Difftest, release binary, through the gate: `run-jit --seeds 0..300 --profiles all --variants tracing,tracing-mir,tracing-mir-profile` on revision 5134571e, 1800 cases, 0 mismatches in every profile.

Suite, whole process (`icount`, 89 benchmarks, the same two binaries): MIR 20,392.6 M before and 20,395.3 M after (+0.01%). The slope of `closure_protos` above is the one change that shows: its whole-process count went from 398.3 M to 371.1 M (-6.8%), Cranelift's from 436.5 M to 436.9 M. Eight benchmarks moved by more than +0.6% under MIR, and Cranelift in the same two binaries moved by about the same (`mineclonia/captured_call` +2.53% MIR and +1.77% Cranelift, `ull_kernel` +1.26% and +0.80%, `utf8_pack` +1.25% and +1.19%, `retained_slots` +0.88% and +0.88%, `short_inner` +0.93% and +0.08%), so the binary moved the count, not the backend; `short_inner` and `captured_call` are the two where MIR moved more (+0.85 and +0.76 points) and both stay under 1% of the difference. Load average at the start of the two runs: 11.8 9.7 12.6.

Luanti (`integrations/luanti`, the oximond variant built from the code of `c0af2c7c`, `mineclonia-emerge`, three reps each, interleaved MIR (backend unset) and `OXIMOND_JIT_BACKEND=cranelift`, `scripts/heavy.sh -x`, CPUs 4 to 7, instructions:u of the server in G):

| Rep | MIR | Cranelift |
|---|---:|---:|
| 1 | 251.4 | 273.9 |
| 2 | 281.8 | 272.7 |
| 3 | 266.0 | 253.2 |
| mean | 266.4 | 266.6 |

The spread of one build is 30 G (11%), more than any difference between the two, so the emerge neither credits nor blames this unit; the stage 8 target (MIR at or below Cranelift) holds in the mean (-0.1%) and in one of three pairs. `mineclonia-startup`, ten runs under MIR: all exit 0, no `ERROR[` line, 78.2 to 78.8 G (the stale-row crash that came about one run in fifteen did not show).

Emerge fallbacks with `OXIMOND_JIT_TRACE` (`--jit-trace`, one run): 294 trees compiled through MIR (15 of them range-versioned), 367 handed back: 350 `numeric operation that needs a conversion`, 8 `function trace`, 7 `ArrayNext`, 2 `table field fill without the in-place store`. No reason of this unit's scope appears: no loop side was handed back for its preheader and no versioned root for a link. The emerge before this unit was not rebuilt (a second Luanti build of `e7a92f7a`), so there is no before count of the same run; the conversion fallback belongs to the numbers unit, the lane arrays to the arrays unit.

Difftest: the 100-seed debug slice (`--seeds 0..100 --profiles all --variants tracing,tracing-mir,tracing-mir-profile --reduce`, 600 cases, the debug binary): no output mismatch; five timeouts at the 20 s default under a load of 13 to 30 (luajit21 seed 15, lua53 seed 15, luajit21 seed 32, the three known ones, and lua51 seed 44 and 51, both only under `tracing-mir-profile`). Rerun alone with `--timeout 120`, seeds 15 (lua53, luajit21), 32 (luajit21) and 51 (lua51) pass; lua51 seed 44 and lua53 seed 15 pass without a mismatch but the harness reports them as a vacuous single-seed run (no native entry in the one case), so they show nothing either way. No mismatch went to the known-bug corpus; nothing was MIR-specific.

Final checks, on commit `d62ca62a` (the last code; the commits after it change this file only): through `scripts/heavy.sh -m 24G`, `cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast` gave 4797 passed and 0 failed with `OXIMOND_JIT_BACKEND` unset and 4797 passed and 0 failed with `OXIMOND_JIT_BACKEND=cranelift`; `cargo clippy --locked --workspace --all-targets` exits 0 with no warning in a line this unit added. The same three runs on the commit before (`18532768`) gave the same counts; the one change since is an `allow` attribute.

**After the merge with the numbers unit.** The numbers unit lowered the same native step of a boxed `next` loop on its own branch. The merge of stage 8 keeps its form (`box_step_of` and `call_helper_stepping` in `lower/steps.rs`, described in "Stage 8, numbers"), which looks at the next slot or entry only and leaves a hole to the helper, and drops the form above with its scans and the step flag. The numbers form measured `table/hash_iteration` at 41.65 M against 53.14 M for this one and 52.72 M under Cranelift. `ArrayNext`, `check_trusted_key` and `scan_value_slots` stay as described. The repro counts of the table below for `next_holes`, `next_hash` and `next_mixed` are of the dropped form and were not measured again.

### Stage 8, numbers

Numeric conversions, numbers of either subtype entering a side, the fill of made tables and shifts by a run-time distance.

The emerge handed back 400 trees on `e7a92f7a` (286 compiled), 385 of them for `numeric operation that needs a conversion`. The recorder emits `Numeric` with `FloatToInteger` (a `math.floor` that gives an integer, an index computed from a float), `TrickBits` (every `bit.*` call of the LuaJIT profile, which Mineclonia's `struct_hash` uses) and `Abs` of an integer; MIR lowered only `Floor`, `Ceil` and `Abs` of a float. This unit lowers the three, the number of run-time subtype that a guard passes to a side, the in-place fill of a table the trace makes, and the shift by a distance known at run time, which the first four uncovered as the next reason (94 of the 111 trees that were left).

**Conversions.** `lower.rs` in the `Numeric` arm:

- `FloatToInteger` is `F2I` and two selects, `fcvt_to_sint_sat`'s saturation: `F2I` gives the minimum for NaN and for what is out of range, so 2^63 and up select the maximum and NaN selects zero. The recorder guards that the float is integral and in range first, so the fixes are never observable in a trace, but `numeric_operations_match_the_reference` (`trace/tests.rs`) runs the reference evaluator over NaN, the infinities and 1e308 on whichever backend is the default, and an earlier version without the selects failed it.
- `TrickBits` is `FAdd` of `1.5 * 2^52`, `Bits` and an `And` with `0xffffffff`.
- `Abs` of an integer is a `Sub` from zero and a select; the minimum wraps to itself, as `iabs` does.
- Any other pair (a `Floor` of an integer) is refused with `numeric operation on a type it is not defined for`; the recorder never emits one.

**Numbers of either subtype entering a side.** A snapshot can say that a register of the root frame holds a float or an integer at run time (`Snapshot::numbers`: the float, and a flag that is 1 for an integer). A side recorded for one subtype takes the float as it is, or its truncation, and runs only if the flag is the one it was recorded for; Cranelift's `side_inputs_at` folds that into a `subtypes` word and `leave` branches to the guard's own exit when it is false. MIR does the same in two places:

- A guard that has such a side (`guard_target`, `guard_cc`): the side's values are made once, with an empty place for each number (`Scratch::convs`), and the guard becomes `If (not cc)` { `Guard flag != 0` to the exit; `Guard 0 == 1` to the side }. The second guard never continues, so it is the entry into the side, with a snapshot made inside the skip (`guard_by_subtype`). The conversions (`Xor` of the flag with 1, `F2I`, the `And` of several flags) are made in that skip too. The first version made them before the guard, in the loop, since a value a snapshot passes to a side is not deferred to the exit stub (only float operations are, see "Exit-only values"); the skip is the only place they cost nothing on the hot path.
- A trace that ends in front of a loop side (`ends_in_exit`): one `Guard` on the flag to the terminal exit before the `Goto`. It runs once per link, so the conversions stay in line.

**Made tables.** A table a `TABLE_NEW` call makes, which a `TABLE_INIT` of the member fills under a root string key or as a short array, gets its value slots from the helper, and each `TABLE_INIT` writes its slot where it is (`lower/made.rs`, the shape of `table_init_in_place`): `ready` is 1 when the helper gave slots; the store is a skip on `ready`, with the tag, the payload (`payload_access`) and the `initialized` count of the control block that the run charges to the collector; the helper call is a second skip on `ready == 0`, a `CallIf` with the guard on its status as `store_by_helpers` has. `TABLE_NEW` is called with the prefill flag when the array part is filled, and the second result word is zeroed before and read after the call. Which tables and which slots is `trace/made_table.rs`, shared with `emit.rs`, which now calls it (`fills_in_place`, `prefills_array`, `made_table_shape`, `init_field`, `init_by_helper_moves`); Cranelift's output is unchanged. Any other helper that writes the heap and names the table, and a fill the helper answers with a key the shape lacks or a nil, mark the table moved, so later fills of it go to the helper.

The recorder never stores a table into a table it makes (`VirtualEscape`), and Cranelift's copy of the lane's value with the collector's barrier check has no test in the repository either. MIR leaves it to Cranelift with `table field fill of a table value` (zero trees on the suite and the emerge).

**Shifts.** `integer_shift` of `emit.rs` is two machine shifts and selects. `oximond-mir` has the shift by a register now: `Shl`, `Shr` and `Sar` with a count that is not a constant are the three-address `shlx`, `shrx` and `sarx` of BMI2, the count reduced modulo 64 as the evaluator has it (`X64::bmi2`, `Target::register_shifts`; the allocator takes the three-address form in `do_alu`). The random programs (`generate_with`) shift by values on a target that has them, and `random_programs_match_the_evaluator_avx` checks that at least 50 do. A target without BMI2 hands the tree back with `shift by a value on a target without BMI2` (a test with `X64::sse2()`). The lowering in `lower.rs` is Cranelift's: the distance negated for `>>` (saturating at the minimum), both shifts of the absolute distance, a select on the sign, and a select on `-63 <= distance <= 63` for the zero.

Tests. `trace/number_tests.rs`, declared from `mir/tests.rs` (`#[path]`) so that it uses the harness: every conversion over the edge floats and integers (NaN, the infinities, -0.0, 1e308, 2^53, 2^63, the integer minimum), a conversion feeding integer arithmetic, a side that takes `no` and `i` as a float or an integer for the four pairs, a loop side entered from a terminal exit with a constant flag either way, the fill of a made table for eight shapes of fill under both backends (the helper calls and their arguments, the fields counted, the slots' contents), and the shifts for 20 distances and 8 values. `new_table_tests.rs` has two Cranelift-level tests for integer, boolean and array fills. `mir_backend.rs` has `NUMBER_SNIPPETS`, each with the list of profiles it runs in (the rounding trick of `bit` in LuaJIT, `bit32` with float and integral-float operands and counts in Lua 5.2 and 5.3, `math.abs`, `math.floor` and `math.ceil` in all six profiles, the integer minimum, shifts by a distance, a made record and a made list, and the boxed `next` loops), compared with the interpreter under MIR and under Cranelift in each of those profiles, and the old `numbers that are integers or floats at run time` snippet lost its fallback.

**The step of a boxed `next` loop.** A loop over a table that has a hash part keeps its control value in a lane of the run's bank (`HelperId::next_box`), and Cranelift makes the step in line before it calls the helper (`call_helper_stepping`). `lower/steps.rs` does the same, with `box_step_of` of `emit.rs` ported for the conditions and `BoxStep` shared. The attempt is a chain of forward skips, since a region has one loop and a load past the end of the array or the snapshot must not run:

- The array part. The lane holds nil (the start) or a whole number inside the array whose slot holds a value, the run's cursor description has no `pending` key, and the next slot holds a value of the kind the call wants. The key is stored in the lane (an integer, or a float converted back and compared in the profiles whose keys are floats), `in_sync` of the description is cleared, a value the body never reads is copied into the lane after the control lane, and the key and the value are the results.
- The hash part, when the tree moves no hash storage (`step.hash`). The description is in sync, the entry after the position holds a value of the kind the call wants, and its key is an integer when the call wants the key. The position moves to the entry, `pending` names it, and the results are read from the entry.
- The helper, in a skip of its own on "no attempt took the step", with the guard on its status as `call_helper_in_vm` makes it. Its results and the attempts' are joined (`end_if`), so the code after reads one value either way.

The one difference from Cranelift is the scan. Cranelift loops over the holes of the array and the deleted entries of the snapshot; here an attempt looks at the next slot or entry only and leaves a hole to the helper, which skips it. The answer is the same and it costs one call for each hole. A trace with a boxed step whose helper takes a string or a table value (`BoxedValue::Kind(String | Table)`), or whose key is not of the profile's key type, calls the helper as Cranelift does, with the row fills of `call_helper_in_vm`.

The row `native next step without the inline form` is gone. Two things were measured on the release build.

- `table/hash_iteration` (128 string keys and 128 integer keys outside the array part, `pairs` with the key unused, so a tree with a hash attempt and no array attempt). The first version of this lowering took the tree and made it slower than before: 57.76 M instructions at scale 3 against 52.73 M for the tree left to Cranelift (+9.5%), 70 instructions per step against 54 for Cranelift's (the slope of scales 1, 3 and 6). Two things cost it. A tree with no array part still went through a skip on the constant 0 flag of the array attempt, and every check was built as a 0/1 value and then compared with 0 (`mov 0, mov 1, cmp, cmov, cmp, je` where one `cmp` and one `jae` do). The tree now goes straight to the hash attempt, and the checks that are one compare (the room left in the snapshot, the tag of the value) open their skip on that compare (`step_passes`). The benchmark is at 41.65 M (-21.0% against before, -21.0% against Cranelift's 52.72 M); the slope is 12.7 M per scale (49 instructions per step) against Cranelift's 13.9 M (54).
- Two repros (a table of 60 integer keys and 60 string keys, and one of 100 string keys, 30,000 traversals each, a debug binary pinned by `perf stat -e instructions:u`, load average 26), measured with the first version: 863.3 M under MIR against 956.4 M under Cranelift, and 706.1 M against 804.3 M.

Tests. `a_boxed_next_step_matches_cranelift` (`number_tests.rs`) runs nine loops under both backends in every profile and requires the interpreter's results, the same iterations and exits as Cranelift, and no fallback for the step: an array part and a hash part, holes in the array, float, boolean and integer values, a loop that reads only the key, one that never reads the value, `next, t`, a table changed during the traversal, and four tables of different sizes taken in turn (the lanes and the snapshot of the run before are reused). `mir_backend.rs` has the same shapes as `NUMBER_SNIPPETS`.

**What stays with Cranelift here.** A guard whose side takes such a number and whose tree has a side that rejoins the loop (`register that is a float or an integer at run time, under a side that rejoins the loop`, 1 compile and 42 IR instructions on the suite: the second tree of `opt/mod_mixed`). With the reason lifted the tree went to MIR and the benchmark ran at 740.5 M instructions against 675.6 M (+9.6%) and 678.2 M under Cranelift; moving the conversions into the skip changed it to 736.4 M (+9.0%), so the conversions were not the cause. The disassembly of the tree shows the rejoin path of "Gaps" item 8: the loop reads the control pointer, the length of the array and three more entry values from frame cells in each iteration (about eight reloads and three spills in a loop of about 100 instructions), where Cranelift keeps them in registers. That is another unit's work, so the case stays with Cranelift under its own reason; the emerge never reached it (no tree of `e7a92f7a` was refused for it, and none of the 14 left after this unit is). The side that does not rejoin (a side that leaves, or a loop side) is in MIR.

**Not done.**

- Function traces (6 suite compiles, 58 IR; 8 emerge trees of 3 to 4 IR instructions, the activation of a recursive function that holds a call and a return). They are a different ABI: an activation is `(control, left, fuel, parameters...) -> (status, result)` that calls itself for `SelfCall`, finds its part of the exit area at a depth, passes the status of an exit on through the activations around it and returns fuel left as a positive status (`FunctionLowering`, `self_call`, `exit_status` in `emit.rs`). MIR has one function per tree with entry registers `(inputs, control, area)`; it would need several functions in one unit with calls between them (`Op::Call` of an address inside the same block, with the callee-saved check), a status return, and snapshots that are stored by the callee into the activation's part of the area. About the size of a stage of its own.
- Counted traces (2 suite compiles, 127 IR; none on the emerge, they come only from a count hook). The budget is a second loop parameter like the fuel: loaded from the control block at the entry, paid `cost` at each loop-back (the `counted` of the poll snapshot, or of the link's snapshot), stored into the control block by every exit and carried by every side entry, join and rejoin. In MIR that is a second value beside `self.fuel` in the 45 places that handle the fuel (`Header`, `Join`, `Rejoin`, `SideEntry`, the snapshots' control slots), and a second control slot in `SnapVal`. Because the cost per loop-back is a constant per path, a tree with no rejoining side could derive the budget from the fuel instead (`budget = entry - cost * (fuel_at_entry - fuel)`), computed in the exit stub, which would need `Mul` among the coldable operations. Neither was built.
- `ArrayNext` (7 emerge trees of 17 to 51 IR instructions) is the arrays unit's.

**Measurements.** Load averages and CPUs are given with each. `e7a92f7a` is "before"; "after" is `247cfbb5` (the code; the commit after it changes this file only), with a release binary and a Luanti build made from it. The intermediate figures of the first checks, on `b3cc3d0f`, are kept where they differ.

Emerge fallbacks by reason, `OXIMOND_JIT_TRACE=1`, one run of each build (the numbers of a log, not of a timing; the count of compiled trees and the small rows move by one or two between runs, since the emerge compiles on a thread):

| Reason | before (`e7a92f7a`) | `b3cc3d0f` | final (`247cfbb5`) |
|---|---:|---:|---:|
| trees compiled through MIR | 286 | 632 | 709 |
| trees handed back | 400 | 17 | 14 |
| numeric operation that needs a conversion | 385 | 0 | 0 |
| shift by a value | 0 | 0 | 0 |
| function trace | 7 | 8 | 8 (28 IR instructions) |
| ArrayNext | 6 | 7 | 6 (167 IR instructions) |
| table field fill without the in-place store | 2 | 0 | 0 |
| native next step without the inline form | 0 | 2 | 0 |

(A build between the two, with the conversions, the tables and the side check but not the shifts, handed back 111 trees, 94 of them for the shift by a value.)

The emerge, `scripts/heavy.sh -x -m 12G python3 integrations/luanti/luanti.py bench --variant oximond --workload mineclonia-emerge --reps 1`, three reps of each, interleaved (MIR, then `OXIMOND_JIT_BACKEND=cranelift`, three times) on the same Luanti build, `instructions:u` of the server in G. The load average read at the end of the series was 10.4, 8.3, 9.4 (the runs are pinned and the counter does not depend on it):

| rep | MIR | Cranelift | MIR against Cranelift |
|---:|---:|---:|---:|
| 1 | 241.0 | 261.6 | -7.9% |
| 2 | 229.5 | 254.3 | -9.8% |
| 3 | 230.2 | 253.7 | -9.2% |
| mean | 233.6 | 256.5 | -9.0% |

The spread of one build is 5.0% under MIR (229.5 to 241.0) and 3.1% under Cranelift (253.7 to 261.6), and every rep has MIR below Cranelift by far more than either spread, so the difference is credited: the emerge under MIR is at or below the same build under Cranelift, which is the goal,. The first checks on `b3cc3d0f` read the other way (254.3 against 250.2, 15% spread), under a machine with other runs on it; the 15% came from what compiles when, and a tree compiled a second earlier runs many more iterations in native code. Against the same build under MIR before this unit (`e7a92f7a`, 236.0, 258.3 and 253.9 G in the earlier series, mean 249.4) the mean is 6.3% lower, which is inside the spread of that series (9%) and is not credited.

`mineclonia-startup`, ten runs under MIR: ten exits with code 0 and no crash, 76.48 G to 76.79 G instructions.

The suite (`benchmarks/mir_backend.py icount`, scale 3, 89 benchmarks, `env -i`, one binary for both backends, under `scripts/heavy.sh -x`; load average 10.4, 8.3 and 9.4 and CPU 6 (sibling 14, idle) for the after run; the before counts are those of `e7a92f7a` that the first checks used):

| | before | after |
|---|---:|---:|
| suite under MIR | 20,392.5 M | 20,369.2 M (-0.11%) |
| suite under Cranelift | 21,036.4 M | 21,000.7 M (-0.17%) |
| MIR against Cranelift | -3.06% | -3.01% |

No benchmark is more than 1% above its before count under MIR except `string/utf8_pack` (+1.09%, and +1.06% under Cranelift between the same two binaries, so the layout of the binary). The changes beyond 1% are `table/hash_iteration` -21.0% (the boxed step, see above; Cranelift's count is unchanged), `string/format` -2.52% and `allocation/gc_cycles` -1.56% (the same under Cranelift). The first version of the boxed step had `table/hash_iteration` at +9.5% (57.76 M); that was found by the final measurement of this unit and is what the change to `step_passes` removed.

Suite coverage, `benchmarks/mir_backend.py coverage`: before 115 of 128 compiles (3,876 of 4,211 IR instructions) through MIR, after 119 (3,984). The refusals before: counted trace 2 (127 IR), table field fill 2 (72), function trace 6 (58), register of either subtype 1 (42), native next step 2 (36). After: counted trace 2 (127), function trace 6 (58), register of either subtype under a rejoining side 1 (42): 9 compiles and 227 IR instructions.

Kernels (`benchmarks/mir_backend.py kernels`, 16 of them): every count per iteration is unchanged under both backends; none of them has a conversion, a shift by a value, a made table or a boxed step.

Difftest at `247cfbb5` (`run-jit --profiles all --variants tracing,tracing-mir,tracing-mir-profile`, the release binary and a debug binary of the same tree, absolute `--out`). The 300-seed release slice, 1,800 cases: 0 mismatches (under `tracing-mir` MIR compiled 6,081 trees and handed back 1,799). The 100-seed debug slice, 600 cases: 1 mismatch, a `timeout` (luajit21 seed 32, flagged flaky, in the `tracing` variant too), and 2 oracle timeouts (lua53 and luajit21 seed 15); the three seeds pass alone with `--timeout 120` (1 of 1, 0 mismatches; lua53 seed 15 alone ends with the harness's "vacuous run" status, since its non-profile variants compile no trace, and no mismatch). The corpus entry `pairs-over-two-tables-lua54` is the pre-existing wrong sum that is not MIR specific (Cranelift gives the same sums); nothing else was added.

Tests at `247cfbb5` (`scripts/heavy.sh -m 24G cargo test -p oximond -p oximond-cli -p oximond-mir -p oximond-difftest --no-fail-fast`): 4,806 passed, 0 failed, 6 ignored with `OXIMOND_JIT_BACKEND` unset, and the same with `OXIMOND_JIT_BACKEND=cranelift`. `cargo clippy --locked --workspace --all-targets` exits 0; its warnings are the ones the branch had, none in a line this unit added (the `type_complexity` of the first version of `Scratch::pending` was removed by naming the entries, `Conv` and `PendingSide`).

**A crash of the unit tests that the checks found.** The first full run on the commit that added the boxed step ended with `SIGSEGV` (or an alignment abort) in the library tests, with either backend, and passed when `number_tests` was skipped. AddressSanitizer (`RUSTFLAGS=-Zsanitizer=address`, nightly) named it: `helper_table_new`, the library's helper, called from a run of `the_fields_of_a_made_table_are_filled_in_place_like_cranelift`, which reads past the test's own buffer. The cause is in the tests, not in the lowering. `install_helper` writes a process-wide address; that test installs stand-ins for `TABLE_NEW`, `TABLE_INIT` and `SET_STRING` and then compiles, and the first test that starts a VM runs `install_helpers` (once) in between and puts the library's helpers over them, so the trace calls the library's helper with the stand-in's context. The boxed-step test started a VM next to it in the alphabetical order, so every full run hit the window. The stand-ins are now installed under a lock, after the first install has run, only for the compile (a trace keeps the addresses it was compiled with), and the library's table helpers are put back at once (`restore_table_helpers`, test builds only). The `resume_tests` stand-ins are of other helpers and are not touched.

### Remaining fallbacks

Stage 9 (below, "Stage 9: the trees MIR declined") lowered every reason in this table and in the emerge's list. The table records what was left after stage 8.

Each reason is the fixed string the counters aggregate, with the compiles and the trace IR instructions it keeps on Cranelift by default (11 compiles, 299 IR instructions of 128 compiles and 4211, `benchmarks/mir_backend.py coverage` on the final build of stage 8, arrays, the probe on), and what the operation needs. With `OXIMOND_JIT_MIR_PROBE=0` two probe rows come back (stage 7 measured 13 compiles and 228 IR instructions for them; the probe-off run was not repeated). The shares are of the 299.

| Reason | Compiles | IR instructions | Share of refused IR | What it needs |
|---|---:|---:|---:|---|
| counted trace | 2 | 127 | 55.9% | a trace with an iteration budget: a second loop parameter and a different loop-back (the budget of the count hook) |
| function trace | 6 | 58 | 25.6% | a different ABI (`-1 - index`), activations and self calls (`SelfCall`) |
| register that is a float or an integer at run time, under a side that rejoins the loop | 1 | 42 | 18.5% | the second tree of `opt/mod_mixed`: a guard that enters a side with a number of run-time subtype, in a tree whose sides rejoin the loop. Lowered in stage 8 (numbers) and handed back again for the cost of the rejoin path, until that is cheaper (see "Stage 8, numbers") |
| HashLoad without the inline probe | 11 | 199 | with `OXIMOND_JIT_MIR_PROBE=0` only | the in-line probe of `lower/probe.rs`, on by default since stage 7. Applies when a tree reads by string index and moves no hash storage (`native_probe`) |
| hash store or delete without the inline probe | 2 | 29 | with `OXIMOND_JIT_MIR_PROBE=0` only | as for `HashLoad`: `SET_PRESENT`, `SET_REVIVE` and `DELETE_STRING` through the probe (`hash_store`, `hash_overwrite`) in a tree that has `native_probe` |

Stage 8, arrays: four rows are gone, `ArraySet without the inline append and hole fill` (3 compiles, 87 IR instructions), `ArraySet under a strict header` (3 compiles, 39), `native next step without the inline form` (2 compiles, 36) and `array load past the end of the array` (1 compile, 18); the first, second and fourth went with the lowering of the first part (`9a9d1db7`, measured at `e7a92f7a`: 13 refused compiles), the third with this unit's native step. `ArrayNext` and `LaneArrayLoad` are lowered (see "Lane array loads, array stores and the array step"). On the emerge the arrays reasons are gone: `ArrayNext` is not logged any more, and `LaneArrayLoad`, `ArraySet under a strict header` and the hole fill are not logged since `e7a92f7a`.

Stage 8, numbers: the rows for the table field fill and for the native next step are gone (the counts of the table are of the build before the step was lowered), the row for a register of either subtype is narrowed to the rejoining case, and the coverage is now 119 of 128 compiles and 3,984 of 4,211 IR instructions through MIR (the numbers are in "Stage 8, numbers"; the table above is the final build's count of that unit for the rows it keeps, the others are the earlier measurement).

Stage 8, merged: the three units together leave the five rows above. The compiles and IR instructions of each row are the units' own counts; the shares are of their sum, 9 compiles and 227 IR instructions, and the merged build was not measured with `coverage` again.

Stage 8, rowfill: no row of this table changes. No benchmark of the suite records a `TABLE_GET_TABLE`, so the trees the lookup row fill moved to MIR (see "The lookup row fill") are not in the suite and its 20 fallbacks are the same. On the emerge the trees with the lookup were never counted, as the gate logged nothing; they now show under `numeric operation that needs a conversion` and `LaneArrayLoad`.

Gone since stage 2b: `closure cell or prototype access without the inline form` (stage 5), `CallHelper` for the helpers Cranelift calls too (the other helpers stay refused above), `trace that sets table entries`, `HostCall`, `side that continues into a loop` (nbody), `NumberLoad`, `ReleaseStrings`, `array load of a nil, table or string element`, `side that rejoins a loop whose parameters do not start from entry inputs` (the rejoin: 2 compiles, 60 IR instructions) and `NumberModulo` (the exit-only values: 2 compiles, 70 IR instructions, one of them now refused for the number of run-time subtype it passes to a side). `HashLoad without the inline probe` and `hash store or delete without the inline probe` are rows only with the probe off (stage 7). Gone in stage 8 (numbers): `numeric operation that needs a conversion` (385 emerge trees), `table field fill without the in-place store`, `shift by a value` and `native next step without the inline form`. Five rows of the table are reasons this backend added for inline paths: `HashLoad without the inline probe`, `ArraySet without the inline append and hole fill`, `table field fill without the in-place store`, `hash store or delete without the inline probe` and `ArraySet under a strict header`. Each names an operation an earlier build of this unit sent to the VM as a plain call and that now stays with Cranelift (see "Inline fast paths that Cranelift has").

Operations the lowering still refuses that no benchmark of the suite reaches (zero compiles in the table): `SelfCall` (function traces), `table field fill of a table value`, `shift by a value on a target without BMI2`, `Read` and `Effect` (Cranelift refuses those two as well), an integer compared with a float, a pattern helper that has a prepared twin, a loop side whose parameters lack a register, a root parameter with no register under a rejoining side (`loop parameter without a register`, as Cranelift's link), a side that rejoins a root that does not loop (`side that rejoins a root that does not loop`). Each keeps its tree on Cranelift with its fixed reason.

`loop side whose preheader calls into the VM under a side that rejoins its loop` is the old `preheader that calls into the VM under a side that rejoins`, narrowed to the one case left: a loop side that passes a parameter on unchanged, with a side that rejoins its loop (`LinkLoop`), keeps its entry values in cells and runs its preheader and loop after its `Join`, so a hoisted call in its preheader would run again at each rejoin, where Cranelift's loop side runs it once when its preheader sits before its header. The root no longer needs it. The gate appeared at stage 2c while testing (`xpcall_runs_a_lua_handler_on_the_caught_path_natively` under `mir`): the preheader had a hoisted concatenation (`'h:' .. e` with an invariant `e`) that, run again each iteration, took a string lane per iteration and ran the run out of lanes every thousand iterations (1,987 exits against 16). A root's preheader runs again per link exactly when Cranelift's does (the loop passes some parameter on unchanged), and once otherwise, so the tree of that test is compiled by MIR with its side. The benchmarks of the suite reach neither case.

### End to end, per benchmark

Whole-process `instructions:u` at scale 3, `benchmarks/mir_backend.py icount` through `scripts/heavy.sh -x`, pinned and under `env -i` as described above. The compile thread and the VM thread are both counted. A loop that runs 2,000,000 iterations is recompiled at the Peak tier, which stays Cranelift, so a total mixes MIR's compile saving, the run time of its code and the Peak recompile.

The benchmarks with at least one tree on MIR (67 of 89; the other 22 compile nothing on MIR at this scale and are identical up to noise, +0.004% in total). Commit `0b2db938`, load average at the start `4.80, 7.20, 14.02`, pinned to CPU 6 (sibling 14 idle):

| Benchmark | Cranelift (M) | MIR (M) | Difference | Compiles accepted, refused |
|---|---:|---:|---:|---|
| table/array_read | 92.29 | 107.76 | +16.8% | 1, 0 |
| opt/range_offset | 474.86 | 554.00 | +16.7% | 1, 0 |
| string/format | 419.58 | 452.31 | +7.8% | 1, 0 |
| string/pattern_run | 1847.65 | 1972.68 | +6.8% | 1, 0 |
| opt/cold_recovery | 244.23 | 257.77 | +5.5% | 1, 0 |
| opt/mod_float | 872.69 | 910.04 | +4.3% | 1, 0 |
| opt/range_float_index | 458.66 | 471.71 | +2.8% | 1, 0 |
| string/utf8_pack | 668.26 | 686.10 | +2.7% | 2, 1 |
| control/loop_in_call | 45.16 | 46.31 | +2.5% | 2, 0 |
| algorithm/sieve | 111.60 | 113.38 | +1.6% | 2, 1 |
| opt/mod_int | 697.87 | 706.58 | +1.2% | 1, 0 |
| opt/arith_mixed | 556.53 | 557.89 | +0.2% | 4, 0 |
| opt/range_dense | 763.84 | 765.37 | +0.2% | 1, 0 |
| string/pattern_many | 759.01 | 758.40 | -0.1% | 1, 0 |
| control/coroutine_pingpong | 603.31 | 600.94 | -0.4% | 1, 0 |
| opt/range_fail | 514.77 | 512.64 | -0.4% | 1, 1 |
| string/naive_concat | 292.61 | 291.28 | -0.5% | 1, 1 |
| stdlib/tonumber_parse | 734.99 | 730.45 | -0.6% | 1, 0 |
| table/metamethod_newindex | 228.48 | 226.03 | -1.1% | 1, 1 |
| calls/metamethod_ops | 632.19 | 624.72 | -1.2% | 1, 0 |
| table/iteration | 66.61 | 65.79 | -1.2% | 1, 0 |
| algorithm/spectral_norm | 109.81 | 107.86 | -1.8% | 2, 2 |
| allocation/ctor_side_exit | 66.67 | 64.97 | -2.5% | 1, 0 |
| control/branching | 108.13 | 105.24 | -2.7% | 2, 0 |
| table/insert_remove | 68.04 | 66.13 | -2.8% | 1, 0 |
| stdlib/math | 142.57 | 138.38 | -2.9% | 1, 0 |
| opt/arith_stable | 153.11 | 148.02 | -3.3% | 1, 0 |
| control/short_inner | 34.29 | 33.08 | -3.5% | 2, 0 |
| allocation/ctor_grow | 46.91 | 45.21 | -3.6% | 1, 0 |
| numeric/for_sum | 60.63 | 57.94 | -4.4% | 1, 0 |
| control/nested_calls | 262.54 | 241.91 | -7.9% | 2, 0 |
| stdlib/native_calls | 9.18 | 8.40 | -8.5% | 1, 0 |
| numeric/while_mix | 74.14 | 67.26 | -9.3% | 1, 0 |
| opt/call_args | 233.64 | 207.31 | -11.3% | 2, 0 |
| opt/dead_slots | 91.40 | 80.02 | -12.5% | 1, 0 |
| mineclonia/open_results | 41.46 | 36.29 | -12.5% | 2, 0 |
| calls/builtin | 17.12 | 14.89 | -13.0% | 1, 0 |
| calls/varargs | 12.00 | 10.38 | -13.5% | 1, 0 |
| table/field_read | 29.29 | 25.30 | -13.6% | 1, 0 |
| opt/field_reload | 33.87 | 29.24 | -13.7% | 1, 0 |
| calls/varargs_forward | 12.38 | 10.67 | -13.8% | 1, 0 |
| opt/alias_holder | 25.20 | 21.37 | -15.2% | 1, 0 |
| allocation/ctor_array | 8.01 | 6.78 | -15.3% | 1, 0 |
| calls/multiple_returns | 19.61 | 16.55 | -15.6% | 1, 0 |
| opt/invariant_field | 35.36 | 29.83 | -15.7% | 1, 0 |
| calls/method | 15.02 | 12.65 | -15.8% | 1, 0 |
| algorithm/nbody | 33.66 | 28.00 | -16.8% | 1, 0 |
| allocation/ctor_fields | 9.79 | 8.05 | -17.7% | 1, 0 |
| allocation/ctor_mixed | 8.77 | 7.20 | -17.9% | 1, 0 |
| table/metamethod_index | 11.77 | 9.66 | -17.9% | 1, 0 |
| algorithm/mandelbrot | 84.70 | 68.41 | -19.2% | 3, 0 |
| table/heap_redundancy | 16.16 | 13.03 | -19.3% | 1, 0 |
| mineclonia/ull_arith | 17.42 | 14.04 | -19.4% | 1, 0 |
| opt/store_load | 49.45 | 39.71 | -19.7% | 1, 0 |
| control/nested_loops | 33.40 | 26.77 | -19.8% | 2, 0 |
| calls/tail_call | 12.72 | 10.18 | -19.9% | 1, 0 |
| allocation/table_churn | 4.92 | 3.91 | -20.5% | 1, 0 |
| string/byte_sub | 81.99 | 63.26 | -22.8% | 1, 0 |
| calls/function | 14.06 | 10.84 | -22.9% | 1, 0 |
| calls/closure | 15.72 | 12.09 | -23.1% | 1, 0 |
| calls/protected | 21.01 | 16.10 | -23.4% | 2, 0 |
| mineclonia/bit_wide | 13.29 | 9.60 | -27.8% | 1, 0 |
| mineclonia/captured_call | 15.80 | 11.16 | -29.4% | 1, 0 |
| mineclonia/ull_kernel | 25.53 | 16.97 | -33.5% | 1, 0 |
| calls/metamethod_call | 23.08 | 15.25 | -33.9% | 1, 0 |
| mineclonia/module_calls | 25.04 | 15.54 | -37.9% | 1, 0 |
| string/plain_find | 3.75 | 2.02 | -46.0% | 1, 0 |

Totals: all 89 benchmarks 21350.0M under Cranelift and 21432.4M under MIR (+0.39%). The 61 benchmarks with every compile accepted: 11392.1M and 11462.3M (+0.62%), 50 of them lower. The 6 with some refused compiles: +0.61%.

Reading the table:

- Warmup-bound benchmarks win the most, as the research predicted: `string/plain_find` -46.1%, `mineclonia/module_calls` -37.9%, `calls/metamethod_call` -33.9%, `mineclonia/ull_kernel` -33.5%, `mineclonia/captured_call` -29.4%, `mineclonia/bit_wide` -27.8%, and nbody -16.8% (see "nbody"). 49 benchmarks are lower by more than 1%, 255.6M instructions in all.
- No benchmark is more than 17% above Cranelift. The previous build of this unit had 12 above +148% (`table/hash_churn` +506.8%, `algorithm/sieve` +354.6%, `opt/closure_protos` +188.5%) and a suite total of +28.6%: they held an operation Cranelift answers in line and the backend called the VM on every execution. Those trees stay with Cranelift now, so they show 0 accepted compiles or are identical up to noise, and the suite is +0.39%.
- `table/array_write`, `allocation/gc_cycles`, `string/table_concat`, `stdlib/sort` and `opt/retained_slots` stay with Cranelift (strict header or append), as before.
- Higher by a smaller amount: `table/array_read` +16.8% and `opt/range_offset` +16.7% were the loops Cranelift lowers twice, the second version without bounds checks (`RangePlan`), which MIR did not lower until stage 4 (see the table below). `opt/cold_recovery` +5.5% (Cranelift computes values only an exit reads in the exit stub, `defer_cold_values`). `opt/mod_float` +4.3%: MIR calls the compat modulo on every iteration where Cranelift's inline path for integral operands makes no call. `string/format` +7.8% and `string/pattern_run` +6.8% are not explained: the last build measured `string/format` at +4.7% on a Cranelift count 3% higher (432.05M against 419.58M now), so part of it is run-to-run variation of the Cranelift baseline (a program that formats and allocates), and I did not trace the rest. Inference. Both are dominated by calls into the VM that MIR and Cranelift make the same way, so the difference is small code-size and compile effects, not the loop.
- A policy that kept trees with a VM call on every iteration on Cranelift is now in the code, for the operations listed under "Remaining fallbacks": it gives up the compile saving on them and takes none of the loss.

#### Stage 4: range-proven loop versions

Whole-process `instructions:u` at scale 3, `benchmarks/mir_backend.py icount` through `scripts/heavy.sh -x`, pinned to a CPU whose SMT sibling is idle, `env -i` with `PAD` making the environment equally long under both backends (the script sets it now), binaries copied to paths of equal length. "Before" is a release build of `275f56a6` (the parent of stage 4), "after" a release build of `5bfcd50c`. Load average at the start of the two-benchmark runs `8.78, 6.71, 7.86` and `8.24, 6.63, 7.83` (CPU 7 and CPU 6); of the whole-suite runs `6.17, 6.27, 7.68` (after) and `5.47, 6.11, 7.60` (before), both on CPU 7.

| Benchmark | Cranelift (M) | MIR before (M) | MIR after (M) | After against Cranelift | After against before |
|---|---:|---:|---:|---:|---:|
| table/array_read | 92.30 | 107.77 | 88.44 | -4.2% | -17.9% |
| opt/range_offset | 474.89 | 554.00 | 463.26 | -2.5% | -16.4% |

Both pass the goal (within 2% of Cranelift or better) with room: MIR's fast loop is shorter than Cranelift's because a proven slot is `[base + scaled - 16]`, with no row add.

Whole suite (89 benchmarks, `suite_after` / `suite_before` runs): Cranelift 21.344G, MIR before 21.419G (+0.35%), MIR after 21.283G (-0.28% against Cranelift, -0.63% against before). The suite total under MIR is below Cranelift's. Against "before", five benchmarks are more than 1% lower: `table/array_read` -17.9%, `opt/range_offset` -16.4%, `control/short_inner` -8.8%, `algorithm/sieve` -4.3% and `opt/range_float_index` -4.1%. One is more than 1% higher: `string/format` +2.3% (441.99M to 452.31M). Cranelift's own count for `string/format` moved the same way between the two suite runs (409.26M to 419.58M, +2.5%), and the benchmark formats and allocates, so I take it for run-to-run variation, not for the range work: the tree has no counted array loop.

Correctness: `oximond-difftest run-jit --seeds 0..300 --profiles all --variants tracing,tracing-mir,tracing-mir-profile` with the release binary: 1800 cases, 0 mismatches, 190 s with 8 jobs. The same run with the debug binary (`target/debug/oximond`, so the MIR validator runs on every function), seeds 0..100: 600 cases, 0 mismatches, 751 s.

### Inline fast paths that Cranelift has

A reviewer found that the previous build accepted trees whose hot operations Cranelift handles in line and turned each into a call of the VM on every execution. 26 tests that assert native behaviour (`the_probe_replaces_the_call_on_every_lookup`, `a_hot_read_loop_asks_the_runtime_for_nothing_after_warmup`, `the_hash_part_is_stepped_natively`, `a_sieve_fills_holes_without_the_set_helper`) failed under `mir`, and 12 benchmarks cost 2.5 to 6 times Cranelift's instructions. That was a regression the backend chose, not a measured shortfall. The fix is a refusal with a fixed reason (in `lower/calls.rs`: `inline_in_cranelift`, and the guards at the top of `hash_load` and `array_set`) for each operation that has an inline path in `emit.rs` and none here:

- `HashLoad` of a `StringRef` key with a scalar expected type, in a tree where Cranelift's `native_probe` holds (a member reads by string index and none moves hash storage). Lowered in stage 5 behind `OXIMOND_JIT_MIR_PROBE`, in line since stage 6 and still declined by default; see "String-index probe".
- `SET_PRESENT`, `SET_REVIVE` and `DELETE_STRING` in such a tree. Lowered in stage 5 behind the same switch.
- `TABLE_INIT` (the fill of a table the trace made, in place in Cranelift).
- Closure prototype and cell helpers that `closure_cell_use` names, and the `next` helpers of a tree with `boxed_steps`.
- `ArraySet` of a number or boolean under a numeric key (Cranelift appends and fills holes in line; the in-place overwrite that MIR has is no longer reachable and stays in the code for the day the append exists).

Each test is the condition Cranelift checks or a conservative superset of it. A superset keeps a tree on Cranelift that Cranelift would have called for anyway (a `TABLE_INIT` on a table with no shape, for example); that costs the compile saving on that tree and nothing else. The result is in the tables above: the 8 failing tests under `mir`, the suite at +0.39%, no benchmark above +17%. The inline forms are not implemented in this unit. Stage 8, arrays, implemented the `ArraySet` forms and the native step of a boxed `next` loop (see "Lane array loads, array stores and the array step"); `TABLE_INIT` and the probe-off refusals stay.

### Steady-state loop cost

The suite hides the loop cost of MIR code behind the Peak promotion, so `benchmarks/mir_backend.py kernels` measures it where the promotion cannot happen: one loop per kernel, run for 300,000 and for 1,500,000 iterations (under the 2,000,000 of the promotion), and the difference of the two instruction counts divided by the difference of the iterations. That is the cost of one iteration of the Optimized tier's code under each backend, with warmup and compile cancelled. Load average when it ran: `2.62, 5.86, 10.78`, pinned to CPU 4 (sibling 12), chosen by the script the same way.

| Kernel | Cranelift | MIR | Difference |
|---|---:|---:|---:|
| float arithmetic | 14.01 | 13.01 | -7.1% |
| float modulo and floor | 18.01 | 15.01 | -16.7% |
| integer arithmetic | 19.01 | 18.01 | -5.3% |
| integer division by 7 | 35.01 | 31.01 | -11.4% |
| integer modulo by a value | 69.01 | 74.01 | +7.2% |
| array sum, float keys | 32.01 | 29.01 | -9.4% |
| array sum, integer keys | 31.01 | 29.01 | -6.4% |
| array update | 30.01 | 30.01 | +0.0% |
| table fields | 13.02 | 12.02 | -7.7% |
| branches and select | 15.01 | 16.01 | +6.7% |

MIR code is equal or lower on eight of ten kernels. Two are above zero and inside the goal of staying within 10%:

- `branches and select`, +6.7%: one instruction an iteration (16 against 15). I did not look for it.
- `integer modulo by a value`, +8.7% (75 against 69), which is the loop of `opt/mod_int` (its +1.5% end to end is this). The disassembly shows two causes. The modulo by a value picks its correction through two selects that each build the constants 0 and 1 in registers, where a mask, an `and` and one select would do. And a function with a division has one general register fewer (rdx is reserved), so the loop's fuel is spilled and reloaded on every pass.

Two kinds of change made the rest equal or lower. Both showed in the first measurement of this unit, before them the kernels read +7% float arithmetic, +11% integer division, +17% array update and +60% branches and select:

- a snapshot-only value takes a spare register instead of a stack slot written on every pass, and a float select on a float compare is `vcmpsd` and `vblendvpd` instead of six instructions through the general registers;
- division saves nothing around `idiv` (rdx is reserved in a function that divides), the fuel guard reads the flags of the subtraction before it, and the elided table counters of `PhantomTable` and the link counter are one `add` to memory each (`opt/dead_slots` went from +19% to -12.5% end to end on that last one).

### nbody

The goal was nbody's whole tree on MIR, with end-to-end instructions well below Cranelift's 33.6M at scale 3 and a loop within 10%. The tree is on MIR and the loop is within the goal. The end-to-end figure is below Cranelift's and short of what the compile saving alone promised. Whole-process instructions at scale 3, same binary, only the variable differs: Cranelift 33,656,524, MIR 27,991,227 (-16.8%, commit `b7003199`, load average `3.54, 7.80, 11.60`, CPU 4 with sibling 12 idle, `env -i`). The whole tree is one accepted compile (`jit-trace: mir backend compiled tree: 98 IR instructions, 329 MIR instructions`) and the program prints the same number.

Inference. The 0.6M fewer instructions than the previous build (28.57M, before the field group check) come from the group check: the pair loop reads and writes several fields of one body, and one mask compare replaces a guard per field. The loop kernel was not measured again.

Where the instructions go, from `OXIMOND_JIT_ICOUNT` (`benchmarks/mir_backend.py icount`): compiling the tree costs 8.11M under Cranelift and 0.42M under MIR (19 times fewer). The three sides that were recorded afterwards are compiled apart from their tree, which is always Cranelift's (4.07M under either backend, one of them 2.10M against 2.23M because it saw a tree of another shape). So the compile thread costs 12.18M under Cranelift and 4.62M under MIR, 7.6M less. The run itself, what is left of the totals, is 21.5M under Cranelift and 23.4M under MIR: 1.9M (9%) more, the MIR tree's code is not quite as short as Cranelift's (the hand-built kernel has 515 instructions an iteration against 476). So the 22M estimate of the previous round assumed the three sides on MIR too, and a loop as cheap as Cranelift's; neither holds. Compiling the sides apart through MIR is the next step for nbody.

What nbody is, from the trace log: the root is the pair loop; its exit leads into a side that runs the outer loop's step and enters the pair loop again (a loop side, `terminal`), and a side nested in that one rejoins the root's loop (`Link`). The tree needed `Op::Goto`, a loop inside a side region, per-region allocator state, and a nested side that rejoins the root while it sits inside a loop side. The array loads of `bodies[i]` are table loads (`ArrayLoad` of a table), which needed the lane admission call.

Loop cost: the stage 1 loop kernel `nbody` (the same pair loop, hand-built) runs 231.0 cycles an iteration under MIR against 230.5 under Cranelift (+0.2%), inside the 10% goal (median of 7 pinned runs, see "The spill-kernel cycle gap").

### The spill-kernel cycle gap

Stage 1 measured equal instruction counts but +15.7% cycles on `pressure` and +13.9% on `guard_restore`. The stage 2b round fixed `guard_restore` (the allocator stored values the loop defines on every iteration: the eviction score now charges them a store) and left `pressure` at +17% with the memory traffic equal to Cranelift's. This round found the cause on `pressure` and fixed it.

Method (`benchmarks/mir_cycles.py`): `bench mir-run K FUEL` and `bench clif-run K FUEL` (the same kernel compiled by each backend and run natively), each at 100,000 and at 500,000 iterations; cycles per iteration are the difference over the 400,000 extra iterations, so the compile and the start-up cancel. `perf stat -e cycles:u,instructions:u`, `taskset` to CPU 3 (sibling 11 idle), `env -i`, through `scripts/heavy.sh -x`, 7 repeats, the median reported (the range of the 7 is at most 5.8% of the median in these rows). Load average at the start: `30.11, 28.86, 28.23`. Events are the AMD Zen 4 counters of `perf list`; the groups had to be run separately, since the core has six.

Where the cycles go. `perf annotate` cannot read the kernels, which live in an anonymous mapping, so the bench writes the machine code of the run to a file and prints the address it ran at (`BENCH_DUMP`), and `benchmarks/mir_profile.py` samples `cycles:u` of the run, subtracts the address, and prints every instruction of the dump with the share of the samples that landed on it. The earlier doc guessed at the header's parallel moves (about 25 moves and reloads at the loop top) and at `mov rax, r8; imul rax, r9` chains through the scratch register. The profile rules both out: no instruction of the loop top has more than 0.1% of the samples, and the products are not on the critical path. The samples are in the last third of the iteration, where the sum of the 18 float products is made. Before the fix, the heaviest instructions of MIR's loop (shares of the samples in the kernel code, 40,000,000 iterations, ungated run on an idle CPU pair at load average 8):

```
  8.7%   5e9: vmovsd xmm6,QWORD PTR [rsp+0xe8]
  8.3%   5f6: vmovsd xmm5,QWORD PTR [rsp+0xe0]
  8.1%   603: vmovsd xmm4,QWORD PTR [rsp+0xd8]
  7.7%   610: vmovsd xmm3,QWORD PTR [rsp+0xd0]
  6.6%   61d: vmovsd xmm2,QWORD PTR [rsp+0xc8]
  6.1%   5dc: vmovsd xmm7,QWORD PTR [rsp+0xf0]
  4.5%   5cf: vmovsd xmm8,QWORD PTR [rsp+0xf8]
  4.1%   62a: vmovsd xmm1,QWORD PTR [rsp+0xc0]
  3.9%   637: vmovsd xmm0,QWORD PTR [rsp+0xb8]
```

They are the loads that sit between the dependent `vaddsd xmm10, xmm10, xmm_k` of the sum, each stalled behind the add before it (the samples skid onto the next instruction). Each add has a latency of 3 cycles and there are 18, one after the other: 54 cycles, which is Cranelift's whole loop (53.9 to 54.5 cycles). MIR's loop took 62.9 to 63.6. The listing says where the other 9 cycles came from. The accumulator `af` is a loop parameter, and the products of the body need more float registers than the file has, so the allocator evicted the value that had the fewest uses before the chain, which was the accumulator. A header parameter is defined at the header, so an evicted one is stored there on every iteration, and loaded again before the chain:

```
 264: vmovsd QWORD PTR [rsp+0x1a8],xmm10      the accumulator, stored at the loop top
 ...
 446: vmovsd xmm10,QWORD PTR [rsp+0x1a8]      loaded again before the 18 additions
```

The store and the load are on the dependency chain from one iteration to the next (the sum of this iteration is the accumulator of the next), so the store-to-load forwarding latency, about 5 cycles, and the register moves around it were added to the 54 cycles of additions every iteration. Cranelift keeps its accumulator in `xmm0` through the whole body and spills the products and the invariants instead.

The fix (`alloc_reg` in `ra.rs`): the eviction score charged a store per iteration to a value the loop defines (`def_at > hdr_idx`), but a header parameter has `def_at == hdr_idx`, so it was not charged. It is now (`>=`), and an invariant, a product or a reload is evicted before the accumulator. After the fix the listing has no store or reload of `xmm10`, the chain starts when the products are ready, and the profile has 45.3% of the samples on its heaviest instruction, inside the chain, as expected of a loop that is as long as its dependent additions:

```
 45.3%   5d3: vmovsd xmm6,QWORD PTR [rsp+0xe8]
 31.3%   5e0: vmovsd xmm5,QWORD PTR [rsp+0xe0]
  5.3%   5ed: vmovsd xmm4,QWORD PTR [rsp+0xd8]
  4.3%   5c6: vmovsd xmm7,QWORD PTR [rsp+0xf0]
  1.7%   5fa: vmovsd xmm3,QWORD PTR [rsp+0xd0]
  1.4%   63e: sub    rbx,0x1
```

(The pinned run through the gate, `benchmarks/mir_profile.py pressure --fuel 40000000`, is the one printed; Cranelift's profile of the same kernel is spread over its own reloads and the same chain.)

Cycles per iteration after the fix, same method:

| Kernel | Cranelift cycles | MIR cycles | Difference | Cranelift instr | MIR instr |
|---|---:|---:|---:|---:|---:|
| pressure | 54.1 | 54.1 | -0.1% | 209 | 189 |
| guard_restore | 28.9 | 25.0 | -13.4% | 139 | 116 |
| nbody | 230.5 | 231.0 | +0.2% | 476 | 515 |
| int_side | 8.6 | 7.6 | -12.3% | 28 | 26 |
| swap | 3.2 | 3.2 | -1.4% | 14 | 15 |

| Per iteration | pressure Cranelift | pressure MIR | guard_restore Cranelift | guard_restore MIR |
|---|---:|---:|---:|---:|
| store-to-load forwards | 4.0 | 4.0 | 0.0 | 0.0 |
| stores | 14.0 | 13.0 | 16.0 | 12.0 |
| loads | 71.0 | 64.0 | 40.0 | 39.0 |
| store queue full stalls | 0.0 | 0.0 | 0.0 | 0.0 |
| backend stall slots | 116.1 | 136.3 | 36.1 | 36.1 |
| store-to-load interlocks | 0.0 | 1.0 | 11.1 | 7.4 |

For reference, the same table before the fix (stage 2b, commit `1650cedb`, same machine and method, load average `5.10, 5.76, 7.82`):

| Kernel | Cranelift cycles | MIR cycles | Difference | Cranelift instr | MIR instr |
|---|---:|---:|---:|---:|---:|
| pressure | 53.9 | 62.9 | +16.7% | 209 | 192 |
| guard_restore | 28.9 | 25.2 | -12.9% | 139 | 116 |
| nbody | 230.4 | 231.0 | +0.2% | 476 | 515 |
| int_side | 8.4 | 7.6 | -9.3% | 28 | 26 |
| swap | 3.2 | 3.2 | -2.6% | 14 | 15 |

| Per iteration | pressure Cranelift | pressure MIR | guard_restore Cranelift | guard_restore MIR |
|---|---:|---:|---:|---:|
| store-to-load forwards | 4.0 | 5.0 | 0.0 | 0.0 |
| stores | 14.0 | 14.0 | 16.0 | 12.0 |
| loads | 70.9 | 65.0 | 40.0 | 40.0 |
| store queue full stalls | 0.0 | 0.0 | 0.0 | 0.0 |
| backend stall slots | 114.7 | 188.2 | 34.8 | 35.7 |
| store-to-load interlocks | 0.0 | 1.0 | 11.0 | 6.7 |

`pressure` is -0.1% against Cranelift (54.1 against 54.1 cycles, with 136 backend stall slots an iteration against 116), where it was +17%. `guard_restore` is -13.4%. `nbody` is +0.2%. `int_side` and `swap` did not change with the fix (three repeated runs of each with and without it gave the same cycles within the noise of an ungated run).

Inference. The lesson for the allocator is that a loop-carried dependency makes a spill expensive in latency, not only in stores: a value on the path from one iteration to the next should be the last to be evicted. The score now says so for parameters. Values that feed the back edge (the argument of `Back`) are loop-defined already and charged.

### Compile cost

Stage 1 measured the per-tree backend figure (this unit did not repeat it), in the table of that unit: 1,788,686 instructions per tree under Cranelift against 81,384 under MIR over the 34 accepted trees, 22.0 times fewer (goal: 10 times; the prototype measured 22 to 48 times on its kernels). The smallest accepted trees (7 to 9 IR instructions) cost about 40,000 under MIR and 770,000 under Cranelift, 19 times. The coverage run's 46 compiles made 3,964 MIR instructions from 1,245 IR instructions (3.2 per IR instruction), so the backend costs roughly 81,000 / 86 = 950 instructions per MIR instruction, or 3,000 per trace IR instruction against 66,000 under Cranelift (the sets differ a little, 34 trees against 46 compiles). The prototype's goal was about 1,000 per MIR instruction.

The steady-state allocation test needs zero allocations inside the backend for a compile after the first few; it passes. Code memory is one 128 MiB mapping per process (reserved, not committed). The 46 trees of the coverage run are 34,065 bytes of machine code and constants in all, about 740 bytes each.

## Gaps, and what the next unit needs

Missing mechanisms, in the order the suite and the Cranelift-only tests say they matter:

1. Inline fast paths in front of the calls, the way to bring the refused families back. The closure rows are done (see "Closure rows"). The string-index probe and the stores through it are done in stage 6 and on by default since stage 7 (see "String-index probe"): they are built on the forward skip (`If` and `EndIf`, see "The forward skip"), which was the missing construct. The rest of the same shape of work is done in stage 8: the append and hole fill of `ArraySet`, the native stepping of `pairs` and the strict-header stores (arrays), and the in-place fill of a table the trace made (numbers). Each is an in-line first case with a skip over the rest and a conditional call of the VM's function behind it (`Op::CallIf`: the in-place array overwrite, the lane probe and the probe do this). The table of "Remaining fallbacks" gives the benchmarks each one would bring back.
2. Counted traces (the count hook's budget as a second loop parameter), function traces and `SelfCall` (a different ABI), and a rejoining side entered with a number of run-time subtype (see "Remaining fallbacks" and, for what they need, "Stage 8, numbers", under "Not done"). All have zero, one or two compiles in the suite and a fixed reason each; the 14 trees the emerge still hands back (stage 8, numbers, final build) are 8 function traces and 6 `ArrayNext`. `NumberModulo` is done in stage 7 (exit-only values).
3. Done in stage 7 (the rejoin): a root parameter that does not start from an entry input, and a preheader that calls into the VM under a side that rejoins, see "Sides that rejoin the root loop". The loop side is done in stage 8 (links): when the loop passes no parameter on unchanged, its `Join` goes below its preheader, as Cranelift's header does, so a rejoin runs the preheader zero times. What is left of the same shape is a loop side that passes a parameter on unchanged: its `Join` still sits before its preheader, so `loop side whose preheader calls into the VM under a side that rejoins its loop` stays for it when a nested side rejoins it.
4. Done in stage 4: the second, range-proven version of a counted array loop (`RangePlan`), see "Range-proven loop versions". Stage 8 (links) did the links: a side that steps like the root rejoins a second `Join` below the checks, one that stays in the guarded loop rejoins a `Join` of region 1, and the guarded loop continues from the held values. What is left: the first join stays, and a root whose preheader calls into the VM and has a range plan stays with Cranelift.
5. Done in stage 7 (exit-only values): values only an exit reads, computed in the exit (`defer_cold_values`), and the float remainder of Lua 5.3 and later in line, see "Exit-only values and the float remainder (stage 7, exit-only values)". What is left of it:
   - A cold value a guard passes to a side, a skip's join value or a conditional call's argument is computed where it stands (the stub form needs the guard to be an exit). Cranelift recomputes it at the guard, so a tree whose exits mostly enter in-tree sides keeps the cost the loop pays.
   - Only float arithmetic with an operation of the stub (`FAdd`, `FSub`, `FMul`, `FDiv`, `FNeg`, `FAbs`, `FFloor`, `FCeil`, `FTrunc`) is deferred. A power and the compat remainders call, so they stay in the loop, and so does everything Cranelift does not defer.
6. The native link of a root to sides compiled apart, if the run loop's call of a side turns out to cost more than Cranelift's native one on a benchmark that takes the side often.
7. Done in stage 7 (the residue): `opt/closure_family` (+0.55% at stage 5, at Cranelift's count since stage 7) and the three string benchmarks of the gap list, see "The residue of the gap list (stage 7, the residue)". What is left of it is items 8 and 9.
8. The rejoin path of `opt/closure_protos` (the residue measured it: 86.8 instructions an iteration in the Optimized tier against Cranelift's 68.5, after the cell homes took it from 104.3). Seven iterations in eight leave the root into a side and rejoin it. What is left, per link:
   - A `Goto` into a side copies each invariant that sits in a slot from that slot into the side's entry slot (a load and a store through `r11`, 16 instructions in the tree of eight members) and rotates the eight closure lanes with a chain of `xchg`. Cranelift passes the same values as block parameters that share the slot. A fix: a side's entry parameter takes the parent's cell home as its slot when the value the guard passes is a cell-home invariant, so the move is empty (the side is compiled before the parent, so the parent's choice has to be planned first).
   - A `Rejoin` moves its arguments into the join parameters (a second chain of eight `xchg`), and the `Join` then moves them into the header's parameters (a third). The join's locations are fixed by `plan_join` before the sides are compiled, the header's incoming locations at the root's back edge, which is compiled last. A fix: choose the header's incoming locations of a root that has a join first and hint the join parameters to them, so the second move is empty.
   - Six float invariants are loaded from their cells into registers after each join, where Cranelift's link leaves them in the registers that dominate the header.
   Stage 8 (links) did the second (the join parameters hinted to the header's registers). The Goto copies stay: homing the side's parameters in the parent's cells was tried and measured slower (see "Direct links to the loop header"). The float reloads are left.
   `xchg` is three micro-ops, so the gap in cycles is probably larger than the gap in instructions; cycles were not measured. The whole process stays 8.8% below Cranelift only because of the compile saving.
9. The compile thread hides run-time loss in whole-process counts (`closure_protos` above). The slope method of the residue (two scales under the Peak threshold with `OXIMOND_JIT_SYNC=1`) is the check for any benchmark whose trees MIR compiles and that has several compiles; the suite table does not do it.


10. Stage 8, rowfill: the lookup row fill is in MIR (see "The lookup row fill"), but the two Mineclonia loops it was written for (structures.lua:855 and :1327) are still handed back, for `numeric operation that needs a conversion` (a `floor`, `ceil` or `abs` of a source that is not a float) and `LaneArrayLoad` behind it. Stage 8 lowered both (numbers and arrays), so the loops should move to MIR; the merged build was not run on the emerge to confirm it. The emerge also hands back trees for `load of a table, closure or string field` (a table, closure or string read out of a bound table's field; 11 trees on main's run, 23 on stage 8's), which this unit does not touch either.

11. Stage 8, arrays: what is left of it. The sets of `table/array_write` and `algorithm/sieve` are 11.8% and 21.1% above Cranelift's count (the tables of "Lane array loads, array stores and the array step"); the cause was not examined. `ArrayNext` along a run of holes is a call (`scan_value_slots`) where Cranelift loops in line. The native step of a boxed `next` loop kept by the merge (the numbers unit's) leaves each hole to the helper's call. The native step is lowered for a call with a number or boolean value only; a `Lane` value (a loop variable the body never reads) is copied as Cranelift copies it, and the other values are the helper's call in both backends.

Smaller points:

- No load fusion: an array load is `movzx`, `cmp`, `jne` and a payload load, where Cranelift's `cmp byte [mem]` and `addsd xmm, [mem]` save two instructions an access.
- A link of a rejoining side moves every loop parameter and the fuel into the join's parameters and loads, from frame cells, what the loop reads of the entry (values, pointers, binding words, the rows pointer), where Cranelift's link keeps those in registers or spills. It is about a load per value the loop reads that has a register, with no store (since the residue a value without a register stays in its cell and costs nothing); the entry pays the stores once. A tree whose sides may move arrays also reads the binding words again after each join. A tree with many such values and a link on a hot path would feel it.
- A cell store at the root's entry runs on every entry of a tree that has a rejoining side, whether or not a link happens. The trees of the suite show no loss from it (the entry is run once per outer iteration at most), but a tree entered millions of times for a few iterations would pay a store per value its loop reads.
- `string/naive_concat`: the loop of its tree is two helper calls with 14 parameters each, and the disassembly shows each call storing seven zero stack arguments. A microbenchmark of that loop alone ran 1% above Cranelift's count on the after build; the whole benchmark equals it (292.45 M against 292.60 M). Not a rejoin cost, and not examined further.
- A float live across a call lives in a frame slot (all xmm registers are caller-saved), so a loop with a call spills the floats it keeps. `CallIf` avoids that for the calls that are conditional.
- `ReleaseStrings` stores the kept lanes into the control block on every iteration, and only the call is conditional.
- `clif_insts` is 0 for MIR code, so the statistics that sum CLIF instructions (`clif_instructions_tree`) read 0 for MIR trees.
- MIR code is x86-64 Linux and Windows only, and needs SSE4.1 (`roundsd`). The region is mapped RWX (on Windows reserved, then committed a megabyte at a time); a system that refuses that falls back to Cranelift with a logged reason.
- On Windows the code follows Win64: arguments by position in rcx, rdx, r8, r9 or xmm0 to xmm3, 32 bytes of shadow space below the stack arguments, rsi, rdi and xmm6 to xmm15 preserved. The prologue saves the callee-saved xmm registers the allocator hands out, plus xmm15, the float scratch, in 16-byte slots at the top of the frame, and probes a frame of a page or more one page at a time.
- The exit-stub tails shared between exits (`ExitPlan::tails`) are ignored: each stub stores its descriptor's slots.
- Sides compiled apart are always Cranelift, so a tree whose root MIR compiled and whose side was compiled apart runs the side through the run loop's `call_side`, one Rust frame per link, not through a native call.
- A loop side that passes a parameter on unchanged and that sides nested in it rejoin runs its preheader again at each `LinkLoop` (it keeps its entry values in cells, then runs the preheader and the loop): fine for what a preheader computes and reads, refused when it calls into the VM. A loop side that passes none runs it zero times per rejoin (stage 8, links).

## Stage 9: the trees MIR declined

Four units, merged on `mir/declines`, took the refusals left after stage 8. On the suite at scale 3 MIR now compiles all 128 of 128 compiles (4,215 of 4,215 trace IR instructions; stage 8 left 9 compiles and 227 IR instructions to Cranelift). On the Luanti Mineclonia workloads no tree falls back: the emerge had 19 fallback trees (1,250 IR instructions) at `481b884e` and has none, Mineclonia startup had 14 and has none. Devtest startup keeps one, a 3-instruction function trace with more arguments than argument registers. The Peak tier and sides compiled apart still use Cranelift, as before.

| Reason that is gone | Suite before | Emerge before | How it is lowered |
|---|---:|---:|---|
| `counted trace` | 2 compiles, 127 IR | 0 | The count hook's budget is a second loop value after the fuel. The root entry loads `TraceControl::budget`, every header, join, side entry and link passes it, a loop-back pays the poll snapshot's `counted` cost (a link its link snapshot's, a `LinkLoop` nothing), and every exit stores the budget the iteration began with. One `sub` per loop-back, as in `emit.rs`. |
| `register that is a float or an integer at run time, under a side that rejoins the loop` | 1 compile, 42 IR | 0 | `NumberLoad` is a forward skip instead of select chains: load the tag and the payload as a float; if the tag is not a float, guard an integer in the exact range and convert. The 14 extra instructions per load that made the rejoin path dear are gone, so the refusal went with them. |
| `function trace` | 6 compiles, 58 IR | 8 trees, 28 IR | One activation function `(left, control, area, fuel, [count], params...) -> (status, result)`, results in rax and rdx (xmm0 for a float), and a wrapper with the trace entry ABI after it in the same image. New MIR operations: `Ret` with two values, `Call` with `CALL_REL` (an offset in the image, 0 for the activation itself), `Entry` with more than three arguments, exits that return a value (`RETURN_SLOT`) and copy a word through `control.inputs` (`INDIRECT_SLOT`). A self call guards the depth once per member and passes `area + stride`; a tail call restarts the activation at a header with `count + 1`. Lowering in `lower/function.rs`. |
| `VmCall` (a host C function called from the trace) | 0 | 5 trees, 311 IR | The helper reads its guard's `Before` state from the exit area, so the code stores it there before the call, the stores `emit.rs` makes. New MIR op `ExitArea`: one load of the frame slot the entry saves the exit-area pointer in (an activation uses its own area). |
| `side that rejoins a root that does not loop` | 0 | 3 trees, 806 IR | A root that ends in front of an inner loop gets a `Join` header anyway, as Cranelift's `lower_root` does; a side's link runs the root's body again from it, the next outer iteration. |
| `loop side with a parameter without a register` | 0 | 3 trees, 105 IR | Such a parameter (a register of the calls' frames, or one that starts from a constant) is carried like any other. A side that rejoins that loop names parameters by register and stays refused, as Cranelift's `LinkLoop` refuses it. |

The outer-root unit also made side entries cheaper for all trees: when every entry into a side passes the same constant, or the same value at two places, the side makes the constant itself or reuses the earlier parameter, and the jump moves nothing for it (`note_entry`); a guard on two integer constants that holds is dropped. A loop side in a call took about 50 moves per entry before, 17 after.

### Measured

Pinned `instructions:u` through `scripts/heavy.sh -x`, each against Cranelift in the same binary. Load averages were 10 to 18; instruction counts, not wall time.

| Case | Cranelift | MIR | MIR against Cranelift |
|---|---:|---:|---:|
| `opt/call_args_hook`, whole process | 238.85 M | 209.51 M | -12.3% |
| `opt/mod_mixed`, whole process | 680.05 M | 662.19 M | -2.6% |
| `calls/recursive_fib`, whole process | 30.74 M | 20.17 M | -34.4% |
| `calls/recursive_fib`, slope per scale unit | 8.96 M | 7.35 M | -18.0% |
| `algorithm/binary_trees`, whole process | 370.68 M | 362.30 M | -2.3% |
| `algorithm/binary_trees`, slope per scale unit | 119.57 M | 120.04 M | +0.4% |
| host call in a loop (`vmcall_probe`), per iteration | 2,380 | 2,369 | -0.5% |
| host call tree, compile thread | 2.88 M | 0.15 M | 19 times fewer |
| outer root, `scan` repro, per outer trip | 285,869 | 292,059 | +2.2% |
| outer root, `siblings` repro, per outer trip | 741 | 605 | -18.4% |

Two steady states are above Cranelift. `binary_trees` reloads integers after each self call (the callee-saved registers are taken) and its in-place table fill is a select and two skips where an if and else would be one branch. `scan` redoes loop-invariant work in each inner iteration (lane-probe selects, control-block loads, about four spill reloads) that Cranelift's optimizer hoists; MIR has no loop-invariant code motion.

The Mineclonia emerge, whole process, Oximond variant only, interleaved with `main` at `481b884e` (load 9 to 19): `mir/declines` 147.6 to 152.2 G (six runs, median 150.6 G), `main` 148.2 to 151.6 G (seven runs, median 149.0 G). The spread between runs of one build is 3%, more than the difference, so the emerge does not move measurably: the trees that changed backend are about 0.5% of the IR compiled, and their compile saving is tens of millions of instructions. Mineclonia startup is 35.75 G against 35.69 G on `main`.

Checks on the merged branch: `cargo test --locked -p oximond -p oximond-cli -p oximond-mir --no-fail-fast` 4,915 passed, 0 failed with the backend unset and with `OXIMOND_JIT_BACKEND=cranelift`; `mir_backend.py diff` equal on all 89 benchmarks; `oximond-difftest run-jit --seeds 0..300 --profiles all` passed; the function-trace unit's tracing conformance run had identical summaries under both backends.

Left:

- A side of a function trace that tail calls (`side of a function trace that tail calls`), and a function trace whose activation needs stack arguments (`function trace with more arguments than argument registers`: more than six integer words on SysV, four on Win64). The devtest startup has one of the second.
- A counted function trace (`counted function trace`): the activation has no place for the budget. Not seen on the suite or Luanti.
- A `LinkLoop` into a loop side with a parameter without a register, refused by both backends.
- Loop-invariant code motion, and an if/else in MIR, for the two steady-state gaps above.

## Commands to rerun

```sh
# Release build, once.
scripts/heavy.sh -m 24G cargo build --locked --release -p oximond-cli -p oximond-conformance

# Tests: both backends (the second run lists the Cranelift-specific failures above).
env -u OXIMOND_JIT_BACKEND scripts/heavy.sh -m 24G cargo test --locked -p oximond -p oximond-cli -p oximond-mir --no-fail-fast
OXIMOND_JIT_BACKEND=mir    scripts/heavy.sh -m 24G cargo test --locked -p oximond -p oximond-cli -p oximond-mir --no-fail-fast
# Narrow, ungated (one crate each):
cargo test -p oximond --lib jit::trace::mir
cargo test -p oximond-mir
cargo test -p oximond-cli --test mir_backend

# Output identity and coverage on the suite.
scripts/heavy.sh -m 16G python3 benchmarks/mir_backend.py diff     out/diff.json
scripts/heavy.sh -m 16G python3 benchmarks/mir_backend.py coverage out/coverage.json

# Conformance, both backends, with this worktree's corpora.
for b in cranelift mir; do
  OXIMOND_JIT_BACKEND=$b scripts/heavy.sh -m 16G target/release/oximond-conformance run --execution tracing --dir .oximond/conformance
done

# Instruction counts (run through the exclusive gate, CPU chosen by the script; copy the binary once so both runs share its path).
cp target/release/oximond out/oximond-rel
scripts/heavy.sh -x python3 benchmarks/mir_backend.py icount  out/icount.json  --bin out/oximond-rel NAME...
scripts/heavy.sh -x python3 benchmarks/mir_backend.py kernels out/kernels.json --bin out/oximond-rel

# Cycles and memory events per iteration of the loop kernels (the spill-kernel study).
scripts/heavy.sh -m 8G cargo build --release -p oximond-mir --example bench
scripts/heavy.sh -x python3 benchmarks/mir_cycles.py out/cycles.json

# Where the cycles of a kernel go, instruction by instruction, MIR code and Cranelift code (needs the bench example built above).
scripts/heavy.sh -x python3 benchmarks/mir_profile.py pressure --fuel 40000000

# Stages 6 and 7: the probe, on by default since stage 7. `--probe 0|1` sets OXIMOND_JIT_MIR_PROBE under both backends, so the environments keep equal lengths; `--probe 0` is the probe-off run and the comparison of two builds should pass `--probe` to both. A binary from before the probe's rewrite (81460ee6) runs with `--probe 0` or `1`, as it has the variable too.
scripts/heavy.sh -x python3 benchmarks/mir_backend.py icount  out/icount.json  --bin out/oximond-rel --probe 1 table/hash_read table/hash_read_small table/hash_read_shaped table/hash_read_indexed table/hash_write table/hash_churn table/metamethod_function table/metamethod_newindex
scripts/heavy.sh -x python3 benchmarks/mir_backend.py kernels out/kernels.json --bin out/oximond-rel --probe 1
scripts/heavy.sh -m 16G python3 benchmarks/mir_backend.py coverage out/coverage-off.json --bin out/oximond-rel --probe 0

# Stage 7, exit-only values and the float remainder: the four benchmarks that moved. Build the two binaries from a `git archive` of each commit into a fresh target directory, copy them to paths of equal length (`bin-a/oximond`, `bin-b/oximond`), run the same command on each.
scripts/heavy.sh -x python3 benchmarks/mir_backend.py icount out/icount.json --bin bin-b/oximond opt/cold_recovery opt/mod_float opt/mod_int opt/mod_mixed
cargo test -p oximond-mir --test cold --test ops
cargo test -p oximond --lib jit::trace::mir::tests::values_only jit::trace::mir::tests::float_remainders jit::trace::mir::tests::number_modulo

# Stage 7, the residue: the six benchmarks of the residue, whole process (before: `bin-a/oximond` from `15b7f761`, after: `bin-b/oximond` from the code of `2055a887`), then the run alone by the slope of two scales (the loop stays in the Optimized tier, `perf stat -e instructions:u`, compile on the VM thread).
scripts/heavy.sh -x python3 benchmarks/mir_backend.py icount out/icount.json --bin bin-b/oximond --cpu 7 opt/closure_protos opt/closure_family string/format string/pattern_run control/loop_in_call string/utf8_pack
for scale in 0.5 1.2; do for backend in mir cranelift; do perf stat -x, -e instructions:u taskset -c 7 env -i OXIMOND_JIT_BACKEND=$backend OXIMOND_JIT_SYNC=1 bin-b/oximond --execution=tracing benchmarks/lua/opt/closure_protos.lua $scale 2>&1 >/dev/null | grep instructions:u; done; done
cargo test -p oximond --lib jit::trace::mir::tests::the_length jit::trace::mir::tests::closures
cargo test -p oximond-mir --test ops --test rejoin
OXIMOND_MIR_SEEDS=100000 cargo test --release -p oximond-mir --test random

# Stage 4: the range-proven loop version, MIR against Cranelift on the two loops it matters for (the script pads the environment).
scripts/heavy.sh -x python3 benchmarks/mir_backend.py icount out/icount.json --bin out/oximond-rel table/array_read opt/range_offset
scripts/heavy.sh -m 16G target/release/oximond-difftest run-jit --seeds 0..300 --profiles all --variants tracing,tracing-mir,tracing-mir-profile --jobs 8 --reduce --oximond target/release/oximond

# One tree: the MIR, then the machine code.
OXIMOND_JIT_BACKEND=mir OXIMOND_JIT_SYNC=1 OXIMOND_JIT_DISASM=1 target/release/oximond prog.lua 2>&1 | python3 benchmarks/mir_disasm.py
OXIMOND_JIT_BACKEND=mir OXIMOND_JIT_SYNC=1 OXIMOND_JIT_TRACE=1 OXIMOND_JIT_STATS=1 target/release/oximond prog.lua

# Stage 8, arrays: the array step and the native next step against Cranelift, the loops of the repro table, and the Luanti emerge.
cargo test -p oximond --lib jit::trace::mir::tests::arrays_tests
cargo test -p oximond-cli --test mir_backend array_steps a_native_next
cargo test -p oximond --test trace_lane_arrays
scripts/heavy.sh -x -m 12G python3 integrations/luanti/luanti.py bench --variant oximond --workload mineclonia-emerge --reps 3 --jit-trace
scripts/heavy.sh -x -m 12G python3 integrations/luanti/luanti.py bench --variant oximond --workload mineclonia-startup --reps 10
```
