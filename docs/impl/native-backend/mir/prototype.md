# MIR prototype: backward allocator and x86-64 emitter

Measured commit: `dde6287a` on branch `flow/mir-research-proto` (base `b33edd2e`). The code is identical to `fc88e785`; the later commit only fixes a newline in `bench.sh`. Crate: `crates/oximond-mir`, `publish = false`, no dependency on any oximond crate. Cranelift 0.130.2 is a dev-dependency, used only by the CLIF side of the tests and the bench.

Marks as in `research.md`: a paragraph that starts with "Inference." is my own reasoning, everything else is code or measurement.

## Result in one paragraph

The hypothesis holds for compile cost and mostly holds for run cost. A compile of the nbody-shaped kernel takes 324,864 host instructions for 507 MIR instructions (641 per MIR instruction). Cranelift's Optimized path takes 7,174,700 for 561 CLIF instructions (12,789 per CLIF instruction), so 22 times more, and 1,000 times the allocations (0 against 448 per compile). The generated loop executes 515 instructions per iteration against Cranelift's 476 (+8.2%), and 234 against 222 cycles (+5.6%). Across the six kernels the instruction count per iteration is within 9% either way, but cycles are +14% to +16% on the two pressure kernels. I did not find out why (see "Gaps").

## What was built

Everything is in `crates/oximond-mir/src`.

| File | Contents |
|---|---|
| `ir.rs` | MIR types, builder, validator, text dump |
| `eval.rs` | Reference evaluator over byte regions |
| `target.rs` | `Target` trait, locations, parallel-move planner, `Sink` trait |
| `x64.rs` | x86-64 encoder and the `Target` impl (SSE2 and AVX) |
| `buf.rs` | Down-growing body buffer, up-growing stub buffer, constant pools |
| `ra.rs` | The backward allocator fused with emission |
| `exec.rs` | mmap RW, copy, mprotect RX; the only unsafe module (plus the bench) |
| `kernels.rs`, `randgen.rs`, `rng.rs` | Hand-written kernels, seeded random programs, SplitMix64 copied from `oximond-difftest` |

### MIR types

- A `Func` is one trace tree: five `Vec`s (`insts`, `val_ty`, `lists`, `snaps`, `snap_vals`) plus the first instruction of each side. `Func::clear` keeps every allocation.
- `Inst` is a 40-byte `Copy` struct: `op`, `ty`, `cc`, `scale`, `dst`, and operands `a b c d` and `imm`. Values are SSA ids (`u32`). Variable-length data (header parameters, back-edge arguments, snapshots) lives in the `lists` and `snap_vals` arenas.
- Types are `I64` and `F64`. Ops: `Const`, `Load` and `Store` (`base + index*scale + disp`), `Add Sub Mul And Or Xor Shl Shr Sar` (shift counts are constants), `FAdd FSub FMul FDiv FSqrt`, `I2F`, `Select` (integer), `Guard`, `Header`, `Back`, `Exit`, `Entry`, `SideEntry`.
- A guard reads a snapshot and is not a block end. It continues when `a cc b` holds and otherwise exits through its snapshot (index = exit index) or passes the snapshot's values to a side. Float compares are ordered, `Ne` is true on NaN, as in Cranelift's `fcmp`.
- Region 0 is the root: `Entry`, prologue, `Header` ... `Back` (or a closing `Exit`). Regions 1.. are sides: `SideEntry`, body, `Exit`.
- ABI: `(inputs, control, exit_area) -> -1 - exit index`, the same as `exit_status` in `emit.rs`. `inputs` uses the `TraceInputSlot` layout (16 bytes, so slot `i` is at word `2i + 1`). `control[0]` is the fuel; the kernels count it down in a loop parameter and guard on it, so fuel needs no special op.

### Target description

`Target` (in `target.rs`) is a trait with three groups of methods.

1. Registers: `alloc_regs`, `caller_saved`, `scratch_gpr`, `scratch_fpr`, `tmp_gpr`, `entry_regs`. Registers are unified ids (GPRs 0..16, FPRs 16..32).
2. Capability table: `two_address(op, ty)`, `imm_ok(op, ty, imm)`, `mem_src_ok(op, ty)`, `fcmp_swapped(cc)`. The allocator asks these questions and never mentions x86.
3. Emission primitives, generic over a `Sink`: `mov`, `swap`, `alu`, `unary`, `load`, `store`, `select`, `guard`, `jump`, `exit_stub`, `prologue`, `epilogue`.

x86-64 SysV is implemented. `X64 { avx }` has two encodings: SSE2 (two-address float ops) and VEX (three-address). `two_address` returns false for floats with AVX, which is the only thing the allocator sees of the difference. Allocatable: 13 GPRs (rax and r11 are scratch, rsp is the stack) and 15 XMM (xmm15 is scratch).

Inference. aarch64 and riscv64 fit without changing MIR or the allocator, for these reasons.

- Registers and the capability table are data. aarch64 has 31 GPRs and 32 vector registers. riscv64 has 31 and 32. Both say `two_address = false`, which the allocator already handles (`use_reg_w` with the destination as a hint instead of `tied`).
- Immediates: `imm_ok` answers per op: 12 bits (shifted or not) for aarch64 add/sub/compare, logical immediates for and/or/xor, 12 bits for riscv64 `addi`. A constant that does not fit becomes a rematerialization into a register, which exists already.
- Memory operands: neither has them, so `mem_src_ok` returns false and spilled values are reloaded by `mov` into a register. The allocator's use of `Src::Slot` is already conditional on the table. Address modes (`base + index*scale + disp`) must be legalized: aarch64 has scaled index and 12-bit scaled offsets, riscv64 has base plus 12-bit offset only. That is an emitter job: the `Addr` the allocator passes has registers for base and index, and the emitter computes `add tmp, base, index<<scale` when the mode does not exist. It needs a scratch register, hence `tmp_gpr`.
- Guards: compare-and-branch is `cmp; b.cond` on aarch64 and `bxx` on riscv64 (which has no flags, and its branch range is ±4 KiB, so the emitter must emit an inverted short branch over a `jal`). Either way the allocator calls `Target::guard` with a `Tgt` and the sink resolves the distance. `Sink::rel` returns `None` when the distance is not known yet and the sink records the fixup; a target with a short-range branch uses that path or inserts a veneer.
- Float constants: both load from a pool, as x86 does here. The pool is a sink concept (`Sink::pool`, `Tgt::Pool`), and the pc-relative load range of aarch64 (`ldr` literal, ±1 MiB) and riscv64 (`auipc` + load) is a per-target emitter detail.
- Select: `csel` on aarch64. riscv64 without Zicond needs a branch, which the emitter can build inside the `select` primitive. The allocator's `do_select` currently assumes a two-address form (destination holds the false value); a three-address `csel` target would need the destination hint instead. This is a small allocator change inside one function, not a MIR change.
- Calls are not implemented in the prototype. The target description has the caller-saved set the allocator needs for them; I have not built or tested call handling.

Inference. The part I did not prove is encodings. Instruction encoding for the other two targets is untested, so "fits" means the abstractions are not x86-shaped, not that a port was tried.

### The passes

`Compiler::compile` runs three things.

1. A prescan over every region. It builds dense per-value arrays (`reg`, `slot`, `uses`, `hint`, `def_at`, `flags`, `kbits`, `lin`), counts strong uses, flags values only snapshots use (weak), and finds the loop invariants (values defined before the header and used inside).
2. Each side, then the root, in one backward pass from the last instruction to the first. At each instruction it allocates and encodes together; no separate emission pass exists.
3. A finish step that patches the constant-pool references once the stub region has a size.

The pieces of the backward pass:

- **Registers.** The first backward use of a value gives it a register (`use_reg`, preferring the hint). Its definition frees the register (`def`) and the instruction writes it. A value nobody holds is computed into the scratch register, and a pure instruction whose result has neither register nor slot is skipped (`dead_skipped`).
- **Two-address.** On a two-address op the first operand is tied to the destination (`tied`). A last-use value takes the destination register without a copy. A live value is copied. Commutative ops swap operands when that saves the copy.
- **Eviction and spills.** When no register is free the allocator evicts the bound value with the fewest earlier uses, preferring one that already has a slot. The victim gets a slot, a reload is emitted right after the evicting instruction, and the victim is unbound for earlier code. The slot is written once, at the value's definition (`def` emits `mov [slot], reg`). A later backward use binds a register again with no reload; the definition then writes both.
- **Weak snapshot uses.** `weak_loc` never takes or evicts a register. It reports where the value is (register, constant, or slot) and, if the value has neither register nor slot, forces a slot written at the definition. Snapshot locations are read after the guard's own operands are allocated, so they match the state at the branch.
- **Constants.** `Const` emits nothing and is never bound. A use is an immediate when `imm_ok` says so, a float constant is a memory operand of the constant pool (`Src::Pool`, a rip-relative reference), and otherwise it is rematerialized into a register just before the use.
- **Loops.** The back edge is processed first. Invariants used in the loop get registers there, most used first, keeping 4 registers per class free for temporaries. Parameters get an incoming register (`lin`) that doubles as a hint for the parameter and for the value the back edge passes. The back-edge arguments are one parallel move. At the header, one parallel move takes the incoming locations to wherever the body allocation left the parameters and invariants. Both moves are usually empty.
- **Parallel moves.** `plan_moves` resolves a set of simultaneous moves: it emits the unblocked ones, breaks a cycle with `xchg` for two GPRs and with a scratch register otherwise (memory involved, or floats), and puts constant sources last. A unit test checks 5,000 random move sets against the parallel semantics.
- **Sides.** Sides are compiled before the root. Their entry locations are then known, and a prescan of the root's guards turns them into hints for the values the guards pass. A guard whose values already sit in the side's registers branches straight into the side (`side_direct`). Otherwise it branches to a trampoline in the stub region that does the moves and jumps to the side (`trampolines`).
- **Exits.** A guard's exit stub is generated when the guard is reached, from the allocation state at that point, because the state exists only then. It stores the snapshot to `exit_area[slot]`, loads `-1 - index` and jumps to the shared epilogue. The exit-area pointer is kept in frame slot 0 by the prologue, so stubs need no register for it and sides need no parameter for it.

### Code layout and branches

The final image is `[body][stubs][epilogue][body pool][stub pool]`. The body is written backward into one reusable vector; a branch to a stub is `distance from the body's top + stub offset` and needs no later patch. The stub region is built forward in a second vector.

Branches to code already emitted (a guard into a side) choose `rel8` or `rel32` from the exact distance. Branches into the stub region always use `rel32`, because the stub follows the body and the body's size is not known until the pass ends. The loop back edge and the epilogue jump are `rel32` and patched when the target is bound.

Inference. `research.md` says forward branches have known distances so that the shortest form can be picked. That holds for branches to code emitted earlier, but a guard's stub depends on allocation state at the guard, so it cannot be emitted before the body. I kept stubs out of the body and paid `rel32` on guards (6 bytes against 2). Hot-path instruction count is unaffected; code size grows by 4 bytes per guard.

## Tests

`cargo test -p oximond-mir` and `cargo clippy --locked -p oximond-mir --all-targets` pass. Tests run on x86-64 Linux; other targets compile the library and skip them.

| Test | What it proves |
|---|---|
| `tests/kernels.rs: every_kernel_matches_the_evaluator` | native result, every byte of inputs, control, exit area and data equal the evaluator, for SSE2 and AVX encodings |
| `exit_indices`, `exit_area_holds_spilled_remat_and_weak_values` | exit indices, and exit-area words for a bound, a spilled, a rematerialized and a snapshot-only value |
| `weak_snapshot_uses_do_not_take_registers` | snapshot-only values get slots (`weak_slots >= 2`), nbody's snapshots need none |
| `loop_parameters_are_hinted_so_the_back_edge_moves_nothing` | nbody: `hint_hits > 0`, `backedge_moves == 0`, `header_moves == 0` |
| `constants_are_rematerialized_not_spilled` | constants take no slot; `remats > 0` and `pool_operands > 0` |
| `spill_is_stored_once_at_the_definition_and_reloaded_often` | pressure kernel: `def_spill_stores <= slots + 1`, `reloads >= evictions` |
| `back_edge_swap_is_a_cycle_and_is_broken` | swap kernel: `cycles_broken >= 1`, result correct for odd and even counts |
| `side_entry_registers_hint_the_parents_values` | int_side: `trampolines == 0`, `side_direct == 1` |
| `plan_moves` unit test | 5,000 random parallel moves, including memory-to-memory and float registers |
| `tests/random.rs` | 1,500 fixed seeds by default (output `i` of SplitMix64 seeded `0x4d49523053454544`), both encodings; `OXIMOND_MIR_SEEDS=N` raises the count. Also asserts that evictions, weak slots, remats, side-direct jumps, trampolines and runs that end inside a side all occur |
| `the_callee_saved_check_catches_a_clobber` | the assembly shim in `exec.rs` sets distinct values in rbx, rbp, r12 to r15 before each native call and fails the call if any changed |
| `tests/allocs.rs` | a reused `Compiler` makes zero allocations in steady state, for every kernel |
| `tests/clif.rs` | the CLIF translation of every kernel and 60 random programs agrees with the evaluator, so the bench compares one program built two ways |

I also ran 40,000 seeds in release mode on both encodings once, with no difference. Bugs the tests found along the way: a missing back-edge jump, a tied copy whose source register was evicted in the same instruction, and callee-saved registers chosen as loop parameter registers but not saved in the prologue. The last one only showed in release builds of the test harness, which is why the callee-saved check exists.

## Measurements

Environment. `scripts/heavy.sh -x -m 8G crates/oximond-mir/bench/bench.sh`, one gate slot set exclusive. `uptime` load average at start `6.12, 9.27, 16.33`, at end `6.12, 9.27, 16.33` (the run took about 2 seconds). Other sessions were not gated; the load was not zero. `perf stat -e instructions:u,cycles:u` counts all threads, and the bench is single-threaded (Cranelift compiles synchronously in the bench; there is no separate compile thread). Pinned with `taskset -c 4` (SMT sibling CPU 12), under `env -i PATH=/usr/bin:/bin`. All modes are the same binary at the same path (`target/release/examples/bench`, built with `scripts/heavy.sh -m 8G cargo build --locked --release -p oximond-mir --example bench`, thin LTO as the workspace release profile).

Method. Each figure is the difference between two runs with different repeat counts divided by the difference of the counts, so process start and one-time setup cancel. Compile: 200 and 2,200 repeats for MIR, 20 and 220 for Cranelift. Run: two fuel values per kernel (nbody 2,000 and 12,000; others 20,000 and 120,000, tiny and swap 100,000 and 1,100,000). Instruction counts are exact and stable. Cycle counts are noisier. Raw output is in the command log below; to rerun:

```sh
scripts/heavy.sh -m 8G cargo build --locked --release -p oximond-mir --example bench
scripts/heavy.sh -x -m 8G crates/oximond-mir/bench/bench.sh        # CPU=4 KERNELS="tiny nbody" to narrow
```

What the Cranelift column includes: the whole per-tree path of `emit.rs`: new `JITBuilder::with_flags` (`opt_level=none`, `regalloc_algorithm=backtracking`, `enable_verifier=false`) and `JITModule`, MIR to CLIF translation, `define_function`, `finalize_definitions`. The MIR column is `Compiler::compile` on a reused compiler. Neither includes the kernel (trace IR) lowering, and neither includes the mmap and mprotect system calls (kernel instructions are not in `instructions:u`).

### Compile cost, host instructions per compile

| kernel | MIR insts | MIR compile | per MIR inst | CLIF insts | Cranelift compile | per CLIF inst | ratio | of which CLIF translation | MIR build |
|---|---|---|---|---|---|---|---|---|---|
| tiny | 10 | 7,198 | 720 | 13 | 210,854 | 16,220 | 29x | 18,377 | 399 |
| nbody | 507 | 324,864 | 641 | 561 | 7,174,700 | 12,789 | 22x | 374,178 | 12,417 |
| int_side | 29 | 17,433 | 601 | 43 | 464,357 | 10,799 | 27x | 44,079 | 1,209 |
| swap | 18 | 14,042 | 780 | 25 | 344,658 | 13,786 | 25x | 28,892 | 674 |
| pressure | 164 | 140,793 | 858 | 171 | 6,224,904 | 36,403 | 44x | 129,499 | 7,322 |
| guard_restore | 84 | 84,749 | 1,009 | 105 | 4,063,322 | 38,698 | 48x | 83,250 | 4,343 |

"MIR insts" counts every MIR instruction including `Entry`, `Const`, `Header` and `Back`. "MIR build" is the cost of the kernel's builder function, a stand-in for trace IR to MIR lowering, which this unit does not implement; "CLIF translation" is the matching stand-in for lowering to CLIF.

Goal: about 1k per MIR instruction. Measured 601 to 1,009. The two pressure kernels cost more per instruction (858, 1,009) because eviction, reloads and slot stores add work. A least-squares fit over the six kernels gives a slope of 636 per MIR instruction and an intercept of 12,084; the intercept is not trustworthy because the cost per instruction varies by kernel. The `tiny` kernel (10 instructions, 7,198 total) bounds the fixed cost of a compile from above at about 7k instructions, against about 211k for Cranelift's fixed path. Inference. Most of the MIR fixed cost is resetting the per-value arrays, the register state and the buffers on each call; I did not profile it.

Inference. The 12k per CLIF instruction the research cites is reproduced by nbody (12.8k). The two pressure kernels show 36k to 39k per CLIF instruction, so backtracking's cost depends strongly on pressure while the MIR pass changes by under 2x.

### Allocations per compile, steady state

| kernel | MIR | Cranelift (bytes) |
|---|---|---|
| tiny | 0 | 173 (78,093) |
| nbody | 0 | 448 (836,581) |
| int_side | 0 | 228 (103,841) |
| swap | 0 | 199 (95,825) |
| pressure | 0 | 367 (406,362) |
| guard_restore | 0 | 346 (247,107) |

MIR figures are after three warm-up compiles. `tests/allocs.rs` asserts zero. Cranelift's include the new `JITModule` per compile.

### Loop run cost, per iteration

| kernel | MIR instr | Cranelift instr | diff | MIR cycles | Cranelift cycles | diff |
|---|---|---|---|---|---|---|
| tiny | 5.0 | 5.0 | +0.0% | 1.0 | 0.9 | not meaningful |
| nbody | 515.0 | 476.0 | +8.2% | 234.4 | 222.0 | +5.6% |
| int_side | 26.2 | 28.3 | -7.4% | 8.2 | 8.7 | -5.7% |
| swap | 15.0 | 14.0 | +7.1% | 3.4 | 3.2 | +6.2% |
| pressure | 205.0 | 209.0 | -1.9% | 62.6 | 54.1 | +15.7% |
| guard_restore | 127.0 | 139.0 | -8.6% | 32.7 | 28.7 | +13.9% |

Both builds produce the same exit and the same checksum of exit area plus data for every kernel (`exit=0` except guard_restore, which exits through its guard 1 for both), so each run is the same computation. Goal: within about 10% of the Optimized tier. Instruction counts meet it on all six. Cycles meet it on four and miss it by 14% and 16% on pressure and guard_restore.

### Generated code for nbody

Loop body only (from the loop header to the back-edge jump, found by `bench/disasm.py`; spill means an operand based on rsp).

| | MIR | Cranelift |
|---|---|---|
| instructions in the loop | 515 | 476 |
| spill or reload instructions | 0 | 0 |
| register-to-register moves | 0 | 0 |
| code bytes, whole function | 3,277 | 3,184 |

Neither keeps anything on the stack: all values stay in XMM registers and the MIR allocator evicts nothing (`evictions: 0`). Cranelift uses AVX three-operand forms too, so it also has no copies. Pressure kernels, same measure:

| kernel | loop instructions MIR / Cranelift | rsp operands MIR / Cranelift | register moves MIR / Cranelift |
|---|---|---|---|
| pressure | 205 / 209 | 89 / 85 | 17 / 24 |
| guard_restore | 127 / 139 | 57 / 55 | 19 / 30 |
| int_side | 13 / 32 | 0 / 0 | 1 / 4 |
| swap | 15 / 14 | 0 / 0 | 5 / 5 |

Beginning of the nbody loop, MIR (left in the file as `nbody.mir.bin`, `objdump -D -b binary -mi386:x86-64 -M intel`):

```
19: cmp    rdx,0x0                          ; fuel guard, exit stub far below
1d: jle    0xc9f
23: vmovsd xmm9,[rcx]                       ; body 0 and 1 loads
27: vmovsd xmm8,[rcx+0x8]
...
59: vsubsd xmm13,xmm9,xmm11                 ; dx, dy, dz
5e: vsubsd xmm12,xmm8,xmm10
63: vsubsd xmm10,xmm6,xmm4
67: vmulsd xmm11,xmm13,xmm13
6c: vmulsd xmm14,xmm12,xmm12
71: vmulsd xmm4,xmm10,xmm10
76: vaddsd xmm11,xmm11,xmm14
7b: vaddsd xmm11,xmm11,xmm4
7f: vucomisd xmm11,[rip+0xc3e]              ; d2 > 0.0, constant from the pool
87: jbe    0xc86                            ; exit stub for guard 1
8d: vsqrtsd xmm4,xmm11,xmm11
92: vmulsd xmm11,xmm11,xmm4
96: vmovsd xmm4,[rip+0xc1f]                 ; dt, rematerialized as a dividend
9e: vdivsd xmm4,xmm4,xmm11
a3: vmulsd xmm5,xmm5,xmm4
...
```

Cranelift's same stretch is nearly identical in shape (`vsubsd`, `vmulsd`, `vaddsd`, `vucomisd` against a rip-relative constant, `vsqrtsd`, `vdivsd`) but it keeps `dt` in `xmm0` for the whole loop and folds loads into arithmetic (`vaddsd xmm11,xmm11,[rax+0x50]`).

Inference. The 39 extra instructions in nbody are 30 loads that Cranelift folds into the following `vaddsd` (a memory operand), and about 10 rematerializations of `dt`. The MIR capability table already has memory source forms (`mem_src_ok`), and the allocator uses them for spilled values and pool constants, but I did not implement load fusion (a load with one use and no store in between becomes the memory operand of its user, as LuaJIT's `asm_fuseload` does). Adding it should close most of the gap.

## Gaps and deviations

These are real and none was hidden by the numbers.

- **Cycles on pressure kernels.** Instructions are equal or lower, cycles are 14% to 16% higher. I did not investigate. Candidates: both reload invariants from the same stack slots, but the order and the use of memory operands differ; store-forwarding or dependence-chain effects. Cycle counts also carry more noise than instruction counts on this machine.
- **No load fusion**, as above.
- **No calls.** The prototype has no call op, no caller-saved clobber handling, and no helper-call ABI. Calls are the biggest remaining design item for the allocator (values live across a call must move to callee-saved registers or slots).
- **Sides end in an exit.** A side cannot rejoin the root's loop. A rejoin needs the root's header locations, which are decided after the side; that is a real ordering problem for sides-before-parents and is not solved here.
- **Float select, variable shifts, integer division** are absent. `Select` is integer only and assumes a two-address target.
- **Heuristics are crude.** Eviction picks the fewest remaining uses; invariants are pinned most-used first up to a cap. A tree with many sides and conflicting entry assignments will need more (the risk in `research.md`). In one run of the 1,500 random seeds (SSE2 only, before the AVX path existed) 1,318 of 1,472 guard-to-side jumps needed a trampoline, because random programs pass arbitrary values; the int_side kernel shows the hint working when the values line up.
- **Two encodings only for the float ops.** Integer code is the same for both.
- **Random programs are shallow.** Most end at one of the first guards: of 1,500 seeds, 679 leave through exit 0. The corpus does reach sides, spills, cycles and trampolines (the test asserts it) but it is not a substitute for real traces.
- **Memory.** The evaluator checks addresses against regions, native code does not.

## Files

- `crates/oximond-mir/` the crate.
- `crates/oximond-mir/bench/bench.sh` and `bench/disasm.py` the measurement scripts.
- `crates/oximond-mir/examples/{bench,dump,seedrun}.rs` bench driver, kernel disassembly dump, single-seed repro with MIR listing.
