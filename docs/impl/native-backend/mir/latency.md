# Time to execution: where the instructions go today, and what that asks of a MIR

Measured on commit b33edd2e (HEAD of main when this unit started), 2026-10-31. Nothing under `crates/` changed. A claim with a number, a file or a URL after it is a measurement or a citation. A paragraph or bullet that starts with "Inference." is my own reasoning from those facts. Background reading is `docs/impl/native-backend/research.md`, which recommends a shared machine IR (MIR), a backward register allocator fused with emission, and Cranelift kept as the Peak tier.

The question for this unit: between the moment a trace tree is submitted for compilation and the moment its loop has run natively, where do the host instructions go, and what does that say about how a MIR and its compiler should be built?

## Summary

- Compile is almost all of nbody's warmup. At scale 3, `algorithm/nbody` is 33.67M instructions, and its four compile submissions take 15.44M (45.9%). Cranelift's `define_function` alone takes 10.98M (32.6%). Installing the results on the VM thread takes another 2.22M (6.6%). Generated code is only about 19.6% of the run (flat samples, section "Whole run").
- The share falls fast as runs get longer: 45.9% for nbody, 23.2% for mandelbrot, 14.3% for spectral_norm, 12.8% for hash_read_indexed, 5.8% for arith_mixed. A MIR has to win on both ends, because the compile share is large only for the shortest runs.
- Cranelift is not the whole bill. Outside `define_function`, the compile spends 4.46M (13.2% of nbody) in Oximond's own code: optimizer passes 1.55M, CLIF building 0.94M, three verifier calls 0.69M, exit planning 0.51M, rebuilding the trace for every pass 0.21M, setup and glue the rest.
- Cost follows the size of the CLIF function, not a fixed price per submission. Over 149 tree compiles, `define_function` costs about 15.4k instructions per CLIF instruction and about nothing fixed (r2 0.97). The smallest compile in the suite, 48 CLIF instructions, costs 0.86M instructions. The Oximond side costs 61k fixed plus 30k per trace IR instruction after the optimizer, or 248k fixed plus 15k per instruction before it.
- Allocation is mostly Oximond's own. nbody makes 16,685 allocator calls. 4,548 of them (27%) come from `snapshot.rs` (cloning snapshots) and 2,085 (12%) from `ir.rs` (cloning `TraceIr`). Cranelift and regalloc2 together are about 1,760 (10.5%). Each compile submission makes about 1,200 to 1,700 allocations whatever its size.
- Latency in wall time is worse than the instruction counts say. The compile thread runs about 2.8G instructions a second against 8G to 18G for the interpreter on the VM thread, so a 10.4M instruction tree compile takes 4.5 ms (3.7 ms of CPU) in the background broker.
- Waiting. The first tree is a stepping stone and the VM blocks for it, so the interpreter runs only 25.8k instructions during the 4.96 ms from submission to first native entry. Side compiles are not awaited; during the first side's 1.65 ms the VM runs a median 8.6M instructions, 5.5M of them interpreter or runtime. A faster backend shortens the blocked wall time of the first tree and cuts the interpreter work during side waits in proportion to compile latency.

## What was measured, and how

Machine: AMD Ryzen 9 8945HS (16 logical CPUs, SMT siblings N and N+8), Linux 7.2.9, perf 7.3.1, glibc malloc. Other sessions use this machine, so load averages were not low. They are given next to each set of numbers below. Instruction counts are not sensitive to load. Wall times are, and they are marked.

Binaries. The measured commit built with `scripts/heavy.sh -m 16G cargo build --release -p oximond-cli` (thin LTO, line tables). Three variants of it:

- Plain: the build above. Used for `benchmarks/icount.sh` totals, for the sampled profiles and for `mallocprof.so`.
- alloc-phases: the same commit with `--features alloc-phases` in its own target directory. Its phase counts reconcile with the malloc profiler (nbody: 16,685 against 16,690; mandelbrot: 7,160 against 7,162).
- Instrumented: a scratch copy of the tree with `tools/instrument.patch` applied. It reads the instruction counter of the calling thread (`perf_event_open`, user only, own thread) at named points and prints it when `OXIMOND_JIT_TRACE` is set. It is a scratch copy because this unit may not change `crates/`.

Instruments.

- Exact stage counts. The instrumented build prints, per thread, the instruction count at every stage boundary of a compile: verify, each optimizer pass, plan_exits, CLIF building, `define_function`, `finalize_definitions`, install. The cost of printing each event is subtracted. These numbers are exact and are the basis of the stage tables. They cover everything between submission and the end of install, on whichever thread ran it.
- Sampled profiles. `perf record -e instructions:u` over many short runs. This machine throttles perf above about 2000 samples a second (`kernel.perf_event_max_sample_rate` is 6000, and above the limit the kernel keeps a few samples per millisecond and none after about 5 ms), so a normal high-rate profile only sees the start of a run. Each run therefore uses a period of one 24th of the run's instructions, with a different period per run, and 125 to 250 runs are merged (`tools/sample.sh`). Flat samples keep the exact IP and are reliable for the split into generated code, libc and Oximond code. Call-chain samples (`--call-graph dwarf`) are used to split Oximond and Cranelift code into phases, with two limits: a sample in generated code has an empty chain, and unwinding fails for about as many other samples, so 15% to 26% of the code samples cannot be given a phase.
- Allocation. `tools/mallocprof.c` is an `LD_PRELOAD` shim that counts malloc, calloc, realloc, memalign and free per call stack. `--features alloc-phases` gives the phase split the runtime itself uses.

Compile ran inline (`OXIMOND_JIT_SYNC=1`, one CPU, as `benchmarks/icount.sh` measures it) for the stage tables, and on the background worker (default broker, two CPUs) for the thread split and for latency. "VM thread" is the thread that runs the interpreter and native code. "Compile thread" is the broker's worker.

Benchmarks: `algorithm/nbody` at scale 3 as asked, and four more at scale 3 chosen for how much compile they do (compile instructions from a first pass over the whole suite): mandelbrot, spectral_norm, hash_read_indexed and arith_mixed. Three benchmarks I looked at first (`opt/field_reload`, `table/field_read`, `calls/method`) compile one tiny tree each (4% to 8% of the run) and were dropped.

Load averages (1, 5, 15 minutes). Exact stage tables and allocation: 8.11, 13.66, 19.50 at 22:24. Sampled profiles: 14.50, 14.55, 20.90 at 23:06, and 5.66 at 23:09 when they finished. Latency runs: 8.89, 9.03, 15.70 at 23:13. Totals per benchmark are `perf stat -e instructions:u` pinned with `taskset -c 4` under `env -i`, run through `scripts/heavy.sh -x`. I did not check that CPU 12 (the sibling of CPU 4) was idle. Ungated sessions were running, and the counts do not depend on it.

## Where the instructions go

### Whole run

Flat samples, compile inline, share of the run's instructions. "Code" is Oximond and Cranelift code, including the interpreter, the recorder and the compiler. `libc malloc and free` is the allocator, wherever it was called from.

| Benchmark | Run instructions | Generated code | libc malloc and free | Code |
|---|---|---|---|---|
| algorithm/nbody | 33.67M | 19.6% | 11.8% | 68.6% |
| algorithm/mandelbrot | 84.68M | 76.1% | 1.5% | 22.5% |
| algorithm/spectral_norm | 109.69M | 73.4% | 2.6% | 24.0% |
| table/hash_read_indexed | 214.01M | 80.7% | 2.2% | 17.1% |
| opt/arith_mixed | 556.46M | 97.1% | 0.1% | 2.7% |

Inference. nbody's 11.8% in malloc and free is 3.98M instructions for 16,685 allocations and about 11,500 frees, about 140 instructions per allocator call. The research note's "malloc about 13%" is the same effect measured on a slightly different mix.

### Compile and install, exact, per stage

Compile inline, one thread. Each cell is instructions and share of the run. The stages add up to the compile total exactly.

| Stage | algorithm/nbody | algorithm/mandelbrot | algorithm/spectral_norm | opt/arith_mixed | table/hash_read_indexed |
|---|---|---|---|---|---|
| Cranelift `define_function` | 10.98M (32.6%) | 15.38M (18.2%) | 12.09M (11.0%) | 24.81M (4.5%) | 24.93M (11.6%) |
| CLIF build (Lowerer, FunctionBuilder) | 0.94M (2.8%) | 1.14M (1.3%) | 1.04M (0.9%) | 1.56M (0.3%) | 1.30M (0.6%) |
| Optimizer passes | 1.55M (4.6%) | 1.51M (1.8%) | 0.91M (0.8%) | 3.45M (0.6%) | 0.35M (0.2%) |
| Verify (three call sites) | 0.69M (2.0%) | 0.52M (0.6%) | 0.39M (0.4%) | 0.88M (0.2%) | 0.24M (0.1%) |
| Draft normalise | 0.21M (0.6%) | 0.17M (0.2%) | 0.14M (0.1%) | 0.34M (0.1%) | 0.08M (0.0%) |
| plan_exits | 0.51M (1.5%) | 0.32M (0.4%) | 0.23M (0.2%) | 0.46M (0.1%) | 0.17M (0.1%) |
| JITBuilder, JITModule, declare, finalize | 0.24M (0.7%) | 0.23M (0.3%) | 0.64M (0.6%) | 0.27M (0.0%) | 0.22M (0.1%) |
| Other glue in the compile | 0.31M (0.9%) | 0.39M (0.5%) | 0.25M (0.2%) | 0.62M (0.1%) | 0.16M (0.1%) |
| **Compile total** | **15.44M (45.9%)** | **19.66M (23.2%)** | **15.69M (14.3%)** | **32.38M (5.8%)** | **27.46M (12.8%)** |
| Install (VM thread) | 2.22M (6.6%) | 0.53M (0.6%) | 0.43M (0.4%) | 0.78M (0.1%) | 0.29M (0.1%) |
| Compiles | 4 (1 tree, 3 sides) | 4 trees | 4 trees | 5 trees | 3 trees |
| Trace IR before / after optimizer | 221 / 154 | 227 / 162 | 182 / 128 | 505 / 393 | 64 / 54 |
| CLIF instructions | 991 | 1165 | 1084 | 1521 | 1387 |

"Verify" is `verify_trace` at `compile_with_plan` entry (skipped for a planned tree in release builds, which is why it is almost zero for nbody), at `optimize_pipeline` entry and at its exit (`optimize.rs`). "Draft normalise" is the `Draft::new(ir).finish()` that `optimize_pipeline` runs once before the passes. The JITBuilder row includes a one-time cost in the first compile of each process: 148k instructions for the first compile against 15k for the others (shared stubs and first-use setup).

### nbody by optimizer pass

Instructions over all rounds and all four members, share of the run, from the same exact counters.

| Pass | Instructions | Share of run |
|---|---|---|
| value numbering | 0.32M | 0.96% |
| dead code | 0.32M | 0.94% |
| copy elimination | 0.26M | 0.78% |
| guard elimination | 0.21M | 0.63% |
| constant folding | 0.20M | 0.60% |
| invariant code motion | 0.15M | 0.46% |
| load forwarding | 0.06M | 0.18% |
| concat length, substrings | 0.03M | 0.08% |

The optimizer costs about 10.4k instructions per trace IR instruction that enters it (fit below, r2 0.94, intercept near zero). The three-pass structure is why. Each pass builds a `Draft` that clones every `TraceOp` (`optimize.rs:575-586`), the pipeline runs up to four rounds of eight passes, and `Draft::finish` renumbers and rebuilds the whole trace (`optimize.rs:666`). A pass that changes nothing still pays for the `Draft`. The first round of nbody's root member changes the trace in copy elimination, folding, guard elimination, value numbering and dead code, so a second round runs.

### By thread, background compile

The same runs with the default broker (worker thread, two CPUs). Flat samples, share of all samples; instructions scaled by the inline run's total, so the thread split is the point, not the absolute figure. A compile that had not finished when the process ended is missing from the worker's share.

| Benchmark | Compile thread: code | Compile thread: malloc and free | VM thread: generated code | VM thread: code | VM thread: malloc and free |
|---|---|---|---|---|---|
| algorithm/nbody | 37.3% | 3.7% | 19.1% | 32.6% | 7.2% |
| algorithm/mandelbrot | 15.5% | 1.4% | 56.3% | 26.5% | 0.2% |
| algorithm/spectral_norm | 12.0% | 1.5% | 75.3% | 10.1% | 1.1% |
| table/hash_read_indexed | 8.6% | 0.4% | 69.1% | 21.2% | 0.7% |
| opt/arith_mixed | 3.2% | 0.2% | 90.0% | 6.5% | 0.1% |

On the VM thread, "code" is what the compile leaves behind: the interpreter, the recorder, tree planning, install and process start. For nbody it is 11.0M instructions, of which 2.22M is install. The exact counter for the nbody worker is 10.39M for the tree and 2.58M for the first side, 13.0M in all, which matches the sampled 41.0% (13.8M).

### Inside `define_function`, sampled

Cranelift's share among the samples that have a Cranelift phase (call-chain samples, allocator and library frames charged to the Cranelift caller when the chain reaches it). "DFG and other" is everything else in `cranelift_codegen`: the data-flow graph, legalization, block order, the dominator tree and so on.

| Benchmark | regalloc2 | ISLE lowering (isa and machinst) | DFG and other | Emission (MachBuffer, instruction encoding) |
|---|---|---|---|---|
| algorithm/nbody | 53% | 25% | 15% | 7% |
| algorithm/mandelbrot | 35% | 40% | 17% | 8% |
| algorithm/spectral_norm | 50% | 14% | 22% | 13% |
| table/hash_read_indexed | 28% | 31% | 30% | 11% |
| opt/arith_mixed | 16% | 30% | 25% | 29% |

The split depends on the shape of the trace, which is the expected result. nbody's trace is float arithmetic with many live values, and regalloc2 takes half. hash_read_indexed and arith_mixed are long and flat, and emission and lowering take more. Inference. Applied to nbody's exact `define_function` figure of 10.98M, regalloc2 is about 5.8M (17% of the run), ISLE about 2.7M (8%), the rest about 2.4M (7%). The sample counts for the smaller benchmarks are lower, and the percentages there carry perhaps five points of noise.

### What is not compile

From call-chain samples, share of the run. Recording is the recorder building trace IR and its first snapshots. Interpreter is everything else the VM does outside native code.

| Benchmark | Recording | Interpreter and runtime |
|---|---|---|
| algorithm/nbody | 7.1% | 7.9% |
| algorithm/spectral_norm | 4.0% | 4.2% |
| table/hash_read_indexed | 0.3% | 4.6% |
| algorithm/mandelbrot | 0.9% | 0.5% |
| opt/arith_mixed | 0.1% | 0.1% |

Planning a tree after recording (`plan_tree_with`, called from `take_queued_tree` and `seed_plan`) clones the whole `TraceIr` for every planned member. In nbody it accounts for 3,898 allocations in the malloc profile, 23% of all allocations.

## Fixed cost per submission against per trace IR instruction

Fits over 159 compiles (149 trees, 10 sides) from all 89 suite benchmarks at scale 3, compile inline, exact counters. `n_pre` is trace IR instructions before the optimizer, `n_ir` after it. Least squares of instructions on size; r2 in the last column. Trees range over 2 to 183 IR instructions and 48 to 1465 CLIF instructions.

| Quantity | Against | Fixed | Per unit | r2 |
|---|---|---|---|---|
| Whole compile | CLIF instruction | -119k | 19,200 | 0.96 |
| Whole compile | trace IR instruction after optimizer | 652k | 116,570 | 0.73 |
| Cranelift `define_function` | CLIF instruction | -275k | 15,386 | 0.97 |
| Everything but `define_function` | trace IR instruction before optimizer | 248k | 15,320 | 0.90 |
| Everything but `define_function` | trace IR instruction after optimizer | 61k | 30,276 | 0.85 |
| Optimizer | trace IR instruction before optimizer | -15k | 10,384 | 0.94 |
| CLIF build | CLIF instruction | 25k | 888 | 1.00 |

Sides (10 compiles, 6 to 22 IR instructions): whole compile 37k fixed plus 12.2k per CLIF instruction (r2 0.97); everything but `define_function` 190k plus 9.1k per instruction before the optimizer (r2 0.82).

What the fit says.

- The negative intercepts are not real. Cranelift's `define_function` has no measurable fixed cost per function in this range. The smallest tree in the suite has 48 CLIF instructions, 11 IR instructions, and costs 862,739 instructions in total, of which 540k is `define_function`. It is a floor set by the minimum trace (entry checks, loop header, back edge, one exit stub per guard), not by a setup cost.
- The costs that do not depend on trace size are small. Setting up the JIT builder, the module and the declarations is about 25k instructions per compile (15k, 1k and 9k), plus 134k once per process. The median tree compile costs 2.94M and the median side 1.22M.
- The number of CLIF instructions per trace IR instruction after the optimizer ranges from 2.4 to 35.6 with a median of 7.1 (6.4 for nbody, 25.7 for hash_read_indexed, whose probes expand inline), which is why `n_ir` alone explains only 73% of the variance and CLIF size explains 96%.
- Inference. A MIR that keeps the expansion in a lowering table instead of generating CLIF can price a compile in trace IR instructions, and if it spends 1k to 3k host instructions per post-optimizer instruction (research.md's order of magnitude is 1k), nbody's 154 instructions compile in about 0.15M to 0.5M instead of 15.4M. That is a target to be tested by the prototype, not a measurement.

## Allocation

### By phase, and per trace IR instruction

From `--features alloc-phases` (phase guards in the runtime), compile inline. Each alloc, alloc_zeroed and realloc counts once.

| Benchmark | Total | Recording | Optimization | Codegen | Installation | Compile per submission | Compile per IR instruction after optimizer | Compile per CLIF instruction |
|---|---|---|---|---|---|---|---|---|
| algorithm/nbody | 16,685 | 6,346 | 2,228 | 3,046 | 3,322 | 1,318 | 34.2 | 5.3 |
| algorithm/mandelbrot | 7,160 | 841 | 1,921 | 3,109 | 548 | 1,258 | 31.0 | 4.3 |
| algorithm/spectral_norm | 9,891 | 2,589 | 1,588 | 3,294 | 545 | 1,220 | 38.1 | 4.5 |
| opt/arith_mixed | 12,478 | 2,003 | 4,265 | 4,373 | 1,021 | 1,728 | 22.0 | 5.7 |
| table/hash_read_indexed | 13,528 | 1,261 | 807 | 3,144 | 357 | 1,317 | 73.2 | 2.9 |

"Compile" is optimization plus codegen. Allocations per trace IR instruction (after the optimizer) are 22 to 73, and 17 to 62 per instruction before it. The per-submission figure is nearly constant at 1,200 to 1,700 while the trace IR size ranges from 54 to 393, so a large part of the allocation count is fixed per submission. nbody's installation phase makes 830 allocations per install (3,322 for four), which `native.rs` `new_skipping` and snapshot clones account for.

### By caller

`tools/mallocprof.c` shim, compile inline, first non-library frame of each allocation's stack (symbolized with `addr2line -i`). Allocations by source file, nbody.

| Source file | Allocations | What it is |
|---|---|---|
| `jit_ir/trace/snapshot.rs` | 4,548 | `Snapshot::clone`, from `TraceIr::clone` in `plan_tree_with`, in the recorder, in `Draft::finish`, and at install |
| `jit_ir/trace/ir.rs` | 2,085 | `TraceIr::clone` (`ir.rs:2298-2319`) |
| `vm/jit/trace/recorder.rs` | 1,563 | recorder finalize and snapshot building |
| `vm/jit/trace/native.rs` | 1,381 | `new_skipping` at install (`native.rs:4159`, `:4309`, `:4310`) |
| `jit_ir/trace/optimize.rs` | 1,286 | `Draft::new` and `finish` (`:586`, `:587`, `:595`, `:718`, `:907`) |
| `jit/trace/compile.rs` | 891 | closures building `LoweredMember` and exit data |
| `jit_ir/trace/tree.rs` | 589 | `plan_tree` windows |
| `cranelift-entity` (`map.rs`, `primary.rs`, `list.rs`) | 260 + 169 + 70 | CLIF's entity maps |
| `regalloc2` (`ion/process.rs`, `liveranges.rs`) | 175 + 83 | live ranges and allocation state |
| `cranelift-codegen` (`machinst/vcode.rs`, `lower.rs`, `ir/extfunc.rs`) | 87 + 81 + 96 | VCode and lowering |

mandelbrot differs: `optimize.rs` is first with 1,350 allocations, then `compile.rs` 622, `native.rs` 572 and `snapshot.rs` 450. arith_mixed has `optimize.rs` 2,657, `snapshot.rs` 1,992, `native.rs` 1,066. In every benchmark the largest callers are Oximond's own passes and clones, not Cranelift. Full tables are in the output of `malloc_report.py` (see "Regenerating the numbers").

Inference. Snapshots are the biggest single cost: each `Snapshot` owns `Vec`s (`snapshot.rs:168`, `:206`), and the tree is cloned whole several times. A MIR whose snapshots are ranges into one flat array removes these clones outright.

## What a MIR pipeline would remove, fuse or keep

Ranked by share of nbody's run (exact counters; the sampled splits are marked). Each row says what the MIR pipeline does with the cost and why. "Remove" means no equivalent exists in the MIR pipeline. "Fuse" means the work stays but moves into another pass. "Keep" means the work stays as it is.

| Rank | Cost today | Share of nbody run | Verdict | Reason |
|---|---|---|---|---|
| 1 | Cranelift regalloc2 (sampled) | about 17% | Remove | Replaced by one backward allocation pass fused with emission. The fastalloc experiment in research.md shows a fast allocator must keep values in registers across guards and the back edge. |
| 2 | Cranelift ISLE lowering (sampled) | about 8% | Remove | Instruction selection becomes a table lookup per MIR op and operand form. |
| 3 | Install on the VM thread | 6.6% | Keep, shrink | The VM-side install stays, but its cost is snapshot clones and `new_skipping` allocations (3,322 allocations). Exit metadata should be a view over the MIR's snapshot array. |
| 4 | Cranelift DFG and other (sampled) | about 5% | Remove | CLIF's data-flow graph, legalization and block order have no counterpart. |
| 5 | Optimizer: copy elimination, constant folding, guard elimination, value numbering | 2.97% | Fuse | One forward pass while building MIR: copy propagation, folding, guard dedupe and CSE are what LuaJIT's FOLD and CSE do while recording. No separate `Draft`. |
| 6 | CLIF build (Lowerer and FunctionBuilder) | 2.8% | Replace | The trace IR to MIR lowering is the same work, minus the builder's bookkeeping. This is the pass that has to be written fast. |
| 7 | Cranelift emission (sampled) | about 2% | Fuse | Emission happens in the backward allocator pass. |
| 8 | Verify at three call sites | 2.0% | Remove two, keep one | Keep one structural check at the trace IR boundary in debug builds. `compile_with_plan` already skips the first in release for planned trees. |
| 9 | plan_exits | 1.5% | Fuse | Exit descriptors and the exit area layout come from the MIR snapshot array when the exit stubs are emitted. |
| 10 | Dead-code elimination | 0.94% | Remove | The backward emitter skips an unreferenced, effect-free instruction (as `lj_asm.c:2572` does). |
| 11 | JITBuilder, JITModule, declarations, finalize | 0.7% | Remove | One code region per process, carved up per compile. No ISA or settings object per submission. |
| 12 | Draft normalise (`Draft::new().finish()`) | 0.62% | Remove | No `Draft` exists in the MIR pipeline. |
| 13 | Optimizer: invariant code motion | 0.46% | Keep | Needs a preheader. Runs while lowering the loop. |
| 14 | Optimizer: load forwarding | 0.18% | Keep | Needs memory aliasing rules. Fuse into the forward pass if it fits. |
| 15 | Optimizer: concat length and substrings | 0.08% | Keep | Rare and cheap. |
| outside the backend | Recording and `plan_tree` clones | about 7.8% | Keep, shrink | Not part of the backend, but `plan_tree_with` makes 23% of nbody's allocations. A MIR that snapshots as ranges makes the clone a copy of two arrays. |

The rows that are removed or fused add up to about 40% of nbody's run (ranks 1, 2, 4, 5, 7, 9, 10, 11 and 12, and two of the three verify calls), almost all of the compile. Rank 3 and the last row are VM-side and remain work for the MIR's neighbours, not for the backend.

Inference. The ranking is stable across the five benchmarks for the passes (the same five passes lead) and for Cranelift as a whole (the biggest row), but the split inside Cranelift is not (table above). A MIR design that assumes regalloc2 is the largest Cranelift cost would be right for float-heavy numeric traces and wrong for hash-probe and long arithmetic traces, where lowering and emission matter as much.

## IR data layout for compile speed

Sources are primary: the compilers' own code or their authors' posts. Line numbers for LuaJIT are in the `luajit-src` 210.7.4 crate (LuaJIT commit c6ffc14, `luajit2/src/`). Line numbers for V8 and WebKit are in the `main` branch copy fetched on 2026-10-31 and will drift. Claims from blog posts say so.

### What each compiler does

LuaJIT, the closest relative: a trace compiler with snapshots and a backward allocator.

- One instruction is 8 bytes: two 16 bit operand references, a type and opcode pair, and a 16 bit `prev` chain that the allocator overwrites with its register and spill bytes (`lj_ir.h:555-580`). Allocation results live in the instruction, so no side table exists.
- References are 16 bit offsets from a bias. Instructions grow up from `REF_BIAS`, constants grow down from it, in the same array (`lj_ir.h:464-469`). A constant is an operand like any other and costs no instruction.
- The recorder writes into one growable buffer that every trace reuses, and only a finished trace is copied into an exact-size object (`lj_trace.c:52-60`, `:450-460`, `:123-155`).
- Assembly state is a single `ASMState` struct with fixed arrays sized by register count (`lj_asm.c:41-130`).
- Machine code grows downward (`mcp`, `lj_asm.c:44`). The driver is one backward loop over the instructions (`lj_asm.c:2565`). There is one limit check per instruction against a fixed red zone (`MCLIM_REDZONE`, `lj_asm.c:139-150`), not a check per emitted byte.
- Dead code costs nothing extra. An instruction that no register or spill slot was allocated for, and that has no side effect, is skipped (`lj_asm.c:2572`; the comment is "Dead-code elimination can be soooo easy").
- A snapshot is a 12 byte header plus an array of 4 byte entries (`lj_jit.h:180-193`). A pair of Bloom filters over snapshot references answers "does a snapshot need this value" without a set (`lj_asm.c:77`).
- Duplicate lookup uses one skip-list chain per opcode through the `prev` field (`IRRef1 chain[IR__MAX]`, `lj_jit.h:478`), not a hash table.

V8 Maglev, an optimizing tier built to be about 10x faster to compile than TurboFan.

- Control-flow-graph SSA, chosen over TurboFan's "more flexible but cache unfriendly sea-of-nodes representation" (https://v8.dev/blog/maglev, blog claim).
- A node's input pointers are stored in the zone directly before the node, so input `i` is `this - sizeof(ptr) * (i + 1)`. A node is one bump allocation (`maglev-ir.h:2000-2010`, `:2450-2490`, `NodeBase::Allocate`).
- One field of a node is a union of the owning block in the front end and a pointer to register allocation info in the back end (`union { owner_; regalloc_info_; }`, `maglev-ir.h` near `:2385`). Fields are reused between phases.
- Register allocation is a prepass for live ranges followed by "a single forward walk over the graph". A value that has to spill "is retroactively told to immediately spill on definition" (blog; the code is `maglev-regalloc.cc:1303-1310`, `AllocateSpillSlot` at `:1736`). That is the same spill-once-at-definition rule LuaJIT uses.
- Work moves into graph building, "as much as possible immediately during graph building" instead of building a generic graph and lowering it later (blog). Maglev's compile time is "roughly 10x slower than Sparkplug, and 10x faster than TurboFan" (blog).

V8 Sparkplug and Liftoff, baseline tiers with no IR.

- Sparkplug "doesn't generate any intermediate representation", compiles "in a single linear pass over the bytecode", and "the entire compiler is a switch statement inside a for loop" (https://v8.dev/blog/sparkplug).
- Liftoff "avoids the time and memory overhead of constructing an IR" and generates machine code in one pass over the Wasm bytecode, with a virtual operand stack in which constants are not materialized until used (https://v8.dev/blog/liftoff).

JavaScriptCore, Air (the machine IR under B3) and its O0 path.

- Air's `Inst` is passed by value with inline storage for three arguments, so "a basic block usually contains a single contiguous slab of memory that completely describes all of its instructions". Procedures are arrays of blocks, values keep no use lists, and edits go through an `InsertionSet` applied in one linear pass per block (https://webkit.org/blog/5852/introducing-the-b3-jit-compiler/, blog claim; `Source/JavaScriptCore/b3/air/AirInst.h`).
- At O0 the pipeline skips CFG simplification, dead code elimination and the optimizing allocator. It runs `lowerAfterRegAlloc`, `lowerEntrySwitch`, `optimizeBlockOrder`, then a combined `GenerateAndAllocateRegisters` object that allocates while it emits (`AirGenerate.cpp:77-95`). The optimizing path runs `simplifyCFG`, `eliminateDeadCode` and `allocateRegistersByGreedy` as separate passes (`AirGenerate.cpp:107-115`).
- B3 as a whole aimed at being "about ten times faster than LLVM" and measured 4.7x on JetStream hot functions (same blog post).

Wasmtime Winch, the baseline compiler beside Cranelift.

- "Single pass over Wasm bytecode", "no intermediate representation", single-pass register allocation with a value stack that tracks locals, registers, constants and memory offsets. The RFC estimates 15x to 20x faster compilation and code 1.1x to 1.5x slower than the optimizing compiler (https://github.com/bytecodealliance/rfcs/blob/main/accepted/wasmtime-baseline-compilation.md; the RFC marks both as approximate).

Makarov's MIR.

- Typed variables in a register IR with no hard registers, and no SSA in the generator pipeline because "building and destroying this form is expensive". Immediates are reduced to moves, and instructions are rewritten into the two-operand forms x86 can encode before allocation (https://developers.redhat.com/blog/2020/01/20/mir-a-lightweight-jit-compiler-project).
- The author's measurement on one benchmark (sieve): MIR-gen compiles in 75 microseconds against 13.35 ms for `gcc -O2` (about 180x), and the generated code is 6% slower. One benchmark is thin evidence, but the order of magnitude agrees with Winch's estimate.

### Oximond's trace IR today, measured

- `TraceInstruction` is 64 bytes, `TraceOp` 40 bytes, `ValueInfo` 24 bytes, `Snapshot` 136 bytes, `TraceIr` 576 bytes (printed by the instrumented build). `TraceOp::CallHelper` owns a `Vec<ValueId>` (`ir.rs:1501-1503`), and every `Snapshot` owns `Vec`s, so cloning a trace clones the heap as well.
- Constants are instructions (`TraceOp::Constant`, `ir.rs:1120`), created by folding (`optimize.rs:1124`) and then numbered and sometimes deleted.
- Value numbering uses a std `HashMap`/`HashSet` with the default SipHash (`optimize.rs:1734`, `:1748`; `sip::Hasher` shows in the profile).
- Every pass starts from `Draft::new`, which clones each op (`optimize.rs:575-586`), and ends in `Draft::finish`, which renumbers values and snapshots and allocates new vectors (`optimize.rs:666-910`).

### Rules for MIR

Inference. These rules follow from the table above and from the measurements in the earlier sections. Each names the cost it removes.

1. One array of fixed-size instructions, 16 bytes each, indexed by `u32`. Opcode and type in one word, two inline operands, one auxiliary word. A call with many arguments keeps them in a side array of `u32` and the instruction holds an index and a count. LuaJIT gets 8 bytes with 16 bit references. 32 bit references cost a little more space and remove LuaJIT's limit on trace size. Today's `TraceInstruction` is 64 bytes with an owned `Vec` in some variants.
2. Constants are operands, not instructions. A constant pool grows down from a bias, as in LuaJIT, or lives in a second array, and an operand tag says which. This removes the `Constant` instructions that folding creates and numbering and DCE then clean up.
3. Allocation results live in the instruction. A register byte and a spill-slot field overlay fields the allocator no longer needs, as in LuaJIT's `r` and `s` and Maglev's owner and regalloc union. No per-value side `Vec` or map.
4. One arena for everything a compile creates, reset between compiles and kept per compile thread. Capacity persists, so a steady-state compile calls the system allocator zero times. This is the direct fix for the 1,200 to 1,700 allocations per submission and the 11.8% malloc share in nbody.
5. No per-instruction owned data. Snapshots are a header array plus one flat `u32` entry array (LuaJIT's `SnapShot` and `SnapEntry`). A guard stores a snapshot index. The VM-side exit and install data reference ranges of the arrays instead of cloning them. This removes the 4,548 snapshot allocations in nbody.
6. Passes edit in place or not at all. No `Draft`. Instructions are never renumbered after construction, deletion is a tombstone opcode, and replacement rewrites an operand entry (Air avoids use lists the same way). This removes the per-pass clone and the 10.4k instructions per IR instruction the optimizer costs now.
7. The emitter is the last pass and does the dead-code elimination: skip an instruction that has no register, no spill and no side effect (`lj_asm.c:2572`). A separate DCE pass is not needed.
8. Size buffers once per compile from the instruction count. The code buffer comes from a bound per instruction, exit stubs from the snapshot count, with one limit check per instruction (`checkmclim`).
9. No hash maps, `BTreeMap`, `String` or `Box<dyn>` on the compile path. Where a lookup is needed (value numbering), use a fixed-size open-addressed table of `u32` keyed by a hash of opcode and operands and cleared with a generation counter. LuaJIT avoids even that and searches a per-opcode chain (`lj_jit.h:478`).
10. No per-compile module, ISA or settings object. The JIT builder that is rebuilt for each submission today (`emit.rs:1369-1461`) has no MIR equivalent. The code region is allocated once per process and carved up.
11. Fold what can be folded while building MIR. A forward builder that applies copy propagation, folding, guard dedupe and value numbering as it appends instructions replaces four passes (2.97% of nbody's run).

## Submission to first native entry

Latency is time from the VM's call to `submit_trace_planned` or `submit_side` (`completions.rs:123,129`) to the first call into the installed code on the VM thread. The instrumented build timestamps each step. Wall times are noisy under load, so this section shows medians and the range over 10 runs per benchmark.

### nbody at scale 3, background broker

10 runs, two CPUs, default broker (`tools/latruns.sh`, load average 8.89 at the start). Medians, with minimum to maximum in brackets.

| Compile | Awaited by the VM | Queue wait | Compile wall (CPU) | Wake after compile end | Install | Submission to first native entry |
|---|---|---|---|---|---|---|
| Root tree, Optimized | yes | 131 us (0 to 180) | 4,532 us (3,815 to 7,921), CPU 3,746 us | 88 us | 187 us | 4,963 us (4,245 to 8,138) |
| Side 1, Optimized | no | 8 us | 1,381 us (1,180 to 2,484), CPU 969 us | 38 us | 140 us | 1,653 us (1,419 to 2,687) to install |
| Side 2, Baseline | no | 6 us | 628 us | 45 us | 129 us | finished in 1 of 10 runs |

The run ends at about 75 ms, so the later sides often are not finished when it does.

- The tree compile has 10.39M instructions and takes 4.5 ms of wall time and 3.7 ms of CPU on the worker, so the worker runs at about 2.8G instructions a second. The VM thread runs the interpreter at 8G to 18G a second (see the table of other benchmarks below). Inference. A compile thread is cold: the code is large and branchy, the caches belong to another core, and a new malloc arena has to fault in its pages. An instruction count under-states compile latency by about the same factor.
- The inline compile of the same tree takes 3.6 ms, so a worker thread does not speed up the first tree. It cannot, because the VM blocks for it.
- Wake-up (a futex wake of an idle worker) is 131 us, and the VM's pickup after the compile ends is 88 us for the tree. Together that is 4.4% of the tree's 4.96 ms.
- The install of the root takes 187 us and 0.65M instructions on the VM thread.

### Interpreter instructions while waiting, nbody

The VM thread's instruction counter, split into the time inside native trace entries (generated code and the helpers it calls) and the rest (interpreter, recorder, runtime).

| Wait | VM instructions during it | Native | Interpreter and runtime |
|---|---|---|---|
| Root tree: submission to install (VM blocked in `wait_trace`) | 25.8k | 0 | 25.8k |
| Root tree: submission to first native entry, includes the install | 0.67M | 0 | 0.67M |
| Side 1: submission to install, background | 8.56M (0.21M to 16.8M) | 3.07M | 5.48M |
| Side 1: submission to install, 4 runs at scale 10 | 6.59M (2.37M to 12.8M) | 2.35M | 4.24M |

- The root is a stepping stone. `awaits_result` is true for a baseline or retree and the VM waits in `wait_trace` for up to 100 ms (`STEPPING_STONE_WAIT`, `completions.rs:14`, `:160-175`). The interpreter therefore runs 25.8k instructions, 0.08% of the run, however slow the compile is. What a slower compile costs here is blocked wall time, not interpreter work.
- Sides are not awaited (`awaits_result` returns false for `CompileTarget::Side`). While a side compiles the VM keeps running its tree natively and interprets side exits. Over the 1.65 ms until the first side is installed, the VM executes a median 8.56M instructions, 5.48M of them interpreter or runtime. That is 16% of the whole run's instructions, and about 3.3M interpreter or runtime instructions per millisecond of compile latency.

Inference. Does a faster backend change how much interpreter time runs while waiting? Two cases.

1. For an awaited compile (the first tree), no. The VM thread blocks, so interpreter instructions during the wait stay at about 26k. What improves is blocked wall time: if a MIR compile took 10x fewer instructions, the 4.96 ms from submission to first native entry would fall to somewhere near 0.9 ms (0.22 ms of queue wait and wake-up and 0.19 ms of install stay, and the compile falls from 4.5 ms to about 0.45 ms). Queue wait, wake-up and install would then be about half of what is left, so those need work too.
2. For a side compile that nobody awaits, yes, roughly in proportion. At nbody's rate of 3.3M interpreter and runtime instructions per millisecond of wait, cutting the first side's compile from 1.38 ms to 0.15 ms would cut the interpreter and runtime instructions during the wait from 5.5M to about 0.7M, and the run reaches the side's native code sooner. This is the effect that matters for the "MIR must win on both" goal, because it shifts work from interpreter instructions to compiled instructions.

### Other benchmarks, background broker

The same measurement for the four other benchmarks (10 runs each). Medians. "Interpreter and runtime to entry" is the VM thread's non-native instructions from submission to the first native entry.

| Benchmark | First compile | Awaited | Queue | Compile wall | Submission to entry | Interpreter and runtime to entry |
|---|---|---|---|---|---|---|
| algorithm/mandelbrot | tree, Optimized | no | 148 us | 1,490 us | 1,757 us | 13.6M |
| algorithm/spectral_norm | tree, Baseline | yes | 122 us | 1,076 us | 1,356 us | 64.7k |
| opt/arith_mixed | tree, Optimized | no | 135 us | 1,815 us | 2,028 us | 32.7M |
| table/hash_read_indexed | tree, Optimized | no | 206 us | 1,779 us | 2,247 us | 40.5M |

These first trees are not awaited, so the VM interprets while they compile: 13.6M to 40.5M instructions of interpreter and runtime work in 1.8 to 2.2 ms, a rate of 8M to 18M instructions per millisecond. Inference. The VM thread retires 8G to 18G instructions a second in the interpreter and the compile thread about 2.8G. Each compile instruction saved therefore removes about three to six interpreter instructions of waiting, whenever the VM is not blocked.

## Regenerating the numbers

All commands run from the worktree root. Anything that measures goes through the gate with `-x`. Scratch output goes under `~/.cache/agent-tmp/mir-research/latency/`. The tool scripts are in `docs/impl/native-backend/mir/tools/`. Take the commit with `git rev-parse HEAD`; the numbers here are for b33edd2e.

Builds.

```sh
S=~/.cache/agent-tmp/mir-research/latency; mkdir -p $S
scripts/heavy.sh -m 16G cargo build --release -p oximond-cli
scripts/heavy.sh -m 16G env CARGO_TARGET_DIR=$PWD/target-ap cargo build --release -p oximond-cli --features alloc-phases
cp target-ap/release/oximond $S/oximond-ap
gcc -O1 -g -shared -fPIC -o $S/mallocprof.so docs/impl/native-backend/mir/tools/mallocprof.c -ldl
# Instrumented scratch copy (this unit does not change crates/):
rsync -a --exclude target --exclude target-ap --exclude .git --exclude docs/impl ./ $S/scratch-src/
W=$PWD; (cd $S/scratch-src && patch -p1 < $W/docs/impl/native-backend/mir/tools/instrument.patch)
scripts/heavy.sh -m 16G env CARGO_TARGET_DIR=$S/scratch-target cargo build --release --manifest-path $S/scratch-src/Cargo.toml -p oximond-cli
cp $S/scratch-target/release/oximond $S/oximond-instr
```

Raw measurements. Each script takes an output directory and, where noted, environment variables.

```sh
# Exact stage counters, malloc profile, alloc-phases, perf-stat totals, per-benchmark icount, suite-wide fit logs.
PLAIN=$PWD/target/release/oximond INSTR=$S/oximond-instr AP=$S/oximond-ap SO=$S/mallocprof.so \
  scripts/heavy.sh -x -m 16G docs/impl/native-backend/mir/tools/collect.sh $S/out/run4
# Sampled profiles (flat and call chain), 250 runs for nbody and 125 for the others.
PLAIN=$PWD/target/release/oximond \
  scripts/heavy.sh -x -m 16G docs/impl/native-backend/mir/tools/sample.sh $S/out/samples2
# Background-broker latency runs.
INSTR=$S/oximond-instr \
  scripts/heavy.sh -x -m 16G docs/impl/native-backend/mir/tools/latruns.sh $S/out/lat4
```

`benchmarks/icount.sh -b target/release/oximond -r tracing,interp algorithm/nbody` gives the headline totals without the rest. `collect.sh` calls it.

Turning raw data into the tables.

```sh
T=docs/impl/native-backend/mir/tools
# Convert perf data (not a measurement, runs outside the gate).
for f in $S/out/samples2/*.bg/*.data $S/out/samples2/*.sync/*.data; do perf script -i $f -F comm,pid,tid,time,period,ip,sym,dso > $f.txt; done
for f in $S/out/samples2/*.bg/*.flat $S/out/samples2/*.sync/*.flat; do perf script -i $f -F pid,tid,time,period,ip,sym,dso > $f.txt; done
# Stage tables (exact), per benchmark; TOTAL is the benchmark's .total file.
python3 $T/report.py $S/out/run4/algorithm_nbody.latsync.log $(cat $S/out/run4/algorithm_nbody.total)
# Fixed and per-instruction fit over the suite.
python3 $T/events.py --fit $S/out/run4/fit/*.log > $S/out/results/fit.csv; python3 $T/fit.py $S/out/results/fit.csv
# Whole-run categories and phase split from samples; add --by-thread for the background runs.
python3 $T/combine.py --total $(cat $S/out/samples2/algorithm_nbody.total) \
  --flat "$S/out/samples2/algorithm_nbody.sync/*.flat.txt" --dwarf "$S/out/samples2/algorithm_nbody.sync/*.data.txt"
# Allocation per phase and per IR instruction; allocation by caller.
python3 $T/allocs.py $S/out/run4/algorithm_nbody.alloc_phases.json $S/out/run4/algorithm_nbody.latsync.log
python3 $T/malloc_report.py $S/out/run4/algorithm_nbody.malloc.txt --top 40
# Latency medians over repeated background runs.
python3 $T/latsummary.py $S/out/lat4/algorithm_nbody.s3.bg*.log
python3 $T/events.py --latency $S/out/lat4/algorithm_nbody.s3.sync.log
```

Tests. `python3 -m unittest docs/impl/native-backend/mir/tools/test_tools.py` checks the parsers, the classifier and the fit, and checks this document against the acceptance criteria of the unit.

## Limits and open questions

- The profile is one commit on one machine under load. Instruction counts are stable to a few hundred between runs (nbody measured 33,667,463 and 33,667,976 in two runs), but the share of the Cranelift internals varies with trace shape (table above) and each benchmark here compiles only three to five trees.
- 15% to 26% of call-chain samples inside Oximond and Cranelift code (26% for nbody) could not be given a phase (truncated unwinding in libc and inlined frames). The per-phase split inside `define_function` and for recording and interpreter rests on the rest. The exact stage counters do not have this limit and carry the compile table.
- The instrumented build changes what it measures a little: it prints to stderr, which costs wall time on both threads. Printing is subtracted from instruction counts, not from wall times. An earlier version that printed on every VM safepoint inflated a side compile from 1.4 ms to 13 ms by lock contention on stderr, which is why the final latency runs print only around submissions.
- Latency wall times are medians of ten runs on a loaded machine. The ranges are wide (4.2 to 8.1 ms for the tree). The ordering of the stages is stable and the instruction counts are exact.
- The native count includes the helpers that generated code calls, so "native" means "inside a native trace entry". Time in the runtime between entries counts as interpreter and runtime.
- Open. How much of the 1,200 to 1,700 allocations per submission are the same ones across compiles (an arena would remove all of them). Whether the `nbody` side compile would be awaited if it were the first compile of a loop. What the compile thread's 2.8G instructions a second becomes with a warm arena and a smaller code footprint; the prototype in the next unit will show it.
