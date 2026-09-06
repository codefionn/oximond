# Trace op census and CLIF expansion

Unit: census. Measured at commit `509c816d` (branch `flow/mir-research-census`, on top of `b33edd2e`). Machine: 16 CPUs, shared, load average 19 (1 minute) when the collection was queued and 4 when it finished (the gate held it behind other jobs for about 20 minutes). Collection counts static IR and CLIF, not time, so load does not move the numbers except through compile timing (see Caveats). Statements marked "Inference." are my reasoning, not measurements.

## What was measured

Every entry of `benchmarks/suite.json` (89 benchmarks, the set `benchmarks/run.py` runs as `oximond-tracing`) at scale 3, `--execution=tracing`, release build, default tier policy. The suite has no named "tracing" subset, so the census takes all of it. That includes nbody, the table and string benchmarks, the Mineclonia kernels and the `opt/` microbenchmarks.

The numbers come from a new statistics dump, `OXIMOND_JIT_CENSUS=1`, added in `crates/oximond/src/jit/trace/census.rs`. No existing facility gave per-op counts, snapshot percentiles, live values or CLIF per op (op-stats keeps sums only). With the variable unset the lowerer pays one `OnceLock` load per lowered operation and generated code is unchanged. Per native compile it prints one `jit-census: {json}` line on stderr:

- per member (root, then sides): preheader and body length, loop parameters, values, operations by variant (with the `op:` sub-kind for arithmetic and numeric ops), peak register pressure, and one row per guard;
- per compile: the tier, the CLIF total of the function, and the CLIF instructions each operation kind lowered to.

Definitions:

- Tree: one root trace id. The last non-detached compile of a root counts, because a tier upgrade or a new side recompiles the whole tree. "Compiles" counts every compile.
- Sides: members after the root in the final compile.
- Guard row: window entries (all registers of all snapshot frames), then entry kinds (value, const, unchanged, root, virtual, slice), distinct non-constant SSA values the snapshot names, values live across the guard through real uses, and snapshot values that only the snapshot keeps alive (the "weak" uses).
- Live across a guard: a non-constant value defined before the guard whose last real use is after it. An input or preheader value read in the body of a member that ends in `LoopBack` counts as live to the back edge. A loop parameter is redefined at the header every iteration, so it is live from the header to its last read, and to the back edge only when a `LoopBack` argument passes it on. Inputs of a member that does not loop end at their last read. Constants are excluded because MIR rematerializes them.
- Peak pressure: the maximum count of simultaneously live non-constant values over the member, integer class (gpr) and float class (fpr), with the same liveness. It counts SSA values, not the table binding words the lowerer also keeps in registers.
- CLIF per op: the CLIF instructions emitted between entering and leaving `Lowerer::lower_instruction`, children excluded (a side lowered inside its parent's instruction is charged to the ops inside the side). It is measured as growth of `dfg.num_insts()`. The "CLIF per op" column divides by the number of lowerings, which exceeds the IR count: for `Arithmetic.Add` the lowerer ran 877 times for 653 IR instructions, so the lowerer emits some ops more than once (loop versioning or preheader copies; Inference, not traced).

Reproduce (the coordinator can rerun):

```sh
scripts/heavy.sh -m 16G cargo build --release -p oximond-cli
scripts/heavy.sh -m 16G python3 benchmarks/trace_census.py collect OUT_DIR
python3 benchmarks/trace_census.py report OUT_DIR --ops algorithm/nbody algorithm/sieve string/byte_sub table/hash_read algorithm/spectral_norm
python3 -m unittest benchmarks/test_trace_census.py
```

`collect` options: `--bin`, `--scale` (default 3), `--timeout`, and benchmark names to restrict the run. `report` regenerates every table below. The raw data of this run is not committed (it is about 1 MB); rerun `collect`.

## Caveats

- The suite benchmarks at scale 3 are short. 100 trees and 3284 IR instructions in total, a mean root of 26 IR instructions (p95 79, max 155) and a mean side of 15. Compile threads run asynchronously, so which trees finish compiling before the program ends varies a little between runs. `string/pattern_gsub` and `string/pattern_hot` compiled nothing, `startup/empty` nothing, as expected (pattern loops wait for 20000 back edges).
- No dynamic weighting. The IR counts are static. Inference: the body of a root runs once per iteration and the preheader once, so static counts of root bodies approximate dynamic ones. Sides are weighted by how often their guard fails, which this census does not know. A dynamic count needs a per-op execution counter in the native code, which would change generated code and is out of scope here.
- Op share is by count of IR instructions, not by cost. A single `ArraySet` costs 85 CLIF instructions on average and a `Constant` costs one.
- Tier mix of final trees: 73 Optimized, 22 Peak, 5 Baseline. CLIF counts depend on tier, and this run does not split by tier.

## Findings in short

1. Trees are small and mostly arithmetic. Arithmetic, Constant, GuardBranch and Compare are 83% of IR instructions. 12 of 32 variants seen reach 95%, 21 reach 99%.
2. Most CLIF comes from a few expensive ops. 71.5% of all CLIF is attributed to operations. The rest (28.5%) is entry checks, loop frame, exit stubs and the epilogue, outside any op. `GuardBranch` (3455), `CallHelper` (2606), `Arithmetic` (2503), `ArrayLoad` (2237) and `HashLoad` (1432) lead. Per op, `ArraySet` costs 85, `HashLoad` 57, `ArrayLoad` 24, `CallHelper` 24, `GuardBranch` 7.2, `Arithmetic` 1.6.
3. Modulo is the one arithmetic op that is not 1:1. `Arithmetic.Modulo` is 6.9% of IR (226 instructions, mostly in call-heavy benchmarks: control/nested_calls 20, opt/call_args 19, mineclonia/module_calls 19) and lowers to 4.0 CLIF on average.
4. Snapshots are wide and mostly cold. A guard's snapshot has a mean window of 30.8 entries (p95 53, max 86), but only 7.4 are values (p95 22, max 30). 6.1 are constants and 17.2 are `Unchanged`, which cost nothing at the exit. Root, virtual and slice entries are rare (0.1, 0.03, 0.01 on average).
5. Weak uses matter. A guard's snapshot names 7.0 distinct non-constant values on average (p95 22, max 26), and 3.6 of them (p95 16, max 21) are kept alive only by the snapshot. A mean of 6.5 values (p95 15, max 21) are live across a guard through real uses. A backward allocator with weak snapshot uses can spill or drop those instead of pinning registers.
6. Register pressure fits x86-64 for gpr and almost always for fpr. Over 146 members, peak gpr averages 3.3 (p95 10, max 16) and peak fpr averages 6.6 (p95 13, max 17). Two roots exceed 16 floats (calls/metamethod_ops and mineclonia/module_calls, 17 each). nbody's root and opt/cold_recovery reach exactly 16, and mandelbrot 15. Spilling must work for floats, but it is rare. An earlier version of this census counted loop parameters live to the end of the member and reported nbody at 17 and a max of 19. Loop parameters per member: mean 2.7, p95 8, max 12.
7. Guard count is high relative to ops: 807 guards for 3284 IR instructions, so one instruction in four is a guard (counting the guards inside `ArrayLoad`, `HashLoad` and similar ops, which carry snapshots too).

## Measured tables

Benchmarks: 89. Trees (last compile per root): 100, sides: 46, trace IR instructions: 3284, guards: 807, tiers: {'Optimized': 73, 'Peak': 22, 'Baseline': 5}.

### Per benchmark

| benchmark | compiles | trees | sides | IR insts | guards | window entries/guard (mean/p95/max) | value entries/guard | live across guard | peak gpr/fpr | CLIF | CLIF/IR |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|
| algorithm/binary_trees | 4 | 2 | 2 | 31 | 16 | 11.0 / 11 / 11 | 3.6 / 7 / 7 | 1.4 / 3 / 3 | 2/3 | 555 | 17.9 |
| algorithm/mandelbrot | 5 | 1 | 3 | 58 | 9 | 39.0 / 39 / 39 | 13.0 / 17 / 17 | 5.2 / 14 / 14 | 1/15 | 413 | 7.1 |
| algorithm/nbody | 3 | 1 | 3 | 98 | 42 | 37.0 / 37 / 37 | 10.1 / 17 / 18 | 8.5 / 16 / 17 | 3/16 | 569 | 5.8 |
| algorithm/sieve | 3 | 2 | 1 | 26 | 10 | 21.0 / 21 / 21 | 6.1 / 13 / 13 | 3.0 / 5 / 5 | 3/6 | 533 | 20.5 |
| algorithm/spectral_norm | 4 | 2 | 4 | 82 | 22 | 28.0 / 28 / 28 | 9.4 / 14 / 14 | 2.3 / 8 / 8 | 3/10 | 850 | 10.4 |
| allocation/ctor_array | 1 | 1 | 0 | 12 | 2 | 16.0 / 16 / 16 | 1.0 / 2 / 2 | 3.0 / 4 / 4 | 1/6 | 59 | 4.9 |
| allocation/ctor_fields | 1 | 1 | 0 | 11 | 2 | 15.0 / 15 / 15 | 1.0 / 2 / 2 | 3.0 / 4 / 4 | 1/7 | 61 | 5.5 |
| allocation/ctor_grow | 1 | 1 | 0 | 20 | 4 | 23.0 / 23 / 23 | 1.8 / 5 / 5 | 4.5 / 7 / 7 | 3/7 | 104 | 5.2 |
| allocation/ctor_mixed | 1 | 1 | 0 | 13 | 2 | 18.0 / 18 / 18 | 1.0 / 2 / 2 | 3.0 / 4 / 4 | 1/7 | 63 | 4.8 |
| allocation/ctor_side_exit | 1 | 1 | 0 | 17 | 3 | 18.0 / 18 / 18 | 1.7 / 3 / 3 | 4.0 / 6 / 6 | 1/7 | 78 | 4.6 |
| allocation/gc_cycles | 1 | 1 | 0 | 17 | 7 | 22.0 / 22 / 22 | 3.0 / 5 / 5 | 6.0 / 8 / 8 | 3/6 | 197 | 11.6 |
| allocation/table_churn | 1 | 1 | 0 | 10 | 2 | 14.0 / 14 / 14 | 1.0 / 2 / 2 | 3.0 / 4 / 4 | 1/6 | 57 | 5.7 |
| calls/builtin | 1 | 1 | 0 | 11 | 3 | 13.0 / 13 / 13 | 0.7 / 2 / 2 | 2.7 / 4 / 4 | 2/6 | 61 | 5.5 |
| calls/closure | 1 | 1 | 0 | 12 | 3 | 15.0 / 15 / 15 | 1.0 / 3 / 3 | 2.7 / 4 / 4 | 2/6 | 68 | 5.7 |
| calls/function | 1 | 1 | 0 | 13 | 3 | 12.0 / 12 / 12 | 0.3 / 1 / 1 | 2.7 / 4 / 4 | 2/5 | 61 | 4.7 |
| calls/metamethod_call | 1 | 1 | 0 | 23 | 3 | 19.0 / 19 / 19 | 0.7 / 2 / 2 | 3.7 / 5 / 5 | 1/9 | 88 | 3.8 |
| calls/metamethod_ops | 1 | 1 | 0 | 130 | 52 | 42.7 / 54 / 54 | 8.8 / 15 / 18 | 15.9 / 21 / 21 | 9/17 | 805 | 6.2 |
| calls/method | 1 | 1 | 0 | 12 | 2 | 17.0 / 17 / 17 | 1.0 / 2 / 2 | 4.0 / 5 / 5 | 2/7 | 63 | 5.2 |
| calls/multiple_returns | 1 | 1 | 0 | 36 | 4 | 23.0 / 23 / 23 | 0.5 / 2 / 2 | 3.0 / 5 / 5 | 3/9 | 106 | 2.9 |
| calls/protected | 3 | 1 | 1 | 48 | 12 | 21.7 / 29 / 29 | 6.4 / 14 / 14 | 3.8 / 7 / 7 | 4/7 | 180 | 3.8 |
| calls/recursive_fib | 2 | 1 | 1 | 9 | 3 | 9.0 / 9 / 9 | 2.7 / 4 / 4 | 1.0 / 1 / 1 | 1/3 | 85 | 9.4 |
| calls/tail_call | 2 | 2 | 0 | 44 | 8 | 13.0 / 13 / 13 | 1.5 / 3 / 3 | 2.0 / 3 / 3 | 4/4 | 170 | 3.9 |
| calls/varargs | 1 | 1 | 0 | 18 | 3 | 22.0 / 22 / 22 | 0.7 / 2 / 2 | 2.7 / 4 / 4 | 2/6 | 67 | 3.7 |
| calls/varargs_forward | 1 | 1 | 0 | 21 | 4 | 23.0 / 23 / 23 | 0.5 / 2 / 2 | 2.8 / 4 / 4 | 3/6 | 77 | 3.7 |
| control/branching | 3 | 1 | 1 | 23 | 5 | 14.0 / 14 / 14 | 4.8 / 7 / 7 | 3.2 / 6 / 6 | 1/7 | 185 | 8.0 |
| control/coroutine_pingpong | 1 | 1 | 0 | 22 | 6 | 24.0 / 24 / 24 | 2.2 / 7 / 7 | 4.3 / 6 / 6 | 4/7 | 149 | 6.8 |
| control/loop_in_call | 2 | 1 | 1 | 21 | 5 | 17.0 / 17 / 17 | 6.4 / 7 / 7 | 2.4 / 6 / 6 | 1/8 | 195 | 9.3 |
| control/nested_calls | 5 | 3 | 1 | 112 | 12 | 22.0 / 22 / 22 | 3.5 / 7 / 7 | 3.8 / 5 / 5 | 2/7 | 508 | 4.5 |
| control/nested_loops | 2 | 1 | 1 | 16 | 4 | 20.0 / 20 / 20 | 5.8 / 7 / 7 | 2.2 / 6 / 6 | 1/8 | 188 | 11.8 |
| control/short_inner | 2 | 1 | 1 | 25 | 6 | 22.0 / 22 / 22 | 8.0 / 9 / 9 | 2.5 / 9 / 9 | 2/10 | 308 | 12.3 |
| mineclonia/bit_wide | 1 | 1 | 0 | 47 | 6 | 27.0 / 27 / 27 | 0.3 / 2 / 2 | 3.3 / 5 / 5 | 6/5 | 139 | 3.0 |
| mineclonia/captured_call | 1 | 1 | 0 | 64 | 17 | 70.2 / 86 / 86 | 4.8 / 16 / 16 | 9.9 / 13 / 13 | 10/11 | 286 | 4.5 |
| mineclonia/module_calls | 1 | 1 | 0 | 155 | 36 | 40.2 / 58 / 58 | 11.0 / 30 / 30 | 14.1 / 16 / 16 | 12/17 | 461 | 3.0 |
| mineclonia/open_results | 2 | 2 | 0 | 75 | 17 | 28.4 / 34 / 34 | 2.1 / 11 / 11 | 5.0 / 8 / 8 | 7/8 | 325 | 4.3 |
| mineclonia/ull_arith | 1 | 1 | 0 | 39 | 5 | 19.0 / 19 / 19 | 2.6 / 4 / 4 | 5.4 / 7 / 7 | 7/6 | 156 | 4.0 |
| mineclonia/ull_kernel | 1 | 1 | 0 | 153 | 40 | 40.5 / 55 / 55 | 9.9 / 20 / 21 | 9.0 / 14 / 16 | 16/6 | 573 | 3.7 |
| numeric/for_sum | 2 | 1 | 0 | 7 | 2 | 11.0 / 11 / 11 | 1.0 / 2 / 2 | 3.0 / 4 / 4 | 1/6 | 49 | 7.0 |
| numeric/while_mix | 2 | 1 | 0 | 11 | 1 | 12.0 / 12 / 12 | 2.0 / 2 / 2 | 3.0 / 3 / 3 | 1/5 | 48 | 4.4 |
| opt/alias_holder | 1 | 1 | 0 | 20 | 2 | 25.0 / 25 / 25 | 1.0 / 2 / 2 | 8.0 / 9 / 9 | 3/9 | 81 | 4.0 |
| opt/arith_mixed | 5 | 1 | 3 | 107 | 18 | 19.0 / 19 / 19 | 7.3 / 10 / 10 | 3.3 / 9 / 9 | 11/5 | 414 | 3.9 |
| opt/arith_stable | 2 | 1 | 0 | 25 | 2 | 17.0 / 17 / 17 | 2.0 / 4 / 4 | 4.5 / 7 / 7 | 1/9 | 84 | 3.4 |
| opt/call_args | 2 | 2 | 0 | 127 | 19 | 49.1 / 53 / 53 | 3.8 / 10 / 10 | 5.4 / 8 / 8 | 6/14 | 478 | 3.8 |
| opt/call_args_hook | 2 | 2 | 0 | 127 | 19 | 50.1 / 54 / 54 | 3.8 / 10 / 10 | 5.4 / 8 / 8 | 6/14 | 497 | 3.9 |
| opt/closure_family | 2 | 1 | 0 | 24 | 8 | 25.5 / 29 / 29 | 11.2 / 15 / 15 | 11.5 / 13 / 13 | 9/7 | 205 | 8.5 |
| opt/closure_protos | 9 | 1 | 7 | 183 | 56 | 39.6 / 43 / 43 | 21.7 / 26 / 26 | 5.4 / 13 / 13 | 9/7 | 1465 | 8.0 |
| opt/cold_recovery | 3 | 1 | 1 | 42 | 5 | 29.0 / 29 / 29 | 9.4 / 15 / 15 | 6.0 / 10 / 10 | 1/16 | 244 | 5.8 |
| opt/dead_slots | 1 | 1 | 0 | 48 | 3 | 14.0 / 14 / 14 | 0.7 / 2 / 2 | 2.7 / 4 / 4 | 2/13 | 148 | 3.1 |
| opt/field_reload | 1 | 1 | 0 | 21 | 2 | 22.0 / 22 / 22 | 1.0 / 2 / 2 | 6.0 / 7 / 7 | 2/8 | 76 | 3.6 |
| opt/invariant_field | 1 | 1 | 0 | 16 | 2 | 18.0 / 18 / 18 | 1.0 / 2 / 2 | 6.0 / 7 / 7 | 1/9 | 71 | 4.4 |
| opt/mod_float | 2 | 1 | 0 | 24 | 4 | 16.0 / 16 / 16 | 2.5 / 5 / 5 | 5.5 / 7 / 7 | 8/4 | 123 | 5.1 |
| opt/mod_int | 2 | 1 | 0 | 30 | 5 | 19.0 / 19 / 19 | 3.8 / 6 / 6 | 7.2 / 10 / 10 | 11/0 | 135 | 4.5 |
| opt/mod_mixed | 3 | 1 | 1 | 42 | 9 | 17.0 / 17 / 17 | 4.6 / 8 / 8 | 5.1 / 10 / 10 | 8/4 | 229 | 5.5 |
| opt/range_dense | 2 | 1 | 0 | 16 | 5 | 21.0 / 21 / 21 | 2.6 / 5 / 5 | 6.2 / 8 / 8 | 10/0 | 154 | 9.6 |
| opt/range_fail | 5 | 1 | 3 | 44 | 12 | 23.0 / 23 / 23 | 9.9 / 15 / 15 | 2.8 / 9 / 9 | 2/10 | 615 | 14.0 |
| opt/range_float_index | 3 | 1 | 1 | 21 | 7 | 21.0 / 21 / 21 | 7.3 / 9 / 9 | 3.1 / 8 / 8 | 2/9 | 296 | 14.1 |
| opt/range_offset | 2 | 1 | 0 | 20 | 6 | 30.0 / 30 / 30 | 3.2 / 5 / 5 | 6.7 / 8 / 8 | 3/8 | 298 | 14.9 |
| opt/retained_slots | 2 | 1 | 0 | 55 | 17 | 35.1 / 39 / 39 | 5.8 / 11 / 11 | 10.6 / 13 / 13 | 7/9 | 472 | 8.6 |
| opt/store_load | 1 | 1 | 0 | 29 | 5 | 22.0 / 22 / 22 | 2.0 / 4 / 4 | 6.4 / 9 / 9 | 4/8 | 132 | 4.6 |
| startup/empty | 0 | 0 | 0 | 0 | 0 | - | - | - | 0/0 | 0 | - |
| stdlib/math | 1 | 1 | 0 | 17 | 2 | 25.0 / 25 / 25 | 1.0 / 2 / 2 | 3.0 / 4 / 4 | 1/8 | 71 | 4.2 |
| stdlib/native_calls | 1 | 1 | 0 | 23 | 5 | 27.0 / 27 / 27 | 0.4 / 2 / 2 | 4.4 / 7 / 7 | 2/8 | 154 | 6.7 |
| stdlib/sort | 1 | 1 | 0 | 11 | 3 | 21.0 / 21 / 21 | 1.7 / 3 / 3 | 4.3 / 5 / 5 | 2/6 | 136 | 12.4 |
| stdlib/tonumber_parse | 1 | 1 | 0 | 30 | 10 | 29.0 / 29 / 29 | 4.7 / 9 / 9 | 7.8 / 10 / 10 | 6/7 | 236 | 7.9 |
| string/byte_sub | 1 | 1 | 0 | 47 | 15 | 28.0 / 28 / 28 | 5.1 / 12 / 12 | 6.3 / 9 / 9 | 4/10 | 275 | 5.9 |
| string/format | 1 | 1 | 0 | 16 | 4 | 22.0 / 22 / 22 | 3.2 / 7 / 7 | 3.5 / 4 / 4 | 1/7 | 131 | 8.2 |
| string/naive_concat | 2 | 1 | 1 | 23 | 7 | 28.0 / 28 / 28 | 7.9 / 11 / 11 | 3.3 / 7 / 7 | 3/8 | 223 | 9.7 |
| string/pattern_gsub | 0 | 0 | 0 | 0 | 0 | - | - | - | 0/0 | 0 | - |
| string/pattern_hot | 0 | 0 | 0 | 0 | 0 | - | - | - | 0/0 | 0 | - |
| string/pattern_many | 2 | 2 | 0 | 10 | 3 | 65.7 / 66 / 66 | 2.0 / 3 / 3 | 1.0 / 2 / 2 | 3/2 | 91 | 9.1 |
| string/pattern_run | 1 | 1 | 0 | 30 | 8 | 24.0 / 24 / 24 | 5.5 / 8 / 8 | 6.4 / 7 / 7 | 4/8 | 212 | 7.1 |
| string/plain_find | 1 | 1 | 0 | 13 | 3 | 24.0 / 24 / 24 | 0.3 / 1 / 1 | 3.3 / 5 / 5 | 2/6 | 79 | 6.1 |
| string/table_concat | 1 | 1 | 0 | 14 | 5 | 22.0 / 22 / 22 | 1.6 / 4 / 4 | 3.8 / 5 / 5 | 3/5 | 152 | 10.9 |
| string/utf8_pack | 3 | 2 | 1 | 118 | 33 | 45.0 / 45 / 45 | 11.2 / 23 / 23 | 7.2 / 15 / 15 | 15/4 | 799 | 6.8 |
| table/array_read | 3 | 1 | 1 | 14 | 5 | 20.0 / 20 / 20 | 6.4 / 7 / 7 | 2.4 / 7 / 7 | 2/8 | 261 | 18.6 |
| table/array_write | 2 | 1 | 0 | 14 | 3 | 13.0 / 13 / 13 | 1.7 / 4 / 4 | 3.7 / 4 / 4 | 2/6 | 137 | 9.8 |
| table/field_read | 1 | 1 | 0 | 17 | 2 | 21.0 / 21 / 21 | 1.0 / 2 / 2 | 7.0 / 8 / 8 | 1/10 | 70 | 4.1 |
| table/hash_churn | 1 | 1 | 0 | 15 | 5 | 17.0 / 17 / 17 | 2.2 / 4 / 4 | 6.4 / 8 / 8 | 3/7 | 197 | 13.1 |
| table/hash_iteration | 4 | 2 | 2 | 58 | 24 | 27.0 / 27 / 27 | 12.5 / 15 / 15 | 4.9 / 9 / 9 | 8/6 | 754 | 13.0 |
| table/hash_read | 2 | 1 | 1 | 15 | 6 | 22.0 / 22 / 22 | 7.0 / 8 / 8 | 3.8 / 8 / 8 | 3/8 | 386 | 25.7 |
| table/hash_read_indexed | 3 | 1 | 2 | 27 | 10 | 24.0 / 24 / 24 | 10.9 / 16 / 16 | 3.0 / 9 / 9 | 3/9 | 598 | 22.1 |
| table/hash_read_shaped | 2 | 2 | 0 | 47 | 22 | 22.0 / 22 / 22 | 3.1 / 4 / 4 | 5.6 / 6 / 6 | 3/6 | 825 | 17.6 |
| table/hash_read_small | 3 | 1 | 1 | 15 | 6 | 22.0 / 22 / 22 | 7.0 / 8 / 8 | 3.8 / 8 / 8 | 3/8 | 402 | 26.8 |
| table/hash_write | 1 | 1 | 0 | 14 | 4 | 17.0 / 17 / 17 | 2.0 / 4 / 4 | 6.0 / 7 / 7 | 3/7 | 158 | 11.3 |
| table/heap_redundancy | 1 | 1 | 0 | 32 | 2 | 24.0 / 24 / 24 | 1.0 / 2 / 2 | 9.0 / 10 / 10 | 2/12 | 94 | 2.9 |
| table/insert_remove | 1 | 1 | 0 | 22 | 4 | 23.0 / 23 / 23 | 3.8 / 7 / 7 | 6.8 / 8 / 8 | 2/9 | 127 | 5.8 |
| table/iteration | 2 | 1 | 0 | 7 | 2 | 25.0 / 25 / 25 | 1.0 / 2 / 2 | 2.0 / 3 / 3 | 3/4 | 70 | 10.0 |
| table/metamethod_function | 2 | 1 | 1 | 27 | 11 | 20.0 / 20 / 20 | 7.3 / 11 / 11 | 5.5 / 9 / 9 | 4/7 | 466 | 17.3 |
| table/metamethod_index | 1 | 1 | 0 | 13 | 2 | 18.0 / 18 / 18 | 1.0 / 2 / 2 | 5.0 / 6 / 6 | 1/8 | 64 | 4.9 |
| table/metamethod_newindex | 2 | 2 | 0 | 32 | 12 | 18.7 / 22 / 22 | 3.0 / 6 / 6 | 7.7 / 9 / 9 | 4/7 | 381 | 11.9 |
| all | 166 | 100 | 46 | 3284 | 807 | 30.8 / 53 / 86 | 7.4 / 22 / 30 | 6.5 / 15 / 21 | 16/17 | 22706 | 6.9 |

### Guard snapshot detail, all benchmarks (mean / p95 / max)

| field | mean / p95 / max |
|---|---|
| window | 30.8 / 53 / 86 |
| value | 7.4 / 22 / 30 |
| const | 6.1 / 15 / 31 |
| unchanged | 17.2 / 45 / 64 |
| root | 0.1 / 0 / 10 |
| virtual | 0.0 / 0 / 3 |
| slice | 0.0 / 0 / 2 |
| snap_values | 7.0 / 22 / 26 |
| live_across | 6.5 / 15 / 21 |
| weak_extra | 3.6 / 16 / 21 |

### All benchmarks, by TraceOp

| TraceOp | IR count | share | cumulative | CLIF emitted | CLIF per op |
|---|---:|---:|---:|---:|---:|
| Arithmetic | 1145 | 34.9% | 34.9% | 2503 | 1.6 |
| Constant | 719 | 21.9% | 56.8% | 750 | 1.0 |
| GuardBranch | 441 | 13.4% | 70.2% | 3455 | 7.2 |
| Compare | 429 | 13.1% | 83.3% | 468 | 1.0 |
| CallHelper | 111 | 3.4% | 86.6% | 2606 | 23.5 |
| ArrayLoad | 81 | 2.5% | 89.1% | 2237 | 23.5 |
| ArrayStore | 49 | 1.5% | 90.6% | 267 | 5.1 |
| IntToFloat | 47 | 1.4% | 92.0% | 47 | 1.0 |
| TableFieldLoad | 37 | 1.1% | 93.1% | 319 | 8.6 |
| FieldLoad | 32 | 1.0% | 94.1% | 32 | 1.0 |
| MetaGuard | 26 | 0.8% | 94.9% | 254 | 9.8 |
| ToWide | 22 | 0.7% | 95.6% | 198 | 9.0 |
| HashLoad | 21 | 0.6% | 96.2% | 1432 | 57.3 |
| Numeric | 20 | 0.6% | 96.8% | 20 | 1.0 |
| FieldStore | 14 | 0.4% | 97.3% | 14 | 1.0 |
| PhantomTable | 14 | 0.4% | 97.7% | 84 | 6.0 |
| ReleaseStrings | 10 | 0.3% | 98.0% | 126 | 12.6 |
| TableFieldStore | 10 | 0.3% | 98.3% | 21 | 2.1 |
| HostCall | 9 | 0.3% | 98.6% | 309 | 34.3 |
| ArraySet | 8 | 0.2% | 98.8% | 677 | 84.6 |
| HelperResult | 7 | 0.2% | 99.0% | 0 | 0.0 |
| NativeUnary | 6 | 0.2% | 99.2% | 10 | 1.7 |
| SelfCall | 6 | 0.2% | 99.4% | 166 | 27.7 |
| StringByte | 6 | 0.2% | 99.6% | 139 | 23.2 |
| NativeBinary | 4 | 0.1% | 99.7% | 8 | 2.0 |
| NumberLoad | 2 | 0.1% | 99.8% | 58 | 29.0 |
| NumberTag | 2 | 0.1% | 99.8% | 0 | 0.0 |
| TableLength | 2 | 0.1% | 99.9% | 0 | 0.0 |
| BoolAnd | 1 | 0.0% | 99.9% | 1 | 1.0 |
| Negate | 1 | 0.0% | 99.9% | 1 | 1.0 |
| Not | 1 | 0.0% | 100.0% | 2 | 1.0 |
| NumberModulo | 1 | 0.0% | 100.0% | 36 | 36.0 |

CLIF attributed to operations: 16240 of 22706 (71.5%); the rest is entry, loop, exit and stub code outside any operation.

Variants for 95% of trace IR instructions: 12. For 99%: 21. Of 32 variants seen.
With each benchmark weighted equally: 95% needs 11 variants, 99% needs 20.

### algorithm/nbody, by TraceOp

| TraceOp | IR count | share | cumulative | CLIF emitted | CLIF per op |
|---|---:|---:|---:|---:|---:|
| Arithmetic | 37 | 37.8% | 37.8% | 37 | 1.0 |
| TableFieldLoad | 20 | 20.4% | 58.2% | 93 | 4.7 |
| Compare | 10 | 10.2% | 68.4% | 10 | 1.0 |
| GuardBranch | 10 | 10.2% | 78.6% | 89 | 8.9 |
| TableFieldStore | 9 | 9.2% | 87.8% | 9 | 1.0 |
| Constant | 6 | 6.1% | 93.9% | 6 | 1.0 |
| ArrayLoad | 3 | 3.1% | 96.9% | 179 | 59.7 |
| TableLength | 2 | 2.0% | 99.0% | 0 | 0.0 |
| NativeUnary | 1 | 1.0% | 100.0% | 1 | 1.0 |

### algorithm/sieve, by TraceOp

| TraceOp | IR count | share | cumulative | CLIF emitted | CLIF per op |
|---|---:|---:|---:|---:|---:|
| GuardBranch | 7 | 26.9% | 26.9% | 68 | 5.7 |
| Compare | 6 | 23.1% | 50.0% | 10 | 1.0 |
| Arithmetic | 5 | 19.2% | 69.2% | 17 | 1.0 |
| Constant | 4 | 15.4% | 84.6% | 6 | 1.0 |
| ArrayLoad | 2 | 7.7% | 92.3% | 152 | 38.0 |
| ArraySet | 1 | 3.8% | 96.2% | 137 | 137.0 |
| Not | 1 | 3.8% | 100.0% | 2 | 1.0 |

### string/byte_sub, by TraceOp

| TraceOp | IR count | share | cumulative | CLIF emitted | CLIF per op |
|---|---:|---:|---:|---:|---:|
| Arithmetic | 12 | 25.5% | 25.5% | 15 | 1.2 |
| Compare | 10 | 21.3% | 46.8% | 10 | 1.0 |
| GuardBranch | 10 | 21.3% | 68.1% | 74 | 7.4 |
| Constant | 9 | 19.1% | 87.2% | 9 | 1.0 |
| StringByte | 5 | 10.6% | 97.9% | 106 | 21.2 |
| ReleaseStrings | 1 | 2.1% | 100.0% | 12 | 12.0 |

### table/hash_read, by TraceOp

| TraceOp | IR count | share | cumulative | CLIF emitted | CLIF per op |
|---|---:|---:|---:|---:|---:|
| Compare | 4 | 26.7% | 26.7% | 7 | 1.0 |
| GuardBranch | 4 | 26.7% | 53.3% | 90 | 12.9 |
| Arithmetic | 3 | 20.0% | 73.3% | 6 | 1.0 |
| Constant | 2 | 13.3% | 86.7% | 3 | 1.0 |
| ArrayLoad | 1 | 6.7% | 93.3% | 45 | 22.5 |
| HashLoad | 1 | 6.7% | 100.0% | 118 | 59.0 |

### algorithm/spectral_norm, by TraceOp

| TraceOp | IR count | share | cumulative | CLIF emitted | CLIF per op |
|---|---:|---:|---:|---:|---:|
| Arithmetic | 30 | 36.6% | 36.6% | 42 | 1.0 |
| Compare | 16 | 19.5% | 56.1% | 16 | 1.0 |
| GuardBranch | 16 | 19.5% | 75.6% | 136 | 8.5 |
| Constant | 14 | 17.1% | 92.7% | 14 | 1.0 |
| ArrayLoad | 2 | 2.4% | 95.1% | 68 | 34.0 |
| ArraySet | 2 | 2.4% | 97.6% | 220 | 110.0 |
| HashLoad | 2 | 2.4% | 100.0% | 58 | 29.0 |

## Proposed MIR op set

Operands are register, immediate or memory per the per-target capability table. "Helper" means a call with the native C ABI to Rust code (`extern "C"`), as the lowerer already uses. Sketches are Inference unless a count is quoted. The IR docs for each op are in `crates/oximond/src/jit_ir/trace/ir.rs`.

MIR core (all targets):

| MIR op | meaning |
|---|---|
| `mov dst, src` | register move, also immediate and spill reload |
| `ld.{i8,i32,i64,f64} dst, [base + index*scale + disp]` | load |
| `st.{i8,i32,i64,f64} [addr], src` | store |
| `add/sub/mul/and/or/xor/shl/shr/sar.{i64}` | integer ALU, register/immediate/memory second operand |
| `fadd/fsub/fmul/fdiv`, `fneg`, `fabs`, `fsqrt`, `ffloor`, `fceil` | f64 ALU, capability table says which exist (`roundsd` needs SSE4.1, else a helper) |
| `cvt.i2f`, `cvt.f2i` | conversions |
| `cmp.{cc}` then `setcc`/`select` | compare producing a boolean, or fused into a guard |
| `select cc, a, b` | conditional move (cmov, csel) |
| `guard.cmp cc, a, b, snap` | compare and branch to the exit stub of snapshot `snap` |
| `guard.tag`, `guard.bits` | compare a loaded tag or mask against an immediate and branch to an exit |
| `call target, args... -> ret` | native C ABI call, clobbers caller-saved registers |
| `loop.back params...` | back edge with parallel move |
| `ret.exit k` | return the exit index |

Lowering per TraceOp, in descending share. "Count" is IR instructions across all 89 benchmarks, "CLIF/op" is the current CLIF per lowering from the all-benchmark table above.

| TraceOp | Count | CLIF/op | MIR lowering sketch | Helper |
|---|---:|---:|---|---|
| Arithmetic (Add, Subtract, Multiply, Divide, BitAnd, BitOr, shifts) | 917 | 1.0 to 1.3 | one `add/sub/mul/fadd/...`; integer ops are wrapping, float ops use the SSE/NEON form | no |
| Constant | 719 | 1.0 | no instruction: an immediate operand, or rematerialized `mov imm` at the use; f64 constants load from a constant pool or `mov` through a gpr | no |
| GuardBranch | 441 | 7.2 | `guard.cmp` fused with the producing `Compare` when the boolean has one use; otherwise `test` plus guard. The 7.2 CLIF include the exit stub emission, which MIR places after the body | no |
| Compare | 429 | 1.0 | fuse into the next `guard.cmp` or `select`; `setcc` only when the boolean escapes | no |
| Arithmetic.Modulo and ModuloWith | 228 | 4.0 | integer path: guard divisor non-zero, `idiv`-based floor modulo or the existing float path; float path: `fmod`-style inline sequence or helper `fmod53` (the lowerer already imports out-of-line `%` helpers) | float case yes, integer case no |
| CallHelper | 111 | 23.5 | `call` of a prepared helper, then `HelperResult` reads the return register; arguments are registers and immediates, results may need an exit-area store | yes, by definition |
| ArrayLoad | 81 | 23.5 | bounds compare against the bound array length word, `guard.cmp`, element load, tag compare for `expected`, `guard.tag`; payload load with `ld.f64` or `ld.i64`. About 5 MIR ops against 24 CLIF | no |
| ArrayStore | 49 | 5.1 | bounds and tag guards as for loads, then `st` of payload | no |
| IntToFloat | 47 | 1.0 | `cvt.i2f` | no |
| TableFieldLoad | 37 | 8.6 | load the row word, compare the key slot, `guard.cmp`, load the payload (`ld`) | no |
| FieldLoad | 32 | 1.0 | one `ld` from the bound field word | no |
| MetaGuard | 26 | 9.8 | load the metatable identity from the table row, `guard.cmp` against the pinned immediate | no |
| ToWide | 22 | 9.0 | `cvt.f2i` with range guard and sign handling, or a reinterpret `mov` between Int64 and UInt64 | no |
| HashLoad | 21 | 57.3 | the lookup is a read-only VM call, so MIR `call` of the lookup helper, then tag guard and payload load. 57 CLIF per op today is mostly argument setup and result decoding | yes |
| Numeric (Floor, Abs) | 20 | 1.0 | `ffloor`, `fabs` | no, or a helper where the target lacks `roundsd` |
| FieldStore | 14 | 1.0 | one `st` | no |
| PhantomTable | 14 | 6.0 | add constants into the run's table and byte counters (two `ld`/`add`/`st` on the control block) | no |
| ReleaseStrings | 10 | 12.6 | `call` of a helper that drops pending string refs | yes |
| TableFieldStore | 10 | 2.1 | `guard.cmp` on the slot, `st` | no |
| HostCall | 9 | 34.3 | `call` through the prepared host entry, then check the result code with `guard.cmp` | yes |
| ArraySet | 8 | 84.6 | in-place overwrite path (bounds, tag guard, `st`) inline; append, pop and everything else through a helper, then reload every array binding word | yes, cold path |
| HelperResult | 7 | 0.0 | no instruction: names the return register of the `call` | no |
| NativeUnary / NativeBinary | 10 | 1.7 to 2.0 | `call` of the pure native function with xmm arguments, so every live value must survive in a callee-saved register or a spill slot | yes |
| SelfCall | 6 | 27.7 | `call` of the trace's own body with an ABI of its own | yes |
| StringByte | 6 | 23.2 | bounds `guard.cmp` on the bound length, `ld.i8` with zero extend, `cvt.i2f` if the profile wants a float | no |
| NumberLoad, NumberTag | 4 | 29.0, 0.0 | as `ArrayLoad` with a tag test that accepts float or int, `NumberTag` reads that tag | no |
| TableLength | 2 | 0.0 | `ld` of the length binding word (constant for the run in most traces, so no code) | no |
| BoolAnd, Not, Negate | 3 | 1.0 | `and`, `xor 1`, `fneg` or `neg` | no |
| NumberModulo | 1 | 36.0 | as `Arithmetic.Modulo` with the float input form | float case yes |
| Move | 0 seen | 0 | `mov`, usually coalesced away | no |
| Select | 0 seen | n/a | `select` | no |
| Truthy | 0 seen | n/a | tag compare against nil and false, `select` | no |
| GuardType | 0 seen | n/a | `guard.tag` | no |
| LaneArrayLoad | 0 seen | n/a | as `ArrayLoad` with one more load for the row's table address | no |
| StringLength | 0 seen | n/a | `ld` of the bound length (constant for the run) | no |
| Read, Effect, ArrayNext | 0 seen | n/a | not in this census: `ArrayNext` is an iteration step (`ld` plus bounds `guard.cmp`), `Read` and `Effect` are ordering markers with no code of their own | `Effect` may call a helper |

The ops marked "0 seen" did not occur in any final tree at scale 3. Inference: the optimizer folds `Move` away and entry guards replace most `GuardType`, but the recorder can still produce them, so a larger workload such as the Mineclonia emerge may show them. They are cheap to lower, so a prototype can reject them and fall back to Cranelift at first (Inference).

### Coverage

Counted as variants, over all 89 benchmarks (3284 IR instructions, 32 variants seen):

- 95%: 12 variants. In order: Arithmetic, Constant, GuardBranch, Compare, CallHelper, ArrayLoad, ArrayStore, IntToFloat, TableFieldLoad, FieldLoad, MetaGuard, ToWide (cumulative 95.6%). With each benchmark weighted equally the set is 11 variants.
- 99%: 21 variants, adding HashLoad, Numeric, FieldStore, PhantomTable, ReleaseStrings, TableFieldStore, HostCall, ArraySet, HelperResult (cumulative 99.0%). With equal weights it is 20.

Of the 12 variants in the 95% set, only `CallHelper` and the guard-bearing table ops need anything beyond `ld`/`st`/ALU/guard. `CallHelper` needs the `call` op with caller-saved clobbering. The set needs no `HashLoad` and no `ArraySet`, which are the two most expensive ops per instruction, but 13 of the 86 benchmarks that compiled a tree contain one (7 each): any tree containing one is rejected by a 95% prototype. The census cannot say how many trees that is without a per-tree op list. The script's `collect` data has it per tree, so a follow-up can count trees fully covered by a given subset (Inference: that share will be well below 95%, because coverage by instruction count is not coverage by tree).

### Implications for the design

- Inference. Compile cost per tree is small in absolute terms (mean root 26 IR instructions), so fixed costs decide: module and context setup, malloc, entry and exit scaffolding. 28.5% of CLIF today sits outside any op. MIR should make entry checks, the loop frame and exit stubs data-driven and table-built, not per-tree emitted code.
- Inference. The fused guard is the right unit. One in four IR instructions carries a guard and the average guard costs 7.2 CLIF, so `guard.cmp` with a snapshot index and a stub placed after the body removes the largest single item of per-instruction expansion.
- Inference. Snapshot reads should be weak and sparse. Of 30.8 window entries per guard, only 7.4 are values, and 2.7 values on average are kept alive only by the snapshot. The exit stub should store only value, virtual and slice entries, and the allocator should let weak uses spill or drop.
- Inference. Constants are 22% of IR instructions. With immediate operands and remat they cost nothing, which cuts the instruction count MIR has to allocate and emit by about a fifth.
- Inference. Float spill support is required even though it is rare, because two roots peak at 17 live floats against 16 xmm registers and nbody's root fills all 16 (before the C-ABI scratch register an emitter needs), and any C-ABI call (`NativeUnary`, `HashLoad`, `CallHelper`) clobbers all of them on SysV.
- Inference. `ArrayLoad`, `ArrayStore` and `HashLoad` expand to 24, 5 and 57 CLIF today. Their MIR forms should be short macro sequences inside the lowering, not extra IR ops, so the MIR op count stays low while the per-op cost drops below one tenth of the CLIF figure.

## Verification

- `scripts/heavy.sh -m 24G cargo test -p oximond -p oximond-cli --no-fail-fast` at `591642e4`: 181 test binaries, 0 failed. The CLI test `trace_census` was added after that run and passed alone (`cargo test -p oximond-cli --test trace_census`), as did `python3 -m unittest benchmarks/test_trace_census.py` (6 tests).
- Liveness test: `census_pins_pressure_and_values_live_across_a_guard` builds a loop with two ordinary parameters, one parameter passed unchanged through `LoopBack`, an invariant input and a guard, and pins peak gpr/fpr and the live-across and weak counts for the looping and the returning variant.
- New tests: `census_counts_the_ops_guards_and_pressure_of_each_member` and `census_clif_accounting_excludes_nested_lowerings_and_reports_json` in `crates/oximond/src/jit/trace/tree_tests.rs`, and `crates/oximond-cli/tests/trace_census.rs`, which checks that the dump is absent by default and has every field when asked for.
- Generated code does not change: the hook only reads `dfg.num_insts()` before and after `lower_instruction` when the flag is set.
