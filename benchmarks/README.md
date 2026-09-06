# Lua runtime benchmarks

This suite compares the same Lua 5.1-compatible programs on stock Lua, LuaJIT,
and Oximond. It covers small VM operations, tables, strings, calls, allocation,
and a few complete algorithms. Each program prints a deterministic checksum.
The runner rejects a measurement if the runtimes disagree.

Build Oximond before running the suite:

```sh
cargo build --release -p oximond-cli
python3 benchmarks/run.py --quick
```

The default run uses one warmup and seven measured processes per benchmark.
Use `--quick` while changing the VM, then run the full suite before keeping an
optimization. Useful focused commands are:

```sh
python3 benchmarks/run.py --filter 'table/' --quick
python3 benchmarks/run.py --filter 'calls/function' --repeat 15 --cpu 4
python3 benchmarks/run.py --runtime lua --runtime oximond-interpreter --quick
python3 benchmarks/run.py --quick --fail-gap 10
python3 benchmarks/run.py --json target/benchmarks/results.json \
  --markdown target/benchmarks/results.md
```

`--fail-gap 10` exits with status 2 when Oximond tracing mode is more than ten
times slower than the faster of Lua and LuaJIT. Correctness failures and
missing executables exit with status 1. Set `LUA_BIN`, `LUA54_BIN`,
`LUAJIT_BIN`, or `OXIMOND_BIN` to override executable discovery.

The `mineclonia/` group reproduces the hot interpreter paths of
Mineclonia's `mcl_levelgen` mod, whose start-up runs PRNG self-tests in
`random.lua`. Each program names the Lua code it mirrors. A suite entry can
set `"compat"` to choose the Oximond profile for that program; the runner
passes it to every Oximond runtime instead of `--compat=lua51`. The
`ull_kernel`, `ull_arith`, and `bit_wide` entries use `luajit21`, because
their hot paths only exist with LuaJIT's 64-bit integer literals (`1ull`)
and `bit` library. Each compiles its kernel with `loadstring`, as
`random.lua` does. Lua 5.1 cannot parse the literals, so it runs a
plain-number version of the same kernel that prints the same checksum. Its
timings measure that version, not the 64-bit one.

```sh
python3 benchmarks/run.py --filter 'mineclonia/' --quick
```

Wall time includes parsing, compilation, runtime setup, and execution. That is
intentional because startup is part of command-line runtime performance. The
programs do enough work to keep startup from dominating most results. For a
single hot operation, increase `--scale` and use `--repeat 15` or higher.

Results depend on CPU frequency, background load, allocator state, and build
flags. Compare before and after results on the same machine. A useful change
should improve the focused median without moving unrelated numeric-loop results
outside normal run-to-run noise.

The nested-call regression also has a retired-instruction check. It uses Lua
5.1, checks the checksum, and fails when tracing execution uses more instructions
than the interpreter. Select a CPU whose physical sibling is idle:

```sh
scripts/heavy.sh -x python3 benchmarks/check_nested_calls.py --cpu 4
scripts/heavy.sh -x python3 benchmarks/check_builtin_calls.py --cpu 4
```

To compare an optimization against a preserved executable, copy the release
binary before rebuilding, then include it in the same shuffled run:

```sh
cp target/release/oximond /tmp/oximond-before
# Make the change and rebuild the release executable.
python3 benchmarks/run.py --before /tmp/oximond-before \
  --filter 'calls/|table/hash_read|numeric/for_sum' --cpu 4 --repeat 15 \
  --json target/benchmarks/comparison.json
```

The report includes `before-interpreter` and `before-tracing` timings for the selected
Oximond modes. JSON reports record executable SHA-256 hashes as well as raw
samples. Use an otherwise idle machine for comparisons.

## JIT environment variables

- `OXIMOND_JIT_STATS=1` prints the JIT, interpreter and trace counters to
  stderr when the script ends.
- `OXIMOND_JIT_TRACE`, `OXIMOND_JIT_TRACE_IR` and `OXIMOND_JIT_TRACE_IR_OPT`
  print trace recording events, the recorded IR and the optimized IR.
- `OXIMOND_JIT_SYNC=1` compiles traces on the recording thread and installs
  them at the same safepoint. By default a background worker compiles, so the
  point where a trace first runs natively depends on thread timing. With the
  variable set, two runs of one script report identical `native_entries`,
  `iterations` and `traces_compiled`. Use it when you compare instruction
  counts. Unset, `0` or empty leaves the background worker in place.
- `OXIMOND_JIT_VERIFY=1` runs Cranelift's verifier in release builds too
  (debug builds always run it). A codegen bug then rejects its trace with
  the verifier's message under `OXIMOND_JIT_TRACE`, where it would otherwise
  reach the register allocator and panic or miscompile. A compile that panics
  rejects only its own trace either way; `compile_panics` in the trace
  counters counts them.
- `OXIMOND_JIT_INJECT_COMPILE_PANICS=0,3` makes the compiles of the listed
  compile requests (counting from 0) panic on purpose. It is for testing the
  panic guard.
- `OXIMOND_JIT_PEAK=0` keeps every tree off the Peak tier (the Cranelift
  recompile after two million native iterations). Any other value sends
  every long-running tree there. Unset, a tree MIR compiled goes there only
  when it has sides compiled apart. Use it to measure what the tier costs
  and gains.
- `OXIMOND_JIT_MIR_PROBE=0` stops the MIR backend from taking the trees that
  probe a table's string index (string-key reads and in-place stores): MIR
  declines them and Cranelift compiles them. The probe is on by default, as
  `table/hash_read_indexed` costs fewer instructions under MIR than under
  Cranelift with it on. `0` or empty turns it off, anything else leaves it
  on. `benchmarks/mir_backend.py --probe 0|1` sets the variable under both
  backends, so the environments keep equal lengths.

- `OXIMOND_ATTRIBUTION=<path>` samples where the interpreter and the native
  traces spend time and writes a tab separated report to `<path>` (see
  `integrations/luanti/README.md`). It uses no debug hook, and with it unset
  the dispatch loop pays one `Option` test per general-loop iteration.
  `OXIMOND_ATTRIBUTION_HZ` sets the tick rate. This is not the opcode counter
  switch `OXIMOND_INTERP_PROFILE`.
- `OXIMOND_PERF_MAP=1` names each compiled trace in `/tmp/perf-<pid>.map` as
  `oximond trace#<index> <source>:<line>`, so `perf report` shows traces by
  loop header.

## Counting instructions and allocations

`run.py` measures wall time. `measure.py` measures what Oximond itself does,
for the table and allocation optimizations. Per benchmark and mode
(`interpreter`, `tracing`, plus `before-interpreter` and `before-tracing` with
`--before BINARY`) it reports:

- retired user instructions from `perf stat -e instructions:u`, pinned with
  `taskset -c CPU`, as the median of `--repeat` runs (`--quick` means three);
- real tables allocated, elided (phantom) table charges and their bytes, and
  the trace statistics: traces compiled, native entries, iterations, exits,
  side exits and recording aborts by reason;
- with `--warmed`, the instructions one more scale unit costs (the difference
  between runs at scale 2N and N), which leaves out start-up, recording and
  compilation. The plain number is end to end;
- SHA-256 hashes of the binaries, `uptime` load, and how busy the pinned CPU
  and its SMT sibling were when the run started.

Every mode must print the same checksum as the stock interpreter for the
entry's profile: `lua5.1` for `lua51`, `lua5.4` for `lua54`, `luajit` for
`luajit21` (override with `LUA_BIN`, `LUA54_BIN`, `LUAJIT_BIN`). Each result
row records the oracle that checked it as `oracle`. When none could run (not
installed, it failed, or `--no-reference`), `oracle` is null, `oracle_reason`
says why, and the markdown lists the row under "NO ORACLE": its modes then
only agree with each other. Run timings through the gate and pin to a CPU
whose sibling (N+8 here) is idle:

```sh
scripts/heavy.sh --status
scripts/heavy.sh -x python3 benchmarks/measure.py --cpu 7 --warmed \
  --filter 'table/hash_read|allocation/' --json out.json --markdown out.md
scripts/heavy.sh -x python3 benchmarks/measure.py --cpu 7 \
  --before /tmp/oximond-before --filter 'table/hash_read'
python3 -m unittest benchmarks/test_measure.py   # unit tests, no binaries
```

`test_measure.py` runs no binary by default. `MEASURE_CHECKSUMS=1` adds a check
of every `opt/` workload against its oracle in both modes (release or debug
binary; about two and a half minutes with a debug build). `MEASURE_LIVE=1`
adds a perf run of `measure.py` itself, which only runs inside the gate:
`MEASURE_LIVE=1 scripts/heavy.sh -x python3 -m unittest benchmarks/test_measure.py`.

### Per-opcode cost with optrace.py

The interpreter's fast loop runs at an IPC near 4 with almost no branch
misses, so it is bound by the instructions it executes. `optrace.py` shows what
each opcode costs in that unit. Use it to judge a change to one opcode arm, or
to the dispatch sequence, before you spend a gated suite run on it. It needs
`gdb` (plus `nm`, `objdump` and `addr2line`) and a release binary:

```sh
python3 benchmarks/optrace.py target/release/oximond numeric/for_sum
python3 benchmarks/optrace.py target/release/oximond numeric/for_sum --dump ADDR
python3 benchmarks/optrace.py BIN path/to/script.lua --profile lua54 --skip 50000 --record 1000
```

It finds the `jmp *%reg` dispatch of `run_fast_activations::<false, false,
false>`, runs the binary under gdb with ASLR off, skips `--skip` dispatches
(default 200000) and single-steps the next `--record` ones (default 400). The
report has one row per dispatch target: how many segments ran, the executed
instructions per segment with the dispatch jump included (a mean when the
opcode has more than one path), the share of all recorded instructions, and the
`fast_path.rs` line. `--dump ADDR` prints the executed path of one segment, where
ADDR is a target from the report (addresses change with every build), with
disassembly. `--save FILE` and `--load FILE` keep a recording. On
`numeric/for_sum` the report has two rows, Add and NumericForLoop. The benchmark
must run longer than `--skip` dispatches. The tool stops with a message when
gdb or the dispatch jump is missing.

`python3 -m unittest benchmarks/test_optrace.py` runs no binary;
`OPTRACE_LIVE=1` adds a gdb run against `target/release/oximond`.

### Flags for the optimizer work

- `--jit-sync` / `--no-jit-sync` (default on) sets `OXIMOND_JIT_SYNC=1` for
  every Oximond run, so tracing counts do not depend on when the background
  compiler finishes. `--no-jit-sync` removes the variable, even when the
  caller's environment sets it. The JSON records the choice as `jit_sync`.
- `loads_forwarded`, `loads_merged` and `loads_hoisted` (optimizer counters)
  and `alloc_count`, `alloc_bytes` and `alloc_peak_bytes` (allocation counters)
  appear in each mode's `summary` in the JSON, and as extra table columns when
  a run has them.
- `--alloc-oximond BIN` and `--alloc-before BIN` take binaries built with the
  `alloc-stats` feature. Each benchmark and mode runs once with them (the
  counts are deterministic) and the result lands under `alloc` in the mode's
  JSON entry. Instruction counts still come from `--oximond` and `--before`,
  which must be normal builds, because the counting allocator adds
  instructions. `--alloc-before` needs `--before` and `--alloc-oximond`.
- `--gate PCT` makes the exit status 3 when the median `instructions:u` of
  any benchmark and mode rises by more than PCT percent against `--before`,
  or when `alloc_count` or `alloc_peak_bytes` rises at all against
  `--alloc-before`. The offending rows go to stderr. Without `--gate` the exit
  status is 0 on success whatever the numbers. Errors still exit 1. `--gate`
  needs `--before`. The gate fails, rather than passes, when a mode was
  measured on one side only, an instruction count is missing, or (with
  `--alloc-before`) an `alloc_count` or `alloc_peak_bytes` is missing.
- Old binaries fail loudly. Every OXIMOND_COUNTERS line must carry
  `tables_created`, `elided_tables`, `elided_table_bytes`, `traces_compiled`,
  `native_entries`, `iterations`, `recording_starts`, `traces_recorded`,
  `loads_forwarded`, `loads_merged` and `loads_hoisted`. A run whose binary
  prints no line, or a line without one of these keys, stops with an error
  that names the binary ("predates ..."). No missing count becomes a zero.
  The `abort.`, `exit.` and `side_exit.` keys are the exception: the CLI
  prints them only when nonzero, so their absence means zero. Any other key
  that some repeats report and others do not is an error.
- `--scales 1,2,4` runs every mode at each listed scale (two or more,
  ascending; the first is the reported scale and replaces `--scale`). The
  mode's `scales` entry in the JSON lists, per scale, the median
  instructions and the trace counts (`traces_compiled`, `recording_starts`,
  `traces_recorded`, total `abort.*` and each abort reason), and the
  marginal instructions per scale unit between consecutive scales. Equal
  counts do not prove the same traces (there are no trace fingerprints);
  different counts prove a different plan. `status` is `inconclusive`, with
  `reasons`, when the trace counts differ between two scales, a marginal is
  not positive, or two consecutive marginals differ by more than
  `--scale-tolerance` percent (default 10). A separate markdown table shows
  the series. `--scales` and `--warmed` exclude each other; `--quick` skips
  both.
- `--warmed` still measures scales N and 2N and reports
  `warmed_instructions_per_scale` as before. It now also runs the trace-plan
  check and stores it as `warmed_check` (`status`, `reasons`); the table
  marks an inconclusive estimate and lists the reasons under it.
- `--op-stats-oximond BIN` and `--op-stats-before BIN` take diagnostic
  builds made with `--features op-stats`. Each benchmark and mode runs once
  with `OXIMOND_OP_STATS` naming a temporary file, under perf and pinned like
  the other runs. The diagnostic build must print the same output as the
  normal build. Its report must have schema `oximond-op-stats/2`, list
  `op-stats` in `build`, carry every counter in `REQUIRED_OP_COUNTERS` (the
  `h1.` to `h6.` names of `crates/oximond/src/runtime/op_stats.rs`, each
  with a kind), all four histograms and the three site tables. A missing
  report stops the run ("printed no op-stats report; build it with
  --features op-stats"). The mode's JSON gets `op_stats`: `counters`,
  `counter_kinds`, `histograms` (all opcodes and operand classes, the 40
  most frequent opcode pairs, plus the pair total and distinct count),
  `sites`, `build`, the report's string fields, the diagnostic run's
  OXIMOND_COUNTERS line, and `instructions`, the diagnostic run's
  `instructions:u`. That number is diagnostic build overhead and is never
  compared with the normal build. The markdown gains one section per
  benchmark and mode with the key counters per hypothesis, their kinds
  (`static_ir`, `dynamic_*`, `gc_event`), derived ratios and the top ten
  opcode pairs. `binaries` records each diagnostic build's path, SHA-256 and
  the features it reported. `--op-stats-before` needs `--before` and
  `--op-stats-oximond`. A unit test fails when `REQUIRED_OP_COUNTERS` and
  the Rust registry disagree, so a new counter has to be added in both.
  A schema 1 report (before the recipe counters) is legacy: `validate_op_stats`
  loads it only with `allow_legacy`, shows the schema 2 counters as
  unavailable (`null`, never 0) and it never satisfies a fresh run. The
  snapshot value counters and `h3.ir.snapshot_entries.other` read as
  unavailable too: schema 1 counted virtual entries as `.other` and left
  recipe operands out of the value classes. The command line always runs
  fresh, so `--op-stats-before` needs a schema 2 build as well; compare
  against an older one through the Python API or the archived runner.

Build the alloc-stats binary and copy it away before the next build replaces
it:

```sh
cargo build --release -p oximond-cli --no-default-features --features alloc-stats   # via scripts/heavy.sh
cp target/release/oximond /tmp/oximond-alloc
```

Do the same for the preserved baseline (`/tmp/oximond-alloc-before`), then:

```sh
scripts/heavy.sh -x python3 benchmarks/measure.py --cpu 7 --repeat 7 \
  --oximond target/release/oximond --before /tmp/oximond-before \
  --alloc-oximond /tmp/oximond-alloc --alloc-before /tmp/oximond-alloc-before \
  --gate 1.0 --filter 'opt/|table/' --json out.json
```

Background compilation threads count in `instructions:u`. When a tracing number
depends on when an install lands, repeat the run and compare the spread in
`instructions_samples` in the JSON.

### OXIMOND_COUNTERS

`OXIMOND_COUNTERS=1 oximond script.lua` prints one line to stderr after a clean
exit and nothing else changes. It follows `OXIMOND_JIT_STATS` and
`OXIMOND_INTERP_PROFILE`. The line is `oximond-counters:` followed by
`key=value` pairs: `tables_created`, `elided_tables`, `elided_table_bytes`
(cumulative; a finished GC cycle does not reset them), `traces_compiled`,
`native_entries`, `iterations`, `recording_starts`, `traces_recorded`, and the
`loads_forwarded`, `loads_merged`, `loads_hoisted`, the `alloc_*` keys of an
`alloc-stats` build, and the non-zero `abort.<Reason>`, `exit.<Kind>` and
`side_exit.<Guard>` counts (a missing one of those means zero). Keys are
added, never renamed.

The only cost when the variable is unset is `Runtime::tables_created`, one
increment per allocated table, and one increment per elided table. Elided bytes
are not counted per charge: a finished GC cycle folds `elided_table_bytes` into
a running total, and the reported figure adds the live remainder. Against a
binary built from 412165f (gated, `--repeat 7`, CPU 3), instructions:u moved
+0.12% on `allocation/table_churn` (about one instruction per charge) and
+0.01% to +0.03% on `algorithm/binary_trees`.

### Pattern benchmarks

| Benchmark | Isolates |
| :-- | :-- |
| `string/pattern_many` | 1024 distinct patterns, each run through `find`, `match`, `gmatch` and `gsub` on four short subjects, so per-call pattern setup dominates |
| `string/pattern_hot` | four fixed patterns (anchored capture parse, bracketed field, word iterator, digit substitution) run about 100000 times over short log lines |

`pattern_many` prints a different checksum under Lua 5.4, whose `gmatch`
and `gsub` skip an empty match that ends where the previous one did.

### UTF-8 and packed records

| Benchmark | Isolates |
| :-- | :-- |
| `string/utf8_pack` | 30000 records packed and unpacked with `string.pack` and `string.unpack`, their names taken apart with `utf8.len`, `utf8.codepoint`, `utf8.offset` and `utf8.char`, then a 2500 character text walked 40 times with `utf8.codes` |

The benchmark needs `string.pack` and the `utf8` library, so the suite runs it
as `lua54` and leaves LuaJIT out of its row (`"skip"` in `suite.json`). The
record loop and the `utf8.codes` loop are two separate hot loops: a
`utf8.codes` loop inside the record loop would make that loop an inner loop
the outer trace cannot hold.

### Dispatch, mutation, and string workloads

These Lua 5.1-compatible workloads also run under LuaJIT and both Oximond
execution modes. Inputs stay bounded as `--scale` increases. Hash iteration
sums values without depending on traversal order, and protected calls count
caught errors without printing runtime-specific error messages.

| Benchmark | Isolates |
| :-- | :-- |
| `calls/multiple_returns` | fixed and open return lists, including nil holes |
| `calls/protected` | successful protected calls and occasional error unwinding |
| `calls/metamethod_call` | callable tables dispatched through `__call` |
| `calls/metamethod_ops` | operator metamethods of a vector class: arithmetic, comparison, equality, `..` and `tostring` |
| `control/coroutine_pingpong` | resume/yield handoffs with values in both directions |
| `control/loop_in_call` | a hot outer loop calling a function that runs a nested loop with bounds from its arguments |
| `control/short_inner` | an outer loop with a 3 to 6 trip inner loop over a table |
| `table/hash_write` | overwrites through dynamic string keys |
| `table/hash_churn` | deletion and reinsertion in a bounded hash table |
| `table/hash_iteration` | `pairs` traversal over mixed hash keys |
| `table/insert_remove` | array shifts with `table.insert` and `table.remove` |
| `table/metamethod_function` | function `__index` lookups, including misses |
| `table/metamethod_newindex` | writes routed through a proxy's `__newindex` |
| `string/byte_sub` | rotating substring extraction and byte reads |
| `stdlib/tonumber_parse` | decimal, hexadecimal, explicit-base, and invalid string conversion |

For a focused run:

```sh
scripts/heavy.sh -x python3 benchmarks/run.py --quick \
  --filter 'calls/(multiple_returns|protected|metamethod_call|metamethod_ops)|control/coroutine_pingpong|table/(hash_write|hash_churn|hash_iteration|insert_remove|metamethod_function|metamethod_newindex)|string/byte_sub|stdlib/tonumber_parse'
```

### Benchmarks for the table and allocation work

| Benchmark | Isolates |
| :-- | :-- |
| `table/hash_read_small` | dynamic string-key reads, ordered table under 16 keys |
| `table/hash_read_shaped` | dynamic reads on a constructor-built (shaped) table |
| `table/hash_read_indexed` | dynamic reads on 3000 keys, one probe in eight missing |
| `allocation/ctor_array` | array-only constructors that die in the iteration |
| `allocation/ctor_fields` | fields-only constructors that die in the iteration |
| `allocation/ctor_mixed` | array plus fields |
| `allocation/ctor_side_exit` | constructors, with a rare branch that keeps the table |
| `table/heap_redundancy` | repeated field reads of one table across a store to another |

The `opt/` benchmarks exercise the load forwarding, merging and hoisting
passes. Each prints a checksum that stock Lua 5.1 and LuaJIT reproduce.

| Benchmark | Isolates |
| :-- | :-- |
| `opt/field_reload` | one field read several times per iteration, with a store to a different field between the reads |
| `opt/store_load` | `t.x = f(i)` then `t.x` in the same iteration, and `a[1]`, `a[2]` stored then read |
| `opt/alias_holder` | two tables sharing an `__index` holder, reads through both, writes to one table's own field |
| `opt/invariant_field` | a config table read in a hot loop and never written |

### Benchmarks for the interpreter and tracing hypotheses

These measure the six hypotheses of the op-stats work: range proofs (h1),
adaptive bytecode (h2), recovery recipes (h3), argument moves (h4), dead-slot
clearing (h5) and closure families (h6). Each comes with a negative control.
`lua54` entries are checked against Lua 5.4. In `run.py` their `lua` column
runs Lua 5.4 instead of Lua 5.1, so it measures integer arithmetic.

| Benchmark | Compat | Isolates |
| :-- | :-- | :-- |
| `opt/range_dense` | lua54 | h1: dense array read and write in a counted loop, integer keys |
| `opt/range_float_index` | lua51 | h1: the same loop with float keys, so each access also checks integrality |
| `opt/range_offset` | lua51 | h1: stencil reads `a[i-1]`, `a[i]`, `a[i+1]` |
| `opt/range_fail` | lua51 | h1 control: up to 3000 of 20000 keys per round fall past the array end |
| `opt/arith_stable` | lua51 | h2: multiply-add chains on non-integral floats only |
| `opt/arith_mixed` | lua54 | h2 control: the same sites see integer, float and mixed operand pairs |
| `opt/cold_recovery` | lua51 | h3: guard-heavy loop whose temporaries are read only on a 1-in-1000 branch |
| `opt/call_args` | lua51 | h4: calls with 0, 1, 4, 8 number arguments, caller-dead and caller-live; a second loop with string arguments |
| `opt/call_args_hook` | lua51 | h4 control: `call_args` with a count hook set |
| `opt/dead_slots` | lua51 | h5: many registers, temporary tables that die at the end of each block |
| `opt/retained_slots` | lua51 | h5 control: tables kept in locals, captured by closures, or kept in a ring |
| `opt/closure_family` | lua51 | h6: eight closures of one prototype rotated through one call site |
| `opt/closure_protos` | lua51 | h6 control: eight different prototypes through one call site |

When these were added (worktree base 28f15ef), several did not trace. Table
constructors abort recording (`UnsupportedOpcode` at `NewTable`), so
`dead_slots` and `retained_slots` run in the interpreter in tracing mode.
String values abort recording, which is why `call_args` keeps its string calls
in a second loop. A count hook stops recording, so `call_args_hook` never
traces. The closure pair compiles traces, but every entry leaves at once
through a `Callee` guard. That is the h6 signal, not a broken workload.

Baselines recorded at 412165f are in `benchmarks/baselines/`.

## File loading pipeline

`file_loading.py` compares complete-input, sequential streaming, and concurrent
file loading. Supply preserved release executables for all three versions.
It measures full CLI launch-to-exit time, checks program output, shuffles the
run order, and records raw elapsed samples plus user-mode instructions.
Choose idle physical cores with unused SMT siblings so the parser and compiler
can run concurrently. Run measurements through the exclusive gate.

```sh
scripts/heavy.sh -x python3 benchmarks/file_loading.py \
  --original /path/to/original --sequential /path/to/sequential \
  --pipeline target/release/oximond --cpus 0,2,3 --repeat 15 \
  --json /path/to/results.json
```

The compiler example adds a controlled delay to each nonempty 8 KiB read.
This measures compilation alone with simulated I/O, separately from real CLI
startup. Its three modes must produce identical bytecode before measurements.

```sh
scripts/heavy.sh cargo build --release -p oximond --example load_pipeline
# Add this argument to file_loading.py:
# --slow-reader target/release/examples/load_pipeline
```
