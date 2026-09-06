# Differential testing

`oximond-difftest` builds valid Lua programs from a seed and runs them in
oximond's execution modes. Each mode is compared with the stock interpreter for
the profile (`lua5.1` through `lua5.5`, or `luajit`), and each JIT mode with
oximond's own interpreter. It saves every mismatch with its source and a replay
command, and reduces it on request.

```sh
cargo build -p oximond-difftest
scripts/heavy.sh -m 8G target/debug/oximond-difftest run --count 500 --oximond target/release/oximond
target/debug/oximond-difftest replay-file target/difftest/failures/lua54-7/case.lua
```

Campaigns default to every profile and to these modes:

| mode | what it runs |
|---|---|
| `interpreter` | the interpreter alone (`--execution interpreter`) |
| `tracing-mir` | the JIT as it ships: MIR, with Cranelift for the trees MIR declines, the Peak tier and sides compiled apart |
| `tracing-mir-profile` | the same, loading a trace profile recorded by an earlier run |
| `tracing-mir-only` | MIR alone (`OXIMOND_JIT_BACKEND=mir-only`): what MIR declines stays in the interpreter, and no tree reaches the Peak tier |
| `tracing` | Cranelift alone (`OXIMOND_JIT_BACKEND=cranelift`) |

Every JIT mode compiles synchronously. Generated programs use in-memory Lua
APIs and do not call filesystem or OS functions. Dynamic loading uses strings or
readers that return strings.

## Two references

The stock interpreter judges every mode. Oximond's interpreter judges the JIT
modes too, exactly and without float tolerance. That comparison shows up in the
signature as `MODE:vs-interpreter:CATEGORY`, such as
`tracing-mir-only:vs-interpreter:stdout@o12`, and in `diff-vs-interpreter.txt`.
It is added when it says something the stock comparison does not:

- With no stock interpreter for the profile, the case is a mismatch if a JIT
  mode differs from the interpreter. Otherwise it stays inconclusive.
- When the interpreter itself differs from stock Lua, a JIT mode that differs
  from the interpreter as well is a JIT bug hidden behind an interpreter bug.
- When the interpreter matches stock Lua, a JIT mode that differs from one
  differs from the other, and the stock entry already names it.

The comparison needs the `interpreter` mode in `--modes`.

## Probes

Each generated program also runs bounded probes once, outside its hot loops.
Seeds vary calls with nil arguments and multiple returns, closure state,
coroutine lifecycle, debug hooks and inspected values, mixed value types,
metatables, globals, protected errors, and control flow with yields or loop
exits. Observations are compared against the stock interpreter.

The probe families are `runtime_functions`, `runtime_coroutines`,
`debug_probes`, `runtime_types`, `runtime_metatables`, `runtime_control`,
`profile_libraries`, and `utf8`. Library probes follow the selected profile:

- Lua 5.3 and later test UTF-8 characters, malformed bytes, byte offsets,
  iterator control arguments and partial ranges. Lua 5.4 and later also test
  strict and lax decoding.
- Modern profiles test string packing, integer math and table moves. Older
  profiles test legacy math and table APIs.
- Lua 5.2 adds `bit32`. LuaJIT adds `bit`.
- Lua 5.5 probes also cover lexical global declarations, named varargs and
  `table.create`.

Dump/load round trips, varied packing formats and weak-table reclamation run as
bounded probes too.

Use `--exclude` to leave selected probe families out while investigating a
different feature. `--exclude utf8` removes both UTF-8 probes and generated
UTF-8 expressions. The original random statements remain part of the program. A
probe stays together during reduction so its setup and cleanup cannot be
removed separately.

## JIT against the interpreter

`oximond-difftest run-jit` tests the tracing JIT against oximond's own
interpreter. No stock Lua is needed. The property: for every generated program
and profile, each variant prints what `--execution interpreter` prints, ends
with the same error message and position if the program errors, and fails or
succeeds the same way.

The oracle is `oximond --compat P --execution interpreter case.lua`
(`--interpreter-oracle` on the other subcommands). An uncaught Lua error in
the oracle is a result to match, not a generator bug.

The variants are execution modes, selected with `--variants` (or `--modes`).
The default is the first four. Each variant is its own process, because
`OXIMOND_JIT_BACKEND` is read once.

| variant | what it runs |
|---|---|
| `tracing-mir` | the JIT as it ships: MIR, the default backend (`OXIMOND_JIT_BACKEND=mir` set explicitly), with Cranelift for what MIR declines |
| `tracing-mir-profile` | the same, loading a profile recorded by an earlier `tracing-mir` run |
| `tracing-mir-only` | MIR with no Cranelift at all (`OXIMOND_JIT_BACKEND=mir-only`) |
| `tracing` | Cranelift alone |
| `tracing-profile` | Cranelift loading a profile recorded by an earlier Cranelift run of the same program |
| `tracing-mir-crossprofile` | MIR loading a profile recorded under Cranelift |
| `tracing-async`, `tracing-mir-async` | background compilation, not in the default set because install timing varies |

Under `OXIMOND_JIT_BACKEND=mir-only` a tree MIR declines is rejected and its
loop stays in the interpreter, a side compiled apart from its tree is declined,
and no tree moves up to the Peak tier, which compiles with Cranelift.

A profile variant records first (`--trace-profile-record`, in
`trace-profiles/` inside the case directory), then runs again with
`--trace-profile-load`. The load run is the one compared. If the recording run
leaves no profile, it stands in for the variant, so a crash there still shows.
MIR is Oximond's default backend. The Cranelift variants set
`OXIMOND_JIT_BACKEND=cranelift` and the MIR variants set `mir`, so neither
depends on the default or an inherited value.

### Hot-loop programs

`run-jit` uses `generate_hot_seed`, an opt-in bias in the generator (`--hot` on
`gen` and `replay`). It drops the probes that run once and builds loops that pass
the hot threshold: kernels of thousands of trips with array and record reads and
writes, calls to local functions, closures and upvalue writes, string building and
string-keyed tables, mid-loop type changes that force guard exits and side
traces, nested loops, branches, protected calls and operator classes. Cold
sections between kernels give loops that rejoin. A `HotKnobs` value (sections,
statements per loop, trips, depth, extra value types, feature mask) sets the
shape. The seed alone draws the knobs, so `replay --seed N --hot` reproduces a
program. Default generation is unchanged.

A comparison in which nothing ran native code proves nothing, so a run reports,
per variant and profile, how many cases entered native code and how many trees MIR
compiled versus handed back to Cranelift (`report.md`, section "JIT variants
against the interpreter"). `run-jit` exits 1 if a variant's native share is below
`--min-native-share` (default 0.6) or a MIR variant compiled no tree through MIR.
`tracing-mir-only` has no share floor: its share measures how much MIR takes,
and the other variants run the same programs.
Tracing runs print statistics only when the program ends without an error, so
cases that end in an error count as not native, and the report lists them.

### Output of a mismatch

A mismatch is saved under `failures/PROFILE-SEED/` like any other: `case.lua`, the
oracle's output (`oracle.stdout`, `oracle.stderr`), each variant's output,
`diff.txt`, and `meta.txt` with the variant in the signature (`tracing-mir:stdout@o12`)
and a `replay_file` and `replay_seed` command. `--reduce` also writes `minimized/`.

```sh
scripts/heavy.sh -m 16G target/debug/oximond-difftest run-jit \
    --seeds 0..300 --profiles lua51,lua54,lua55,luajit21 --jobs 8 --reduce
oximond-difftest replay-file failures/lua54-17/case.lua --profile lua54 \
    --interpreter-oracle --modes tracing-mir --oximond target/release/oximond
```

### Property test

`cargo test -p oximond-difftest --test jit_property` runs the same property with
proptest. The strategy draws a seed, a profile and the knobs. When a case fails,
proptest shrinks the knobs, the difftest reducer shrinks the program, and the
failure message has the reduced program, its signature and a one-line replay
command. Every program starts with a fixed anchor loop that the trace compiler
always compiles. The 60% check does not count it: it counts native entries
beyond what the anchor alone gives under the same variant and profile (measured
once per profile in each run), so the random sections have to compile too. The
default is 4 cases under each `run-jit` variant plus 8 light cases under `tracing` and
`tracing-mir`, from proptest seed 0, so it is deterministic. It takes 40 to
105 s with the debug binary on a loaded machine (load average 13). The run also
fails if a variant entered native code in less than 60% of the cases or no MIR
variant compiled a tree beyond the anchor's.

| variable | meaning |
|---|---|
| `OXIMOND_DIFFTEST_OXIMOND` | binary under test, absolute path (use the release build for long runs); default `target/debug/oximond`, built if missing |
| `OXIMOND_PROPTEST_CASES` | number of programs run under every variant (default 4) |
| `OXIMOND_PROPTEST_LIGHT_CASES` | extra programs run under `tracing` and `tracing-mir` only (default 8) |
| `OXIMOND_PROPTEST_SEED` | proptest's own seed, to draw other programs (default 0) |
| `OXIMOND_PROPTEST_ASYNC` | `1` adds `tracing-async` and `tracing-mir-async` (background compilation) to the four synchronous variants |
| `OXIMOND_PROPTEST_MIN_NATIVE` | least native share per variant (default 0.6) |
| `OXIMOND_PROPTEST_MIN_TRIPS`, `OXIMOND_PROPTEST_MAX_TRIPS` | trip-count knob bounds (default 4000 and 8000) |
| `OXIMOND_PROPTEST_VERBOSE` | print the knobs and native entries of each case |

The same file tests the harness against fake binaries: a variant that prints
something else is caught, shrunk to the minimal knobs and reduced; a wrong
profile load is caught; a run with no native code, or a MIR variant with no MIR
compile, fails.

### Long campaigns

```sh
scripts/heavy.sh -m 16G target/release/oximond-difftest run-jit \
    --duration 2h --master-seed 1 --jobs 8 --reduce --oximond target/release/oximond
OXIMOND_DIFFTEST_OXIMOND=$PWD/target/release/oximond OXIMOND_PROPTEST_CASES=500 \
OXIMOND_PROPTEST_SEED=7 scripts/heavy.sh -m 16G cargo test --release -p oximond-difftest --test jit_property
```

The property draws from all six profiles (lua51 to lua55 and luajit21).
`run-jit` defaults to the four oracle profiles; pass `--profiles all` for the
six. Pass `--size small|medium|large` to change the section count, and
`--variants` to cover one variant or the async ones. A run that loads a trace
profile gets four times the per-process timeout: loading a profile keeps the
interpreter off its fast loop for the whole program, so such a run took about
7 times as long as the oracle on a program that is mostly interpreted. That
speed cliff is an open profile-path issue (see the MIR proptest notes). A mismatch is a bug in the JIT,
the profile path or the MIR backend. Reduce it, fix it or add it to the known-bug
corpus with the interpreter's output as `.stdout`.

## Other subcommands

`run-rejections` compares compiler acceptance of invalid chunks, `run-fixtures`
checks isolated filesystem and OS behavior, and `run-workloads` reports native
activity for separate hot workloads.

## Trace profiles

Pass `--trace-profile-record-dir DIR` to a difftest run to record a separate
profile for each tracing case and mode. A later run with
`--trace-profile-load-dir DIR` loads those profiles. Interpreter cases are
unaffected.

## Known-bug corpus

`crates/oximond-difftest/corpus` holds reduced repros of known bugs with their
stock output. `cargo test -p oximond-difftest` replays them and fails when a
bug's signature changes or goes away. See `crates/oximond-difftest/README.md`.
