# oximond-difftest

Seeded differential testing. A seed and a profile pick one valid, bounded
Lua program. The stock interpreter for that profile runs it and its output
is the expectation. Oximond then runs the same file in the interpreter and
in tracing mode, and every difference is saved with what it takes to replay
it.

```sh
cargo build -p oximond-difftest
target/debug/oximond-difftest gen --seed 7 --profile lua54            # print one program
scripts/heavy.sh -m 8G target/debug/oximond-difftest run --count 500  # a campaign
scripts/heavy.sh -m 8G target/debug/oximond-difftest run --duration 2h # random seeds for 2 hours
target/debug/oximond-difftest replay --seed 7 --profile lua54         # one seed, with diffs
target/debug/oximond-difftest replay-file target/difftest/failures/lua54-7/case.lua
target/debug/oximond-difftest corpus-check crates/oximond-difftest/corpus
```

`run` tests `target/release/oximond` when it exists, otherwise
`target/debug/oximond`. Pass `--oximond PATH` to pick a build, and rebuild
it first: the report records its SHA-256, not its revision.

`--duration D` stops starting cases after `D` (`90` or `90s`, `30m`, `2h`,
`1d`). Cases already running finish, so the campaign overruns by up to one
case's time, longer with `--reduce`. Without `--seed`, `--seeds`, `--count`
or `--start`, the campaign runs random seeds until then. Seed `n` is output
`n` of `SplitMix64` seeded with a master seed, fresh each run. The campaign
prints the master seed at the start and the report records it, and
`--master-seed N` replays the same order. With a seed list, `--duration`
only cuts the list short. Either way each seed runs on every profile before
the next seed starts, so a cut-short campaign covers the profiles evenly.
Failures record their own seed, so `replay --seed` works as usual.

## Oracles

| profile  | oracle   |
|----------|----------|
| lua51    | `lua5.1` |
| lua52    | `lua5.2` |
| lua53    | `lua5.3` |
| lua54    | `lua5.4` |
| lua55    | `lua5.5` |
| luajit21 | `luajit` |

The runner checks each oracle's `-v` banner and refuses one that reports a
different version. Stock LuaJIT can disagree with itself: its trace compiler
turns `x ^ 2` into `x * x` while its interpreter calls `pow`. When a mode
differs from `luajit`, the runner also runs `luajit -joff`, and a mode that
matches either run passes. `meta.txt` and `oracle-alternate.stdout` record
that second run. `--oracle lua53=/opt/lua53/bin/lua` points a profile at
another binary. Without an oracle, Oximond's modes still run and its
interpreter judges the JIT modes: a JIT mode that differs from it makes the case
a mismatch, and a case whose modes agree is `inconclusive`. Agreement never
counts as a pass. With an oracle, the interpreter also judges the JIT modes when
it differs from the oracle itself (`MODE:vs-interpreter:CATEGORY` in the
signature; see `docs/difftest.md`). Campaigns default to all six profiles. Missing oracles produce inconclusive
cases. `--profiles oracle` selects the smaller reference-machine set
(lua51, lua54, lua55 and luajit21).

The oracle can also be oximond's own interpreter (`--interpreter-oracle`, and
`run-jit`), which tests the tracing JIT variants, including MIR and trace
profiles, without a stock interpreter. See `docs/difftest.md`.

## The programs

`src/generate.rs` builds programs from typed, scope-aware choices. Each
variable keeps one kind (integer, number, string, boolean, array, record,
object), so operations only meet operands they accept. Errors happen only
where the generator plants them, inside `pcall` or `xpcall`.

A program has fixed helpers, two classes with shared method names and
comparison metamethods, a proxy table with `__index` and `__newindex`
functions, program-wide state captured as upvalues, a few globals, fixed
library functions (bounded recursion, a tail call, mutual recursion,
varargs, a dispatch table, a closure counter, a `coroutine.wrap`
generator, an `_ENV` or `setfenv` function), generated functions, and
sections. Half the sections are general: a loop that runs hot (100 to 400
trips), warm (20 to 90) or cold (1 to 5), in one of seven shapes (numeric,
reverse, `while`, `repeat`, `ipairs`, float step, nested), using every
statement family and printing every iteration. Statements inside them
include phase changes that fire after the loop has had time to compile: a
number turning float, an object switching metatable, a record changing
shape, an array replaced, a global set and cleared.

Two statement families probe the edges of the runtime. A yield boundary runs
a coroutine that yields from inside a metamethod (`__index`, `__newindex`,
arithmetic, comparison, `__concat`, `__call`, `__len`), a generic `for`
iterator, a `pcall` inside a metamethod, or a C function's callback
(`table.sort`, `string.gsub`, `tostring`, `load`'s reader), then resumes it
to the end. Stock 5.2 and later and LuaJIT allow some of these and reject the
rest with an exact message; 5.1 rejects them all. A standard-library edge
calls one corner case through `pcall`: bad `gsub` captures, malformed
patterns, `%s` with `__tostring`, `dofile` of a missing file,
`xpcall(load, handler, reader)`, 5.1's `table.foreach`, and 5.3+ table
functions on a proxy with `__index`, `__newindex` and `__len`.

Every program also probes the libraries its profile provides. Lua 5.3 and
later cover UTF-8 encoding, decoding, iteration, byte offsets, embedded NULs,
invalid byte sequences and boundary code points. Lua 5.4 and 5.5 also cover
strict versus lax decoding of surrogates and extended code points. These
profiles test string packing, integer math and overlapping table moves.
Packing probes vary integer and float widths, byte order, alignment,
length prefixes, unpack offsets, truncation and range errors. Every profile
tests `string.dump` by loading and executing its result; dump bytes are not
compared. Lua 5.5 also probes `table.create`.
Older profiles test legacy math functions; Lua 5.1 and LuaJIT also test
`table.maxn`, `table.foreachi` and global `unpack`. Lua 5.2 tests `bit32`,
and LuaJIT tests `bit`. `--exclude profile_libraries` removes the library
probe, and `--exclude utf8` removes UTF-8 probes and expressions.

Language probes include Lua 5.5 lexical global declarations and named
varargs, captured const locals, close handlers on return, break, goto and
error exits, and bounded backward jumps. Weak-table probes discard both
keys and values before collection and observe reclamation. These probes
are removed with `--exclude runtime_control`.

About a fifth of the sections are protected kernels: a hot loop that calls named functions
through `pcall` or `xpcall` (a Lua message handler), where the function raises
on every call or on every second, third, seventh or eleventh. It raises with
`error` at levels 0 to 2 (a constant string, a computed one, a table, nil, a
number, a boolean), a failed `assert`, or an error the runtime raises. The
trace compiler inlines the functions, so these loops exercise errors caught
without leaving native code and the ones that fall back. `--exclude
protected_kernel` removes them.

The rest are kernel sections, written for the trace compiler. Their
loop runs 3 000 to 40 000 times, because the compiler skips a loop whose
remaining work cannot repay a compile (`COMPILE_REPAYING_GUEST_WORK` in
`oximond::vm`'s trace policy). It only does arithmetic, array and record reads
and writes, numeric branches, short inner loops and method calls, with no
output inside. A checksum folds in an integer every iteration. A forced
exit changes a number's subtype, replaces an array and drops a record field
at one iteration, which fails a guard in compiled code, and prints the
loop state right there. Everything prints again after the loop. General
loops print through `print` and work on strings, which the recorder does
not follow, so most compiled traces come from kernel sections.

Integer expressions also call `rawget`, `unpack` (or `table.unpack`) and
`getmetatable` of objects and strings. On LuaJIT and Lua 5.2, one in three
bitwise expressions calls a `bit` or `bit32` function with operands at the
edges of its conversion (word boundaries, ties, `-0.0`, and on LuaJIT NaN and
the infinities) and shift counts around 32, which the trace recorder turns
into integer operations. Features `rawget`, `unpack`, `getmetatable` and
`bit_library_edges` mark the programs that do.

Seeds use SplitMix64 (`src/rng.rs`) with the profile index and
`GENERATOR_VERSION` mixed into the stream, so a seed means the same program
on every machine until the version changes. Saved failures keep their
source and replay with `replay-file` regardless of version.

`--exclude` leaves out feature families that hit a known open bug, so a
campaign can look past them. It changes the programs, and the header line
records it. Report excluded runs next to an unfiltered baseline, never in
place of one. The families are `call_result_to_variable` (results of calls
go only to fresh locals), `float_modulo`, `format_c_zero`, `tonumber_base`,
`math_random`, `rep_separator`, `format_q`, `coroutine_isyieldable`,
`yield_across_pcall`,
`error_number`, `error_level2`, `error_no_integer`, `error_bad_argument`,
`error_string_arith`, `coroutine_error`, `xpcall_lua_handler`,
`yield_boundaries`, `stdlib_edges`, `runtime_functions`,
`runtime_coroutines`, `debug_probes`, `runtime_types`,
`runtime_metatables`, `runtime_control`, `profile_libraries`, `utf8`,
`patterns`, `string_arithmetic`, `text_calls` and `table_traversal`.

### What the generator avoids

Lua leaves some behaviour unspecified, and comparing it would report noise:

- Evaluation order of operands, arguments and assignment targets.
  Expressions only call functions without side effects. A call that writes
  state is its own statement, assigned only to a plain variable.
- Table traversal order. `rec` sorts keys before printing, and `pairs` only
  sums integers.
- Length of tables with holes. Arrays never get holes, and removal keeps
  at least one element.
- Addresses. Tables, functions and coroutines print as their type name.
- Unbounded work. Generated calls carry a cost estimate and are skipped
  past a per-call and per-iteration budget. Recursion depth is bounded by
  reducing arguments, strings are cut to 48 bytes when stored, arrays stop
  growing at 12 elements.
- The sign of zero in Lua 5.1 and 5.2. Stock 5.1 folds constant
  expressions and keys its constant table by numeric value, so a folded
  `-0` and a later `0` share one constant and whichever came first wins.
  That is an artefact of stock's compiler, not a language rule. The
  generator writes no `-0.0` literal for those profiles, and their
  observations print every zero as `n0`. Generated numeric text conversions
  also use positive zero for these profiles. Other profiles preserve its sign.

In Lua 5.1 and LuaJIT, integer values are reduced modulo 1000003 when
stored, so they stay exact doubles and valid `bit` operands. Lua 5.3 and
later keep integer wrap-around.

### The observation format

`out(tag, ...)` prints one line per observation: the statement's tag, `#`,
the value count, then each value. Integers print as `i5`, floats as
`f2.5` with 17 significant digits, and 5.1 and LuaJIT numbers as `n2.5`.
Every NaN prints as `nan`: its sign depends on how an engine produced it
(x86 makes `0/0` negative, a Rust constant fold positive), and Lua does not
define it. Before a computed number reaches concatenation, `table.concat`, `tostring`, or
`string.format`, generated expressions use `math.abs` to make NaNs positive.
The formatter still handles a NaN, and its resulting string is compared exactly.
Explicit formatter tests keep their raw inputs and exact comparisons.
Strings print quoted, with control characters, quotes, backslashes and
bytes from 128 up written as `\ddd`. The helper does not use `%q`, which is
under test elsewhere. Stdout is compared byte for byte by default.

`--float-compare approximate` compares finite `f` and `n` numeric observations
with `|actual - expected| <= max(absolute, relative * max(|actual|, |expected|))`.
Both tolerances default to `1e-12`. Set them with `--float-abs-tol` and
`--float-rel-tol`; values must be finite and nonnegative. For example:

```sh
scripts/heavy.sh -m 8G target/debug/oximond-difftest run --count 1000 --float-compare approximate
```

This keeps float operations in the generated programs. A numeric difference
outside the tolerance still fails. Integer `i` observations, tags, value
counts, strings, errors and termination stay exact. That includes arrays and
records serialized as strings, and `tostring` or formatting results. NaNs,
infinities and the sign of zero also stay exact. Unknown output formats use
exact comparison.

Reports count passing cases that required approximation. Artifact metadata
and replay commands record the mode and tolerances. Replay, oracle-based
reduction and `corpus-check` use the selected comparison policy. The
checked-in corpus tests remain exact, and a known-failure entry still fails
if its expected signature changes. Agreement between Oximond's modes and
stability across reruns always use exact output.

## Verdicts

| verdict | meaning |
|---|---|
| `pass` | every mode printed exactly what the oracle printed and ended the same way |
| `mismatch` | some mode differs |
| `oracle_syntax_error`, `oracle_runtime_error` | the oracle rejected the program: a generator bug |
| `oracle_crash`, `oracle_timeout` | the oracle died or hung |
| `inconclusive` | no oracle for the profile |
| `harness_error` | the runner itself failed for this case |

A mismatch has a signature per mode, such as `tracing:stdout@L12` (stdout
first differs at observation `L12`), `interpreter:unexpected_error@C4`,
`tracing:crash` or `tracing:timeout`. Exit codes are not compared (stock
exits 1 on an uncaught error, oximond 70). An uncaught error is compared by
its first stderr line with the interpreter-name prefix removed. Every
program runs as `case.lua` in its own directory, so positions in error
messages agree.

Every mismatching mode reruns (`--reruns`, default 2). One that prints
something else on a rerun is marked flaky.

## Artifacts

A campaign writes `cases.tsv` with one line per attempted case, including
its feature list, and `report.md` with verdict counts, distinct signatures,
JIT coverage and the feature ledger. Each failure gets a directory under
`failures/` (oracle rejections under `invalid/`) with `case.lua`, every
engine's raw stdout and stderr, `diff.txt`, and `meta.txt`. `meta.txt`
records the seed, generator version, profile, features, oracle path and
banner, oximond path and SHA-256, commands with their environment, limits,
statuses, JIT counters and replay commands.

`--reduce` shrinks each mismatch: it deletes statements and blocks,
replaces blocks with their bodies and halves loop counts. An edit is kept
only if the oracle still runs the program to completion and the same mode
still fails at the same tag. The result goes to `minimized/` next to the
original, with `reduction.log`. `--reduce-attempts` and `--reduce-seconds`
bound the work.

## JIT coverage

Synchronous tracing runs set `OXIMOND_JIT_SYNC=1` so traces install at the
same point on every run. Every tracing mode sets `OXIMOND_JIT_STATS=1`. A case counts as JIT-productive when
compiled code ran at least once: a trace was entered (`native_entries`).
The report's JIT table sums, per profile, traces compiled, native entries,
iterations completed in native code, exits back to the interpreter, guard
failures and side traces installed.
Campaigns default to `interpreter,tracing-mir,tracing-mir-profile,tracing-mir-only,tracing`:
the interpreter, the JIT as it ships, the same with a loaded trace profile, MIR
alone and Cranelift alone. `--modes ...,tracing-async` adds background
compilation, which varies in install timing, so check the stability of its
failures.

The feature ledger counts features present in source. Its native-activity
column means compiled code ran somewhere in that case, even if that feature
was cold or never executed. It does not prove native execution of a feature.
`run-workloads` runs separate hot loops for math, strings, table access,
closures and metamethods and reports native entries for each workload.
Even there, entries do not prove that every operation was lowered.

## Scalar library campaign

Generator 21 added the scalar library edges that traces call through
helpers: `string.sub` with indices past either end, negative or zero, the
second result of `math.modf`, `math.log` with bases 2, 10, e, 1, 0 and
negative ones, `math.pow` where installed, `tostring(nil)`, `tonumber` of
strings such as `" 0x1p4 "`, `"1e"`, `"9223372036854775808"` and `"nan"`,
and `math.random` draws. A draw changes the generator, so it is a statement
of its own, and the prelude seeds the generator with `math.randomseed(42)`
before any program runs. Draws appear only in the lua52, lua53, lua54 and
lua55 profiles. Lua 5.4 and 5.5 give the stock sequence after a seed, and
5.2 and 5.3 have no oracle. `--exclude math_random` leaves them out there too.

Stock Lua 5.1 and LuaJIT draw from libc and from LuaJIT's own generator,
which Oximond does not reproduce bit for bit, so those profiles generate no
draws. Since generator 22 `math.log (x, b)` follows each profile's reference
(`log2` and `log10` for bases 2 and 10 from Lua 5.3, only `log10` in 5.2, and
`log2 (x) * (1 / log2 (b))` in LuaJIT), so every profile draws every base, and
`math.pow` is absent from the Lua 5.5 profile as in stock Lua 5.5. A
ten-minute oracle campaign on lua54, lua55 and luajit21 at generator 22
(`--master-seed 777 --jobs 4`, all three modes) ran 14549 cases with no
mismatch and none between the interpreter and the tracing modes. The same
campaign at master seed 4242 before `math.modf` kept the sign of a zero
fraction in the Lua 5.1 family found 34 LuaJIT mismatches, every one a
negative zero (`math.modf (-3)` gives `-3, -0` there), and none in `math.log`
or `math.pow`.
The check of the tracing JIT is
the campaign on the profiles without an oracle, which counts whether the
modes agree, plus a comparison of `interpreter.stdout` and `tracing.stdout`
in every failure directory of the oracle profiles:

```sh
scripts/heavy.sh -m 8G target/debug/oximond-difftest run --count 400 \
    --profiles lua52,lua53 --modes interpreter,tracing
scripts/heavy.sh -m 8G target/debug/oximond-difftest run --count 150 \
    --profiles oracle --modes interpreter,tracing
cmp failures/*/interpreter.stdout failures/*/tracing.stdout   # per directory
```

A run on the scalar helper commit: 800 cases on lua52 and lua53, all 800
with the modes in agreement, 749 of them with native code running. 600 cases
on the oracle profiles: 371 passed, 229 differed from the oracle, and in 228
of those the two modes printed the same thing. The remaining case (`lua51`
seed 14) does not involve a scalar helper: `reduce-modes` cuts it down to a
nested loop that stores a table field and updates a global, and the
reduced program differs on the branch head before these helpers too (the
pre-flow build agrees with itself). A rerun of 150 seeds on lua52, lua53,
lua54 and luajit21 with the final build showed no case where the two modes
printed different output.

## Rejection and fixture suites

`run-rejections` checks invalid chunks for missing block endings, unknown
labels, jumps into local scope, const writes and Lua 5.5 named-vararg writes.
`reject-file` checks a supplied invalid chunk. Both compile through
`loadstring` or `load` without executing the source and compare acceptance,
not diagnostic wording. If stock Lua accepts the source, the case fails as
`oracle_unexpected_acceptance`. Crashes and timeouts never count as rejection.
Saved failures replay with `reject-file`.

`run-fixtures` covers binary file I/O, seek and lines, closed handles,
`loadfile` and `dofile`, preload-module caching and errors, and fixed UTC
OS date operations. Files stay in the isolated case directory and successful
cases remove their own fixture files. Generated seed programs still avoid
filesystem and OS calls.

```sh
target/debug/oximond-difftest run-rejections --oximond target/debug/oximond
target/debug/oximond-difftest reject-file invalid.lua --profile lua55 --oximond target/debug/oximond
target/debug/oximond-difftest run-fixtures --oximond target/debug/oximond
scripts/heavy.sh target/debug/oximond-difftest run-workloads --oximond target/debug/oximond
```

All three suites accept `--profiles`, `--modes`, `--oracle`, `--out` and
process limits. Missing oracles are inconclusive. Suite cases run serially;
`--jobs`, seed, exclusion and reduction options apply to seeded campaigns.

## Resources

Each process runs in its own process group under `prlimit --as`
(`--memory`, default 2048 MiB), a deadline (`--timeout`, default 20 s) and
an 8 MiB output cap. A timeout or output overflow kills the group.
Campaigns default to 4 parallel cases. Run long campaigns through
`scripts/heavy.sh`.

## The corpus

`corpus/` holds generated programs with the stdout their oracle printed.
`cargo test -p oximond-difftest` replays them through embedded oximond in
both modes. An entry with a `NAME.known-failure` file lists the exact
`mode:category` lines an open bug causes. The test fails when that set
changes, including when the bug is fixed, so the fix updates the entry.
`corpus-add DIR --seed N --profile P` generates a seed and records its
oracle output. A reduced failure goes in by copying `minimized/case.lua`
and `minimized/oracle.stdout`.

`output-repeats-after-concat-exits-lua51` is a real open bug that `run-jit`
found: output printed before a hot loop is written again on each guard exit.

`float-modulo-lua51` stays a known failure on purpose. Lua 5.1 and 5.2
compute `a % b` as the unfused `a - floor(a / b) * b`, like LuaJIT. The
`lua5.1` oracle on the reference machine (CachyOS, built for x86-64-v3 with
FP contraction) fuses the multiply and subtract, so its last digits differ
(`o1`, `o2` and `o3` all show it). That is a build artefact, not Lua 5.1
behavior, and a fused `%` would cost an out-of-line call in every tier.

## Pattern campaign

Generator 23 added `patterns`: `string.find`, `string.match`, `string.gmatch`
and `string.gsub` (a string or a table for the replacement, with and without
a limit) over a fixed set of patterns (classes, captures, position captures,
anchors, `%b`, `%f`, empty matches) and a mix of literal and generated
subjects. The result is folded into an integer variable, so the statement
runs in hot loops, where the tracing JIT calls the helpers
(`string.find_pattern`, `string.match`, `gmatch.next`, `string.gsub`).
`--exclude patterns` leaves it out. Programs of generator 22 and earlier
replay with it off.

The campaign on the commit that added the helpers (release build, both
Oximond modes against the stock interpreters, `--jobs 4`):

```sh
scripts/heavy.sh -m 8G target/release/oximond-difftest run --count 150 \
    --profiles oracle --modes interpreter,tracing
scripts/heavy.sh -m 8G target/release/oximond-difftest run --count 400 \
    --start 150 --profiles all --modes interpreter,tracing
```

550 seeds on each of lua54, lua55 and luajit21 passed against the oracle. The
400 seeds on lua52 and lua53 have no oracle: all 400 agreed between the two
modes on each. lua51 differed from `lua5.1` on 57 of 550 seeds, the same 57 the
pre-flow tracing build differs on (the float and `math.modf` cases of the
scalar campaign above), and the two Oximond modes printed the same output on
all of them. No case printed different output in the interpreter and in the
tracing mode. The `patterns` family was in 2928 of the 3000 cases, and 2742
of those ran native code.

## UTF-8 and pack campaign

Generator 26 added `text_calls`, on the profiles that have `utf8` and
`string.pack` (lua53, lua54 and lua55): `utf8.len`, `utf8.codepoint`,
`utf8.offset`, `utf8.char`, loops over `utf8.codes` (with the lax flag where
the profile has one), and `string.pack`, `string.unpack` and
`string.packsize` over a fixed set of formats (every integer size and
endianness, floats, `s1`, `s2`, `z`, `c3`, alignment, padding), values at the
edges of their ranges, and strings with truncated, overlong, surrogate and
out-of-range sequences. A call that may raise goes through `pcall`, so the
exit path of a trace is drawn too and the program goes on. The result is
folded into an integer variable, so the statement runs in hot loops, where
the tracing JIT calls the helpers (`utf8.len`, `utf8.codepoint`,
`utf8.offset`, `utf8.char`, `utf8.next`, `string.pack`, `string.unpack`,
`string.packsize`). `--exclude text_calls` leaves it out. Programs of
generator 25 and earlier replay with it off.

The campaign on the commit that added the helpers (release build, all three
modes, `--jobs 4`):

```sh
scripts/heavy.sh -m 8G target/release/oximond-difftest run --count 2000 \
    --profiles lua53,lua54,lua55 --modes interpreter,tracing,tracing-async
```

2000 seeds on each profile passed: lua54 and lua55 against `lua5.4` and
`lua5.5`, and lua53, which has no oracle here, with the three modes printing
the same output on all 2000. The `text_calls` family was in 5583 of the 6000
cases, and 5380 of those ran native code. A first campaign found that
`string.unpack` raised for an initial position of zero or one before the start
of the string, where `lua5.4` and `lua5.5` clamp it to 1 (`posrelatI`); Lua 5.3
raises, and the library now does the same per profile.

## Table traversal campaign

Generator 27 added `table_traversal`, on every profile. A statement builds a
small table of its own and walks it with `pairs (t)` or `next, t`, then folds
the result into an integer variable, so it runs in hot loops. The table has an
array part, string keys (`k1`, `k2`, ...), sparse integer keys, a float key and
a boolean key, or a mix, sometimes with holes in the array part. The body
sums the values and a weight of each key (the number, the length of the
string, 1), or writes while it walks:

- clears the key it reached (`t[k] = nil`), then checks `next (t) == nil`;
- overwrites its value (`t[k] = v * 2 + 1`), or deletes it when the value is
  even and bumps it otherwise;
- copies every entry to a second table and clears that one as it walks it;
- walks the table in an outer loop and a copy of it in an inner loop, clearing
  the outer key;
- steps by hand with `local k, v = next (t)` and `k, v = next (t, k)` in a
  `while` loop, clearing some of the keys on the way;
- walks three times, decrementing or clearing each value.

Every write goes to the key just reached, which Lua allows, and every result
is a sum or a count. The order of the keys (unspecified in Lua, and not stock's)
never shows, so the output compares with the stock interpreters and the order
only has to agree between Oximond's modes. The loops that fill the tables name
their keys at run time (`tq["k" .. i]`), and the writes use the keys the
traversal gives, so the traces store under keys no constant names.
`--exclude table_traversal` leaves the family out. Programs of generator 26 and
earlier replay with it off.

The campaign on the commit that added the helpers for these stores (release
build, all three modes, `--jobs 3`, the machine busy with other sessions):

```sh
scripts/heavy.sh -m 8G target/release/oximond-difftest run --start 1000 \
    --count 2000 --profiles all --modes interpreter,tracing,tracing-async --jobs 3
```

2000 seeds on each of the six profiles, 12000 cases, no case where the modes
printed different output. lua54, lua55 and luajit21 passed against `lua5.4`,
`lua5.5` and `luajit` on all 6000. lua52 and lua53 have no oracle here: all
4000 printed the same output in the three modes. lua51 differed from `lua5.1`
on 254 of 2000 seeds, the float and `math.modf` cases of the scalar campaign
above, with all three Oximond modes printing the same output on every one.
Two checks that the family adds none of them: 600 lua51 seeds with
`--exclude table_traversal` differed on 75 and 600 without the exclusion on
67, and a lua51 case reduced to 273 lines (seed 4, a `L25` observation)
contains no traversal statement. The `table_traversal` family was in 10828 of
the 12000 cases, and 10304 of those ran native code.

An earlier run of the same seeds, on the commit that sent every store under a
computed key to one general helper, had one lua53 case (seed 1195) where
`tracing-async` hit the 20 second limit with no trace compiled, while the
machine's load average was 30. The same case finished in 2 to 4 seconds in 86
reruns (async and sync, with and without the `prlimit` memory cap), and all
2000 lua53 seeds agreed in the final run above. I put it down to the
overloaded machine, not to the traversals, but I did not find the cause.

## String arithmetic campaign

Generator 24 added `string_arithmetic`: numeric expressions that take one or
two operands from the `numeric_strings` table (`"10"`, `"42"`, `"7"`, `" 7 "`,
`"0x10"`, `"2.5"`, `"1e2"`, `"-3"`, `"0.5"`, `"3."`) and combine them with a
number or with each other using `+`, `-`, `*`, `/`, `%` and `^` (`//` where
the profile has integers), or negate one. Generator 25 also draws the family
inside kernel sections, where most compiled traces come from. Each kernel
shadows the table with a local copy (`local numeric_strings = numeric_strings`)
so a locals-only kernel can read it. Hot loops with these expressions make
the tracing JIT convert the string with the string-arithmetic helper.
Integer and float results differ per profile (5.3 and later keep an integer
for `"10" + 1` where the string looks like one), so the programs check the
conversion rules as well as the recorder. `--exclude string_arithmetic` leaves
the family out. Programs of generator 23 and earlier replay with it off.

The campaign on the commit that moved the family into kernels (debug build,
both Oximond modes, `--jobs 4`):

```sh
scripts/heavy.sh -m 8G target/debug/oximond-difftest run --seeds 5000..5060 \
    --profiles lua51,lua53,lua54,luajit21 --modes interpreter,tracing \
    --oximond target/debug/oximond
```

240 cases. The interpreter and the tracing mode printed the same output in
every case. lua54 and luajit21 agreed with their stock interpreters. lua53 has
no oracle, so those cases are inconclusive but the two modes agreed. lua51
differed from `lua5.1` on 11 seeds (5003, 5005, 5009, 5010, 5013, 5015, 5023,
5028, 5038, 5045, 5046): the fused-`%` artefact of the `lua5.1` build seen in
the scalar campaign. With `--exclude float_modulo` all 60 lua51 cases pass,
and the pre-flow build prints the same output as ours on those seeds.

To see that the helper really ran, each generated program was run with
`OXIMOND_JIT_TRACE_IR=1` and searched for `HelperId(176)` or `HelperId(177)`
(`string.number_integer` and `string.number_float`) in a recorded trace.
Programs with a `numeric_strings[` operand that recorded the conversion
helper, by profile: lua51 13 of 53, lua53 11 of 54, lua54 12 of 54, luajit21
19 of 53. Before the kernel change the count was 0 of 45.

## Operator metamethod campaign

Generator 28 added `operator_classes`: a section of its own that builds two
classes with every operator metamethod a trace can run as an inlined Lua call
and applies them in a hot loop. The handlers cover arithmetic (`+ - * / % ^`,
and `//` with the bitwise operators on the profiles with integers), `-x`,
`~x`, `..`, `==`, `<` and `<=`, `tostring`, `#` and `pairs` (`__len` and
`__pairs` on the profiles that ask for them). The loop applies them to an
object on either side of an object, a number or a numeric string, to
two objects of one class or of both, and to one object twice. A handler answers
with a number, a string, a boolean, nothing, two values (the second is dropped),
or the object itself, some after a store into it. The comparison handlers are
shared by both classes, as Lua 5.1 and 5.2 need to call them across classes,
and the `__le` handler is left out where the profile falls back on `not
(b < a)` (every profile but Lua 5.5, which has no fallback) about half the
time. Part way through the loop objects change class and a handler is
replaced, so guards fail and side traces grow, and in one section in four a
handler raises, which a `pcall` around the loop catches. A `pairs` loop with
a `__pairs` handler has a loop of its own: inside the hot loop it would turn
the loop into an outer loop, which no trace covers. Every result folds into a
checksum, so nothing prints inside the loop.

Three quarters of the sections have objects of one class, so a trace serves
them all and the loop runs natively; the rest mix both classes, which leaves
the trace at every second object until a side trace takes the second class.
The second class may return a number from `__tostring`, which a trace leaves
to the interpreter on the profiles that check (it formats the number, or
raises for anything else); the first class never does.

`--exclude operator_classes` leaves the section out. The section draws from
the random stream only when it is on, so with it off a seed gives the program
generator 26 gave: the test `the_version_26_streams_survive_the_operator_switch`
pins twelve of them (six profiles, two seeds) by fingerprints taken from the
generator 26 binary. A frozen regression that replays an older stream sets
`operator_classes = false` beside the other switches.

The campaigns on the commit that added the section (release build, `--jobs 4`):

```sh
scripts/heavy.sh -m 8G target/debug/oximond-difftest run --seeds 0..1000 \
    --profiles all --modes interpreter,tracing --oximond target/release/oximond
scripts/heavy.sh -m 8G target/debug/oximond-difftest run --seeds 1000..1400 \
    --profiles all --modes interpreter,tracing,tracing-async \
    --oximond target/release/oximond
```

The first ran 6000 cases and the second 2400. The section was in 4042 and 1625
of them, and compiled code ran in 4041 and 1624. No case printed different
output in the interpreter and in a tracing mode: of the failure directories,
none held differing `interpreter.stdout`, `tracing.stdout` and
`tracing-async.stdout`, and no failure was at a tag of the section (`OP`).
By profile:

- lua54, lua55 and luajit21 agreed with their stock interpreters on all 1400
  seeds, except one lua54 seed (1055) of the second run that stock Lua 5.4
  could not finish ("not enough memory" under the 2 GiB limit; the program has
  no operator section, and the case is oracle-invalid, not a mismatch).
- lua52 and lua53 have no oracle here, so their cases are inconclusive; the
  modes agreed on all 1400 seeds of each.
- lua51 passed 894 of 1000 and 353 of 400, and differed from `lua5.1` on 106 and
  47 seeds. In every one of them the interpreter and the tracing modes printed
  the same, at the tags of the older families (the float and `math.modf`
  cases of the scalar campaign above and the fused-`%` artefact of the
  reference build); none at an `OP` tag.

To see how much of a section runs natively, each section was cut out of its
program with the prelude and run by itself under `OXIMOND_COUNTERS=1` (seeds
100 to 139, about 40 sections per profile). Every profile compiled traces for
nearly every loop: 74 to 118 traces and 15 000 to 25 000 native entries per
profile, with the remaining aborts the inner `pairs` loop around which no outer
loop is covered (`InnerLoop`, `LoopNotClosed`), the number-returning
`__tostring` of the second class (`abort_native.__tostring`), and loops with
more than sixteen distinct handlers (`LimitExceeded`).
