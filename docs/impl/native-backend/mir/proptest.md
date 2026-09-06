# Differential testing of the JIT against the interpreter

Stage 3 added a property test to `oximond-difftest`. For each generated
hot-loop program and profile, five tracing variants must print what oximond's
interpreter prints, end with the same error and succeed or fail alike. See
`docs/difftest.md` ("JIT against the interpreter") for the commands.

## What exists

- Oracle: `Oracle::interpreter`, which runs `oximond --compat P --execution
  interpreter`. No stock Lua is needed. An uncaught Lua error in the oracle is
  a result to match.
- Variants (modes): `tracing` (Cranelift), `tracing-profile`, `tracing-mir`,
  `tracing-mir-profile`, `tracing-mir-crossprofile`, plus `tracing-async` and
  `tracing-mir-async`. A profile variant records in an earlier run, then loads
  the file. The cross variant records under Cranelift and loads under MIR.
- Generator bias: `generate_hot` and `HotKnobs` (opt-in, default programs are
  unchanged). Kernels with thousands of trips, calls, closures, upvalues,
  string building and string keys, guard-exit phase changes, nested loops.
- `oximond-difftest run-jit`: seeded campaign, saves mismatches with source,
  variant, both outputs and replay commands, `--reduce` on request. It exits 1
  if a variant enters native code in under 60% of the cases or a MIR variant
  compiles no tree through MIR.
- `tests/jit_property.rs`: proptest strategy over seed, profile and knobs,
  shrinking, then the difftest reducer, then a report with the reduced program
  and a replay command. Four cases under every variant plus 8 light cases under
  `tracing` and `tracing-mir`, from proptest seed 0 by default (40 to 105 s with
  the debug binary, depending on machine load).
  `OXIMOND_PROPTEST_CASES` and `OXIMOND_PROPTEST_LIGHT_CASES` raise it. Fake-binary tests show that a
  wrong variant, a wrong profile load, a run with no native code and a MIR
  variant with no MIR compile all fail.

## Non-vacuity

Native entry needs a loop with enough guest work to repay a compile
(`COMPILE_REPAYING_GUEST_WORK`, 262144), and no construct the recorder rejects.
Two things dominate. Plain `tracing` often does not compile a short loop that a
loaded profile compiles at once, so the profile variants always report a higher
native share. And `rawget` and a key chosen with `and`/`or` abort recording for
good, so the hot generator leaves them out unless the `UNSUPPORTED` knob bit is
on. Tracing runs print statistics only when the program ends without an error.

The stage-3 review found the default run vacuous for proptest seed 1: six cases,
2 of them native (33%). Four programs never entered native code. Two had the
`UNSUPPORTED` bit (about a quarter of the seeds keep it through the feature
mask). One had 6000 trips of 4 statements, under the 262144 guest instructions a
compile has to repay. One had only kernels the recorder rejects without that
bit: `unpack` over an open call (`UnsupportedOperand`), indexing a table
constructor (`VirtualEscape`) or a short inner loop (`InnerLoop`). The generator
draws those on purpose, and the campaign tolerates it because its native share
is over many cases (82% for plain tracing). A property run of 4 to 6 cases does
not, so for some seed it fails.

The first fix (`adf3ac7f`) was structural. `HotKnobs::anchor` puts a fixed
kernel (`ANCHOR_KERNEL`, 12000 trips of integer arithmetic, one array read and
write, one branch, about 30 guest instructions each) before the random
sections, and the property strategy turns it on and removes the `UNSUPPORTED`
bit. The review found that this made the 60% check measure the anchor: the
share was 100% for every variant, and a JIT that stopped compiling every
generated section would still have passed. The anchor stays, because it keeps
the profile and MIR checks meaningful in a short run, but the check no longer
counts it.

Now the harness runs the anchor alone (`HotKnobs::anchor_only`, zero random
sections) under every variant for each profile in the run, and counts a case as
native only when its native entries exceed that baseline. MIR trees are counted
the same way. A fake binary whose only native code is the anchor fails the
check (`a_run_whose_only_native_code_is_the_anchor_is_vacuous`).

That exposed the real rate. Probing 40 strategy draws of seed 7 in plain
tracing without the anchor, 34 entered native code (85%). The six that did not
had either too little guest work (a short kernel never reaches the 262144
instructions a compile has to repay: `recording_starts` was 0) or a construct
the recorder rejects (`recording_aborts` and `unroll_restarts` set). With four
cases at 85% per case, about one seed in ten falls under 60% for plain tracing;
seed 7 did (2 of 4). The profile variants are not the weak ones, since a loaded
profile compiles at once.

So the property draws a larger sample for the weak variants. After the four
full cases it runs 8 light cases under `tracing` and `tracing-mir` only. A
profile variant costs several times a plain one, so the light cases add
12 draws to those two variants for about a third of the price of 8 full cases.
At 85% per case, 12 draws reach 60% with probability 0.98. The random sections
stay free to draw kernels the recorder rejects, because that is part of what
the campaign tests. The same change lowers the strategy caps (sections 2 to 3,
statements 3 to 6, depth 1 to 2, trips 4000 to 8000), because a loaded profile
makes long loops expensive (see "Open profile-path issues"). Before the cap
change a default case took 30 to 40 s, with one case at over 4 minutes in the
three profile variants.

## First campaign

Command, with the release `oximond` of this branch (built 1 Nov 18:05, sha256
722d2662e0d0f8041bfe8e1d1fb65e962a1f74f1a91de999d9d5eb2289266fd5):

```sh
scripts/heavy.sh -m 16G target/debug/oximond-difftest run-jit \
    --seeds 0..300 --profiles lua51,lua54,lua55,luajit21 --jobs 8 --reduce
```

1200 cases (300 seeds on each of four profiles), 316 s. Native entry share and
MIR compile share are over the cases per variant and profile.

| profile | variant | native share | MIR cases | MIR trees compiled / fell back |
|---|---|---|---|---|
| lua51 | tracing | 81% | 0 | 0 / 0 |
| lua51 | tracing-mir | 81% | 224 | 655 / 304 |
| lua51 | tracing-profile | 94% | 0 | 0 / 0 |
| lua51 | tracing-mir-profile | 94% | 265 | 988 / 1038 |
| lua51 | tracing-mir-crossprofile | 94% | 265 | 988 / 1038 |
| lua54 | tracing | 86% | 0 | 0 / 0 |
| lua54 | tracing-mir | 86% | 178 | 612 / 798 |
| lua54 | tracing-profile | 97% | 0 | 0 / 0 |
| lua54 | tracing-mir-profile | 97% | 207 | 1186 / 1319 |
| lua54 | tracing-mir-crossprofile | 97% | 207 | 1186 / 1319 |
| lua55 | tracing | 84% | 0 | 0 / 0 |
| lua55 | tracing-mir | 84% | 156 | 516 / 762 |
| lua55 | tracing-profile | 95% | 0 | 0 / 0 |
| lua55 | tracing-mir-profile | 95% | 196 | 940 / 1251 |
| lua55 | tracing-mir-crossprofile | 95% | 196 | 940 / 1251 |
| luajit21 | tracing | 81% | 0 | 0 / 0 |
| luajit21 | tracing-mir | 81% | 212 | 589 / 401 |
| luajit21 | tracing-profile | 96% | 0 | 0 / 0 |
| luajit21 | tracing-mir-profile | 96% | 251 | 898 / 1074 |
| luajit21 | tracing-mir-crossprofile | 96% | 251 | 898 / 1074 |

Every case had a successful interpreter run (no
errored cases).

Result: 1199 pass, 1 mismatch (lua51 seed 43). Every variant failed it,
Cranelift included, so it is not a MIR bug.

### Mismatch: output printed again on guard exits

A program prints a line, then runs a hot loop that concatenates two objects
(`o1 .. "tt" .. o2 .. "tt"`, both with `__concat`) and swaps the metatable of two
of them part way through. Under tracing the line printed before the loop appears
14 times, once per exit of one guard kind (`side_exits[1]`), where the
interpreter prints it once. 14 lines of output become 46 in the generated
program. A 20-line repro is the corpus entry
`crates/oximond-difftest/corpus/output-repeats-after-concat-exits-lua51` with
`tracing:stdout@?` as its known failure (the in-process replay shows it too, so
it is not the CLI's output buffering). It needs the metatable swap, two different
objects and four concat operands. It is fixed in the long campaign below (cause 1).

## Long campaign

Command, with the release `oximond` built from `d462bd64` (sha256
722d2662e0d0f8041bfe8e1d1fb65e962a1f74f1a91de999d9d5eb2289266fd5), through the
gate:

```sh
scripts/heavy.sh -m 20G target/debug/oximond-difftest run-jit --seeds 0..3000 \
    --profiles all --jobs 8 --reduce --oximond PATH-TO-OXIMOND \
    --variants tracing,tracing-profile,tracing-mir,tracing-mir-profile,tracing-mir-crossprofile,tracing-async,tracing-mir-async
```

3000 seeds on each of the six profiles, seven variants (synchronous and
background compilation for both backends), 18000 cases and 6319 s. Each case
runs the oracle and seven variants, and the four profile variants record first.

| variant | cases | native share | native entries | cases with MIR trees | MIR trees compiled / fell back |
|---|---|---|---|---|---|
| tracing | 18000 | 82% | 15367881 | 0 | 0 / 0 |
| tracing-async | 18000 | 80% | 23639213 | 0 | 0 / 0 |
| tracing-profile | 18000 | 95% | 16617080 | 0 | 0 / 0 |
| tracing-mir | 18000 | 82% | 15367881 | 11238 (62%) | 37023 / 37865 |
| tracing-mir-async | 18000 | 81% | 23102893 | 11182 (62%) | 36572 / 35476 |
| tracing-mir-profile | 18000 | 95% | 16617121 | 13763 (76%) | 63257 / 77333 |
| tracing-mir-crossprofile | 18000 | 95% | 16618561 | 13763 (76%) | 63269 / 77408 |

Result: 18000 cases, 34 mismatches (lua51 4, lua52 9, lua53 7, lua54 4, lua55 5,
luajit21 5). No case failed in a MIR variant alone. Every mismatch failed the
Cranelift variants too, so none is a MIR backend bug.

### Bugs

| cause | what | cases | reduced repro | fix |
|---|---|---|---|---|
| 1. recorder | The guard on the second `__concat` handler of a chain used the pc of the first handler's return. A failing guard resumed the interpreter in the main chunk at that pc, which printed earlier lines again or called the wrong register. | 16: lua51 43, 2892; lua52 223; lua53 1318, 1776, 2641; lua54 1294, 2605, 2884; lua55 291, 392, 414, 514; luajit21 2634, 2770, 2850 | `concat-chain-guard-after-handler-lua54`, `output-repeats-after-concat-exits-lua51` | `c50d4457`, `inline_operator.rs` |
| 2. recorder | A `..` chain such as `1.5 .. o .. "2.5"` rebuilds the handler frame at an exit and reads the operands not yet folded from the caller's registers. Those are dead after the instruction, so the snapshot gave nil and the return failed with "attempt to concatenate a nil value". | 4: lua52 581, 860; lua53 2984; luajit21 950 | `concat-pending-operand-in-handler-lua52` | `c50d4457`, `recorder.rs` (`held` registers in snapshots) |
| 3. optimizer | A loop side that a parent trace links into natively got a branch guard hoisted into its preheader. The guard exits through an entry-state snapshot that leaves registers `Unchanged`, but the interpreter does not hold the parent's registers yet. The inner numeric loop resumed with a function as its limit. | 4: lua51 1109; lua52 876, 1444, 2613 | `hoisted-guard-entry-state-after-link-lua51` | `c50d4457`, `optimize.rs`: only the root hoists guards |
| 4. harness (profile load) | Loading a trace profile keeps the interpreter off its fast loop for the whole program (`has_loaded_profile` in `fast_path_eligible_hooked` and `native_entry_possible`), so an interpreter-bound program takes about 7 times as long and hit the 20 s limit. Wrong answers: none. | 7: lua51 2759; lua52 2017, 2157, 2584; lua53 579, 1905; lua55 2364 | none (timeouts) | profile-load runs get 4 times the timeout. The speed cliff stays open, see "Open profile-path issues" |
| 5. recorder or native code, open | A hot loop rebinds a table at one iteration and then reads it with `unpack` or writes it by a computed index. Values differ from the interpreter's afterwards, or `math.random` continues on another stream. MIR and Cranelift fail alike. Not the optimizer: both repros fail the same at `--opt-level=0`, and not a profile problem, since they fail without one. They need synchronous compilation (`OXIMOND_JIT_SYNC=1`, what the harness sets): the debug CLI with background compilation prints the interpreter's output. | 3: lua53 1226, lua54 728, luajit21 2723 | `table-rebound-in-hot-loop-luajit21`, `math-random-stream-diverges-lua54` (both `.known-failure`) | none; not found small and local |
| 6. verifier (optimized trace) | A loop recorded through an unrolled inner loop calls an effect (`math.random`) at one pc several times, with only the invariant constants of the `forloop` check between the calls. Loading a profile recompiles the saved trace, and an optimizer pass (invariant code motion, value numbering) took those constants out. The verifier finds a guest step as a run of neighbours at one site, so the calls then read as one step and a call's guard looked like a replay of the previous call's effect: "snapshot 1 resumes before an instruction whose effect already committed". The debug validator panicked (exit 101). Release skips that validator. From reading `optimize_pipeline` (not run with a release build), its final verify then rejects the trace and returns the unoptimized one, so release would run correct but slower code. The optimized trace was right in every case. | 2: proptest seed 3 (lua55, in `tracing-profile`, `tracing-mir-profile` and `tracing-mir-crossprofile`), campaign lua54 44 (value numbering, same three variants) | `math.randomseed(42)` and `for i=1,52 do for j=1,2 do k = math.random(7) end end` (record-then-load test `a_loaded_profile_compiles_an_unrolled_loop_around_an_effect` in `crates/oximond/tests/trace_profile.rs`), and the lua54 44 reduction, 15 nodes (`..._a_longer_unrolled_loop_...`, same file), plus `an_optimized_trace_may_put_two_runs_of_one_effectful_pc_side_by_side` in `optimize_tests.rs`. The corpus replays only interpreter against tracing, never a profile, so it has no entry | `09fe6da9`: `TraceIr::verify_optimized` skips the two adjacency checks on optimizer output; an earlier pin of the separator in `hoist_invariants` (`071c7d41`) fixed one pass and was replaced |
| 7. harness (vacuous run) | The default property run, 6 cases from proptest seed 1, had 2 native cases (33%) and failed its own 60% check. Random kernels are often rejected by the recorder or too short to repay a compile. | proptest seed 1 | none (a property of the strategy) | `adf3ac7f`: `HotKnobs::anchor`, no `UNSUPPORTED` bit in the strategy, lower caps; then the anchor baseline in the native count and 8 light cases (the anchor alone had made the share 100%) |

Cause 6 is not related to cause 5. It needs a loaded profile and an effect at
one pc several times, and it is fixed. Cause 5 fails without a profile and at
`--opt-level=0`. The `math.random` symptom of cause 5 is a stream that draws one
value too many or too few, which is also what a replayed effect would do, but
nothing here shows a replay: the verifier accepts those traces, and the
recorded trace of cause 6 never replayed anything either.

An earlier fix for cause 3 named every entry input in the snapshot. It also
made the tree builder fail, so native entries fell from 80% to 33% of the cases
and hid the cause 5 cases. The narrower rule above keeps the native share where
it was. Check the native share after any change to the optimizer.

Each fixed bug except causes 4 and 6 has a corpus entry that fails with the
binary from before the fix and passes after it, under both backends (checked by
running the entries with the baseline binary and with `OXIMOND_JIT_BACKEND=mir`).
Cause 4 is a timeout and cause 6 needs a loaded profile, which the corpus replay
never loads. `cargo test -p oximond-difftest` replays the corpus, and replays the
two open entries as known failures.

### Rerun after the fixes

The same 3000 seeds on six profiles and seven variants with the fixed binary
(5350 s): 18000 cases, 3 mismatches, all of cause 5 (lua53 1226, lua54 728,
luajit21 2723). Native share and MIR share match the table above (82% for
`tracing`, 95% for the profile variants, 62% of the `tracing-mir` cases
compiled at least one tree through MIR).

### Open profile-path issues

`has_loaded_profile` turns off the interpreter's fast loop and native entry for
the whole run once any record is loaded, whether or not the running chunk has
one. On the stage-3 campaign this was a 7 times slowdown on interpreter-bound
programs (cause 4). A default-property case measured here: the interpreter
3.9 s, the recording run 4.0 s, the run that loads that profile 43 s. The
harness answers it with a 4 times timeout and with smaller programs. The right
fix is a rule that asks whether the loaded profile still has an unattempted
record for the chunk that runs, and that gives the fast loop back to every other
chunk. That changes the interpreter dispatch rule and needs a benchmark
count, so it is open.

### Debug-binary slice with the optimizer validator on

The stage-3 campaign ran a release binary, which skips the per-pass verifier.
This slice reruns the campaign with a debug build of the branch (`target/debug/oximond`
at `adf3ac7f`, sha256 f9e163cb35a9ab12...; it includes the first cause 6 pin),
so every optimizer pass is verified. Through the gate, at 8 jobs:

```sh
scripts/heavy.sh -m 20G target/debug/oximond-difftest run-jit --seeds 0..300 \
    --profiles all --jobs 8 --reduce --timeout 90 --oximond target/debug/oximond \
    --variants tracing,tracing-profile,tracing-mir,tracing-mir-profile,tracing-mir-crossprofile,tracing-async,tracing-mir-async
```

300 seeds on each of the six profiles and seven variants: 1800 cases in 6416 s
(the machine ran other sessions' tests and builds throughout). 1798 passed and
2 mismatched, both cause 6 (lua54 44 through value numbering, lua53 204 through
guard elimination): the three profile variants crashed with exit 101 on a pass
that "produced an invalid trace", and the other four variants passed. Both
reduce to a few lines (lua54 44 to 15 nodes) and are tests in
`crates/oximond/tests/trace_profile.rs`. With the binary at `09fe6da9` both
seeds pass in all seven variants.

| variant | cases | native share | native entries | cases with MIR trees | MIR trees compiled / fell back |
|---|---|---|---|---|---|
| tracing | 1800 | 84% | 1556810 | 0 | 0 / 0 |
| tracing-async | 1800 | 83% | 3018020 | 0 | 0 / 0 |
| tracing-profile | 1800 | 96% | 1884688 | 0 | 0 / 0 |
| tracing-mir | 1800 | 84% | 1556810 | 1169 (65%) | 3939 / 3531 |
| tracing-mir-async | 1800 | 83% | 2800238 | 1162 (65%) | 3843 / 3272 |
| tracing-mir-profile | 1800 | 96% | 1884688 | 1394 (77%) | 6771 / 7646 |
| tracing-mir-crossprofile | 1800 | 96% | 1884688 | 1394 (77%) | 6771 / 7646 |

Cause 5 did not show: it needs more seeds than 300 (3 of 18000 before). The
shares match the release run.

### Raised-count proptest, debug binary

`OXIMOND_PROPTEST_CASES=200 OXIMOND_PROPTEST_SEED=7 OXIMOND_PROPTEST_ASYNC=1`
through the gate (`scripts/heavy.sh -m 16G cargo test -p oximond-difftest --test
jit_property tracing_variants_match`), debug binary built from `adf3ac7f`: 200
programs over six profiles and seven variants passed in 2692 s. Native share
was 100% for every variant except `tracing-async` (196 of 200). The MIR variants
compiled 729 trees through MIR and handed back 359 (sync), 723 / 331 (async);
the two profile variants 749 / 821.

### Proptest run (stage 3, release)

`OXIMOND_PROPTEST_CASES=400 OXIMOND_PROPTEST_SEED=7 OXIMOND_PROPTEST_ASYNC=1`
with the fixed release binary: 400 programs over six profiles and seven variants
passed in 615 s. Native share was 82% for `tracing`, `tracing-async`,
`tracing-mir` and `tracing-mir-async`, and 95% for the three profile variants.
The MIR variants compiled 773 trees through MIR and handed back 798 (sync), and
766 / 767 (async); the profile variants 1294 / 1642. The strategy draws the
seed's base shape the way a campaign does and caps it. The caps shrink toward
two sections, three statements, depth one and the fewest trips. The value-type mix shrinks to none and the feature mask toward zero (the
strings feature stays on, so a program still has output to compare).

### Test suites

`scripts/heavy.sh -m 24G cargo test -p oximond -p oximond-cli -p oximond-mir -p
oximond-difftest --no-fail-fast` at the final commit:

- `OXIMOND_JIT_BACKEND` unset: 4595 passed, 0 failed. An earlier run of the same
  command failed one test, `a_good_fake_passes_and_counts_native_runs`: the oracle
  run of the fake shell script failed once, while the real property test ran in
  the same process. The suspected cause is ETXTBSY (a forked child of another
  test thread held the freshly written script open for writing). It did not
  reproduce in six reruns, three without the fix. The test now runs the fake
  once and retries on errno 26 before it counts anything.
- `OXIMOND_JIT_BACKEND=mir`: 4587 passed and the 8 known failures, nothing else
  (`float_fmod_uses_fma_only_when_the_isa_has_it`,
  `float_modulo_forms_call_a_helper_only_off_the_inline_path`,
  `clif_instructions_of_a_side_compiled_apart_are_fewer_than_the_tree_that_holds_it`,
  `a_side_that_links_back_is_compiled_alone_after_its_tree_runs`,
  `compiling_a_side_costs_less_than_compiling_the_tree_with_it`,
  `native_code_calls_the_side_that_returns_into_the_calling_loop`,
  `the_loop_variables_after_a_run_that_native_links_left_are_the_interpreters` and
  `a_hot_nest_native_code_links_in_still_stops_for_a_count_hook`).

The default property test (4 full and 8 light cases, proptest seed 0) passes
for every seed from 0 to 10 with the debug binary, 39 to 105 s each at load
average 13, counting only native entries beyond the anchor. No seed fails, so
none needs a bug-table entry. The earlier failures are accounted for: seed 1 was
vacuous (cause 7) and seed 3 was cause 6. Before the light cases, seed 7 failed
the corrected check with 2 of 4 plain-tracing cases native; that is the
sampling rate above, not a bug. The default seed is 0 because it is the neutral
value, not because it passes; seed 4, the previous default, was picked for that
reason. The generator output for a given proptest seed changed with this work,
so the seeds of the old runs do not draw the same programs.

Cause 7 (harness, vacuous default run) is the seed 1 finding in "Non-vacuity".
Fixes: `adf3ac7f` (anchor kernel, no `UNSUPPORTED` bit, lower caps) and the
anchor baseline plus light cases that follow it.

### Commands to rerun

```sh
cargo build --release -p oximond-cli && cargo build -p oximond-difftest
scripts/heavy.sh -m 20G target/debug/oximond-difftest run-jit --seeds 0..3000 \
    --profiles all --jobs 8 --reduce --oximond $PWD/target/release/oximond \
    --variants tracing,tracing-profile,tracing-mir,tracing-mir-profile,tracing-mir-crossprofile,tracing-async,tracing-mir-async
# the debug slice: the same command with --seeds 0..300 --timeout 90 and
# --oximond $PWD/target/debug/oximond, which verifies every optimizer pass
OXIMOND_PROPTEST_CASES=400 OXIMOND_PROPTEST_SEED=7 OXIMOND_PROPTEST_ASYNC=1 \
OXIMOND_DIFFTEST_OXIMOND=$PWD/target/release/oximond \
scripts/heavy.sh -m 16G cargo test -p oximond-difftest --test jit_property
```

## Caveats

- The default property test uses the debug binary unless
  `OXIMOND_DIFFTEST_OXIMOND` names another, and builds it if missing.
- A concurrent heavy job can push a case past the 60 s per-process timeout.
- The async variants are covered by `--variants tracing-async,tracing-mir-async`
  but were not part of this first campaign, because install timing varies.
