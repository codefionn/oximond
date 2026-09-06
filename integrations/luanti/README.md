# Luanti engine comparison

This integration pins Luanti 5.17.0 to commit
`c0e6812b1a4260bb25a1f606f70f55f4962bb97d`. The patch changes only the
Luanti checkout under `target/luanti/source`. Build output stays under `target`.
The upstream release is at <https://github.com/luanti-org/luanti/releases/tag/5.17.0>.

Build requirements include Rust and Cargo, a C++17 compiler, CMake, pkg-config,
LuaJIT development files, and Luanti's system libraries. On this Linux host,
Luanti uses SDL2, OpenGL, zlib, zstd, GMP, jsoncpp, SQLite3, PNG, JPEG, and
FreeType. CMake reports a missing required dependency during configuration.

Run from the Oximond repository root:

```sh
python3 integrations/luanti/luanti.py prepare
python3 integrations/luanti/luanti.py build both
python3 integrations/luanti/luanti.py smoke both
python3 integrations/luanti/luanti.py world-smoke both
python3 integrations/luanti/luanti.py run both
```

`build both` creates separate `build-luajit` and `build-oximond` trees. The
binary version strings include their engine names, so the client windows can
be distinguished. `run both` assigns ports 30041 and 30042 unless `--port`
is given.

The LuaJIT variant requires a system LuaJIT development package. The Oximond
variant builds the `oximond-capi` Rust static library, then links it with
Luanti's bundled Lua 5.1 auxiliary library, bundled bit library, and the C++
exception bridge. It does not link Luanti's bundled Lua interpreter.
Oximond runs in tracing mode (interpreter plus JIT) here. The C API speaks the
LuaJIT 2.1 dialect by default: Lua 5.1 plus `goto`, `\x`/`\z`/`\u{}` escapes
and the `bit` library. It installs no `jit` table, so mods take their portable
paths. `build oximond --capi-profile lua51` selects stock Lua 5.1 instead; it
maps to the `oximond-capi` Cargo features `luajit` (default) and `lua51`. The C API bridge is built for this
pinned Luanti comparison and is not a general Lua library replacement.

Each launch sets `LUANTI_USER_PATH` to its own `target/luanti/user-<engine>`
directory. Settings and worlds therefore stay separate. Add engine arguments
after `--`, for example `run oximond -- --gameid devtest`. `run both` starts two
clients at once. `smoke both` checks each binary's `--version` engine marker,
then runs Luanti's `TestLua` module without a display.
`world-smoke` starts each variant's separate `devtest` world as a headless
server. It waits up to 90 seconds for the server to listen, allows one second
of running time, and checks the log for errors. Its test config binds to
`127.0.0.1`. It needs local socket access.
`world-smoke --game mineclonia` does the same for Mineclonia, a much larger
game. The first run shallow-clones Mineclonia at a pinned tag into
`target/luanti/games` and links it into both user directories. `--game all`
runs every game. Raise `--timeout` for large games.

The driver refuses a checkout at an unexpected Luanti commit or one with
unrecognized changes. It can be rerun after the integration patch is already
applied.

## Benchmark

```sh
scripts/heavy.sh -x -m 12G python3 integrations/luanti/luanti.py bench \
  --variant both --workload mineclonia-emerge --reps 1
```

`bench` compares LuaJIT and Oximond on fixed workloads: `devtest-startup`,
`mineclonia-startup` and `mineclonia-emerge` (or `all`). Each run gets a fresh
world under `target/luanti/bench/user-<variant>`, with seed 1234567,
`num_emerge_threads = 1` and the `bench/oxbench` mod installed as a worldmod.
The mod times the workload and shuts the server down. The server runs under
`perf stat -e instructions:u,task-clock` pinned with `taskset`. Each run prints
one JSON line (variant, game, workload, rc, wall_s, instructions, cpu_ms,
emerge_us, blocks, and any `ERROR[` lines), and a ratio table follows.

Options:

- `--area R,YLO,YHI` sets the emerge area, x and z in `[-R, R-1]` and y in
  `[YLO, YHI]`. The default `64,-64,63` is 512 blocks.
- `--cpus` is the taskset list (default `4-7`). CPUs N and N+8 share a physical
  core, so pick CPUs whose siblings are idle.
- `--jit-trace` sets `OXIMOND_JIT_TRACE=1` for the Oximond run and prints the
  top 40 trace aborts grouped by opcode, reason and source:line.
- `--profile` replaces `perf stat` with `perf record -g` and prints, for the
  main and `Emerge-0` threads, the share of samples in the interpreter, C API
  boundary, JIT machine code, parse/compile, runtime, libc and engine C++.
  Instruction counts are not reported in that run. With `kernel.perf_event_paranoid = 2`
  perf refuses call chains; set `OXI_PERF_CALLGRAPH=none` to record flat
  user-space samples instead. The perf data lives in a
  temp directory that is removed afterwards.
- `--lua-profile` turns on a count-hook line sampler (one sample per 1000 VM
  instructions) through `secure.trusted_mods = oxbench` and prints the top
  lines per Lua environment (`main`, `mapgen`). It slows the run, so do not
  read counts from it.

- `--attribution` (Oximond only) runs the server under `perf record` with the
  interpreter sampler and the perf map on, and prints per environment thread
  (`Emerge-0` for the mapgen env): the top `--top` (default 30) interpreted
  functions and lines with their share of interpreter samples and the trace
  state of the loop around them (never-hot, counting, recording, aborted,
  rejected, retired, compiled with its exit count and its top interpreter
  exits by kind and guard, compiled-exited when abnormal exits reach a quarter
  of the entries, or covered by an outer tree) with the last abort reason and source:line; the top uncovered
  functions (every function with interpreter samples; under a compiled loop the
  row says the trace was compiled but this code still ran interpreted); the ticks each compiled trace spent in native code; the JIT
  samples per trace header from perf; and the exits that went back to the
  interpreter by kind and guard. Counts are samples, not instructions, so
  read shares, not totals.

The sampler behind it is the VM's own: `OXIMOND_ATTRIBUTION=<path>` attaches
a collector to every VM of the process (`%p` in the path is the process id),
ticks 1000 times a second (`OXIMOND_ATTRIBUTION_HZ`), and writes the report
about once a second, at `lua_close` and at VM drop. It uses no debug hook, so
traces record and run as they do without it. A tick is acted on at the next
backedge or call the interpreter reaches, so the line of a sample is the line
of that backedge or call, and time in native code is charged to the header the
trace was entered at. `OXIMOND_PERF_MAP=1` names each compiled trace in
`/tmp/perf-<pid>.map` as `oximond trace#<index> <source>:<line>` for
`perf report`, replacing the generic names Cranelift writes under `perf
record`. Both are off by default and cost nothing then.

Every run logs a probe line (`jit` table, `ffi`, ULL literals). The bench fails
if the two engines disagree on it.

Measurement caveats. The metric is `instructions:u`; wall time on this machine
is noise. Run through `scripts/heavy.sh -x` so no other gated job shares the
CPUs, and report `uptime` next to the numbers, since sessions outside the gate
still use CPU. Rebuild the Oximond variant after Rust changes
(`scripts/heavy.sh -m 16G python3 integrations/luanti/luanti.py build oximond --jobs 8`)
and measure the committed build. The first Mineclonia run clones the pinned game
into `target/luanti/games`; a worktree reuses the main checkout's clone when it
is at the pinned commit. Whole-process counts include startup. The reference
for the default emerge on the 5.17.0 pair is about 93.3 G instructions for
LuaJIT and 370 G for Oximond.
