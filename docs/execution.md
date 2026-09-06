# Execution modes and the CLI

Compatibility and execution are separate settings. `--compat` accepts `lua51`,
`lua52`, `lua53`, `lua54`, `lua55`, and `luajit21`. The modern profiles include
version-gated operators, escapes, local attributes, goto and labels, Lua 5.5
global declarations and named varargs, and their generation-specific
standard-library additions. `--execution` accepts `interpreter` and `tracing`.
The interpreter is the default.

```sh
cargo run -p oximond-cli -- --compat lua51 --execution interpreter -e 'print(_VERSION)'
cargo run -p oximond-cli -- --execution tracing --opt-level 2 bench.lua
```

## Tracing

Tracing mode starts in the interpreter and records traces of hot loops. A trace
is typed SSA with guards. It inlines calls to Lua closures, reads and writes
table slots it bound at entry, and grows side traces from guards that fail
often. A side for a guard that fails rarely is compiled apart and called from
the exit it continues, so growing a tree does not compile the tree again (see
the trace policy's module documentation). Cranelift compiles each trace tree
on a background thread, or at the next safepoint when the process may run on
one CPU only, where a worker would just wait for the interpreter's time slice.
A guard that fails, or an operation a trace cannot hold, restores the
interpreter's frame from a snapshot and the interpreter continues. Native loops
return to VM garbage-collection safe points on a fuel budget.

## Optimization levels and flags

`--opt-level` sets how much compile time tracing mode may spend on a trace. It
accepts `0`, `1`, and `2`, and the default is `0`. Level 0 holds every trace
optimization Oximond does today, including both tiers of the trace optimizer.
Levels 1 and 2 are for advanced optimizations of frequently run code, which
spend longer compiling a hot trace to make it run faster. Both add bit 3 and
nothing else so far. The interpreter ignores the setting. Embedders set it with
`Builder::tracing_opt_level(TracingOptLevel::O2)`.

Each level stands for a 64-bit set of optimization flags, one bit per optional
pass. Today bit 0 hoists invariant guards and bit 1 runs the memory passes
(store-to-load forwarding, load merging, invariant table reads). Bit 2 checks
loop array bounds once. Bit 3 prepares constant patterns of `string.find`,
`string.match`, `string.gmatch` and `string.gsub`, and constant format strings
of `string.format`, once when the trace compiles instead of on every call.
`--opt-flags=BITS` replaces the level's flags, so `--opt-flags=0x1` turns the
memory passes off and `--opt-flags=0` keeps only the passes every trace needs.
Embedders use
`Builder::tracing_opt_flags(TracingOptLevel::O0.flags().without(TracingOptFlags::MEMORY))`.
A flag only allows a pass. Traces at the baseline compile tier still skip the
expensive ones.

## Trace profiles

A tracing run can record which traces it compiled and a later run can load them:

```sh
cargo run -p oximond-cli -- --execution tracing --trace-profile-record bench.profile bench.lua
cargo run -p oximond-cli -- --execution tracing --trace-profile-load bench.profile bench.lua
```

The CLI also accepts `OXIMOND_TRACE_PROFILE_RECORD` and
`OXIMOND_TRACE_PROFILE_LOAD` as path alternatives to the flags. Profile files
are written even when the Lua program fails or calls `os.exit`.

On replay, a matching trace compiles at its first recorded header. The VM
interprets setup code and any paths the profile does not cover. Profiles are
build-specific, so regenerate them after changing the trace compiler or the
installed native libraries.

Embedders use the same API on `Vm`; see [embedding](embedding.md#trace-profiles).
For per-test profiles in conformance and difftest runs, see
[conformance](conformance.md) and [differential testing](difftest.md).

## Standard library notes

The full library includes `io.popen` and the interactive `debug.debug` console.
Lua 5.2 and later provide `package.searchpath` and debug uservalue access. Lua
5.4 and 5.5 support indexed uservalues. Package search uses the configured host
source reader, including virtual files. Lua 5.3 and later support
`math.atan(y, x)`. Lua 5.4/5.5 and LuaJIT use their respective seeded random
generators. LuaJIT's optional `table.new` and `table.clear` modules load through
`require`.

`os.clock` reports process CPU time on Linux when `/proc` and `getconf` are
available. Other native targets use elapsed wall time. `os.setlocale` supports
the `"C"` locale. The Rust-only package loader cannot load native Lua C modules.
The C API build provides native `package.loadlib` and C module searchers.

The standard library also includes Rust implementations of Redis's `cjson`,
`cjson.safe`, `struct`, `cmsgpack`, and `cmsgpack_safe` modules. Load them with
`require` under any compatibility profile.

## Source loading

File scripts, standard input, `loadfile`, `dofile`, and path-based `require`
read source in chunks of up to 8 KiB. Regular files larger than one read use
three concurrent stages. The calling thread reads, one worker lexes and parses,
and another compiles completed top-level statements. The two queues each hold
at most two batches. Statements move between workers without AST cloning.
Source text and the complete tree remain available for diagnostics and any
final compilation pass. Cancellation joins both workers.

Regular files of at most 8 KiB use the complete-input parser. The bounded probe
switches to the pipeline if the file grows beyond that size. Stdin, pipes,
virtual source hooks, and C API reader callbacks pull synchronously. A parser
failure can stop those inputs before another blocking read. Execution starts
after the complete chunk is compiled and validated. Serialized binary bytecode
still buffers the whole file. Hosts without native threads, and failed worker
creation, use the synchronous path.
