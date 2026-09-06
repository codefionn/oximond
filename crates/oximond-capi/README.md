The C API crate builds a static library for one Lua ABI at a time. Select
`lua51`, `lua52`, `lua53`, `lua54`, or `lua55`. The default `luajit` feature
uses Lua 5.1's ABI and the LuaJIT language profile.

```sh
cargo build -p oximond-capi --release --no-default-features \
  --features lua54,bundled-bridge,small-blocks
```

The `small-blocks` feature, on by default, makes `oximond::alloc::SmallBlocks`
the Rust global allocator, which serves the runtime's small strings and
table parts from per-thread lists. Leave it out when a Rust program that
links the crate brings its own allocator. It has no effect on Windows.

On this shared checkout, run release builds through `scripts/heavy.sh`.
Link `liboximond_capi.a` with a C++ linker and `ffi`, `dl`, `pthread`, and
`m`. Compile the client against the matching `vendor/lua54` headers.
Each vendor directory records the upstream release, archive checksum, and
file checksums. The upstream auxiliary library and library registration
code run over Oximond's Rust VM. No upstream Lua interpreter is linked.

The selected ABI controls function signatures, registry indices, numeric
types, debug records, continuation contexts, and coroutine resume results.
The modern constructors start with unopened libraries, as stock Lua does.
Call `luaL_openlibs` or `luaL_requiref` to open them. The CLI continues to
export the Lua 5.1 ABI regardless of its selected language profile.

Native callbacks can create, resume, and yield coroutines. Lua 5.2 onward
also supports continuations for yielding calls and protected calls. The
C++ bridge catches errors and yields before returning to Rust. Code built
with `bundled-bridge` must not replace these operations with longjmp.

Run the native tests with:

```sh
bash crates/oximond-capi/tests/run.sh
bash crates/oximond-capi/tests/run-modern.sh
cargo test -p oximond-capi --features bundled-bridge
```

The last command matters because the CLI enables `lazy-bridge`, which embeds
the bridge as a shared object loaded on the first call into native code.
Cargo unifies features, so a workspace-wide `cargo test` builds the crate
with `lazy-bridge` and compiles out the unit tests that call bridge-defined
`lua_*` functions directly.

The modern runner checks exported core, auxiliary, and library symbols
against the public headers,
then loads a shared native module compiled for that ABI. Its tests cover
stack operations, integer precision, userdata, library opening, loading
and dumping, continuations, hooks, warnings, and closing coroutine resources.
These checks do not establish full stock Lua C API conformance. Oximond
bytecode has its own format. A custom Lua allocator does not control the
Rust VM's entire heap, and Lua 5.5 external string bytes are copied into
Rust storage even though the native interface preserves the supplied
pointer and release callback.
