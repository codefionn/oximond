# Embedding Oximond

The `oximond` crate wires the VM, compiler, and standard library behind one
builder:

```rust
use oximond::{CompatibilityProfile, ExecutionMode, Lua, TracingOptLevel};

let mut lua = Lua::builder()
    .profile(CompatibilityProfile::Lua54)
    .execution_mode(ExecutionMode::Tracing)
    .tracing_opt_level(TracingOptLevel::O0)
    .build()?;
let results = lua.exec("return 40 + 2")?;
```

See [execution modes](execution.md) for what the profile, execution mode, and
optimization settings do.

## Standard library selection

The builder installs the full standard library by default. For code that should
have no host access by default, use `StandardLibrary::Pure`. It keeps functions
such as `load`, `math`, `string`, and `table`, but initially omits `io`, `os`,
`print`, `loadfile`, `dofile`, `require`, `package`, and `debug`.

```rust
use oximond::{Lua, StandardLibrary};

let mut lua = Lua::builder()
    .standard_library(StandardLibrary::Pure)
    .build()?;
let results = lua.exec("return math.abs(-42)")?;
```

Hosts can add back only the functions a project needs:

- `io_installer` and `os_installer` register replacement tables.
- `host_installer` can register `print`, `warn`, or other callbacks and
  configure runtime output writers.
- `load_installer` and `require_installer` replace those Lua-visible functions.
- `library_installer` accepts other named globals.
- `source_reader` supplies bytes to `loadfile`, `dofile`, and path-based
  `require`, for virtual files.
- `module_resolver` resolves a module name before path search.

These hooks never fall back to host files when `Pure` is selected. The
`package_path` option controls virtual module paths too.

```rust
use oximond::{Lua, StandardLibrary, Value};

let mut lua = Lua::builder()
    .standard_library(StandardLibrary::Pure)
    .package_path("project/?.lua")
    .source_reader(|path| match path {
        Some(b"project/main.lua") => Ok(b"return require('answer')".to_vec()),
        Some(b"project/answer.lua") => Ok(b"return 42".to_vec()),
        _ => Err(std::io::ErrorKind::NotFound.into()),
    })
    .build()?;
assert_eq!(lua.exec("return dofile('project/main.lua')")?, vec![Value::Number(42.0)]);
```

With `Full`, replacement `io` and `os` tables take the place of their defaults.
Custom source readers likewise replace filesystem reads in the VM. Other full
host functions remain available unless replaced. `StandardLibrary::None`
installs no standard library globals. The older `.stdlib(bool)` option still
selects full or none.

## `os.exit`

The full `os.exit` does not end the host process. It unwinds the whole Lua
stack, past `pcall`, `xpcall`, and coroutines, and returns an error whose
`VmError::exit_request()` holds the status and the `close` flag. A closing
request has already run the main thread's pending `__close` handlers. Call
`Vm::run_close_finalizers` to run the finalizers as well. The `oximond` command
does that, flushes its output, and exits with the status.

## Trace profiles

Call `lua.vm_mut().start_trace_profile_recording()` before `exec`, then write
`lua.vm().trace_profile()?` to a file after execution. On a later tracing VM,
read the file and call `lua.vm_mut().load_trace_profile(&bytes)?` before
`exec`. See [trace profiles](execution.md#trace-profiles) for replay behavior.

## WebAssembly

The Rust embedding crate supports `wasm32-unknown-unknown`. Use the interpreter
and `StandardLibrary::Pure` in browsers or other hosts without filesystem
access:

```sh
rustup target add wasm32-unknown-unknown
cargo build -p oximond --target wasm32-unknown-unknown --release
```

Embed this crate in your host's WASM entry points. The build produces a Rust
library, so JavaScript bindings belong in the embedding application. Use
`module_resolver`, `source_reader`, and host installers for virtual modules and
host services.

Native JIT compilation and native FFI calls are unavailable on WebAssembly. JIT
and tracing execution fall back to the interpreter. The LuaJIT profile installs
`bit` and `jit` on WASM and omits native `ffi`. Explicitly requesting the FFI
extension returns an unsupported-target error.

Bare WASM has no built-in clock, so `os.clock` and `os.time()` return Lua
errors there. Replace the `os` table if your host supplies time. Explicit
calendar timestamps use UTC as the local timezone on bare WASM. Full host
libraries require services that bare WASM does not provide.

The smoke test builds a standalone WASM module and runs Lua under Node.js across
all six compatibility profiles:

```sh
cargo build --manifest-path tests/wasm-smoke/Cargo.toml --target wasm32-unknown-unknown --release --locked
node tests/wasm-smoke/run.mjs
```

## LuaJIT extension modules

Use `luajit_extension` to add `bit`, `ffi`, or `jit` without changing the Lua
language profile. For example, a Lua 5.4 state can use the LuaJIT bit library:

```rust
use oximond::{CompatibilityProfile, Lua, LuaJitExtension};

let mut lua = Lua::builder()
    .profile(CompatibilityProfile::Lua54)
    .luajit_extension(LuaJitExtension::Bit)
    .build()?;
let result = lua.exec("return bit.bxor(6, 3)")?;
```

`Jit` installs global `jit.on()`, `jit.off()`, `jit.status()`, and
`jit.flush()`. They control Oximond's selected JIT engine. An interpreter-only
state reports JIT disabled. Function-specific controls and LuaJIT optimizer
settings are not implemented.

`Ffi` uses native libffi and the process dynamic loader. `ffi.cdef` accepts
scalar and pointer function declarations, typedef aliases, C qualifiers, and
comments. Calls use `ffi.C` or `ffi.load`. Short library names such as `"m"`
are accepted. `ffi.new` creates scalars and fixed scalar arrays. Arrays support
indexed reads and writes and can be passed to native functions. `ffi.typeof`
returns a callable type object, and `ffi.istype` checks cdata types. The module
also provides `ffi.cast`, `ffi.sizeof`, `ffi.alignof`, `ffi.abi`, `ffi.errno`,
`ffi.copy`, `ffi.fill`, and `ffi.string`.

Native `char *` results stay as pointers. Use `ffi.string(ptr)` to read a
zero-terminated string or `ffi.string(ptr, length)` to copy a fixed number of
bytes, including embedded zero bytes. Structs, unions, callbacks, C variables,
`ffi.gc`, `ffi.offsetof`, and metatypes are not supported yet. Pointer type
compatibility is limited because the current native type model does not retain
every pointee type. Native calls trust the declarations and pointers supplied
by Lua.

Only enable `Ffi` for trusted code. Adding it to a `Pure` state gives that
state native process access.

## C API

Oximond's compiler, VM, and embedding crate are Rust-only. Building the CLI also
requires a C++17 compiler for native module calls. The CLI exports the Lua 5.1 C
ABI. Embedders can build `oximond-capi` with `lua52`, `lua53`, `lua54`, or
`lua55` to select the matching C ABI and public headers. See
[the C API guide](../crates/oximond-capi/README.md) for builds and tests.
