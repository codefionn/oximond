# Windows

Oximond builds for `x86_64-pc-windows-gnu` and runs there with all of its
execution modes: the interpreter, and tracing with either native backend
(MIR, the default, or Cranelift). The `ffi` module calls C functions in DLLs.
The build is a cross-compile from Linux with zig as the C compiler and linker,
and wine runs the result, so no Windows machine or MinGW toolchain is needed.

## Build and run

You need `zig`, `cargo-zigbuild` and `wine`:

```sh
rustup target add x86_64-pc-windows-gnu
cargo zigbuild --release --target x86_64-pc-windows-gnu -p oximond-cli
wine target/x86_64-pc-windows-gnu/release/oximond.exe -e 'print(_VERSION)'
wine target/x86_64-pc-windows-gnu/release/oximond.exe --execution tracing bench.lua
```

`OXIMOND_JIT_BACKEND=cranelift` or `=mir` picks the backend as on Linux.
`scripts/ci.sh windows` does the release build and runs
`tests/windows-smoke/smoke.lua` under the interpreter and both backends, with
the native module `tests/windows-smoke/module.c` built next to it. The three
outputs must match, the module must load, and the MIR run must have compiled a
tree. The
target is not part of `all`, because it needs the three tools above.

## Running the tests under wine

`cargo zigbuild` has no `test` subcommand. Point plain `cargo test` at the
wrappers it generates under `~/.cache/cargo-zigbuild/<version>/wrappers/<hash>/`
(run one `cargo zigbuild` first so they exist), and make wine the runner:

```sh
ZW=$(dirname ~/.cache/cargo-zigbuild/*/wrappers/*/zigcc-x86_64-pc-windows-gnu-*.sh)
PATH=$ZW:$PATH \
CC_x86_64_pc_windows_gnu=$(echo $ZW/zigcc-*.sh) \
CXX_x86_64_pc_windows_gnu=$(echo $ZW/zigcxx-*.sh) \
AR_x86_64_pc_windows_gnu=$ZW/ar \
CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=$(echo $ZW/zigcc-*.sh) \
CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=wine \
cargo test --target x86_64-pc-windows-gnu -p oximond-mir
```

`PATH` gives rustc the `x86_64-w64-mingw32-dlltool` wrapper that the
`windows-sys` import libraries need. The same works for `-p oximond` and
`-p oximond-cli` (the CLI tests spawn `oximond.exe`, which wine runs too).

## What differs from Linux

- MIR emits Win64 code: arguments by position in rcx, rdx, r8 and r9 or
  xmm0 to xmm3, 32 bytes of shadow space, rsi, rdi and xmm6 to xmm15 kept
  across calls. Its code region is reserved with `VirtualAlloc` and committed
  a megabyte at a time. See `docs/impl/native-backend/mir/backend.md`.
- The `ffi` module has no libffi. `ffi_native/native_windows.c` calls through
  a Win64 trampoline that loads each of the first four arguments into both its
  integer and its float register, which covers variadic callees. `ffi.C`
  searches the executable, `ucrtbase.dll`, `kernel32.dll`, then `msvcrt.dll`
  (the universal C runtime exports no `printf` family), `user32.dll` and
  `gdi32.dll`. `ffi.load("name")` also tries `name.dll`. Struct arguments are
  unsupported here as on Linux.
- Native Lua modules load through `LoadLibraryExA` and `GetProcAddress`.
  `package.cpath` defaults to `.\?.dll;.\loadall.dll`. The C++ bridge is
  linked into the executable instead of loaded from an embedded shared object.
- A module built for LuaJIT or Lua 5.1 imports the Lua API from `lua51.dll`
  (or `lua5.1.dll`). The build generates that DLL: one thunk per API function,
  each a jump through a slot. The executable embeds it, and before the first
  native module it writes it to `%TEMP%\oximond-lua51-<hash>\`, loads it and
  fills the slots with its own functions. Windows binds a module's imports to
  an already loaded DLL of the same name, so the module calls into Oximond. A
  `lua51.dll` that the process loaded before keeps its own imports. To link a
  module, make an import library from the generated `lua51.def` in
  `oximond-capi`'s build output (`zig dlltool -m i386:x86-64 -d lua51.def -l
  liblua51.a`), as `scripts/ci.sh windows` does for
  `tests/windows-smoke/module.c`. A LuaJIT `lua51.lib` works as well.
- `require` joins module paths with `\`, as stock Lua does on Windows.
- I/O errors report the C runtime's `errno` and `strerror` text ("No such file
  or directory", 2), not the Win32 code and message, as stock Lua does there.
- `x ^ y` in the profiles that call `pow` gives the Windows C runtime's
  answer, which can differ from glibc's in the last bit.

## Toolchain notes

zig links its `compiler_rt` into every windows-gnu executable, and its weak
`fma` rounds twice (`fma(-1, -1e300, -1e300)` is `2^-78`, not 0). Rust's
`f64::mul_add` calls `fma` in code compiled without the FMA instructions, which
includes the float `%` of a debug build. `native_windows.c` defines a strong
`fma` that forwards to the universal C runtime's, which is correct.
