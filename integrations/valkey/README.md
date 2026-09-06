# Valkey scripting integration

This integration builds Valkey 8.1.3 twice. The stock build keeps its embedded
Lua 5.1 implementation. The Oximond build uses the Lua 5.1 C ABI, Rust standard
library and serializers, and Valkey's native BitOp module. `source.json` pins
the upstream archive by SHA-256. Both builds apply the same checked-in patch;
its Oximond changes are enabled only when `OXIMOND_LIBRARY` is set.

The Oximond server selects its language profile and execution mode through the
`oximond_newstate_with_options` constructor. The harness runs `lua51`, `lua52`,
`lua53`, `lua54`, `lua55`, and `luajit21` with interpreter and tracing execution.
All profiles use the server's Lua 5.1 ABI. They compare Lua 5.1-compatible
scripting operations against the same stock Valkey server.

## Build and run

The Linux dependencies are Python 3.12 or newer, a C/C++ compiler, GNU make,
binutils, and the repository's Rust toolchain and native dependencies. The
server uses libc allocation, disables TLS and RDMA, and builds with two jobs.

```sh
scripts/heavy.sh --status
CARGO_BUILD_JOBS=2 scripts/heavy.sh python3 integrations/valkey/valkey.py \
  build all --jobs 2
scripts/heavy.sh python3 integrations/valkey/valkey.py run
```

`build` restores the pinned source in a separate directory for each variant.
Its patch removes stock Lua VM objects from the Oximond build and keeps only
the native BitOp library. The server links the Rust static library with its
native dependencies. It checks constructor symbols in each resulting binary.
Compile commands, errors, source pins, and patch hashes stay in
`target/valkey/build-stock.json` and `target/valkey/build-oximond.json`.

By default, the Oximond build compiles a private Lua 5.1 ABI static library
with the bundled bridge. Its Cargo output stays in `target/valkey/cargo`,
separate from the shared CLI and other ABI builds. The harness retains the
linked library by content hash under `target/valkey/artifacts`.
`--toolchain 1.98.1` selects an installed named Rust toolchain. To use an
existing Lua 5.1 ABI library with the bundled bridge instead, pass
`build --library /path/to/liboximond_capi.a`. This skips the Cargo build.
`--build-library` forces a private build even when `--library` is supplied.
`--source-root` builds the static library from an isolated source checkout.
`--dir` chooses another generated build and run directory. Commands sharing
that directory wait on a lock. `prepare all` downloads and applies the patch
without building.

```sh
scripts/heavy.sh python3 integrations/valkey/valkey.py run \
  --profile lua55 --engine tracing --report target/valkey/lua55.json
```

## Workload

Every server listens only on a private UNIX socket. TCP is disabled. Persistence
is disabled, and each variant, profile, and engine has its own working directory.
Each server has a 2 GiB address-space limit. The harness terminates every server
after its workload and kills an unresponsive process after five seconds.

The workload executes real server commands and embedded Lua:

- `EVAL` writes and retrieves a string containing zero and non-ASCII bytes.
- `SCRIPT LOAD`, `SCRIPT EXISTS`, and `EVALSHA` test compilation, cache lookup,
  argument arrays, key arrays, and reply conversion. `SCRIPT FLUSH` removes the
  cached script; subsequent lookup and invocation must fail.
- Script errors and `server.pcall` test exception handling and command errors.
- Global checks exclude host IO, package loaders, FFI, and JIT controls. Ordinary
  assignments and `rawset` must fail on readonly library tables.
  The `os` table must expose only `clock`, matching Valkey's stock sandbox.
- Rust CJSON, struct, and MessagePack serialize data inside Valkey; native BitOp
  checks a C module call through Oximond's bridge.
- `FUNCTION LOAD` and `FCALL` register and invoke a Lua function.
  `FUNCTION FLUSH` removes its library and prevents another invocation.
- A second connection observes `BUSY` from a running infinite script, issues
  `SCRIPT KILL`, and checks that the server remains responsive.

Oximond rows also verify `_VERSION` for their selected profile. A bounded numeric
Lua kernel proves tracing execution through the actual VM's native execution
counter. Interpreter rows require that counter to remain zero. This does not
claim that every server callback or serializer runs in JIT code.
The harness disables Valkey's timeout hook only for this final bounded kernel.
Hooked scripts run in the interpreter so Oximond can preserve instruction-hook
behavior. The infinite-script kill check runs with the timeout hook enabled.

`target/valkey/report.json` records all twelve profile and engine results,
binary hashes, source and patch pins, exact binary replies, and native counters.
Common replies must match stock Valkey. Expected error categories are checked
by their message fragments because call locations differ between Lua runtimes.
There are no expected failures or skipped profiles. Server and process logs
stay under `target/valkey/runs/`.

The patch retains Valkey's table readonly checks and global allow lists. It
removes extra globals from Oximond's full standard library after library
registration, before the script environments are frozen. It replaces the full
OS library with Valkey's clock-only table.
