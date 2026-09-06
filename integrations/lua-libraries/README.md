# Lua library integration tests

This harness runs Teal 0.24.8, LPeg 1.1.0, and LuaSocket 3.1.0 under all six
Oximond compatibility profiles with interpreter and tracing execution. It also
compares the built-in Rust `cjson`, `cjson.safe`, `struct`, and `cmsgpack` modules
against Redis 7.4.2's C implementations.

`sources.json` pins every downloaded archive by SHA-256. It includes stock Lua
5.1.5, 5.2.4, 5.3.6, 5.4.9, 5.5.1, an exact LuaJIT 2.1 commit, and Teal's
compat53 0.14.4 dependency. The harness builds these reference runtimes itself.
System Lua modules and LuaRocks search paths do not participate.

## Build and run

The Linux dependencies are Python 3.12 or newer, a C compiler, GNU make,
`nm`, `ldd`, readline and ncurses development libraries for stock Lua's Linux build,
and the repository's Rust toolchain. Downloads need HTTPS access to lua.org,
GitHub's codeload host, and PUC-Rio. `CC` selects the C compiler.

```sh
python3 integrations/lua-libraries/libraries.py prepare
scripts/heavy.sh --status
scripts/heavy.sh python3 integrations/lua-libraries/libraries.py build --jobs 2
CARGO_BUILD_JOBS=2 scripts/heavy.sh cargo build --release -p oximond-cli
scripts/heavy.sh python3 integrations/lua-libraries/libraries.py run
```

`prepare` replaces only the generated source directories inside
`target/lua-libraries/source`. `build` restores those pinned sources first.
`run` verifies the archives and each extracted source file before testing.
It records every compile command, compiler diagnostic, undefined native symbol,
and module linkage in `target/lua-libraries/build-report.json`.

The full run contains 144 rows. Every case runs for `lua51`, `lua52`, `lua53`,
`lua54`, `lua55`, and `luajit21` with both execution modes. A failed reference,
assertion, missing completion marker, signal, or deadline makes the row fail.
There are no expected failures or skipped profiles. Each child process has a
2 GiB address-space limit and a 300-second deadline. Change those with
`--memory-mib` and `--timeout` when investigating a failure.

The JSON report defaults to `target/lua-libraries/report.json`. It records the
tested CLI's SHA-256, manifest and module hashes, complete process output, and
the native JIT execution count. Per-process scripts and result files stay in
`target/lua-libraries/runs/<profile>/<variant>/<engine>/<case>/`.

Focus a run without changing its pass criteria:

```sh
scripts/heavy.sh python3 integrations/lua-libraries/libraries.py run \
  --profile lua55 --engine tracing --case socket-loopback
scripts/heavy.sh python3 integrations/lua-libraries/libraries.py run \
  --reference-only --report target/lua-libraries/reference-report.json
```

`--dir` selects a separate download, source, module, and run directory.
`--oximond` selects another CLI build. Rebuild the CLI after changing Rust code.
Commands sharing a `--dir` wait on a lock, so a build cannot replace sources
under a running suite.

## What runs

LPeg runs its complete upstream `test.lua`, including the `re` module. A
focused fixture checks pattern userdata, operator metamethods, table and named
captures, recursive grammars, binary strings, and Lua match-time callbacks.

LuaSocket runs upstream `excepttest.lua`, `ltn12test.lua`, `mimetest.lua`, and
`urltest.lua`. The TCP and UDP fixture opens ephemeral loopback ports. It checks
binary sends, reads with an existing partial buffer, immediate timeouts,
line reads, closed sockets, byte counters, MIME round trips, and protected
errors. A Python HTTP server bound to `127.0.0.1` supplies a binary response to
the real `socket.http` client. These tests do not contact public servers.
The other upstream LuaSocket programs require separate services or external
network access and are outside this local integration workload.

Teal loads the upstream Lua compiler, typechecks a generic typed program,
executes its generated Lua, and compares generated source and diagnostic
positions/messages against the matching stock runtime. The bootstrap case
typechecks and generates the complete `tl.tl` compiler twice. It loads the
first generated compiler, verifies byte-identical second-pass output, and
uses that compiler on a typed fixture. The report also compares SHA-256 hashes
of the emitted compiler across Oximond and stock Lua. Teal documents support
through Lua 5.4. Lua 5.5 still runs both cases here. Generated output explicitly
targets Lua 5.1, which all six runtimes can execute.

The Rust module fixture checks recursive JSON with null and Unicode escapes,
binary strings, independent CJSON settings, protected errors and sparse arrays,
fixed big/little endian struct bytes, and exact MessagePack integer/array bytes.
An extended differential fixture probes invalid input, floating-point formatting,
JSON settings, struct alignment and size quirks, and MessagePack integer bounds.
It compares 167 probes. Error argument positions and details are preserved;
the VM's caller function name is normalized because Lua 5.1 and modern Lua
report those names differently.
Its oracle is Redis's Lua 5.1 runtime and C modules for every Oximond profile.
Redis defines these module formats independently of Oximond's language profile.

## Native ABI and tracing execution

This integration builds against Oximond's default Lua 5.1 C API. Its LPeg, LuaSocket, and compat53
modules compile against `crates/oximond-capi/vendor/lua51`, with the pinned Lua
5.1 `lualib.h` as a fallback. This remains true in Lua 5.2 through 5.5 language
profiles. The reference modules compile against each stock runtime's matching
headers. No module links another Lua library. `ldd` checks that boundary.

Every process runs a bounded numeric Lua kernel before its library workload.
Tracing rows require `OXIMOND_JIT_STATS` to report `native_entries > 0`. This proves
the selected engine executed native Lua code in that process. It does not
claim that Teal's entire compiler or a native module's C code passed through
Oximond's JIT. Upstream suite logs can contain random chunk details or timing
text, so their assertions and completion markers decide success. Focused
fixtures compare exact stdout with their reference process.
