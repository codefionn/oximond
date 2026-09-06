# Conformance suites

The interpreter runs pinned LuaJIT tests and official Lua 5.1 through 5.5 suites
with exact file ledgers. The official suites use checked-in patches that remove
tests requiring Lua's internal `testC` fixture. Their required slices cover
varargs, large generated programs, Lua 5.2 syntax, locals, math, and goto
behavior through Lua 5.5. Other modern tests remain expected failures until they
pass. Files devoted to the C API, stock Lua opcodes, or fixture-controlled
allocation failures remain deferred.

```sh
cargo run -p oximond-conformance -- prepare
cargo build --release -p oximond-cli -p oximond-conformance
scripts/heavy.sh -m 16G target/release/oximond-conformance run
```

## Which binary runs

`run` tests `target/release/oximond` because its 60-second per-test deadline is
calibrated for optimized code. Rebuild that binary first, or a stale CLI is
tested. To test another build, pass it explicitly, for example
`run --oximond target/debug/oximond`, and raise the deadline with
`OXIMOND_CONFORMANCE_TIMEOUT_SECS` if needed. Concurrent runs against the same
corpus directory wait for each other because each run rebuilds the `patched`
trees.

The Lua 5.4 and 5.5 CLI tests build `lib2-v2.so` with `CC`, or `cc` when `CC` is
unset.

## Memory limit

Each test process runs under a 4 GiB address-space limit, so stress files such
as `heavy.lua` fail with an allocation error instead of exhausting the machine.
Set `OXIMOND_CONFORMANCE_MEMORY_LIMIT_MIB` to change the limit, or to `0` to
disable it.

## Trace profiles

To record or load a separate profile for every test, run
`oximond-conformance run --execution tracing --trace-profile-record-dir DIR` or
`--trace-profile-load-dir DIR`. See [trace profiles](execution.md#trace-profiles).

## Patches

Each suite in `tests/conformance/sources.toml` can reference a unified patch set
with a path relative to that manifest, for example:

```toml
patches = "patches/lua51-no-testc.patch"
```

`prepare` and `run` rebuild a separate `patched` directory from the pinned input
and apply the patch set without fuzz. The downloaded archive and `source`
directory stay intact. Reports include the patch path and SHA-256, so a patched
suite's results can be traced to the exact changes. Suites without a `patches`
field run from their original source directory.

The patches remove fixture-only sections from mixed tests and omit calls to
C-only files in `all.lua`. They keep ordinary Lua assertions. The C-only files
remain in the ledger as deferred tests and never count as passes. Any remaining
`testC` self-skip still fails an executable ledger entry.
