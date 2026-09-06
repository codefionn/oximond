# Oximond

Oximond (*oxi* = oxidiced, *Mond* = German for lua so moon) is a Rust
implementatiopn of stock Lua 5.1 through Lua 5.5.1 and LuaJIT 2.1.

## Non-goals

This project does not seek to reimagine the Lua programming language and is
solely designed to be a fast Rust implementation of Lua and LuaJIT.

## Goals

- Functional multi-profile Lua interpreter and jit
- Passes conformance suites (excluding specific c test fixtures specific to the
  Lua interpreter itself)
- Wasm compatible
- Drop in replacement for `lua` and `luajit`

## Run it

```sh
cargo run -p oximond-cli -- -e 'local answer = 40 + 2; assert(answer == 42)'
cargo run -p oximond-cli -- --compat lua51 -e 'print(_VERSION)'
cargo run -p oximond-cli -- --execution tracing bench.lua
cargo run -p oximond-cli -- --compat lua54
```

The last line has no script, so on a terminal it starts an interactive session,
as `lua` does. `-i` starts one after a script. A failed line prints its error
and traceback and the session goes on. Ctrl-C stops a running chunk or drops a
half-typed one, and Ctrl-D leaves. When standard input is not a terminal, a
bare `oximond` runs it as a script.

## Embed it

```rust
use oximond::{CompatibilityProfile, Lua};

let mut lua = Lua::builder()
    .profile(CompatibilityProfile::Lua54)
    .build()?;
let results = lua.exec("return 40 + 2")?;
```

## Differential testing

`oximond-difftest` compares seeded random Lua programs between Oximond and the
stock interpreters. See [differential testing](docs/difftest.md).

## Benchmark it

The checked-in suite runs 35 Lua 5.1-compatible workloads on stock Lua,
LuaJIT, and Oximond's interpreter and tracing modes. It checks deterministic
output before recording process medians.

```sh
cargo build --release -p oximond-cli
python3 benchmarks/run.py --quick
```

See `benchmarks/README.md` for pinned CPU runs, focused filters, JSON output,
and performance-gap gates.

## AI Disclosure

This project is being developed with AI assistance.

## License

MIT. See [LICENSE](LICENSE). Third-party material keeps its own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
