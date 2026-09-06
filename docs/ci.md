# CI

Every pipeline runs its jobs through `scripts/ci.sh`, the same entry point a
local run uses, so CI and a developer's machine cannot drift apart.

```sh
scripts/ci.sh              # everything (the default)
scripts/ci.sh check        # fmt + clippy + ledger
scripts/ci.sh test wasm    # the workspace tests and the wasm smoke test
```

## Targets

| target | what it runs |
| --- | --- |
| `fmt` | `cargo fmt` over the workspace |
| `clippy` | `cargo clippy` over every workspace target |
| `ledger` | validates `tests/conformance` sources and ledger manifests |
| `test` | `cargo test` over the workspace |
| `wasm` | wasm32 build of the library and `tests/wasm-smoke` under node |
| `windows` | Windows build of the CLI and `tests/windows-smoke` under wine (not in `all`) |
| `conformance` | the release CLI and the pinned conformance corpora |
| `package` | packages every publishable crate, no build and no upload |
| `publish` | publishes every publishable crate to crates.io |
| `check` | `fmt` + `clippy` + `ledger` |
| `ci` | `test` + `wasm` + `conformance` |
| `all` | `check` + `ci` (the default) |

The `conformance` target downloads the pinned corpora into
`.oximond/conformance` (never vendored) and then runs `run` against
`target/release/oximond`, whose per-test deadline is calibrated for optimized
code. `OXIMOND_CI_REPORT=FILE` additionally writes the run report to `FILE`;
the GitLab and GitHub pipelines upload it as an artifact. See
[conformance suites](conformance.md) for the ledgers.

The `package` target packages every workspace member without `publish =
false` and checks the metadata crates.io requires and the version
requirements `cargo publish` rewrites the workspace path edges to. The
`publish` target uploads the same set to crates.io in dependency order,
waiting out the registry's new-crate rate limit and resuming past crates
that are already in the index, so a rerun after a partial upload picks up
where the last attempt stopped. `publish` needs `CARGO_REGISTRY_TOKEN` and
jq on the `PATH`; only the GitHub release workflow runs it. The published
set is the product crates `oximond`, `oximond-capi` and `oximond-cli`; the
`oximond-conformance` and `oximond-difftest` harnesses carry `publish =
false` — they embed checkout files and only build inside the repository.

## Pipelines

- `.gitlab-ci.yml` — the full pipeline: `version` (tags only), `check`
  (`fmt`, `clippy`, `ledger`), `test` (`test`, `wasm`, `conformance`). Job
  caches keep the cargo registry and the corpora between runs, and the
  conformance report is uploaded as an artifact.
- `.github/workflows/` — the same full pipeline as GitHub Actions:
  `check.yml` and `ci.yml` on branches and pull requests, `release.yml` on
  `v*.*.*` tags. Jobs run on GitHub-hosted runners with the toolchain pinned
  through `dtolnay/rust-toolchain`; no shared cache is assumed, so corpora
  are downloaded per run, and the conformance report is uploaded as a run
  artifact. `check.yml` additionally packages the publishable crates, and
  `release.yml` publishes them to crates.io after its gate passes (below).
- `.forgejo/workflows/wasm.yml` — the `wasm` target on Forgejo Actions: one
  job in a `rust:1.97.1-trixie` container on the `docker` runner label. The
  report stays in the job log there.

Clippy reports the workspace's pedantic lints without failing on them; add
`-- -D warnings` to the `clippy` target once the warnings are gone.

## Version gate

A `v*.*.*` tag only passes the release gate when it names the workspace
version from `Cargo.toml` — the `version` job in `.gitlab-ci.yml` and the one
in `.github/workflows/release.yml` refuse the pipeline otherwise. The tag
pipeline then re-runs the full gate. On GitHub the `publish` job then
releases every workspace crate without `publish = false` to crates.io; the
GitLab and Forgejo pipelines still only gate the tag.

## Runner requirements

- a Rust toolchain with a C/C++ toolchain (tests compile native modules, the
  Lua 5.4/5.5 conformance tests build `lib2-v2.so`) and node for the wasm
  target — the GitLab and Forgejo jobs get these from their `rust` image, the
  GitHub jobs from the runner image plus `dtolnay/rust-toolchain`;
- internet access for the pinned corpus downloads (lua.org tarballs and one
  pinned GitHub checkout);
- 4 GiB of memory minimum for the `test` and `conformance` jobs — each
  conformance test process runs under its own 4 GiB address-space cap — with
  8 GiB comfortable for the thin-LTO release build.

No CI/CD variables or secrets are required to gate; the GitHub `publish` job
takes a crates.io token from the `crates-io` environment's
`CARGO_REGISTRY_TOKEN` secret.

## Not in CI

Differential testing, benchmarks, and the Luanti, Neovim, Valkey and Lua
library integrations are local heavy workloads; run them through
`scripts/heavy.sh` as described in the repository guides.
