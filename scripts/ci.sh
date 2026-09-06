#!/usr/bin/env bash
# CI entry point. .gitlab-ci.yml, .github/workflows/* and .forgejo/workflows/*
# all run their jobs through this script, and so does a local full-gate run:
# one place decides what "the pipeline passes" means.
#
#   scripts/ci.sh [TARGET...]
#
#   fmt           cargo fmt over the workspace
#   clippy        cargo clippy over every workspace target
#   ledger        validate the conformance sources and ledger manifests
#   test          cargo test over the workspace
#   wasm          wasm32 build of the library and tests/wasm-smoke under node
#   windows       x86_64-pc-windows-gnu build of the CLI (cargo-zigbuild) and
#                 tests/windows-smoke under wine; not part of `all`
#   conformance   release CLI and the pinned conformance corpora
#   package       package every publishable crate, no build and no upload
#   publish       publish every publishable crate to crates.io
#   check         fmt + clippy + ledger
#   ci            test + wasm + conformance
#   all           check + ci (the default)
#
# Environment:
#   OXIMOND_CI_REPORT     also write the conformance run report to this file;
#                         CI jobs upload it as an artifact.
#   OXIMOND_CI_NO_GATE=1  never queue heavy commands behind scripts/heavy.sh.
#   CARGO_REGISTRY_TOKEN  crates.io token for the publish target; the GitHub
#                         release workflow takes it from the crates-io
#                         environment's CARGO_REGISTRY_TOKEN secret.
#
# The corpora the conformance target downloads (pinned lua.org tarballs and one
# pinned git checkout) land in .oximond/conformance and are never vendored;
# CI caches that directory. See docs/conformance.md for the ledgers, and
# docs/ci.md for how the pipelines use this script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CARGO_INCREMENTAL=0
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"

# Outside CI the workspace shares the machine with other sessions and
# worktrees, so the heavy commands (workspace clippy and tests, the release
# build, the conformance run) queue behind the repository gate instead of
# racing everything else; see AGENTS.md. A command that already holds a gate
# slot (OXIMOND_HEAVY_ACTIVE, exported by scripts/heavy.sh) does not queue
# again. CI containers own their machine and skip the gate.
gate=(scripts/heavy.sh)
if [ -n "${CI:-}" ] || [ -n "${OXIMOND_HEAVY_ACTIVE:-}" ] || [ "${OXIMOND_CI_NO_GATE:-}" = "1" ]; then
    gate=()
fi

# heavy [-m MEMORY] COMMAND [ARGS...]
#
# With the gate: its own argument syntax. Without it: the memory flag is only a
# gate concern and is dropped before the command runs.
heavy() {
    if [ "${#gate[@]}" -eq 0 ]; then
        while [ "$#" -gt 0 ]; do
            case "$1" in
                -m) shift 2 ;;
                --) shift; break ;;
                *) break ;;
            esac
        done
        "$@"
    else
        "${gate[@]}" "$@"
    fi
}

target_fmt() {
    cargo fmt --all -- --check
}

target_clippy() {
    # Compiles build scripts, so a C toolchain must be installed.
    heavy cargo clippy --locked --workspace --all-targets
}

target_ledger() {
    cargo run --locked -p oximond-conformance -- check-ledger
}

target_test() {
    heavy cargo test --locked --workspace
}

target_wasm() {
    command -v node >/dev/null || {
        echo "scripts/ci.sh: the wasm target needs node (>= 14.8) to run tests/wasm-smoke" >&2
        exit 2
    }
    if ! rustup target list --installed | grep -qx wasm32-unknown-unknown; then
        rustup target add wasm32-unknown-unknown
    fi
    cargo build --locked --release -p oximond --target wasm32-unknown-unknown
    cargo build --locked --release --manifest-path tests/wasm-smoke/Cargo.toml \
        --target wasm32-unknown-unknown
    node tests/wasm-smoke/run.mjs
}

target_windows() {
    local tool
    for tool in zig cargo-zigbuild wine; do
        command -v "$tool" >/dev/null || {
            echo "scripts/ci.sh: the windows target needs zig, cargo-zigbuild and wine (missing $tool)" >&2
            exit 2
        }
    done
    local triple=x86_64-pc-windows-gnu
    if ! rustup target list --installed | grep -qx "$triple"; then
        rustup target add "$triple"
    fi
    # zig is the C and C++ compiler and the linker, so no MinGW toolchain is
    # needed.
    heavy cargo zigbuild --locked --release --target "$triple" -p oximond-cli
    local exe="$PWD/target/$triple/release/oximond.exe"
    local smoke="$PWD/tests/windows-smoke/smoke.lua"
    local out
    out="$(mktemp -d "${TMPDIR:-/tmp}/oximond-windows.XXXXXX")"
    trap "rm -rf '${out}'" EXIT
    # The test module imports from the lua51.dll that oximond-capi's build
    # script generates, through an import library made from its lua51.def
    # (docs/windows.md).
    local def
    def="$(find "target/$triple/release/build" -path '*oximond-capi*' -name lua51.def \
        -printf '%T@ %p\n' | sort -rn | head -n 1 | cut -d' ' -f2-)"
    zig dlltool -m i386:x86-64 -d "$def" -l "$out/liblua51.a"
    zig cc -target x86_64-windows-gnu -shared -O2 -I crates/oximond-capi/vendor/lua51 \
        -o "$out/smoke_module.dll" tests/windows-smoke/module.c "$out/liblua51.a"
    export WINEDEBUG="${WINEDEBUG:--all}"
    (
        cd "$out"
        wine "$exe" "$smoke" >interpreter.txt
        OXIMOND_JIT_SYNC=1 OXIMOND_JIT_BACKEND=cranelift \
            wine "$exe" --execution tracing "$smoke" >cranelift.txt
        OXIMOND_JIT_SYNC=1 OXIMOND_JIT_BACKEND=mir OXIMOND_JIT_TRACE=1 \
            wine "$exe" --execution tracing "$smoke" >mir.txt 2>mir.log
    )
    diff -u "$out/interpreter.txt" "$out/cranelift.txt"
    diff -u "$out/interpreter.txt" "$out/mir.txt"
    grep -q "mir backend compiled tree" "$out/mir.log" || {
        echo "scripts/ci.sh: the MIR backend compiled no tree under wine" >&2
        exit 1
    }
    grep -q "^module" "$out/interpreter.txt" && ! grep -q "module missing" "$out/interpreter.txt" || {
        echo "scripts/ci.sh: the native test module did not load under wine" >&2
        exit 1
    }
    cat "$out/interpreter.txt"
}

target_conformance() {
    # prepare downloads the pinned corpora and applies the declared patch sets;
    # it is idempotent and network-bound only on a cold cache.
    cargo run --locked -p oximond-conformance -- prepare
    # `run` tests target/release/oximond with a deadline calibrated for
    # optimized code (docs/conformance.md), so the release build comes first.
    heavy cargo build --locked --release -p oximond-cli -p oximond-conformance
    if [ -n "${OXIMOND_CI_REPORT:-}" ]; then
        heavy -m 16G target/release/oximond-conformance run | tee "$OXIMOND_CI_REPORT"
    else
        heavy -m 16G target/release/oximond-conformance run
    fi
}

target_package() {
    # Packages every workspace member without `publish = false` and checks the
    # metadata crates.io requires and the version requirements `cargo publish`
    # rewrites the workspace path edges to. No build and no upload. A local
    # tree is always dirty, so this target carries --allow-dirty; the publish
    # target stays strict and packages committed state only.
    cargo publish --workspace --dry-run --no-verify --locked --allow-dirty
}

target_publish() {
    # Publishes every member without `publish = false` in dependency order.
    # crates.io lets an account create about one new crate per 10 minutes
    # after a short burst and answers 429 past that. Each pass excludes the
    # crates whose version is already in the index, so a 429 or a rerun after
    # a partial upload resumes where the last attempt stopped.
    command -v jq >/dev/null || {
        echo "scripts/ci.sh: the publish target needs jq to read cargo metadata" >&2
        exit 2
    }
    if [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
        echo "scripts/ci.sh: the publish target needs CARGO_REGISTRY_TOKEN" >&2
        exit 2
    fi
    local log attempt name version pending
    local excludes=()
    log="$(mktemp "${TMPDIR:-/tmp}/oximond-publish.XXXXXX")"
    # Expand the path now: the trap fires at script exit, past this function's
    # locals.
    trap "rm -f '${log}'" EXIT
    for attempt in $(seq 1 30); do
        excludes=()
        pending=0
        while read -r name version; do
            # Sparse index path for names of four or more characters.
            if curl -fsS "https://index.crates.io/${name:0:2}/${name:2:2}/${name}" 2>/dev/null \
                | jq -e --arg v "${version}" 'select(.vers == $v)' >/dev/null; then
                excludes+=(--exclude "${name}")
            else
                pending=$((pending + 1))
            fi
        done < <(cargo metadata --format-version 1 --no-deps \
            | jq -r '.packages[] | select(.publish != []) | "\(.name) \(.version)"')
        if [ "${pending}" -eq 0 ]; then
            echo "all crates are on crates.io"
            return 0
        fi
        echo "attempt ${attempt}: ${pending} crates left"
        if heavy cargo publish --workspace --locked "${excludes[@]}" 2>&1 | tee "${log}"; then
            continue
        fi
        if grep -q 'status 429' "${log}"; then
            # The wait sleeps outside the gate: a held slot would idle the
            # machine for ten minutes at a time.
            sleep 600
        elif grep -q 'already exists on crates.io index' "${log}"; then
            # The sparse index lags a fresh upload by a few seconds.
            sleep 30
        else
            return 1
        fi
    done
    echo "scripts/ci.sh: crates still unpublished after 30 attempts" >&2
    return 1
}

targets=("$@")
if [ "${#targets[@]}" -eq 0 ]; then
    targets=(all)
fi

for target in "${targets[@]}"; do
    case "$target" in
        fmt) target_fmt ;;
        clippy) target_clippy ;;
        ledger) target_ledger ;;
        test) target_test ;;
        wasm) target_wasm ;;
        windows) target_windows ;;
        conformance) target_conformance ;;
        package) target_package ;;
        publish) target_publish ;;
        check)
            target_fmt
            target_clippy
            target_ledger
            ;;
        ci)
            target_test
            target_wasm
            target_conformance
            ;;
        all)
            target_fmt
            target_clippy
            target_ledger
            target_test
            target_wasm
            target_conformance
            ;;
        *)
            echo "scripts/ci.sh: unknown target \`$target\`" >&2
            echo "known targets: fmt clippy ledger test wasm windows conformance package publish check ci all" >&2
            exit 2
            ;;
    esac
done
