#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../../.." && pwd)
build_dir="${OXIMOND_CAPI_TEST_DIR:-$repo_root/target/modern-capi-tests}"
mkdir -p "$build_dir"
exec 9>"$build_dir/.lock"
flock -n 9 || { echo "Modern C API tests already running in $build_dir" >&2; exit 1; }
cargo_command=(cargo)
if [[ -n "${OXIMOND_RUST_TOOLCHAIN:-}" ]]; then
    cargo_command+=("+$OXIMOND_RUST_TOOLCHAIN")
fi
profiles=("$@")
if (( ${#profiles[@]} == 0 )); then
    profiles=(lua52 lua53 lua54 lua55)
fi
for profile in "${profiles[@]}"; do
    case "$profile" in lua52|lua53|lua54|lua55) ;; *) echo "Unknown profile: $profile" >&2; exit 1 ;; esac
    header_dir="$repo_root/crates/oximond-capi/vendor/$profile"
    "${cargo_command[@]}" build --manifest-path "$repo_root/Cargo.toml" -p oximond-capi \
        --no-default-features --features "$profile,bundled-bridge"
    "${cargo_command[@]}" test --manifest-path "$repo_root/Cargo.toml" -p oximond-capi \
        --no-default-features --features "$profile,bundled-bridge" -- --test-threads=2
    archive="${CARGO_TARGET_DIR:-$repo_root/target}/debug/liboximond_capi.a"
    cp "$archive" "$build_dir/$profile.a"
    python3 - "$header_dir" "$build_dir/$profile.a" <<'PY'
import pathlib, re, subprocess, sys
required = set()
for name in ("lua.h", "lauxlib.h", "lualib.h"):
    header = pathlib.Path(sys.argv[1]) / name
    active = subprocess.check_output(["g++", "-E", "-fdirectives-only", "-x", "c++",
                                      "-I" + sys.argv[1], str(header)], text=True)
    declarations = re.findall(r"^(?:LUA_API|LUALIB_API|LUAMOD_API)[ \t]+([^;]+);", active, re.MULTILINE)
    required.update(match.group(1) for declaration in declarations
                    if (match := re.search(r"\b(lua(?:L_|open_|_)[a-z]\w*)\s*\)?\s*\(", declaration)))
symbols = subprocess.check_output(["nm", "--defined-only", "-g", sys.argv[2]], text=True, stderr=subprocess.DEVNULL)
exported = set(re.findall(r"\b(lua\w+)$", symbols, re.MULTILINE))
missing = sorted(required - exported)
if missing:
    sys.exit("Missing public Lua API symbols: " + ", ".join(missing))
print(f"Checked {len(required)} public Lua API symbols")
PY
    g++ -std=c++17 -fPIC -shared -I"$header_dir" \
        "$repo_root/crates/oximond-capi/tests/modern_native_module.cpp" -o "$build_dir/$profile-module.so"
    g++ -std=c++17 -rdynamic -I"$header_dir" "$repo_root/crates/oximond-capi/tests/modern_api_test.cpp" \
        "$build_dir/$profile.a" -ldl -lpthread -lm -lffi -o "$build_dir/$profile-test"
    (ulimit -v 2097152; timeout 30 "$build_dir/$profile-test" "$build_dir/$profile-module.so")
done
