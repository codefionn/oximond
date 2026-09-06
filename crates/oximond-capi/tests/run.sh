#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
lua_src="$repo_root/target/luanti/source/lib/lua/src"
archive="${CARGO_TARGET_DIR:-$repo_root/target}/debug/liboximond_capi.a"
build_dir="${OXIMOND_CAPI_TEST_DIR:-$repo_root/target/capi-bridge-tests}"
mkdir -p "$build_dir"
exec 9>"$build_dir/.lock"
flock -n 9 || { echo "C API tests already running in $build_dir" >&2; exit 1; }

if [[ ! -f "$lua_src/lua.h" ]]; then
    lua_src="$repo_root/crates/oximond-capi/vendor/lua51"
fi

cargo_command=(cargo)
if [[ -n "${OXIMOND_RUST_TOOLCHAIN:-}" ]]; then
    cargo_command+=("+$OXIMOND_RUST_TOOLCHAIN")
fi
"${cargo_command[@]}" build --manifest-path "$repo_root/Cargo.toml" -p oximond-capi \
    --no-default-features --features lua51
mkdir -p "$build_dir"

# lauxlib.c uses only the public Lua API. Compile it as C++ so luaL_error can
# unwind through its frames when the bridge turns lua_error into an exception.
g++ -std=c++17 -x c++ -I"$lua_src" -c "$lua_src/lauxlib.c" -o "$build_dir/lauxlib.o"
g++ -std=c++17 -I"$lua_src" -c "$repo_root/crates/oximond-capi/bridge.cpp" -o "$build_dir/bridge.o"
g++ -std=c++17 -I"$lua_src" -c "$repo_root/crates/oximond-capi/tests/bridge_test.cpp" -o "$build_dir/bridge_test.o"
g++ -std=c++17 -fPIC -shared -I"$lua_src" "$repo_root/crates/oximond-capi/tests/native_module.cpp" -o "$build_dir/oxitest.so"
g++ "$build_dir/bridge_test.o" "$build_dir/lauxlib.o" "$build_dir/bridge.o" "$archive" \
    -Wl,--export-dynamic -lffi -ldl -lpthread -lm -o "$build_dir/bridge_test"
(ulimit -v 2097152; timeout 30 "$build_dir/bridge_test" "$build_dir/oxitest.so")
