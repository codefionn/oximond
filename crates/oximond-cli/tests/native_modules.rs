use std::path::{Path, PathBuf};
use std::process::Command;

fn build_fixture(profile: &str, directory: &Path) -> PathBuf {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let headers = crate_root.join("../oximond-capi/vendor/lua51");
    let module = directory.join("lib2-v2.so");
    let output = Command::new(std::env::var_os("CC").unwrap_or_else(|| "cc".into()))
        .args(["-shared", "-fPIC", "-O2"])
        .arg("-I")
        .arg(&headers)
        .arg("-include")
        .arg(headers.join("modern-module.h"))
        .arg(crate_root.join(format!("tests/fixtures/lib22-{profile}.c")))
        .arg("-o")
        .arg(&module)
        .output()
        .expect("compile native fixture");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    module
}

#[test]
fn native_version_suffix_modules_run_in_the_cli_vm() {
    for profile in ["lua54", "lua55"] {
        let directory =
            std::env::temp_dir().join(format!("oximond-native-{profile}-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let module = build_fixture(profile, &directory);
        let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
        command.args(["--compat", profile, "-l", "lib2-v2", "-e"])
            .arg(if profile == "lua55" {
                "assert(lib2.id()); local s = string.rep('abc', 1000); assert(lib2.newstr(s) == s); print(lib2.id())"
            } else {
                "assert(lib2.id()); assert(x == 'lib2-v2'); print(lib2.id())"
            })
            .env("LUA_CPATH", directory.join("?.so"))
            .env_remove("LUA_CPATH_5_5")
            .env_remove("LUA_INIT")
            .env_remove("LUA_INIT_5_5");
        let output = command.output().expect("run native module");
        assert!(
            output.status.success(),
            "{profile}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, b"true\n");
        std::fs::remove_file(module).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}

/// libstdc++.so costs ~2.4M instructions of loader work per start. The bridge that needs it
/// is loaded on the first call into native code, so the executable must not depend on it.
#[cfg(target_os = "linux")]
#[test]
fn binary_does_not_link_libstdcxx() {
    let binary = env!("CARGO_BIN_EXE_oximond");
    let output = Command::new("readelf")
        .args(["-d", binary])
        .output()
        .expect("run readelf");
    assert!(output.status.success(), "{output:?}");
    let needed = String::from_utf8_lossy(&output.stdout);
    assert!(!needed.contains("libstdc++"), "{needed}");
}

const UNWIND_C: &str = r#"
#include "lua.h"
#include "lauxlib.h"
static int fail(lua_State *L) { return luaL_error(L, "boom %d", (int)luaL_checkinteger(L, 1)); }
static int raw(lua_State *L) { lua_pushinteger(L, 4242); return lua_error(L); }
static int through(lua_State *L) { luaL_checktype(L, 1, LUA_TFUNCTION); lua_call(L, 0, 0); return 0; }
int luaopen_unwindc(lua_State *L) {
    lua_newtable(L);
    lua_pushcfunction(L, fail); lua_setfield(L, -2, "fail");
    lua_pushcfunction(L, raw); lua_setfield(L, -2, "raw");
    lua_pushcfunction(L, through); lua_setfield(L, -2, "through");
    return 1;
}
"#;

const UNWIND_CPP: &str = r#"
extern "C" {
#include "lua.h"
#include "lauxlib.h"
}
#include <stdexcept>
struct Guard { int *flag; ~Guard() { ++*flag; } };
static int destroyed = 0;
static int lua_fail(lua_State *L) { Guard g{&destroyed}; return luaL_error(L, "rt %d", destroyed); }
static int cxx_throw(lua_State *L) {
    Guard g{&destroyed};
    try { throw std::runtime_error("rt thrown"); }
    catch (const std::exception &e) { lua_pushstring(L, e.what()); return 1; }
}
static int cxx_escape(lua_State *L) {
    Guard g{&destroyed};
    throw std::runtime_error("rt escaped");
}
static int count(lua_State *L) { lua_pushinteger(L, destroyed); return 1; }
static int rethrow(lua_State *L) {
    try { lua_call(L, 0, 0); } catch (...) { throw; }
    return 0;
}
static int uncaught(lua_State *L) { lua_pushinteger(L, std::uncaught_exceptions()); return 1; }
extern "C" int luaopen_unwindcpp(lua_State *L) {
    lua_newtable(L);
    lua_pushcfunction(L, lua_fail); lua_setfield(L, -2, "lua_fail");
    lua_pushcfunction(L, cxx_throw); lua_setfield(L, -2, "cxx_throw");
    lua_pushcfunction(L, cxx_escape); lua_setfield(L, -2, "cxx_escape");
    lua_pushcfunction(L, count); lua_setfield(L, -2, "count");
    lua_pushcfunction(L, rethrow); lua_setfield(L, -2, "rethrow");
    lua_pushcfunction(L, uncaught); lua_setfield(L, -2, "uncaught");
    return 1;
}
"#;

fn compile_unwind_module(directory: &Path, name: &str, cxx: bool, source: &str) {
    let headers = Path::new(env!("CARGO_MANIFEST_DIR")).join("../oximond-capi/vendor/lua51");
    let file = directory.join(format!("{name}.{}", if cxx { "cpp" } else { "c" }));
    std::fs::write(&file, source).unwrap();
    let compiler = if cxx {
        std::env::var_os("CXX").unwrap_or_else(|| "c++".into())
    } else {
        std::env::var_os("CC").unwrap_or_else(|| "cc".into())
    };
    let output = Command::new(compiler)
        .args(["-shared", "-fPIC", "-O1"])
        .arg("-I")
        .arg(&headers)
        .arg(&file)
        .arg("-o")
        .arg(directory.join(format!("{name}.so")))
        .output()
        .expect("compile unwind fixture");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Exceptions raised by `lua_error` must unwind through module frames, including catch(...) and
/// rethrow, without disturbing the module runtime's uncaught-exception count.
#[test]
fn errors_unwind_through_native_module_frames() {
    let directory = std::env::temp_dir().join(format!("oximond-unwind-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    compile_unwind_module(&directory, "unwindc", false, UNWIND_C);
    compile_unwind_module(&directory, "unwindcpp", true, UNWIND_CPP);
    let script = r#"
        local c = require("unwindc")
        local cpp = require("unwindcpp")
        for i = 1, 500 do
            local ok, msg = pcall(c.fail, 7)
            assert(not ok and msg:find("boom 7", 1, true), msg)
            local ok, v = pcall(c.raw)
            assert(not ok and v == 4242, tostring(v))
            local ok, msg = pcall(c.through, function() error("lua side " .. i, 0) end)
            assert(not ok and msg == "lua side " .. i, tostring(msg))
            local ok, msg = pcall(cpp.lua_fail)
            assert(not ok and msg:find("rt ", 1, true), tostring(msg))
            assert(cpp.cxx_throw() == "rt thrown")
            local ok, msg = pcall(cpp.cxx_escape)
            assert(not ok, "escaped C++ exception must surface as an error")
            local ok, msg = pcall(cpp.rethrow, function() error("through catch", 0) end)
            assert(not ok and msg == "through catch", tostring(msg))
        end
        assert(cpp.uncaught() == 0, "uncaught_exceptions drifted: " .. cpp.uncaught())
        assert(cpp.count() >= 1500, cpp.count())
        print("ok")
    "#;
    for mode in ["interpreter", "tracing"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat", "lua51", "--execution", mode, "-e", script])
            .env("LUA_CPATH", directory.join("?.so"))
            .env_remove("LUA_INIT")
            .output()
            .expect("run unwind script");
        assert!(
            output.status.success() && output.stdout == b"ok\n",
            "{mode}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

const CXX_EXCEPTIONS: &str = r#"
extern "C" {
#include "lua.h"
#include "lauxlib.h"
}
#include <exception>
#include <stdexcept>
#include <string>
struct Deep : std::runtime_error { int depth; Deep(int d) : std::runtime_error("deep"), depth(d) {} };
struct Guard { int *flag; ~Guard() { ++*flag; } };
static int destroyed = 0;
static int descend(int n) {
    Guard g{&destroyed};
    if (n == 0) throw Deep(destroyed);
    return descend(n - 1) + 1;
}
/* Throws through n frames of destructors and catches in the module. */
static int deep(lua_State *L) {
    int n = (int)luaL_checkinteger(L, 1), before = destroyed;
    try { descend(n); } catch (const Deep &e) { lua_pushinteger(L, destroyed - before); return 1; }
    return luaL_error(L, "not caught");
}
static int escape(lua_State *L) { throw std::runtime_error("inner escaped"); }
/* A C++ exception escapes an inner C function, the bridge turns it into a Lua
 * error for lua_pcall, and the outer function then throws and catches its own. */
static int nested(lua_State *L) {
    lua_pushcfunction(L, escape);
    int status = lua_pcall(L, 0, 0, 0);
    std::string inner = status ? lua_tostring(L, -1) : "none";
    try { throw std::logic_error(inner); }
    catch (const std::logic_error &e) { lua_pushstring(L, e.what()); return 1; }
}
static int stored(lua_State *L) {
    std::exception_ptr p;
    try { throw std::out_of_range("stored"); } catch (...) { p = std::current_exception(); }
    try { std::rethrow_exception(p); }
    catch (const std::out_of_range &e) { lua_pushstring(L, e.what()); return 1; }
    return 0;
}
static int uncaught(lua_State *L) { lua_pushinteger(L, std::uncaught_exceptions()); return 1; }
extern "C" int luaopen_cxxexceptions(lua_State *L) {
    lua_newtable(L);
    lua_pushcfunction(L, deep); lua_setfield(L, -2, "deep");
    lua_pushcfunction(L, nested); lua_setfield(L, -2, "nested");
    lua_pushcfunction(L, stored); lua_setfield(L, -2, "stored");
    lua_pushcfunction(L, uncaught); lua_setfield(L, -2, "uncaught");
    return 1;
}
"#;

/// The executable links the unwinder statically (`libgcc_eh`) and exports only
/// the C API, so the bridge loads `libgcc_s` and the process holds two
/// unwinders. C++ exceptions are thrown, resumed and caught in `libgcc_s`; this
/// checks that holds for exceptions that never leave the module, for one the
/// bridge turns into a Lua error inside another C++ call, and for a stored
/// `exception_ptr`. Lua errors through C++ frames are covered by
/// `errors_unwind_through_native_module_frames`.
#[test]
fn cxx_exceptions_are_caught_inside_cxx_modules() {
    let directory = std::env::temp_dir().join(format!("oximond-cxxexc-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    compile_unwind_module(&directory, "cxxexceptions", true, CXX_EXCEPTIONS);
    let script = r#"
        local m = require("cxxexceptions")
        for i = 1, 300 do
            local depth = i % 40
            assert(m.deep(depth) == depth + 1, "deep " .. depth)
            local msg = m.nested()
            assert(msg:find("inner escaped", 1, true), msg)
            assert(m.stored() == "stored")
        end
        assert(m.uncaught() == 0, "uncaught_exceptions drifted: " .. m.uncaught())
        local maps = io.open("/proc/self/maps"):read("*a")
        print(maps:find("libgcc_s", 1, true) and "two unwinders" or "one unwinder")
    "#;
    let needed = Command::new("readelf")
        .args(["-d", env!("CARGO_BIN_EXE_oximond")])
        .output()
        .expect("run readelf");
    let links_libgcc_s = String::from_utf8_lossy(&needed.stdout).contains("libgcc_s");
    for mode in ["interpreter", "tracing"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat", "lua51", "--execution", mode, "-e", script])
            .env("LUA_CPATH", directory.join("?.so"))
            .env_remove("LUA_INIT")
            .output()
            .expect("run C++ exception script");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{mode}: {stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        // Without libgcc_s among the executable's needed libraries, only the
        // bridge can have mapped it: the test ran with two unwinders.
        if !links_libgcc_s {
            assert_eq!(stdout, "two unwinders\n", "{mode}");
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}
