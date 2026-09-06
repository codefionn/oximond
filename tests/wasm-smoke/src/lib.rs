use oximond::{CompatibilityProfile, ExecutionMode, Lua, LuaJitExtension, StandardLibrary, Value};

/// Exercise runtime creation, compilation, GC, and virtual module loading.
#[unsafe(no_mangle)]
pub extern "C" fn smoke() -> i32 {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
        CompatibilityProfile::LuaJit21,
    ] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let Ok(mut lua) = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .standard_library(StandardLibrary::Pure)
                .module_resolver(|name| {
                    if name == b"answer" {
                        Ok(Some((b"return 42".to_vec(), b"answer".to_vec())))
                    } else {
                        Ok(None)
                    }
                })
                .build()
            else {
                return 1;
            };
            let Ok(values) = lua.exec(
                "local t = {}; for i = 1, 1000 do t[i] = i end; collectgarbage(); \
             assert(t[1000] == 1000); return require('answer') + math.abs(-8)",
            ) else {
                return 2;
            };
            if !matches!(
                values.as_slice(),
                [Value::Number(50.0)] | [Value::Integer(50)]
            ) {
                return 3;
            }
        }
    }
    let Ok(mut lua) = Lua::builder().build() else {
        return 4;
    };
    if lua
        .exec("assert(not pcall(os.clock)); assert(not pcall(os.time))")
        .is_err()
    {
        return 5;
    }
    if Lua::builder()
        .luajit_extension(LuaJitExtension::Ffi)
        .build()
        .is_ok()
    {
        return 6;
    }
    0
}
