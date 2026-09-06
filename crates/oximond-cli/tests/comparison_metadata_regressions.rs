use oximond::{CompatibilityProfile, ExecutionMode, Lua};

fn run(source: &str) {
    for profile in CompatibilityProfile::ALL {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let source = source
                .replace(
                    "MODERN_EQUALITY",
                    if profile.is_lua53_or_later() {
                        "true"
                    } else {
                        "false"
                    },
                )
                .replace(
                    "MODERN_ORDERING",
                    if profile.is_lua51_family() {
                        "false"
                    } else {
                        "true"
                    },
                )
                .replace(
                    "LE_FALLBACK",
                    if profile == CompatibilityProfile::Lua55 {
                        "false"
                    } else {
                        "true"
                    },
                )
                .replace(
                    "PARAMETER_INFO",
                    if profile == CompatibilityProfile::Lua51 {
                        "false"
                    } else {
                        "true"
                    },
                );
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            lua.exec(source)
                .unwrap_or_else(|error| panic!("{profile}/{mode}: {error:?}"));
        }
    }
}

#[test]
fn equality_selects_handlers_by_profile_and_operand_order() {
    run(r#"
        local left_calls, right_calls = 0, 0
        local left = setmetatable({}, {__eq = function() left_calls = left_calls + 1; return "true" end})
        local right = setmetatable({}, {__eq = function() right_calls = right_calls + 1; return false end})
        for i = 1, 256 do
            assert((left == right) == MODERN_EQUALITY)
            assert((right == left) == false)
            assert((left == {}) == MODERN_EQUALITY)
            assert(({} == left) == MODERN_EQUALITY)
            assert(left == left and right == right)
            assert(left ~= nil and left ~= false and left ~= 1 and left ~= "string")
        end
        assert(left_calls == (MODERN_EQUALITY and 768 or 0), left_calls)
        assert(right_calls == (MODERN_EQUALITY and 256 or 0), right_calls)
        local common = {__eq = function() return true end}
        assert(setmetatable({}, common) == setmetatable({}, common))
    "#);
}

#[test]
fn ordering_and_table_sort_select_left_then_right_handlers_in_modern_profiles() {
    run(r"
        local left = setmetatable({value = 1}, {__lt = function(a,b) return a.value < b.value end})
        local right = setmetatable({value = 2}, {__lt = function(a,b) return a.value < b.value end})
        local ok, result = pcall(function() return left < right end)
        assert(ok == MODERN_ORDERING)
        if ok then assert(result) end
        local other = setmetatable({value = 0}, {})
        ok, result = pcall(function() return other < right end)
        assert(ok == MODERN_ORDERING)
        if ok then assert(result) end
        local list = {right, left}
        ok = pcall(table.sort, list)
        assert(ok == MODERN_ORDERING)
        if ok then assert(list[1] == left and list[2] == right) end
        local le_left = setmetatable({value = 1}, {__le = function(a,b) return a.value <= b.value end})
        local le_right = setmetatable({value = 2}, {})
        ok, result = pcall(function() return le_left <= le_right end)
        assert(ok == MODERN_ORDERING)
        if ok then assert(result) end
    ");
}

#[test]
fn comparison_handlers_keep_their_own_native_dispatch() {
    run(r#"
        local add = getmetatable("").__add
        if type(add) == "function" then
            local left = setmetatable({}, {__eq = add, __lt = add})
            local right = setmetatable({}, {
                __eq = function() return false end,
                __lt = function() return false end,
                __add = function() return true end,
            })
            assert(left == right)
            assert(left < right)
        end
    "#);
}

#[test]
fn lua55_requires_le_instead_of_falling_back_to_lt() {
    run(r"
        local calls = 0
        local mt = {__lt = function(a,b) calls = calls + 1; return a.value < b.value end}
        local left, right = setmetatable({value = 1}, mt), setmetatable({value = 2}, mt)
        local ok, result = pcall(function() return left <= right end)
        assert(ok == LE_FALLBACK)
        assert(calls == (LE_FALLBACK and 1 or 0), calls)
        if ok then assert(result) end
        mt.__le = function(a,b) return a.value <= b.value end
        assert(left <= right and not (right <= left))
        assert(1 <= 2 and not (2 <= 1))
    ");
}

#[test]
fn luajit_reports_lua_and_native_parameter_shapes() {
    run(r#"
        local info = debug.getinfo(function(a,b,...) end, "u")
        local fixed = debug.getinfo(function() end, "u")
        local native = debug.getinfo(print, "u")
        if PARAMETER_INFO then
            assert(info.nparams == 2 and info.isvararg == true)
            assert(fixed.nparams == 0 and fixed.isvararg == false)
            assert(native.nparams == 0 and native.isvararg == true)
        else
            assert(info.nparams == nil and info.isvararg == nil)
            assert(fixed.nparams == nil and fixed.isvararg == nil)
            assert(native.nparams == nil and native.isvararg == nil)
        end
        info = debug.getinfo(function(a,...) end, "S")
        assert(info.nparams == nil and info.isvararg == nil)
    "#);
}

#[test]
fn failed_coroutine_metadata_and_open_upvalues_remain_available_until_close() {
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.exec(
                r#"
                local closer, observed
                local function subject()
                    local value = {tag = 17}
                    closer = function() observed = value end
                    local item <close> = setmetatable({}, {__close = closer})
                    error("failed", 0)
                end
                local co = coroutine.create(subject)
                assert(not coroutine.resume(co))
                assert(debug.getinfo(co, 0, "S").what == "C")
                assert(debug.getinfo(co, 1, "f").func == subject)
                local changed = false
                for index = 1, 16 do
                    local name = debug.getupvalue(closer, index)
                    if name == "value" then
                        assert(debug.setupvalue(closer, index, {tag = 23}) == "value")
                        changed = true
                        break
                    end
                end
                assert(changed)
                collectgarbage("collect")
                assert(not coroutine.close(co))
                assert(observed.tag == 23)
                assert(debug.getinfo(co, 0) == nil and debug.getinfo(co, 1) == nil)
            "#,
            )
            .unwrap_or_else(|error| panic!("{profile}/{mode}: {error:?}"));
        }
    }
}
