use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
];

fn check(source: &str, profiles: &[CompatibilityProfile]) {
    for &profile in profiles {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            if let Err(error) = lua.exec(source) {
                panic!("{profile} {mode:?}: {error}");
            }
        }
    }
}

/// Element-by-element references for the one-move shift.
const REFERENCE: &str = r"
    local function ref_insert(t, pos, v)
        local n = #t
        for i = n + 1, pos + 1, -1 do rawset(t, i, rawget(t, i - 1)) end
        rawset(t, pos, v)
    end
    local function ref_remove(t, pos)
        local n = #t
        local removed = rawget(t, pos)
        for i = pos, n - 1 do rawset(t, i, rawget(t, i + 1)) end
        rawset(t, n, nil)
        return removed
    end
    local function same(a, b, upto)
        for i = 1, upto do assert(rawget(a, i) == rawget(b, i), i) end
        assert(#a == #b)
    end
";

#[test]
fn proxies_with_element_metamethods_are_honoured_from_lua53_and_ignored_before() {
    let proxy = r#"
        local store, log = {}, {}
        local proxy = setmetatable({}, {
            __index = function(_, k) log[#log + 1] = "get" return store[k] end,
            __newindex = function(_, k, v) log[#log + 1] = "set" store[k] = v end,
            __len = function() return #store end,
        })
    "#;
    check(
        &format!(
            "{proxy}
        table.insert(proxy, 'a')
        table.insert(proxy, 1, 'b')
        assert(store[1] == 'b' and store[2] == 'a')
        assert(rawget(proxy, 1) == nil and rawlen(proxy) == 0)
        assert(#log > 0)
        assert(table.remove(proxy, 1) == 'b')
        assert(store[1] == 'a' and store[2] == nil)
        "
        ),
        &[CompatibilityProfile::Lua53, CompatibilityProfile::Lua54],
    );
    check(
        &format!(
            "{proxy}
        table.insert(proxy, 'a')
        table.insert(proxy, 1, 'b')
        assert(rawget(proxy, 1) == 'b' and rawget(proxy, 2) == 'a')
        assert(next(store) == nil and #log == 0)
        assert(table.remove(proxy, 1) == 'b')
        assert(rawget(proxy, 1) == 'a' and rawget(proxy, 2) == nil)
        "
        ),
        &[CompatibilityProfile::Lua51],
    );
}

#[test]
fn holes_and_borders_inside_the_array_part_shift_like_the_element_loop() {
    check(
        &format!(
            "{REFERENCE}
        local function builds()
            local list = {{}}
            list[#list + 1] = {{1, 2, 3, 4}}
            list[#list + 1] = {{1, 2, 3, 4, 5, 6}}
            list[#list + 1] = {{1, 2, 3, nil, nil, 6}}
            list[#list + 1] = {{nil, 2, 3}}
            list[#list + 1] = {{1, 2, 3, nil}}
            list[#list + 1] = {{}}
            list[2][2] = nil
            list[5][3] = nil
            return list
        end
        for index = 1, #builds() do
            for pos = 1, 8 do
                local a, b = builds()[index], builds()[index]
                local n = #a
                if pos <= n + 1 then
                    table.insert(a, pos, 'x')
                    ref_insert(b, pos, 'x')
                    same(a, b, 9)
                end
                a, b = builds()[index], builds()[index]
                n = #a
                if pos <= n then
                    assert(table.remove(a, pos) == ref_remove(b, pos))
                    same(a, b, 9)
                end
            end
            local a, b = builds()[index], builds()[index]
            assert(table.remove(a) == ref_remove(b, #b))
            same(a, b, 9)
        end
        "
        ),
        &PROFILES,
    );
}

#[test]
fn removing_during_a_next_traversal_matches_the_element_loop() {
    check(
        &format!(
            "{REFERENCE}
        local function walk(remove)
            local t = {{10, 20, 30, 40, 50, name = 'n'}}
            local visited = {{}}
            for k, v in next, t do
                visited[#visited + 1] = tostring(k) .. '=' .. tostring(v)
                if k == 2 then remove(t, 3) end
            end
            local final = {{}}
            for i = 1, 6 do final[i] = tostring(rawget(t, i)) end
            return table.concat(visited, ','), table.concat(final, ','), #t
        end
        local a1, a2, a3 = walk(table.remove)
        local b1, b2, b3 = walk(ref_remove)
        assert(a1 == b1 and a2 == b2 and a3 == b3, a1 .. ' | ' .. b1)
        "
        ),
        &PROFILES,
    );
}
