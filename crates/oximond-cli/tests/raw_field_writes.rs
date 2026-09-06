//! Field and global writes that overwrite a stored value take a raw fast
//! path in the interpreter. These programs pin the writes that must not:
//! absent keys, stored nils, `__newindex`, finalizer fields, and the
//! collector's view of the replaced and the stored value.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const ALL_PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(source: &str) {
    run_on(&ALL_PROFILES, source);
}

fn run_on(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            if let Err(error) = lua.exec(source) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

#[test]
fn global_overwrites_delete_and_recreate() {
    run(r#"
        counter = 0
        for i = 1, 500 do counter = counter + 1 end
        assert(counter == 500)
        counter = nil
        assert(counter == nil and rawget(_G, "counter") == nil)
        for i = 1, 3 do counter = i end
        assert(counter == 3)
        local seen = 0
        for k in pairs(_G) do if k == "counter" then seen = seen + 1 end end
        assert(seen == 1)
    "#);
}

#[test]
fn newindex_runs_only_for_absent_globals_and_fields() {
    run(r#"
        local log = {}
        local environment = _G
        local original = getmetatable(environment)
        setmetatable(environment, {__newindex = function(t, k, v)
            log[#log + 1] = k
            rawset(t, k, v)
        end})
        present = 1
        for i = 1, 200 do present = present + 1 end
        absent = 1
        for i = 1, 200 do absent = absent + 1 end
        present = nil
        present = 7
        setmetatable(environment, original)
        assert(present == 7 and absent == 201)
        assert(table.concat(log, ",") == "present,absent,present", table.concat(log, ","))

        local fields = {}
        local record = setmetatable({count = 0}, {__newindex = function(t, k, v)
            fields[#fields + 1] = k
            rawset(t, k, v)
        end})
        for i = 1, 200 do
            record.count = record.count + 1
            record.extra = i
        end
        record.count = nil
        record.count = 5
        assert(record.count == 5 and record.extra == 200)
        assert(table.concat(fields, ",") == "extra,count", table.concat(fields, ","))
    "#);
}

#[test]
fn stored_values_stay_reachable_across_collection() {
    run(r#"
        local holder = {slot = false}
        for i = 1, 2000 do
            holder.slot = {i, tostring(i)}
            if i % 50 == 0 then collectgarbage("step", 20) end
            assert(holder.slot[1] == i and holder.slot[2] == tostring(i))
        end
        collectgarbage()
        assert(holder.slot[1] == 2000)

        scratch = false
        for i = 1, 2000 do
            scratch = {i}
            if i % 50 == 0 then collectgarbage("step", 20) end
            assert(scratch[1] == i)
        end
    "#);
}

#[test]
fn replaced_values_become_collectable() {
    run(r#"
        local weak = setmetatable({}, {__mode = "v"})
        local holder = {slot = false}
        do
            local object = {}
            weak[1] = object
            holder.slot = object
        end
        holder.slot = 1
        collectgarbage()
        collectgarbage()
        assert(weak[1] == nil)

        kept = false
        do
            local object = {}
            weak[2] = object
            kept = object
        end
        kept = 1
        collectgarbage()
        collectgarbage()
        assert(weak[2] == nil)
    "#);
}

#[test]
fn finalizer_field_overwrites_keep_their_meaning() {
    // Tables gain `__gc` in Lua 5.2.
    run_on(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"
        local log = {}
        local mt = {__gc = function() log[#log + 1] = "first" end}
        local object = setmetatable({}, mt)
        mt.__gc = function() log[#log + 1] = "second" end
        object = nil
        collectgarbage()
        collectgarbage()
        assert(#log == 1 and log[1] == "second", table.concat(log, ","))
    "#,
    );
}

#[test]
fn string_values_and_keys_compare_by_content() {
    run(r#"
        local t = {name = "a"}
        local key = ("na") .. ("me")
        for i = 1, 100 do t[key] = "v" .. i end
        assert(t.name == "v100")
        local count = 0
        for _ in pairs(t) do count = count + 1 end
        assert(count == 1)
        glob = "x"
        _G[("g") .. ("lob")] = "y"
        assert(glob == "y")
    "#);
}
