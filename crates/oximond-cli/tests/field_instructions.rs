//! `t.name` reads and writes compile to field instructions that name their
//! key by a string constant. These programs pin what those instructions must
//! keep from the register-keyed table access: absent fields, `__index` and
//! `__newindex` in every form, non-table receivers and their error names,
//! and chunks whose field names do not fit the instruction's byte operand.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};
use std::fmt::Write;

const ALL_PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(source: &str) {
    for profile in ALL_PROFILES {
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
fn absent_fields_read_nil_unless_index_applies() {
    run(r#"
        local plain = {present = 1}
        local misses = 0
        for i = 1, 300 do
            if plain.absent == nil then misses = misses + 1 end
        end
        assert(misses == 300 and plain.present == 1)

        local log = {}
        local proxy = setmetatable({own = 2}, {__index = function(t, k)
            log[#log + 1] = k
            return k .. "!"
        end})
        for i = 1, 200 do assert(proxy.own == 2) end
        assert(proxy.foo == "foo!" and proxy.bar == "bar!")
        assert(table.concat(log, ",") == "foo,bar")

        local chain = setmetatable({}, {__index = setmetatable({mid = 2},
            {__index = {deep = 3}})})
        local sum = 0
        for i = 1, 200 do sum = sum + chain.mid + chain.deep end
        assert(sum == 1000 and chain.none == nil)
    "#);
}

#[test]
fn field_writes_reach_newindex_only_for_absent_fields() {
    run(r#"
        local seen = {}
        local record = setmetatable({}, {__newindex = function(t, k, v)
            seen[#seen + 1] = k
            rawset(t, k, v)
        end})
        record.a = 1
        for i = 1, 200 do record.a = record.a + 1 end
        record.a = nil
        record.a = 5
        assert(record.a == 5)
        assert(table.concat(seen, ",") == "a,a", table.concat(seen, ","))

        local constructed = {x = 1, y = 2, x2 = 3}
        constructed.y = nil
        local keys = {}
        for k in pairs(constructed) do keys[#keys + 1] = k end
        table.sort(keys)
        assert(table.concat(keys, ",") == "x,x2")
    "#);
}

#[test]
fn non_table_receivers_keep_their_metatables_and_error_names() {
    run(r#"
        local text = "abc"
        assert(text.len == string.len and text:upper() == "ABC")
        local function message(f)
            local ok, error = pcall(f)
            assert(not ok)
            return error
        end
        local nothing
        assert(message(function() return nothing.field end):find("upvalue 'nothing'"))
        local nested = {}
        assert(message(function() return nested.inner.leaf end):find("field 'inner'"))
        assert(message(function() nested.inner.leaf = 1 end):find("field 'inner'"))
        assert(message(function() return missing_global.field end)
            :find("global 'missing_global'"))
    "#);
}

#[test]
fn field_names_past_the_byte_operand_still_resolve() {
    let mut source = String::from("local numbers = {");
    for value in 0..300 {
        write!(source, "{value}.5, ").expect("formatting into a String cannot fail");
    }
    source.push_str(
        "} local t = {late = 1} \
         function t.method(self) return self.late end \
         for i = 1, 200 do t.late = t.late + 1 end \
         assert(t:method() == 201 and t.absent == nil and #numbers == 300)",
    );
    run(&source);
}
