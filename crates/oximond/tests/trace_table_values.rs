//! Tables, strings and functions read out of fields and array slots, and
//! tables that differ from one iteration to the next.

use oximond::runtime::Value;
use oximond::vm::{TraceAbortReason, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, LuaJitExtension};

const ALL: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn outcome(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .luajit_extension(LuaJitExtension::Bit)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

fn show(result: &Result<Vec<Value>, String>) -> String {
    format!("{result:?}")
}

/// Recordings abandoned for good because of a table access. A `TableMiss` is
/// a retry the recorder asks for, not a refusal.
fn table_aborts(stats: &TraceStatistics) -> u64 {
    stats.recording_aborts[TraceAbortReason::UnsupportedOperand.index()]
}

/// Both modes agree, and the loop compiled and ran natively before the change
/// the source makes late in its run, which the trace leaves for the
/// interpreter at the access.
fn check_late(source: &str) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    }
}

/// Both modes agree. With `native`, the loop also compiled and ran natively
/// with no table abort.
fn check(source: &str, native: bool) {
    check_on(&ALL, source, native);
}

/// [`check`] over some profiles.
fn check_on(profiles: &[CompatibilityProfile], source: &str, native: bool) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        if std::env::var("TV_PRINT").is_ok() {
            eprintln!(
                "{profile:?} compiled={} entries={} table_aborts={} {:?}",
                stats.traces_compiled,
                stats.native_entries,
                table_aborts(&stats),
                stats.recording_aborts
            );
        }
        if native {
            assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
            assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
            assert_eq!(table_aborts(&stats), 0, "{profile:?} {source} {stats:?}");
        }
    }
}

#[test]
fn a_table_read_out_of_a_field_of_a_table_from_an_array() {
    check(
        r"
        local recs = {}
        for i = 1, 50 do recs[i] = { n = i, inner = { x = i * 2, y = 1 } } end
        local s = 0
        for rep = 1, 200 do
          for i = 1, #recs do
            local r = recs[i]
            local t = r.inner
            s = s + t.x + t.y
          end
        end
        return s
        ",
        true,
    );
}

#[test]
fn a_table_field_of_a_table_the_loop_holds_fixed() {
    check(
        r"
        local region = { solids = { a = 1, b = 2 }, scale = 3 }
        local s = 0
        for rep = 1, 100 do
          for i = 1, 50 do
            local t = region.solids
            s = s + t.a * region.scale + t.b + i
          end
        end
        return s
        ",
        true,
    );
}

#[test]
fn a_table_field_that_changes_type_or_goes_nil_leaves_to_the_interpreter() {
    check(
        r#"
        local recs = {}
        for i = 1, 40 do recs[i] = { inner = { x = i } } end
        recs[7].inner = nil
        recs[9].inner = 5
        recs[11].inner = setmetatable({}, { __index = function(_, k) return 100 end })
        recs[13].inner = setmetatable({}, { __index = { x = 1000 } })
        recs[15] = { inner = { x = "7" } }
        local s = 0
        for rep = 1, 100 do
          for i = 1, #recs do
            local ok, v = pcall(function() return recs[i].inner.x end)
            if ok then s = s + v else s = s + 1 end
          end
        end
        local n = 0
        for rep = 1, 100 do
          for i = 1, 6 do local t = recs[i].inner; n = n + t.x end
        end
        return s, n
        "#,
        false,
    );
}

#[test]
fn a_table_swapped_mid_loop_and_fields_added() {
    check(
        r"
        local holder = { inner = { x = 1, y = 2 } }
        local s = 0
        for i = 1, 100000 do
          local t = holder.inner
          s = s + t.x + t.y
          if i % 1000 == 0 then holder.inner = { x = i, y = 1 } end
          if i == 3000 then holder.inner = { y = 5, x = 6, z = 1 } end
        end
        return s
        ",
        false,
    );
}

#[test]
fn numeric_reads_of_a_table_met_by_lane() {
    check(
        r"
        local recs = {}
        for i = 1, 30 do recs[i] = { cells = { i, i + 1, i + 2, i + 3 } } end
        local s = 0
        for rep = 1, 100 do
          for i = 1, #recs do
            local c = recs[i].cells
            for k = 1, 4 do s = s + c[k] end
          end
        end
        return s
        ",
        true,
    );
}

/// A table met by lane, read at integer keys it mostly lacks (the
/// stronghold starts Mineclonia's concentric ring `chunk_test` reads,
/// structures.lua:406): `edit` runs before iteration `i`.
fn absent_keys_source(edit: &str) -> String {
    format!(
        r"
        local sets = {{}}
        for k = 1, 8 do
          local starts = {{}}
          starts[k * 1000 + 7] = true
          sets[k] = {{ starts = starts }}
        end
        local found = 0
        for i = 1, 40000 do
          {edit}
          local starts = sets[i % 8 + 1].starts
          if starts[i] then found = found + 1 end
        end
        return found
        "
    )
}

#[test]
fn absent_integer_keys_of_a_lane_table_read_nil_natively() {
    let source = absent_keys_source("");
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.iterations > 35_000, "{profile:?} {stats:?}");
        assert_eq!(table_aborts(&stats), 0, "{profile:?} {stats:?}");
    }
}

/// The absent key turns up, holds `false`, or the table gains an `__index`
/// (a table or a function): the read leaves to the interpreter, which gives
/// the same result.
#[test]
fn an_absent_key_of_a_lane_table_that_turns_up_or_gains_index_leaves_to_the_interpreter() {
    for edit in [
        "if i == 20000 then for k = 1, 8 do sets[k].starts[20000 + k * 3] = true end end",
        "if i == 20000 then sets[3].starts[20003] = false end",
        "if i == 20000 then setmetatable(sets[2].starts, { __index = function(t, k) return k % 5 == 0 end }) end",
        "if i == 20000 then setmetatable(sets[4].starts, { __index = { [20004] = 1, [20012] = 2 } }) end",
    ] {
        check_late(&absent_keys_source(edit));
    }
}

#[test]
fn a_table_read_by_a_string_key_out_of_a_lane_table() {
    check(
        r#"
        local registry = { items = { a = { w = 1 }, b = { w = 2 }, c = { w = 3 } } }
        local names = { "a", "b", "c", "a", "c" }
        local s = 0
        for rep = 1, 300 do
          for i = 1, #names do
            local def = registry.items[names[i]]
            s = s + def.w
          end
        end
        return s
        "#,
        true,
    );
}

#[test]
fn strings_and_functions_out_of_fields() {
    check(
        r#"
        local function f(x) return x + 1 end
        local defs = {}
        for i = 1, 20 do defs[i] = { name = "n" .. i, fn = f, groups = { a = i } } end
        local s, len = 0, 0
        for rep = 1, 100 do
          for i = 1, #defs do
            local d = defs[i]
            len = len + #d.name
            s = s + d.fn(d.groups.a)
          end
        end
        return s, len
        "#,
        true,
    );
}

#[test]
fn the_length_of_a_lane_table() {
    check(
        r"
        local recs = {}
        for i = 1, 25 do local l = {}; for k = 1, i do l[k] = k end; recs[i] = { list = l } end
        local s = 0
        for rep = 1, 100 do
          for i = 1, #recs do s = s + #recs[i].list end
        end
        return s
        ",
        true,
    );
}

#[test]
fn storing_tables_and_strings_into_fields_of_lane_tables() {
    check(
        r#"
        local recs = {}
        for i = 1, 30 do recs[i] = { slot = false, label = "x", n = 0 } end
        local shared = { v = 1 }
        local s = 0
        for rep = 1, 100 do
          for i = 1, #recs do
            local r = recs[i]
            r.slot = shared
            r.label = "label"
            r.n = r.n + 1
            s = s + r.slot.v + r.n
          end
        end
        return s, recs[3].label, recs[4].slot == shared
        "#,
        true,
    );
}

#[test]
fn storing_into_array_slots_of_lane_tables() {
    check(
        r#"
        local recs = {}
        for i = 1, 30 do recs[i] = { list = { 0, 0, 0 } } end
        local fill = { 9 }
        local s = 0
        for rep = 1, 100 do
          for i = 1, #recs do
            local l = recs[i].list
            l[1] = rep
            l[2] = fill
            l[3] = "s"
            s = s + l[1] + l[2][1]
          end
        end
        return s, recs[2].list[3]
        "#,
        true,
    );
}

#[test]
fn fresh_tables_and_strings_stored_in_a_loop_stay_alive() {
    // Fresh tables and strings go into fields of tables met by lane, with a
    // full collection between rounds. Every value must survive and read back.
    check(
        r#"
        local recs = {}
        for i = 1, 40 do recs[i] = { tag = "", kid = false } end
        local s = 0
        for rep = 1, 200 do
          for i = 1, #recs do
            local r = recs[i]
            r.kid = { rep, i }
            r.tag = "t" .. rep
          end
          collectgarbage()
        end
        for i = 1, #recs do s = s + recs[i].kid[1] + recs[i].kid[2] + #recs[i].tag end
        return s
        "#,
        true,
    );
}

#[test]
fn fresh_values_stored_through_numeric_keys_stay_alive() {
    check(
        r#"
        local recs = {}
        for i = 1, 40 do recs[i] = { kids = {}, tag = "" } end
        local s = 0
        for rep = 1, 200 do
          for i = 1, #recs do
            local r = recs[i]
            r.kids[(rep % 5) + 1] = { i }
            r.tag = "t" .. rep
          end
          collectgarbage()
        end
        for i = 1, #recs do s = s + recs[i].kids[1][1] + #recs[i].tag end
        return s
        "#,
        false,
    );
}

#[test]
fn method_calls_on_objects_met_by_lane() {
    check(
        r"
        local Class = {}
        Class.__index = Class
        function Class:get() return self.v end
        function Class:bump(d) self.v = self.v + d; return self end
        local objs = {}
        for i = 1, 30 do objs[i] = setmetatable({ v = i }, Class) end
        local s = 0
        for rep = 1, 100 do
          for i = 1, #objs do
            local o = objs[i]
            o:bump(1)
            s = s + o:get()
          end
        end
        return s
        ",
        true,
    );
}

// Changes the loops meet after the trace ran. Each source warms the loop up
// for 150 rounds of 200, then changes one thing a table access of the trace
// depends on and runs 50 more rounds.

#[test]
fn a_table_field_goes_nil_late() {
    check_late(
        r"
        local recs = {}
        for i = 1, 30 do recs[i] = { inner = { x = i } } end
        local s = 0
        for rep = 1, 200 do
          if rep == 151 then recs[5].inner = nil; recs[9].inner = false end
          for i = 1, #recs do
            local t = recs[i].inner
            if t then s = s + t.x else s = s - 1 end
          end
        end
        return s
        ",
    );
}

#[test]
fn a_table_gains_an_index_function_late() {
    check_late(
        r"
        local recs = {}
        for i = 1, 30 do recs[i] = { inner = { x = i } } end
        local s = 0
        for rep = 1, 200 do
          if rep == 151 then
            recs[7].inner = setmetatable({}, { __index = function(_, k) return 1000 end })
            recs[8].inner = setmetatable({}, { __index = { x = 77 } })
            recs[9].inner = setmetatable({ x = 5 }, { __index = function() return 1 end })
          end
          for i = 1, #recs do s = s + recs[i].inner.x end
        end
        return s
        ",
    );
}

#[test]
fn a_field_changes_type_late() {
    check_late(
        r#"
        local recs = {}
        for i = 1, 30 do recs[i] = { inner = { x = i, name = "n" .. i } } end
        local s, len = 0, 0
        for rep = 1, 200 do
          if rep == 151 then
            recs[4].inner.x = 2.5
            recs[5].inner.x = "12"
            recs[6].inner.name = 77
            recs[7].inner = 3
          end
          for i = 1, #recs do
            local t = recs[i].inner
            if type(t) == "table" then
              s = s + t.x
              local n = t.name
              if type(n) == "string" then len = len + #n end
            end
          end
        end
        return s, len
        "#,
    );
}

#[test]
fn a_record_is_swapped_for_one_with_another_layout_late() {
    check_late(
        r"
        local recs = {}
        for i = 1, 30 do recs[i] = { a = 1, inner = { x = i } } end
        local s = 0
        for rep = 1, 200 do
          if rep == 151 then
            recs[3] = { z = 1, y = 2, inner = { y = 1, x = 40 } }
            recs[4] = setmetatable({ inner = { x = 1 } }, { __index = { other = 1 } })
            recs[5] = { inner = setmetatable({ x = 9 }, { __index = function() return 0 end }) }
          end
          for i = 1, #recs do s = s + recs[i].inner.x end
        end
        return s
        ",
    );
}

#[test]
fn array_slots_of_a_lane_table_go_nil_or_change_late() {
    check_late(
        r"
        local recs = {}
        for i = 1, 30 do recs[i] = { cells = { i, i + 1, i + 2 } } end
        local s = 0
        for rep = 1, 200 do
          if rep == 151 then
            recs[2].cells[2] = nil
            recs[3].cells[1] = 0.5
            recs[4].cells = { 1, 2 }
            recs[5].cells = setmetatable({}, { __index = function(_, k) return k * 100 end })
          end
          for i = 1, #recs do
            local c = recs[i].cells
            for k = 1, 3 do s = s + (c[k] or 0) end
          end
        end
        return s
        ",
    );
}

#[test]
fn the_length_of_a_lane_table_with_len_late() {
    check_late(
        r"
        local recs = {}
        for i = 1, 30 do local l = {}; for k = 1, 3 do l[k] = k end; recs[i] = { list = l } end
        local s = 0
        for rep = 1, 200 do
          if rep == 151 then
            recs[3].list = setmetatable({ 1 }, { __len = function() return 42 end })
            recs[4].list[10] = 1
            recs[5].list[4] = 4
          end
          for i = 1, #recs do s = s + #recs[i].list end
        end
        return s
        ",
    );
}

#[test]
fn a_method_is_replaced_late() {
    check_late(
        r"
        local Class = {}
        Class.__index = Class
        function Class:get() return self.v end
        local objs = {}
        for i = 1, 30 do objs[i] = setmetatable({ v = i }, Class) end
        local s = 0
        for rep = 1, 200 do
          if rep == 151 then
            Class.get = function(self) return -self.v end
            objs[3].get = function(self) return 1000 end
            objs[4] = setmetatable({ v = 4 }, { __index = { get = function() return 7 end } })
          end
          for i = 1, #objs do s = s + objs[i]:get() end
        end
        return s
        ",
    );
}

#[test]
fn a_string_field_is_replaced_late_and_keys_come_from_strings() {
    check_late(
        r#"
        local defs = {}
        for i = 1, 20 do defs[i] = { name = "item" .. i, groups = { a = i, b = 1 } } end
        local reg = { items = {} }
        for i = 1, 20 do reg.items[defs[i].name] = defs[i] end
        local s = 0
        for rep = 1, 200 do
          if rep == 151 then
            defs[2].name = "changed"
            reg.items.item3 = nil
            reg.items.item4 = 5
          end
          for i = 1, #defs do
            local d = reg.items[defs[i].name]
            if type(d) == "table" then s = s + d.groups.a + #d.name end
          end
        end
        return s
        "#,
    );
}

/// The profiles that have `newproxy`, which makes a userdata.
const WITH_USERDATA: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21];

#[test]
fn a_userdata_read_out_of_a_field_is_a_lane() {
    check_on(
        &WITH_USERDATA,
        r#"
        local u = newproxy()
        local objs = {}
        for i = 1, 30 do objs[i] = { obj = u, n = i } end
        local c = 0
        for rep = 1, 100 do
          for i = 1, #objs do
            local r = objs[i]
            local o = r.obj
            if o then c = c + 1 end
            if type(o) == "userdata" then c = c + r.n end
          end
        end
        return c, objs[1].obj == u
        "#,
        true,
    );
}

#[test]
fn a_userdata_read_out_of_an_array_slot_is_a_lane() {
    check_on(
        &WITH_USERDATA,
        r#"
        local u, w = newproxy(), newproxy()
        local list = {}
        for i = 1, 40 do list[i] = (i % 2 == 0) and u or w end
        local c = 0
        for rep = 1, 200 do
          for i = 1, #list do
            local o = list[i]
            if type(o) == "userdata" then c = c + 1 end
          end
        end
        return c, list[2] == u, list[3] == w
        "#,
        true,
    );
}

#[test]
fn a_userdata_slot_goes_nil_late() {
    check_on(
        &WITH_USERDATA,
        r"
        local u = newproxy()
        local objs = {}
        for i = 1, 30 do objs[i] = { obj = u } end
        local c = 0
        for rep = 1, 200 do
          if rep == 180 then objs[7].obj = nil end
          for i = 1, #objs do
            local o = objs[i].obj
            if o then c = c + 1 end
          end
        end
        return c
        ",
        false,
    );
}

#[test]
fn userdata_eq_metamethod_is_not_skipped() {
    // Two distinct userdata that gain `__eq` late must compare through it.
    check_on(
        &WITH_USERDATA,
        r"
        local u = newproxy(true)
        local w = newproxy(u)
        local objs = {}
        for i = 1, 30 do objs[i] = { a = u, b = w } end
        local c = 0
        for rep = 1, 200 do
          if rep == 150 then getmetatable(u).__eq = function() return true end end
          for i = 1, #objs do
            local o = objs[i]
            if o.a == o.b then c = c + 1 end
          end
        end
        return c
        ",
        false,
    );
}

#[test]
fn identical_userdata_compare_natively() {
    check_on(
        &WITH_USERDATA,
        r"
        local u = newproxy()
        local objs = {}
        for i = 1, 30 do objs[i] = { a = u, b = u } end
        local c = 0
        for rep = 1, 200 do
          for i = 1, #objs do
            local o = objs[i]
            if o.a == o.b then c = c + 1 end
            if o.a ~= o.b then c = c - 1000 end
          end
        end
        return c
        ",
        true,
    );
}

#[test]
fn a_string_read_out_of_an_array_slot_of_a_bound_table() {
    check(
        r#"
        local words = {}
        for i = 1, 50 do words[i] = "w" .. i end
        local n = 0
        for i = 1, 100000 do
          local s = words[(i % 50) + 1]
          n = n + #s
        end
        return n, words[1], words[50]
        "#,
        true,
    );
}

#[test]
fn a_table_register_the_loop_swaps_is_read_through_its_lane() {
    check(
        r"
        local a, b = { 1.5, x = 1 }, { 2.5, y = 2 }
        local t, s = a, 0
        for i = 1, 100000 do
          s = s + t[1]
          if t == a then t = b else t = a end
        end
        return s, t == a
        ",
        true,
    );
}

#[test]
fn a_table_register_swapped_for_one_with_a_metatable_late() {
    check_late(
        r"
        local a, b = { 1.5 }, { 2.5 }
        local c = setmetatable({}, { __index = function(_, k) return 100 end })
        local t, s = a, 0
        for rep = 1, 200 do
          if rep == 180 then b = c end
          for i = 1, 200 do
            s = s + t[1]
            if t == a then t = b else t = a end
          end
        end
        return s
        ",
    );
}

#[test]
fn a_numeric_store_into_a_lane_table_meeting_newindex_late() {
    // The objects gain a `__newindex` for absent keys late in the run: the
    // store leaves before the instruction and the interpreter runs it.
    check_late(
        r"
        local log = 0
        local objs = {}
        for i = 1, 300 do objs[i] = { list = { 0, 0, 0 } } end
        for rep = 1, 200 do
          if rep == 180 then
            for i = 1, #objs do
              objs[i].list = setmetatable({}, { __newindex = function(t, k, v)
                log = log + 1
                rawset(t, k, v)
              end })
            end
          end
          for i = 1, #objs do
            local l = objs[i].list
            l[(rep % 3) + 1] = rep
          end
        end
        return log, objs[1].list[1], objs[2].list[2]
        ",
    );
}

#[test]
fn a_numeric_store_into_a_lane_table_with_a_table_newindex() {
    check_late(
        r"
        local sink = {}
        local objs = {}
        for i = 1, 300 do objs[i] = { list = { 1, 2, 3 } } end
        for rep = 1, 200 do
          if rep == 180 then
            for i = 1, #objs do
              objs[i].list = setmetatable({}, { __newindex = sink })
            end
          end
          for i = 1, #objs do
            objs[i].list[(rep % 3) + 1] = rep
          end
        end
        local n = 0
        for k, v in pairs(sink) do n = n + k + v end
        return n
        ",
    );
}

#[test]
fn numeric_stores_that_grow_and_rehash_beside_lane_reads() {
    check(
        r#"
        local recs = {}
        for i = 1, 20 do recs[i] = { list = {}, tag = "t" } end
        local s = 0
        for rep = 1, 200 do
          for i = 1, #recs do
            local r = recs[i]
            local l = r.list
            l[#l + 1] = rep
            l[-rep] = i
            s = s + l[#l] + #l + (l[-rep] or 0)
          end
        end
        return s, #recs[1].list
        "#,
        true,
    );
}

#[test]
fn a_store_that_rehashes_beside_lane_reads_retries_without_rows() {
    // New string keys go into the objects (the hash part rehashes) while
    // other fields of the same objects are read through lanes: the first
    // recording is refused for the pair, the retry reads and writes through
    // lookups and runs natively.
    let source = r#"
        local objs = {}
        for i = 1, 30 do objs[i] = { a = 1, b = 2 } end
        local s = 0
        for rep = 1, 200 do
          for i = 1, #objs do
            local o = objs[i]
            s = s + o.a + o.b
            o["k" .. (rep % 40)] = rep
          end
        end
        return s, objs[1].k5
    "#;
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        // One refusal for the pair (the rows of lanes beside a rehash), then
        // the retry that reads and writes through lookups.
        let misses = stats.recording_aborts[TraceAbortReason::TableMiss.index()];
        assert!((1..=2).contains(&misses), "{profile:?} {stats:?}");
        let called = lua.vm().trace_helper_calls();
        assert!(
            called.iter().any(|name| name.starts_with("table.get_")),
            "{profile:?} {called:?}"
        );
        assert!(
            called.contains(&"table.set_string"),
            "{profile:?} {called:?}"
        );
    }
}

#[test]
fn fresh_values_stored_while_a_collection_is_in_progress() {
    // The collector steps all the time (`setpause 20`, `setstepmul 1000`) and
    // the loop makes garbage, so an incremental cycle is under way while the
    // trace stores fresh tables and strings into old tables, by field and by
    // number. Every value must read back.
    check(
        r#"
        if pcall(collectgarbage, "setpause", 20) then collectgarbage("setstepmul", 1000)
        else collectgarbage("incremental", 20, 1000, 0) end
        local recs = {}
        for i = 1, 40 do recs[i] = { kid = false, tag = "", kids = {} } end
        for rep = 1, 300 do
          for i = 1, #recs do
            local r = recs[i]
            r.kid = { rep, i }
            r.tag = "t" .. rep .. "_" .. i
            r.kids[(rep % 4) + 1] = { i, rep }
            r.kids[5] = "s" .. i
            local junk = { rep, { i } }
          end
        end
        local s = 0
        for i = 1, #recs do
          local r = recs[i]
          s = s + r.kid[1] + r.kid[2] + #r.tag + r.kids[5]:len()
          for j = 1, 4 do if r.kids[j] then s = s + r.kids[j][1] + r.kids[j][2] end end
        end
        return s, recs[7].tag, recs[40].kid[1]
        "#,
        true,
    );
}

/// The compiled traces of `source` call `helper`.
fn calls(profiles: &[CompatibilityProfile], source: &str, helper: &str) {
    for &profile in profiles {
        let (_, lua) = outcome(profile, ExecutionMode::Tracing, source);
        let called = lua.vm().trace_helper_calls();
        assert!(
            called.contains(&helper),
            "{profile:?} {called:?} lacks {helper}"
        );
    }
}

#[test]
fn the_new_lookups_and_stores_are_calls_of_their_own() {
    calls(
        &ALL,
        r#"
        local recs = {}
        for i = 1, 30 do recs[i] = { list = { 0, 0, 0 } } end
        for rep = 1, 100 do
          -- A number into a number slot is stored in place
          -- (`LaneArrayStore`); a string is the call's.
          for i = 1, #recs do recs[i].list[2] = "r" end
        end
        return recs[1].list[2]
        "#,
        "table.set_index",
    );
    calls(
        &WITH_USERDATA,
        r"
        local u = newproxy()
        local objs = {}
        for i = 1, 30 do objs[i] = { obj = u } end
        local c = 0
        for rep = 1, 100 do
          for i = 1, #objs do if objs[i].obj then c = c + 1 end end
        end
        return c
        ",
        "table.get_userdata",
    );
}

#[test]
fn functions_stored_into_fields_and_slots_run_natively() {
    // A closure and a native function go into fields and number slots of a
    // table met by lane, and of one the loop holds fixed.
    check(
        r"
        local f = function(x) return x + 1 end
        local g = math.abs
        local recs = {}
        for i = 1, 40 do recs[i] = { fn = false, list = { false, false } } end
        local bound = { fn = false, list = { false, false, false } }
        for rep = 1, 200 do
          for i = 1, #recs do
            local r = recs[i]
            r.fn = f
            r.list[1] = f
            r.list[2] = g
            bound.fn = g
            bound.list[2] = f
          end
        end
        local s = 0
        for i = 1, #recs do
          local r = recs[i]
          s = s + r.fn(i) + r.list[1](i) + r.list[2](-i)
        end
        return s, bound.fn == g, bound.list[2] == f
        ",
        true,
    );
}

#[test]
fn functions_stored_while_a_collection_is_in_progress() {
    // Closures made outside the loop are stored into old tables while the
    // collector steps all the time and the loop makes garbage. A closure
    // dropped from its pool then lives through the table alone.
    check(
        r#"
        if pcall(collectgarbage, "setpause", 20) then collectgarbage("setstepmul", 1000)
        else collectgarbage("incremental", 20, 1000, 0) end
        local pool = {}
        for k = 1, 8 do pool[k] = function() return k end end
        local recs = {}
        for i = 1, 40 do recs[i] = { fn = false, list = { false } } end
        for rep = 1, 300 do
          local f = pool[(rep % 8) + 1]
          for i = 1, #recs do
            local r = recs[i]
            r.fn = f
            r.list[1] = f
            local junk = { rep, { i } }
          end
        end
        pool = nil
        collectgarbage()
        collectgarbage()
        local s = 0
        for i = 1, #recs do
          s = s + recs[i].fn() + recs[i].list[1]()
        end
        return s
        "#,
        true,
    );
}
