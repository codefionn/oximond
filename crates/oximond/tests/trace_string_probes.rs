//! Lookups by a string key that is a runtime value, in traces: the probe of a
//! table's string index compiled in line, the key hash two probes of one key
//! share, the metatable class check a loop makes once before it starts, and
//! the reads of strings and tables out of an array whose index the loop
//! bounds prove.
//!
//! Every source runs in both modes of every profile and has to give the same
//! results. The ones marked native also have to run their iterations in
//! compiled code, so a lookup that fell back to the interpreter shows.

use oximond::runtime::Value;
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

/// Both modes agree. With `iterations`, at least that many loop iterations
/// also completed in compiled code.
fn check(source: &str, iterations: u64) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?} {source}"
        );
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats.iterations >= iterations,
            "{profile:?} {source}: {} iterations {stats:?}",
            stats.iterations
        );
    }
}

/// A loop that reads `values[keys[i]]` for `rounds` rounds over `keys`, after
/// `setup` has made both tables, and counts the hits and the misses.
fn lookups(setup: &str, rounds: u32) -> String {
    format!(
        "local values, keys = {{}}, {{}}
         {setup}
         local total, missing = 0, 0
         for _ = 1, {rounds} do
             for i = 1, #keys do
                 local value = values[keys[i]]
                 if value then total = total + value else missing = missing + 1 end
             end
         end
         return total, missing"
    )
}

#[test]
fn keys_that_are_the_strings_the_table_holds() {
    // Each key is the allocation the table was filled with: the first bucket
    // or a bucket further on holds its address. 12 keys stay in the ordered
    // representation, 700 are hashed and some of them share a first bucket.
    for count in [1, 12, 700] {
        let setup = format!(
            "for i = 1, {count} do
                 local key = 'field_' .. i
                 keys[i] = key
                 values[key] = i * 3 + 1
             end"
        );
        check(&lookups(&setup, 40_000 / count), 2_000);
    }
}

#[test]
fn keys_with_the_same_bytes_in_another_allocation() {
    // The probe finds no bucket with the key's address, then one with its
    // hash, and compares the bytes: packed in the string for a short key,
    // behind a pointer for a long one, which the runtime's walk compares.
    for (count, prefix) in [
        (12, "k"),
        (700, "k"),
        (
            12,
            "a_key_that_is_too_long_to_be_packed_into_the_string_itself_",
        ),
        (
            300,
            "a_key_that_is_too_long_to_be_packed_into_the_string_itself_",
        ),
    ] {
        let setup = format!(
            "for i = 1, {count} do
                 values['{prefix}' .. i] = i % 101
                 keys[i] = '{prefix}' .. i
             end"
        );
        check(&lookups(&setup, 40_000 / count), 2_000);
    }
}

#[test]
fn keys_the_table_does_not_have() {
    // One key in four is absent: a short one, a long one, and one that
    // differs from a present key only in its last byte.
    for count in [8, 900] {
        let setup = format!(
            "for i = 1, {count} do
                 values['item_' .. i] = i % 101
                 if i % 4 == 0 then
                     keys[i] = 'absent_' .. i
                 elseif i % 4 == 1 then
                     keys[i] = 'item_' .. i .. 'x'
                 elseif i % 4 == 2 then
                     keys[i] = string.rep('long', 20) .. i
                 else
                     keys[i] = 'item_' .. i
                 end
             end"
        );
        check(&lookups(&setup, 40_000 / count), 2_000);
    }
    // Every key is absent, of a table with string keys and of an empty one.
    check(
        &lookups(
            "for i = 1, 16 do keys[i] = 'no_' .. i end values.one = 1",
            2_000,
        ),
        2_000,
    );
    check(
        &lookups("for i = 1, 16 do keys[i] = 'no_' .. i end", 2_000),
        2_000,
    );
}

#[test]
fn keys_that_are_not_strings_among_string_keys() {
    // A number or a boolean among the keys leaves the string lookup through
    // its type guard, and the results still agree.
    check(
        &lookups(
            "for i = 1, 12 do
                 keys[i] = 'field_' .. i
                 values[keys[i]] = i
             end
             keys[5] = 5
             keys[9] = true
             values[5] = 500
             values[true] = 900",
            3_000,
        ),
        0,
    );
}

#[test]
fn a_table_that_gains_and_loses_keys_while_the_loop_runs() {
    // A new key changes the layout of the index, a removed one leaves a
    // bucket without a value, and a key stored again brings it back.
    check(
        "local values, keys = {}, {}
         for i = 1, 24 do
             keys[i] = 'slot_' .. i
             if i % 2 == 0 then values[keys[i]] = i end
         end
         local total, missing = 0, 0
         for round = 1, 3000 do
             for i = 1, 24 do
                 local value = values[keys[i]]
                 if value then total = total + value else missing = missing + 1 end
             end
             if round == 1000 then values.slot_3 = 300 end
             if round == 1500 then values.slot_4 = nil end
             if round == 2000 then values.slot_4 = 4000 end
             if round == 2500 then
                 for i = 30, 90 do values['slot_' .. i] = i end
             end
         end
         return total, missing",
        2_000,
    );
}

#[test]
fn two_lookups_of_one_key() {
    // The second lookup reuses the hash the first one read out of the key.
    check(
        "local first, second, keys = {}, {}, {}
         for i = 1, 12 do
             keys[i] = 'field_' .. i
             first[keys[i]] = i
             if i % 3 ~= 0 then second['field_' .. i] = i * 100 end
         end
         local total = 0
         for _ = 1, 3000 do
             for i = 1, 12 do
                 local key = keys[i]
                 total = total + first[key] + (second[key] or 7)
             end
         end
         return total",
        2_000,
    );
    // A store between the two lookups replaces the string in the slot the
    // key was read from, or the table the second lookup reads.
    check(
        "local first, second, keys = {}, {}, {}
         for i = 1, 12 do
             keys[i] = 'field_' .. i
             first[keys[i]] = i
             second[keys[i]] = i * 100
         end
         local total = 0
         for round = 1, 3000 do
             for i = 1, 12 do
                 local key = keys[i]
                 local a = first[key]
                 keys[i] = keys[i % 12 + 1]
                 second[key] = (second[key] or 0) + 1
                 total = total + a + second[key] + (first[keys[i]] or 0)
             end
         end
         return total",
        0,
    );
}

/// A loop that reads `object[keys[i]]` through the `__index` function of the
/// metatable of `object`, with `edit` run at the end of each round.
fn proxied(edit: &str) -> String {
    format!(
        "local fallback = {{alpha = 3, beta = 5, gamma = 7}}
         local keys = {{'alpha', 'beta', 'missing', 'gamma'}}
         local meta = {{__index = function(_, key) return fallback[key] end}}
         local other = {{__index = function(_, key) return 1000 end}}
         local object = setmetatable({{}}, meta)
         local spare = setmetatable({{}}, meta)
         local total, missing = 0, 0
         local function swap(to) setmetatable(object, to) end
         for round = 1, 6000 do
             for i = 1, 4 do
                 local value = object[keys[i]]
                 if value then total = total + value else missing = missing + 1 end
             end
             {edit}
         end
         return total, missing"
    )
}

#[test]
fn the_class_of_a_table_the_loop_holds_fixed_is_checked_before_the_loop() {
    check(&proxied(""), 2_000);
}

#[test]
fn a_metatable_that_changes_while_the_loop_runs_is_seen() {
    // Each edit makes the class check fail from then on: at the entry of the
    // inner loop where it was moved to, or in the loop where a call that may
    // change a metatable keeps it.
    for edit in [
        "if round == 3000 then setmetatable(object, other) end",
        "if round == 3000 then swap(other) end",
        "if round == 3000 then setmetatable(object, nil) end",
        "if round == 3000 then meta.__index = other.__index end",
        "if round == 3000 then meta.__index = fallback end",
        "if round == 3000 then rawset(object, 'missing', 11) end",
        "if round == 3000 then object, spare = spare, object end",
        "if round % 1000 == 0 then swap(round % 2000 == 0 and meta or other) end",
    ] {
        check(&proxied(edit), 0);
    }
    // The metatable changes inside the loop that makes the lookups.
    check(
        "local fallback = {alpha = 3, beta = 5}
         local keys = {'alpha', 'beta', 'missing', 'alpha'}
         local meta = {__index = function(_, key) return fallback[key] end}
         local other = {__index = function(_, key) return 1000 end}
         local object = setmetatable({}, meta)
         local total, missing = 0, 0
         for i = 1, 40000 do
             local value = object[keys[i % 4 + 1]]
             if value then total = total + value else missing = missing + 1 end
             if i % 9000 == 0 then
                 setmetatable(object, i % 18000 == 0 and meta or other)
             end
         end
         return total, missing",
        0,
    );
}

#[test]
fn strings_and_tables_out_of_an_array_the_loop_bounds_cover() {
    // The loop runs over `1, #list`, so its index is proven inside the array
    // and the read of a string, a table or a function checks only the type.
    check(
        "local names, rows, calls, sizes = {}, {}, {}, {}
         for i = 1, 40 do
             names[i] = 'name_' .. i
             rows[i] = {weight = i}
             calls[i] = function(x) return x + i end
             sizes[names[i]] = i
         end
         local total = 0
         for _ = 1, 1500 do
             for i = 1, #names do
                 total = total + sizes[names[i]] + rows[i].weight + calls[i](1)
             end
         end
         return total",
        2_000,
    );
}

#[test]
fn an_array_the_loop_bounds_do_not_cover_leaves_the_fast_read() {
    for source in [
        // The bound is past the end of the array: the last reads are nil.
        "local names, sizes = {}, {}
         for i = 1, 40 do names[i] = 'name_' .. i sizes[names[i]] = i end
         local total, absent = 0, 0
         for _ = 1, 1500 do
             for i = 1, 44 do
                 local name = names[i]
                 if name then total = total + sizes[name] else absent = absent + 1 end
             end
         end
         return total, absent",
        // The array shrinks, and then holds a number where a string was.
        "local names, sizes = {}, {}
         for i = 1, 40 do names[i] = 'name_' .. i sizes[names[i]] = i end
         local total, absent = 0, 0
         for round = 1, 1500 do
             for i = 1, 40 do
                 local name = names[i]
                 if name then total = total + (sizes[name] or 2) else absent = absent + 1 end
             end
             if round == 700 then names[40] = nil names[39] = nil end
             if round == 1100 then names[7] = 7 sizes[7] = 70 end
         end
         return total, absent",
        // A step that is not one, a start that is not one, and a loop that
        // counts down.
        "local names, sizes = {}, {}
         for i = 1, 40 do names[i] = 'name_' .. i sizes[names[i]] = i end
         local total = 0
         for _ = 1, 1500 do
             for i = 3, 40, 2 do total = total + sizes[names[i]] end
             for i = 40, 1, -1 do total = total + sizes[names[i]] end
         end
         return total",
        // A fractional step reads no element, and the lookups read nil.
        "local names, sizes = {}, {}
         for i = 1, 40 do names[i] = 'name_' .. i sizes[names[i]] = i end
         local total, absent = 0, 0
         for _ = 1, 1500 do
             for i = 1, 20, 0.5 do
                 local name = names[i]
                 if name then total = total + sizes[name] else absent = absent + 1 end
             end
         end
         return total, absent",
    ] {
        check(source, 0);
    }
}

#[test]
fn a_branch_on_the_value_takes_what_the_lookup_found() {
    // The branch that follows a lookup is recorded from the lookup's type
    // guard and starts with the same lookup, which it takes from the guard
    // instead of probing again. Every kind of answer goes that way: an
    // integer, a float, a boolean and an absent key, by the table's own
    // string and by another with the same bytes, packed in the string and
    // too long for that (the runtime's walk and the VM answer those).
    for prefix in [
        "slot_",
        "a_key_that_is_too_long_to_be_packed_into_the_string_itself_",
    ] {
        for count in [12, 400] {
            let source = format!(
                "local values, keys = {{}}, {{}}
                 for i = 1, {count} do
                     local key = '{prefix}' .. i
                     keys[i] = i % 2 == 0 and key or '{prefix}' .. i
                     if i % 4 == 0 then
                         values[key] = i
                     elseif i % 4 == 1 then
                         values[key] = i + 0.5
                     elseif i % 4 == 2 then
                         values[key] = i % 8 == 2
                     end
                 end
                 local total, flags, missing = 0, 0, 0
                 for _ = 1, {rounds} do
                     for i = 1, #keys do
                         local value = values[keys[i]]
                         if value == nil then
                             missing = missing + 1
                         elseif value == true then
                             flags = flags + 1
                         elseif value == false then
                             flags = flags + 2
                         else
                             total = total + value
                         end
                     end
                 end
                 return total, flags, missing",
                rounds = 48_000 / count
            );
            check(&source, 2_000);
        }
    }
}

#[test]
fn a_branch_sees_the_table_as_it_is_when_it_runs() {
    // The values change type, keys come and go and the table moves from the
    // ordered representation to the hashed one while the loop and its
    // branches run: what a guard hands its branch is the answer of that
    // iteration's probe.
    check(
        "local values, keys = {}, {}
         for i = 1, 10 do
             keys[i] = 'slot_' .. i
             if i % 2 == 0 then values['slot_' .. i] = i end
         end
         local total, flags, missing = 0, 0, 0
         for round = 1, 4000 do
             for i = 1, #keys do
                 local value = values[keys[i]]
                 if value == nil then
                     missing = missing + 1
                 elseif value == true then
                     flags = flags + 1
                 else
                     total = total + value
                 end
             end
             if round == 500 then values.slot_2 = nil end
             if round == 1000 then values.slot_3 = true end
             if round == 1500 then values.slot_4 = 4.25 end
             if round == 2500 then values.slot_6 = true end
             if round == 3000 then values.slot_2 = 22 end
             if round == 3500 then values.slot_3 = nil end
             if round == 2000 then
                 for i = 1, 40 do values['more_' .. i] = i end
             end
         end
         return total, flags, missing",
        // The recorder gives up on the body of the outer loop after a while
        // (with either backend's probe), so only part of the run is native.
        500,
    );
}

#[test]
fn a_miss_in_a_proxy_that_asks_a_function() {
    // A read that misses the proxy calls its `__index` function, which looks
    // the key up in another table. The branch for a key that table lacks
    // repeats both lookups: the proxy's miss, which it does not make again,
    // and the other table's, which the guard hands it. Keys the proxy holds
    // itself, keys that appear in either table while the loop runs and a
    // metatable that is replaced all leave that path.
    for edit in [
        "",
        "if round == 900 then backing.name_3 = 7 backing.name_1 = nil end",
        "if round == 900 then rawset(proxy, 'name_6', 50) end",
        "if round == 900 then rawset(proxy, 'name_3', false) end",
        "if round == 900 then
             setmetatable(proxy, { __index = function(_, key) return #key end })
         end",
        "if round == 900 then setmetatable(proxy, nil) end",
    ] {
        let source = format!(
            "local backing, keys = {{}}, {{}}
             for i = 1, 16 do
                 local key = 'name_' .. i
                 keys[i] = i % 2 == 0 and key or 'name_' .. i
                 if i % 3 == 1 then
                     backing[key] = i
                 elseif i % 3 == 2 then
                     backing[key] = i + 0.5
                 end
             end
             local proxy = setmetatable({{}}, {{
                 __index = function(_, key) return backing[key] end,
             }})
             proxy.name_2 = 100
             proxy.name_5 = 1.5
             local total, missing = 0, 0
             for round = 1, 1800 do
                 for i = 1, #keys do
                     local value = proxy[keys[i]]
                     if value then total = total + value else missing = missing + 1 end
                 end
                 {edit}
             end
             return total, missing"
        );
        check(&source, 2_000);
    }
}

#[test]
fn a_key_hashed_by_its_first_lookup_is_found_by_the_second() {
    // A string built in the loop has no hash until its first lookup asks the
    // VM, which computes it. The second lookup of the same key then has to
    // start from that hash, not from the empty one the first read.
    for count in [6, 40, 400] {
        let source = format!(
            "local first, second = {{}}, {{}}
             for i = 1, {count} do
                 first['key_' .. i] = i
                 if i % 3 ~= 0 then second['key_' .. i] = i * 2 end
             end
             local total, missing = 0, 0
             for _ = 1, {rounds} do
                 for i = 1, {count} do
                     local key = 'key_' .. i
                     local one, two = first[key], second[key]
                     if two == nil then
                         missing = missing + one
                     else
                         total = total + one + two
                     end
                 end
             end
             return total, missing",
            rounds = 24_000 / count
        );
        check(&source, 2_000);
    }
}

#[test]
fn a_key_out_of_an_array_hashed_by_its_first_lookup_is_found_by_the_second() {
    // The same with keys an array holds, none of them hashed before the
    // loop reads it: the loop is compiled long before it reaches the last.
    let source = "local first, second, keys = {}, {}, {}
         for i = 1, 6000 do
             first['a_key_that_is_too_long_to_be_packed_into_the_string_itself_or_cached_anywhere_at_all_' .. i] = i
             if i % 3 ~= 0 then second['a_key_that_is_too_long_to_be_packed_into_the_string_itself_or_cached_anywhere_at_all_' .. i] = i * 2 end
             keys[i] = 'a_key_that_is_too_long_to_be_packed_into_the_string_itself_or_cached_anywhere_at_all_' .. i
         end
         local total, missing = 0, 0
         for i = 1, #keys do
             local key = keys[i]
             local one, two = first[key], second[key]
             if two == nil then
                 missing = missing + one
             else
                 total = total + one + two
             end
         end
         return total, missing";
    check(source, 2_000);
}

// A slot that holds a key with the bytes of a table's key in another
// allocation is given the table's string by its first compiled lookup, so
// that the next one ends at the first compare (`trace_adopt_key` in
// `mir/lower/probe.rs`). Nothing in Lua tells the two strings apart, so the
// tests below ask for what a program can see: the lookups, and every later
// use of the strings the slots hold.

#[test]
fn a_slot_given_the_string_of_a_table_is_still_the_same_string() {
    // After the loop the strings of the array are read in every way a
    // program reads a string, used as keys of other tables, and outlive the
    // table they were looked up in.
    for count in [12, 700] {
        let source = format!(
            "local values, keys = {{}}, {{}}
             for i = 1, {count} do
                 values['k_' .. i] = i % 101
                 keys[i] = 'k_' .. i
             end
             local total = 0
             for _ = 1, {rounds} do
                 for i = 1, #keys do
                     total = total + values[keys[i]]
                 end
             end
             local seen, lengths, equal = {{}}, 0, 0
             for i = 1, #keys do
                 local key = keys[i]
                 seen[key] = i
                 lengths = lengths + #key
                 if key == 'k_' .. i and key .. '' == 'k_' .. i then equal = equal + 1 end
             end
             values = nil
             collectgarbage()
             collectgarbage()
             local found, bytes = 0, 0
             for i = 1, #keys do
                 if seen['k_' .. i] == i and seen[keys[i]] == i then found = found + 1 end
                 bytes = bytes + string.byte(keys[i], #keys[i])
             end
             local joined = table.concat(keys, ',')
             return total, lengths, equal, found, bytes, #joined, string.sub(joined, 1, 11)",
            rounds = 40_000 / count
        );
        check(&source, 2_000);
    }
}

#[test]
fn two_tables_with_equal_keys_in_two_allocations_read_through_one_array() {
    // Each lookup of a slot in the other table would swap the slot's string
    // again. The answers are the same whether or not the code stops
    // swapping, in one loop that reads both and in loops that take turns.
    for count in [10, 500] {
        let both = format!(
            "local a, b, keys = {{}}, {{}}, {{}}
             for i = 1, {count} do
                 a['k_' .. i] = i
                 b['k_' .. i] = 2 * i
                 keys[i] = 'k_' .. i
             end
             local total = 0
             for _ = 1, {rounds} do
                 for i = 1, #keys do
                     local key = keys[i]
                     total = total + a[key] + b[key]
                 end
             end
             return total",
            rounds = 40_000 / count
        );
        check(&both, 2_000);
        let turns = format!(
            "local tables, keys = {{}}, {{}}
             for t = 1, 3 do
                 tables[t] = {{}}
                 for i = 1, {count} do tables[t]['k_' .. i] = i * t end
             end
             for i = 1, {count} do keys[i] = 'k_' .. i end
             local function sum(values)
                 local total = 0
                 for i = 1, #keys do total = total + values[keys[i]] end
                 return total
             end
             local total = 0
             for round = 1, {rounds} do
                 total = total + sum(tables[round % 3 + 1])
             end
             return total",
            rounds = 60_000 / count
        );
        // A short loop in a function called in turn for three tables
        // compiles for few of its runs: the long one has to run compiled.
        check(&turns, if count < 100 { 0 } else { 2_000 });
    }
}

#[test]
fn a_key_removed_and_added_again_as_another_string() {
    // The table's key string changes under a slot that took the old one:
    // the slot's string is then one more equal string in another
    // allocation. Some keys stay away for a round.
    for count in [12, 300] {
        let source = format!(
            "local values, keys = {{}}, {{}}
             for i = 1, {count} do
                 values['k_' .. i] = i
                 keys[i] = 'k_' .. i
             end
             local total, missing = 0, 0
             for round = 1, {rounds} do
                 for i = 1, #keys do
                     local value = values[keys[i]]
                     if value then total = total + value else missing = missing + 1 end
                 end
                 local at = round % {count} + 1
                 values['k_' .. at] = nil
                 if round % 3 ~= 0 then values['k_' .. at] = round end
                 if round % 7 == 0 then keys[at] = 'k_' .. at end
             end
             return total, missing",
            rounds = 60_000 / count
        );
        check(&source, 2_000);
    }
}

#[test]
fn keys_made_by_every_function_that_makes_a_string() {
    // The slots are refilled between rounds with equal strings from
    // concatenation, `string.sub`, `tostring`, `string.format`,
    // `string.rep`, `table.concat` and a pattern capture, and with numbers
    // and near-equal strings that are other keys.
    let source = "local values, keys = {}, {}
         for i = 1, 64 do values[tostring(i)] = i end
         for i = 1, 64 do values['n' .. i] = 1000 + i end
         local makers = {
             function(i) return '' .. i end,
             function(i) return string.sub('x' .. i .. 'y', 2, -2) end,
             function(i) return tostring(i) end,
             function(i) return string.format('%d', i) end,
             function(i) return string.rep(tostring(i), 1) end,
             function(i) return table.concat({ i }) end,
             function(i) return string.match('<' .. i .. '>', '<(%d+)>') end,
             function(i) return 'n' .. i end,
             function(i) return i end,
             function(i) return i .. ' ' end,
             function(i) return '0' .. i end,
         }
         local total, missing = 0, 0
         for round = 1, 66 do
             local make = makers[round % #makers + 1]
             for i = 1, 64 do
                 if (i + round) % 5 ~= 0 then keys[i] = make(i) end
             end
             for _ = 1, 10 do
                 for i = 1, 64 do
                     local value = values[keys[i]]
                     if value then total = total + value else missing = missing + 1 end
                 end
             end
         end
         return total, missing";
    check(source, 2_000);
}

#[test]
fn weak_tables_and_collections_between_the_rounds() {
    // The table read is weak in its keys or values, or the array is weak in
    // its values, and a collection runs between rounds. A string is never
    // collected out of a weak table, whichever allocation holds it.
    for (values_mode, keys_mode) in [("k", ""), ("v", ""), ("kv", ""), ("", "v"), ("k", "kv")] {
        let source = format!(
            "local values = setmetatable({{}}, {{ __mode = '{values_mode}' }})
             local keys = setmetatable({{}}, {{ __mode = '{keys_mode}' }})
             for i = 1, 200 do
                 values['w_' .. i] = i
                 keys[i] = 'w_' .. i
             end
             local total = 0
             for round = 1, 120 do
                 for i = 1, 200 do
                     total = total + (values[keys[i]] or -1)
                 end
                 if round % 10 == 0 then
                     collectgarbage()
                     values['w_' .. round] = nil
                     values['w_' .. round] = round
                 end
             end
             local alive = 0
             for i = 1, 200 do
                 if type(keys[i]) == 'string' and values[keys[i]] then alive = alive + 1 end
             end
             return total, alive"
        );
        check(&source, 2_000);
    }
}

#[test]
fn keys_a_host_put_into_the_array() {
    // The array's strings come from the embedder, not from Lua code: each
    // is an allocation of its own that the table has never met.
    let source = "local values = {}
         for i = 1, 300 do values['host_' .. i] = i end
         local total = 0
         for _ = 1, 60 do
             for i = 1, #host_keys do
                 total = total + (values[host_keys[i]] or 0)
             end
         end
         local again = 0
         for i = 1, #host_keys do
             if host_keys[i] == 'host_' .. ((i - 1) % 300 + 1) then again = again + 1 end
         end
         return total, again";
    for profile in ALL {
        let run = |mode: ExecutionMode| {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            let runtime = lua.vm_mut().runtime_mut();
            let table = runtime.create_table().unwrap();
            for index in 0..450_i64 {
                // Every third string is one allocation stored in two slots.
                let text = Value::from(format!("host_{}", index % 300 + 1));
                runtime
                    .table_set(table, &Value::Integer(index + 1), text.clone())
                    .unwrap();
                if index % 3 == 0 && index + 450 < 600 {
                    runtime
                        .table_set(table, &Value::Integer(index + 451), text)
                        .unwrap();
                }
            }
            runtime.set_global("host_keys", Value::Table(table));
            let results = lua.exec(source).map_err(|error| format!("{error:?}"));
            let iterations = lua.vm().jit_statistics().trace.iterations;
            (results, iterations)
        };
        let (expected, _) = run(ExecutionMode::Interpreter);
        let (actual, iterations) = run(ExecutionMode::Tracing);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?}"
        );
        assert!(iterations >= 2_000, "{profile:?}: {iterations} iterations");
    }
}
