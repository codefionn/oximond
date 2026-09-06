//! A function metamethod of a table's own metatable (`__index`, `__newindex`
//! or an operator) is dispatched by a class guard: the trace pins the
//! metatable once per run, entry checks what it holds for the event, and each
//! access compares only the metatable the table's row names.
//!
//! Every program runs under the interpreter and under tracing execution in
//! the profiles that differ in dispatch, and the results must match. The
//! first tests show that the stable loops run natively; the others change the
//! class under the trace in every way the guard has to notice: the handler
//! replaced between runs and in the middle of a run, a metatable set on an
//! operand, objects of another class, chains through tables, keys that name
//! events, and operands of different metatables.

use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.vm_mut().ignore_remaining_work();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` in every profile under both engines, checks the results
/// agree, and returns each profile's tracing statistics.
fn agree(source: &str) -> Vec<(CompatibilityProfile, TraceStatistics)> {
    PROFILES
        .iter()
        .map(|&profile| {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
            let (traced, stats) = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(traced, expected, "{profile:?}\n{source}\n{stats:?}");
            (profile, stats)
        })
        .collect()
}

fn assert_ran_natively(runs: &[(CompatibilityProfile, TraceStatistics)], least: u64) {
    for (profile, stats) in runs {
        assert!(
            stats.iterations >= least,
            "{profile:?}: {} native iterations, wanted {least}\n{stats:?}",
            stats.iterations
        );
        assert!(stats.traces_compiled >= 1, "{profile:?}\n{stats:?}");
    }
}

#[test]
fn a_function_index_runs_natively_for_every_key() {
    let runs = agree(
        "local fallback = {alpha = 3, beta = 5, gamma = 7}
         local keys = {'alpha', 'beta', 'missing', 'gamma'}
         local object = setmetatable({}, {__index = function(_, key) return fallback[key] end})
         local total, missing = 0, 0
         for i = 1, 4000 do
           local value = object[keys[(i - 1) % 4 + 1]]
           if value then total = total + value else missing = missing + 1 end
         end
         return total, missing",
    );
    assert_ran_natively(&runs, 1000);
}

#[test]
fn a_function_newindex_runs_natively_for_every_key() {
    let runs = agree(
        "local backing, keys = {}, {}
         for i = 1, 64 do keys[i] = 'item' .. i end
         local proxy = setmetatable({}, {__newindex = function(_, key, value) backing[key] = value end})
         for i = 1, 4000 do proxy[keys[(i - 1) % 64 + 1]] = i % 17 end
         local total = 0
         for i = 1, 64 do total = total + (backing[keys[i]] or 0) end
         return total, rawget(proxy, 'item1')",
    );
    assert_ran_natively(&runs, 1000);
}

#[test]
fn operators_of_a_class_run_natively() {
    let runs = agree(
        "local Vec = {}
         Vec.__add = function(a, b)
           return (type(a) == 'number' and a or a.x) + (type(b) == 'number' and b or b.x)
         end
         Vec.__eq = function(a, b) return a.x == b.x end
         Vec.__lt = function(a, b) return a.x < b.x end
         Vec.__unm = function(a) return -a.x end
         local pool = {}
         for i = 1, 16 do pool[i] = setmetatable({x = i}, Vec) end
         local big = setmetatable({x = 100}, Vec)
         local total, less, equal = 0, 0, 0
         for i = 1, 4000 do
           local a, b = pool[i % 16 + 1], pool[(i * 5) % 16 + 1]
           total = total + (a + b) + (a + 1) + (2 + b) + (-a)
           if a < big then less = less + 1 end
           if a == b then equal = equal + 1 end
         end
         return total, less, equal",
    );
    assert_ran_natively(&runs, 500);
}

#[test]
fn a_handler_replaced_between_runs_is_noticed_at_entry() {
    agree(
        "local calls = 0
         local mt = {__index = function(_, key) calls = calls + 1; return 1 end}
         local object = setmetatable({}, mt)
         local function loop()
           local total = 0
           for i = 1, 3000 do total = total + object.k + object[i] end
           return total
         end
         local first = loop()
         mt.__index = function(_, key) calls = calls + 100; return 2 end
         local second = loop()
         mt.__index = nil
         local third = pcall(loop)
         return first, second, calls, third",
    );
}

#[test]
fn a_handler_replaced_in_the_middle_of_a_run_is_noticed() {
    agree(
        "local mt = {}
         local function one(_, key) return 1 end
         local function two(_, key) return 20 end
         mt.__index = one
         local object = setmetatable({}, mt)
         local total = 0
         for i = 1, 3000 do
           total = total + object.x
           if i == 1500 then mt.__index = two end
           if i == 2200 then rawset(mt, '__index', one) end
         end
         return total",
    );
}

#[test]
fn a_handler_replaced_through_a_dynamic_key_is_noticed() {
    agree(
        "local mt = {}
         local function one(_, key) return 1 end
         local function two(_, key) return 20 end
         mt.__index = one
         local object = setmetatable({}, mt)
         local names = {'a', 'b', '__index'}
         local total = 0
         for i = 1, 3000 do
           total = total + object.x
           if i == 1500 then mt[names[3]] = two end
         end
         return total",
    );
}

#[test]
fn a_handler_replaced_by_a_callback_is_noticed() {
    agree(
        "local mt = {}
         mt.__newindex = function(t, k, v) rawset(t, k, v) end
         local object = setmetatable({}, mt)
         local swap = function() mt.__newindex = function(t, k, v) rawset(t, k, (v or 0) * 2) end end
         local seen = {}
         for i = 1, 3000 do
           object['k' .. (i % 3)] = nil
           object.x = i
           if i == 700 then pcall(swap) end
           seen[#seen + 1] = rawget(object, 'x')
           rawset(object, 'x', nil)
         end
         return #seen, seen[1], seen[#seen], rawget(object, 'x')",
    );
}

#[test]
fn setmetatable_on_an_operand_inside_the_loop_is_noticed() {
    agree(
        "local first = {__index = function(_, key) return 1 end}
         local second = {__index = function(_, key) return 10 end}
         local object = setmetatable({}, first)
         local total = 0
         for i = 1, 3000 do
           total = total + object.k
           if i % 500 == 0 then
             setmetatable(object, i % 1000 == 0 and first or second)
           end
         end
         return total",
    );
}

#[test]
fn a_metatable_set_in_the_run_changes_the_class_the_next_access_sees() {
    // The access after `setmetatable` is guarded by the class the recording
    // saw, and the table's row names the metatable the run just gave it.
    agree(
        "local A = {__index = function(_, key) return 1 end}
         local B = {__index = function(_, key) return 1000 end}
         local sequence = {A, A, A, B, A, A}
         local object = setmetatable({}, A)
         local total = 0
         for i = 1, 3000 do
           setmetatable(object, A)
           total = total + object.k
           setmetatable(object, sequence[i % 6 + 1])
           total = total + object.k
         end
         return total",
    );
}

#[test]
fn setmetatable_beside_operator_handlers_is_noticed() {
    agree(
        "local A = {__add = function(a, b) return 1 end}
         local B = {__add = function(a, b) return 100 end}
         local object = setmetatable({}, A)
         local total = 0
         for i = 1, 3000 do
           total = total + (object + 1)
           if i % 300 == 0 then setmetatable(object, i % 600 == 0 and A or B) end
         end
         return total",
    );
}

#[test]
fn objects_of_another_class_leave_the_trace() {
    agree(
        "local A = {__index = function(_, key) return 1 end}
         local B = {__index = function(_, key) return 100 end}
         local C = {__index = A.__index}
         local objects = {}
         for i = 1, 30 do
           objects[i] = setmetatable({}, i % 3 == 0 and B or (i % 3 == 1 and A or C))
         end
         local total = 0
         for i = 1, 6000 do
           local object = objects[i % 30 + 1]
           total = total + object.field
         end
         return total",
    );
}

#[test]
fn objects_of_another_class_beside_operators_leave_the_trace() {
    agree(
        "local A = {__sub = function(a, b) return a.v - b end}
         local B = {__sub = function(a, b) return a.v * 1000 - b end}
         local objects = {}
         for i = 1, 20 do objects[i] = setmetatable({v = i}, i % 4 == 0 and B or A) end
         local total = 0
         for i = 1, 6000 do total = total + (objects[i % 20 + 1] - 1) end
         return total",
    );
}

#[test]
fn an_index_chain_through_a_table_keeps_its_walk() {
    agree(
        "local base = setmetatable({}, {__index = function(_, key) return 5 end})
         local object = setmetatable({}, {__index = base})
         local direct = setmetatable({}, {__index = function(_, key) return 7 end})
         local total = 0
         for i = 1, 3000 do
           total = total + object.k + direct.k
           if i == 2000 then rawset(base, 'k', 100) end
         end
         return total",
    );
}

#[test]
fn a_key_the_object_holds_never_reaches_the_handler() {
    agree(
        "local object = setmetatable({held = 1}, {__index = function(_, key) return 10 end})
         local keys = {'held', 'absent', 'held', 'absent2'}
         local total = 0
         for i = 1, 3000 do
           total = total + object[keys[i % 4 + 1]]
           if i == 1500 then object.absent = 1000 end
           if i == 2500 then object.held = nil end
         end
         return total",
    );
}

#[test]
fn keys_that_name_events_are_ordinary_keys_of_the_object() {
    agree(
        "local calls = 0
         local mt = {__index = function(_, key) calls = calls + 1; return 1 end}
         local object = setmetatable({}, mt)
         local keys = {'__index', '__newindex', '__len', 'x'}
         local total = 0
         for i = 1, 3000 do total = total + object[keys[i % 4 + 1]] end
         rawset(object, '__index', 50)
         for i = 1, 3000 do total = total + object[keys[i % 4 + 1]] end
         return total, calls",
    );
}

#[test]
fn events_stored_into_the_metatable_by_a_handler_are_noticed() {
    agree(
        "local mt = {}
         mt.__index = function(t, key)
           if key == 'flip' then mt.__index = function() return 99 end end
           return 1
         end
         local object = setmetatable({}, mt)
         local keys = {'a', 'b', 'flip', 'c'}
         local total = 0
         for i = 1, 3000 do total = total + object[keys[i % 4 + 1]] end
         return total",
    );
}

#[test]
fn eq_of_operands_with_different_metatables_follows_the_interpreter() {
    agree(
        "local A = {__eq = function(a, b) return a.v == b.v end}
         local B = {__eq = function(a, b) return true end}
         local C = {__eq = A.__eq}
         local one = setmetatable({v = 1}, A)
         local two = setmetatable({v = 1}, B)
         local three = setmetatable({v = 1}, C)
         local four = setmetatable({v = 2}, A)
         local list = {one, two, three, four}
         local equal = 0
         for i = 1, 6000 do
           local a, b = list[i % 4 + 1], list[(i * 3) % 4 + 1]
           if a == b then equal = equal + 1 end
         end
         return equal",
    );
}

#[test]
fn numbers_and_strings_beside_a_class_keep_their_handlers() {
    agree(
        "local V = {}
         V.__concat = function(a, b) return 'cat' end
         V.__add = function(a, b) return 1 end
         local object = setmetatable({}, V)
         local count = 0
         for i = 1, 3000 do
           local s = 'x' .. object
           local t = object .. 'y'
           if s == 'cat' and t == 'cat' then count = count + 1 end
           count = count + (object + i) + (i + object)
         end
         return count",
    );
}

#[test]
fn string_arithmetic_beside_a_class_is_not_a_table_handler() {
    agree(
        "local V = {__add = function(a, b) return 5 end}
         local object = setmetatable({}, V)
         local total = 0
         for i = 1, 3000 do
           total = total + (object + '10') + ('3' + 4)
         end
         return total",
    );
}

#[test]
fn a_metatable_given_a_handler_before_the_run_pins_it_at_entry() {
    agree(
        "local mt = {}
         local object = setmetatable({}, mt)
         local total = 0
         local function loop()
           local sum = 0
           for i = 1, 2000 do sum = sum + (object.k or 3) end
           return sum
         end
         total = total + loop()
         mt.__index = function() return 4 end
         total = total + loop()
         mt.__index = function() return 5 end
         total = total + loop()
         mt.__index = nil
         total = total + loop()
         return total",
    );
}

#[test]
fn an_absent_key_of_a_small_table_reads_nil_through_the_scan() {
    agree(
        "local small = {a = 1, b = 2, c = 3}
         local keys = {'a', 'b', 'c', 'd', 'e', 'abc', ''}
         local object = setmetatable({}, {__index = function(_, key) return small[key] end})
         local hits, misses = 0, 0
         for i = 1, 4000 do
           if object[keys[i % 7 + 1]] then hits = hits + 1 else misses = misses + 1 end
         end
         return hits, misses",
    );
}

#[test]
fn a_newindex_on_a_table_that_holds_the_key_stores_raw() {
    agree(
        "local log = 0
         local object = setmetatable({held = 0}, {__newindex = function(t, k, v)
           log = log + 1
           rawset(t, k, v)
         end})
         local keys = {'held', 'fresh1', 'held', 'fresh2'}
         for i = 1, 3000 do
           object[keys[i % 4 + 1]] = i
           if i % 100 == 0 then object.fresh1, object.fresh2 = nil, nil end
         end
         return log, object.held, object.fresh1, object.fresh2",
    );
}

#[test]
fn a_global_receiver_that_gains_the_key_stops_reaching_the_handler() {
    // The receiver lives in a global, so it has no window of its own; the
    // key-absence check must still read the receiver and not the table of
    // another object of the class.
    agree(
        "local A = {__index = function() return 1 end}
         obj = setmetatable({}, A)
         local other = setmetatable({}, A)
         local out = {}
         for round = 1, 6 do
           local t = 0
           for i = 1, 3000 do t = t + obj.k + other.x end
           out[round] = t
           if round == 3 then rawset(obj, 'k', 1000) end
         end
         return out[1], out[3], out[4], out[6]",
    );
}

#[test]
fn an_upvalue_receiver_that_gains_the_key_stops_reaching_the_handler() {
    agree(
        "local A = {__index = function() return 1 end}
         local obj = setmetatable({}, A)
         local other = setmetatable({}, A)
         local function run()
           local t = 0
           for i = 1, 3000 do t = t + obj.k + other.x end
           return t
         end
         local a = run(); run(); run()
         rawset(obj, 'k', 1000)
         return a, run(), run()",
    );
}

#[test]
fn equal_strings_of_other_allocations_find_the_key_of_a_small_table() {
    // The probe keys are built at run time, so they are equal to the stored
    // keys without being the same allocation.
    agree(
        "local small = {key1 = 1, key2 = 2, key3 = 3}
         local keys = {}
         for i = 1, 5 do keys[i] = 'key' .. i end
         local object = setmetatable({}, {__index = function(_, key) return small[key] end})
         local total = 0
         for i = 1, 4000 do total = total + (object[keys[i % 5 + 1]] or 100) end
         return total",
    );
}
