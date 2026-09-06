//! A captured local lives in its register while the function that declared it
//! runs, and its upvalue reads and writes that register (an open upvalue, as
//! in PUC-Lua). The upvalue closes, taking the value with it, when the
//! variable's scope ends, the function returns or tail-calls, an error
//! unwinds it, or its coroutine dies or is collected. These programs check
//! every way of observing such a variable, in every profile and both
//! execution modes.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL: [CompatibilityProfile; 5] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];
const FROM_52: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
];
const MODES: [ExecutionMode; 2] = [ExecutionMode::Interpreter, ExecutionMode::Tracing];

fn as_number(value: &Value) -> f64 {
    match value {
        Value::Number(n) => *n,
        Value::Integer(n) => *n as f64,
        Value::Boolean(true) => 1.0,
        Value::Boolean(false) => 0.0,
        other => panic!("not a number: {other:?}"),
    }
}

/// Runs `source` under every listed profile and both modes. The results must
/// be numbers (or booleans) and must agree across all of them.
fn run_in(profiles: &[CompatibilityProfile], source: &str) -> Vec<f64> {
    let mut first: Option<Vec<f64>> = None;
    for &profile in profiles {
        for mode in MODES {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            let results: Vec<f64> = lua
                .exec(source)
                .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}"))
                .iter()
                .map(as_number)
                .collect();
            match &first {
                Some(first) => assert_eq!(&results, first, "{profile:?} {mode:?} disagrees"),
                None => first = Some(results),
            }
        }
    }
    first.unwrap()
}

fn run(source: &str) -> Vec<f64> {
    run_in(&ALL, source)
}

#[test]
fn closures_share_one_upvalue_with_each_other_and_with_the_frame() {
    let results = run(r"
        local n = 0
        local function inc() n = n + 1; return n end
        local function get() return n end
        local function set(v) n = v end
        inc(); inc()
        local a = get()          -- 2
        n = n + 10               -- the frame writes its own register
        local b = get()          -- 12
        set(100)
        local c = n              -- 100
        inc()
        return a, b, c, n, get()
    ");
    assert_eq!(results, [2.0, 12.0, 100.0, 101.0, 101.0]);
}

#[test]
fn a_function_that_returns_keeps_its_variables_in_the_closures() {
    let results = run(r"
        local function counter(start)
          local n = start
          return function() n = n + 1; return n end,
                 function() return n end
        end
        local inc1, get1 = counter(10)
        local inc2, get2 = counter(100)
        inc1(); inc1(); inc2()
        return get1(), get2(), inc1(), inc2()
    ");
    assert_eq!(results, [12.0, 101.0, 13.0, 102.0]);
}

#[test]
fn every_loop_iteration_has_its_own_upvalue() {
    let results = run(r"
        local fns = {}
        for i = 1, 4 do fns[i] = function() i = i + 100; return i end end
        local a = fns[1]() + fns[1]()      -- 101 + 201
        local b = fns[2]()                 -- 102
        local gs = {}
        for _, v in ipairs({10, 20, 30}) do gs[#gs + 1] = function() v = v + 1; return v end end
        local c = gs[1]() + gs[1]() + gs[3]()   -- 11 + 12 + 31
        local hs = {}
        local k = 0
        while k < 3 do
          k = k + 1
          local x = k * 2
          hs[k] = function() x = x + 1; return x end
        end
        local d = hs[1]() + hs[1]() + hs[3]()   -- 3 + 4 + 7
        local es = {}
        local j = 0
        repeat
          j = j + 1
          local y = j
          es[j] = function() y = y * 2; return y end
        until y >= 3
        local e = es[1]() + es[1]() + es[3]()   -- 2 + 4 + 6
        return a, b, c, d, e
    ");
    assert_eq!(results, [302.0, 102.0, 54.0, 14.0, 12.0]);
}

#[test]
fn a_loop_counter_is_not_changed_by_a_closure_over_it() {
    // The closure writes its own copy of the control variable in 5.2 and
    // later; in 5.1 and LuaJIT as well, since the body sees a fresh local.
    let results = run(r"
        local seen = 0
        local fns = {}
        for i = 1, 3 do
          local f = function() i = i + 10; return i end
          f()
          seen = seen + 1
          fns[#fns + 1] = f
        end
        return seen, fns[1](), fns[3]()
    ");
    assert_eq!(results, [3.0, 21.0, 23.0]);
}

#[test]
fn break_and_goto_close_the_upvalues_of_the_scope_they_leave() {
    let results = run_in(
        &FROM_52,
        r"
        local fns = {}
        for i = 1, 10 do
          local v = i
          fns[#fns + 1] = function() v = v + 1; return v end
          if i == 3 then break end
        end
        local a = fns[1]() + fns[1]() + fns[3]()      -- 2 + 3 + 4
        local gs = {}
        do
          local n = 0
          ::again::
          local w = n
          gs[#gs + 1] = function() w = w + 5; return w end
          n = n + 1
          if n < 3 then goto again end
        end
        local b = gs[1]() + gs[1]() + gs[3]()          -- 5 + 10 + 7
        return a, b
    ",
    );
    assert_eq!(results, [9.0, 22.0]);
}

#[test]
fn upvalues_of_a_suspended_coroutine_stay_live() {
    // `g` is made inside the coroutine and used by the main thread while the
    // coroutine is suspended: the upvalue is open on a stack that is not
    // running. The coroutine sees what `g` wrote after it resumes.
    let results = run(r"
        local co = coroutine.wrap(function()
          local y = 5
          local function g(v) if v then y = v end return y end
          coroutine.yield(g)
          local after = y
          coroutine.yield(after)
          return y
        end)
        local g = co()
        local a = g()
        local b = g(40)
        local c = g(41)
        local d = co()        -- the coroutine read 41
        g(77)
        local e = co()        -- the coroutine returns y: 77
        local f = g()         -- the coroutine is dead: the variable is closed
        g(78)
        return a, b, c, d, e, f, g()
    ");
    assert_eq!(results, [5.0, 40.0, 41.0, 41.0, 77.0, 77.0, 78.0]);
}

#[test]
fn a_collected_suspended_coroutine_leaves_its_variables_to_the_closures() {
    let results = run(r"
        local g
        do
          local co = coroutine.create(function()
            local t = {value = 3}
            local function f(v) if v then t = {value = v} end return t.value end
            g = f
            coroutine.yield()
            return t.value
          end)
          coroutine.resume(co)
        end
        g(9)
        collectgarbage(); collectgarbage()
        local a = g()
        g(11)
        collectgarbage()
        return a, g()
    ");
    assert_eq!(results, [9.0, 11.0]);
}

#[test]
fn a_value_stored_through_a_suspended_threads_upvalue_survives_collection() {
    // The only reference to the table is the coroutine's register, written
    // by the main thread through the open upvalue.
    let results = run(r"
        local co = coroutine.create(function()
          local box
          coroutine.yield(function(v) box = v end)
          coroutine.yield()
          return box.n
        end)
        local _, set = coroutine.resume(co)
        set({n = 123})
        set = nil
        collectgarbage(); collectgarbage()
        local ok1 = coroutine.resume(co)
        local ok2, n = coroutine.resume(co)
        return ok1, ok2, n
    ");
    assert_eq!(results, [1.0, 1.0, 123.0]);
}

#[test]
fn an_error_closes_the_upvalues_of_the_frames_it_unwinds() {
    let results = run(r"
        local escaped
        local ok = pcall(function()
          local n = 1
          local function bump() n = n + 1; return n end
          escaped = bump
          bump()
          local deeper = function()
            local m = 50
            local function grow() m = m + 1; return m end
            escaped2 = grow
            error('x')
          end
          deeper()
        end)
        local a = escaped()      -- 3
        local b = escaped()      -- 4
        local c = escaped2()     -- 51
        collectgarbage()
        return ok and 1 or 0, a, b, c, escaped2()
    ");
    assert_eq!(results, [0.0, 3.0, 4.0, 51.0, 52.0]);
}

#[test]
fn an_error_in_a_coroutine_closes_its_upvalues() {
    let results = run(r"
        local f
        local co = coroutine.create(function()
          local n = 7
          f = function() n = n + 1; return n end
          error('boom')
        end)
        local ok = coroutine.resume(co)
        local a = f()
        local b = f()
        return ok and 1 or 0, a, b
    ");
    assert_eq!(results, [0.0, 8.0, 9.0]);
}

#[test]
fn closing_a_coroutine_closes_its_upvalues() {
    let results = run_in(
        &[CompatibilityProfile::Lua54],
        r"
        local f
        local co = coroutine.create(function()
          local n = 1
          f = function() n = n * 3; return n end
          coroutine.yield()
        end)
        coroutine.resume(co)
        local a = f()
        local ok = coroutine.close(co)
        return a, f(), ok and 1 or 0
    ",
    );
    assert_eq!(results, [3.0, 9.0, 1.0]);
}

#[test]
fn a_tail_call_closes_the_locals_it_replaces() {
    let results = run(r"
        local function make()
          local n = 10
          local function bump() n = n + 1; return n end
          return bump          -- the frame returns normally
        end
        local function tail(g) return g() end
        local function frame_with_local()
          local v = 5
          local function inc() v = v + 1; return v end
          return tail(inc)     -- v is closed by the tail call, then inc runs
        end
        local b = make()
        local first = b()
        return first, b(), frame_with_local()
    ");
    assert_eq!(results, [11.0, 12.0, 6.0]);
}

#[test]
fn varargs_frames_with_captured_locals_return_and_tail_call() {
    let results = run(r"
        local function pack_sum(...)
          local total = 0
          local function add(v) total = total + v end
          for i = 1, select('#', ...) do add((select(i, ...))) end
          return total, ...
        end
        local function fwd(...)
          local c = select('#', ...)
          local function get() return c end
          return pack_sum(...), get()
        end
        local t, a, b = pack_sum(1, 2, 3)
        local u = fwd(4, 5)
        return t, a, b, u
    ");
    assert_eq!(results, [6.0, 1.0, 2.0, 9.0]);
}

#[test]
fn a_recursion_deeper_than_the_stack_keeps_every_frames_upvalue() {
    // The value stack moves while it grows. Each frame's captured local is
    // an open upvalue of a stack slot, not of an address.
    let results = run(r"
        local fns = {}
        local function rec(depth)
          local mine = depth
          fns[depth] = function(v) if v then mine = v end return mine end
          if depth < 3000 then rec(depth + 1) end
          mine = mine + 1
        end
        rec(1)
        local sum = 0
        for i = 1, 3000 do sum = sum + fns[i]() - i end
        fns[2500](7)
        return sum, fns[2500](), fns[1]()
    ");
    assert_eq!(results, [3000.0, 7.0, 2.0]);
}

#[test]
fn a_native_callback_and_a_metamethod_write_the_frames_variables() {
    let results = run(r#"
        local calls, picked = 0, 0
        local function cmp(a, b) calls = calls + 1; return a < b end
        local t = {5, 3, 9, 1, 7}
        table.sort(t, cmp)
        local mt = setmetatable({}, {__index = function(_, k) picked = picked + k; return k end,
                                     __add = function(a, b) calls = calls + 1000; return 5 end})
        local x = mt[3] + mt[4]
        local z = mt + 1
        local replaced = 0
        string.gsub("a b c", "%a", function() replaced = replaced + 1 end)
        return calls > 1000 and 1 or 0, picked, x, z, replaced, t[1], t[5]
    "#);
    assert_eq!(results, [1.0, 7.0, 7.0, 5.0, 3.0, 1.0, 9.0]);
}

#[test]
fn debug_functions_see_open_and_closed_upvalues() {
    let results = run_in(
        &FROM_52,
        r"
        local a, b = 1, 2
        local function f() return a, b end
        local function g() return b end
        local n1, v1 = debug.getupvalue(f, 1)
        local n2, v2 = debug.getupvalue(f, 2)
        debug.setupvalue(f, 1, 10)                  -- writes the running frame's local
        local direct = a
        a = 20                                      -- and the frame's write reaches f
        local seen = f()
        local same = debug.upvalueid(f, 2) == debug.upvalueid(g, 1)
        local differ = debug.upvalueid(f, 1) ~= debug.upvalueid(f, 2)
        local function make()
          local p, q = 3, 4
          local function h() return p, q end
          return h, function() return q end
        end
        local h, k = make()
        debug.setupvalue(h, 2, 40)                  -- a closed upvalue
        local joined = 0
        debug.upvaluejoin(h, 1, k, 1)               -- h's p now is k's q (40)
        joined = (h())
        return v1, v2, direct, seen, same and 1 or 0, differ and 1 or 0, joined, (k())
    ",
    );
    assert_eq!(results, [1.0, 2.0, 10.0, 20.0, 1.0, 1.0, 40.0, 40.0]);
}

#[test]
fn upvaluejoin_across_open_upvalues_of_different_frames() {
    let results = run_in(
        &FROM_52,
        r"
        local function pair()
          local x = 1
          return function() return x end, function(v) x = v end
        end
        local read1, write1 = pair()
        local read2, write2 = pair()
        local result = {}
        local function outer()
          local mine = 100
          local function peek() return mine end
          local function poke(v) mine = v end
          -- join read1's upvalue to the open local of this running frame
          debug.upvaluejoin(read1, 1, peek, 1)
          result[1] = read1()          -- 100: reads this frame's register
          mine = 150
          result[2] = read1()          -- 150
          poke(160)
          result[3] = read1()          -- 160
          result[4] = mine             -- 160
        end
        outer()
        write1(7)                       -- read1 no longer shares write1's variable
        result[5] = read1()             -- the closed variable of outer: 160
        write2(9)
        result[6] = read2()
        return result[1], result[2], result[3], result[4], result[5], result[6]
    ",
    );
    assert_eq!(results, [100.0, 150.0, 160.0, 160.0, 160.0, 9.0]);
}

#[test]
fn debug_getlocal_and_setlocal_reach_captured_locals() {
    let results = run_in(
        &FROM_52,
        r"
        local captured = 5
        local plain = 6
        local function get() return captured end
        local function set(v) captured = v end
        local _, got = debug.getlocal(1, 1)
        set(9)
        local _, after = debug.getlocal(1, 1)
        debug.setlocal(1, 1, 70)
        local viaClosure = get()
        local function inner()
          local mine = 1
          local function peek() return mine end
          debug.setlocal(1, 1, 33)
          return peek(), mine
        end
        local p, m = inner()
        return got, after, viaClosure, p, m
    ",
    );
    assert_eq!(results, [5.0, 9.0, 70.0, 33.0, 33.0]);
}

#[test]
fn to_be_closed_variables_and_captured_locals_close_in_order() {
    let results = run_in(
        &[CompatibilityProfile::Lua54],
        r"
        local log = 0
        local keep
        do
          local cap = 1
          local function bump() cap = cap + 1; return cap end
          keep = bump
          local res <close> = setmetatable({}, {__close = function()
            log = log * 10 + bump()          -- cap is still alive while closing
          end})
        end
        local first = keep()
        local ok = pcall(function()
          local cap2 = 100
          local function bump2() cap2 = cap2 + 1; return cap2 end
          local res <close> = setmetatable({}, {__close = function() log = log * 10 + bump2() end})
          error('x')
        end)
        return log, first, ok and 1 or 0
    ",
    );
    assert_eq!(results, [2.0 * 10.0 + 101.0, 3.0, 0.0]);
}

#[test]
fn collection_keeps_what_open_upvalues_hold() {
    let results = run(r"
        local held = {tag = 42}
        local function get() return held.tag end
        local function swap(t) held = t end
        for i = 1, 5 do
          local junk = {}
          for j = 1, 200 do junk[j] = {j} end
          collectgarbage()
        end
        local a = get()
        swap({tag = 7})
        held = {tag = held.tag + 1}
        collectgarbage(); collectgarbage()
        return a, get(), held.tag
    ");
    assert_eq!(results, [42.0, 8.0, 8.0]);
}

#[test]
fn a_hot_loop_over_captured_locals_agrees_in_every_mode() {
    let results = run(r"
        local sum, count = 0, 0
        local function add(v) sum = sum + v; count = count + 1 end
        local function total() return sum end
        for i = 1, 20000 do
          add(i)
          if i % 1000 == 0 then sum = sum - total() % 3 end
        end
        return sum, count
    ");
    assert_eq!(results.len(), 2);
    assert_eq!(results[1], 20000.0);
    assert!(results[0] > 2.0e8);
}

#[test]
fn wrapped_generators_share_variables_with_their_callers() {
    let results = run(r"
        local produced = 0
        local function gen(n)
          return coroutine.wrap(function()
            local i = 0
            local function step() i = i + 1; produced = produced + 1; return i end
            while i < n do coroutine.yield(step()) end
          end)
        end
        local total = 0
        for v in gen(5) do total = total + v end
        local g = gen(3)
        local a, b = g(), g()
        return total, produced, a + b
    ");
    assert_eq!(results, [15.0, 7.0, 3.0]);
}

#[test]
fn a_loop_that_makes_many_closures_collects_and_keeps_every_variable() {
    // Closure creation allocates in the fast loop and may have to leave it
    // for a collection; the variables the closures share must survive that.
    let results = run(r"
        local keep = {}
        local total = 0
        for i = 1, 60000 do
          local v = i
          local f = function() v = v + 1; return v end
          if i % 5000 == 0 then keep[#keep + 1] = f end
          total = total + f()
        end
        collectgarbage()
        local after = 0
        for _, f in ipairs(keep) do after = after + f() end
        return total, #keep, after
    ");
    let total: f64 = (1..=60000).map(|i| f64::from(i) + 1.0).sum();
    let after: f64 = (1..=12).map(|k| f64::from(k * 5000) + 2.0).sum();
    assert_eq!(results, [total, 12.0, after]);
}

#[test]
fn hooks_see_the_same_captured_variables() {
    // Count and call hooks run the hooked variant of the fast loop; a line
    // hook runs the general loop. Each must read and write the same register.
    let results = run_in(
        &FROM_52,
        r"
        local calls, lines = 0, 0
        local n = 0
        local function bump() n = n + 1; return n end
        local function make(k)
          local own = k
          return function() own = own + n; return own end
        end
        debug.sethook(function() calls = calls + 1 end, '', 7)
        local fns = {}
        for i = 1, 20 do fns[i] = make(i); bump() end
        local a = fns[1]() + fns[20]()
        debug.sethook(function() lines = lines + 1 end, 'l')
        local b = fns[1]() + bump()
        debug.sethook()
        debug.sethook(function() calls = calls + 1 end, 'cr')
        local c = fns[2]() + n
        debug.sethook()
        return a, b, c, n, calls > 0 and 1 or 0, lines > 0 and 1 or 0
    ",
    );
    // fns[1]: own 1 + 20 = 21, fns[20]: 20 + 20 = 40; then fns[1] again: 21 + 20
    // = 41, bump() = 21; fns[2]: 2 + 21 = 23, n = 21.
    assert_eq!(results, [61.0, 62.0, 44.0, 21.0, 1.0, 1.0]);
}
