//! A frame keeps a captured local in its register and mirrors it into the
//! cell that closures share. The VM only writes back registers that the chunk
//! can still write after a closure captured them, so these programs check
//! that every way of observing a captured local still sees its current value.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];
const MODES: [ExecutionMode; 2] = [ExecutionMode::Interpreter, ExecutionMode::Tracing];

fn as_number(value: &Value) -> f64 {
    match value {
        Value::Number(n) => *n,
        Value::Integer(n) => *n as f64,
        other => panic!("not a number: {other:?}"),
    }
}

/// Runs `source` under every profile and mode and returns its results,
/// which must be numbers and must agree across all of them.
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
    run_in(&PROFILES, source)
}

fn assert_numbers(results: &[f64], expected: &[f64]) {
    assert_eq!(results, expected);
}

#[test]
fn a_closure_made_before_a_loop_sees_every_update_when_called_mid_loop() {
    let results = run(r"
        local total = 0
        local seen = 0
        local function peek() return total end
        for i = 1, 50 do
            total = total + i
            seen = seen + peek() - total
        end
        local x = 1
        local function getx() return x end
        local sum = 0
        for i = 1, 10 do
            x = x * 2
            sum = sum + getx()
        end
        return total, seen, sum
    ");
    assert_numbers(&results, &[1275.0, 0.0, 2046.0]);
}

#[test]
fn a_callee_that_sets_an_upvalue_is_seen_by_the_caller_after_return() {
    // `count` is written once before any closure captures it, so the chunk
    // never writes it back; the callee's store must still reach the caller.
    let results = run(r"
        local count = 10
        local function bump(n) count = count + n end
        local function read() return count end
        bump(5)
        local a = count
        bump(1)
        local b = count + 0
        local c = read()
        for i = 1, 20 do bump(1) end
        return a, b, c, count, read()
    ");
    assert_numbers(&results, &[15.0, 16.0, 16.0, 36.0, 36.0]);
}

#[test]
fn a_call_result_and_a_callee_write_both_survive_the_return_into_a_captured_frame() {
    // `count` is captured and never written by the caller after capture
    // (not volatile): the return writes the result, then syncs. `slot` is
    // captured and receives the call result (volatile): sync, write, flush.
    // Fixed-count, open (`return ...` of a call) and zero-result returns.
    let results = run(r"
        local count = 0
        local function bump(x) count = count + 1; return x + count end
        local function bump_open(x) count = count + 1; return (function(...) return ... end)(x, count) end
        local function bump_none() count = count + 1 end
        local slot = 0
        local function read_slot() return slot end
        local sum = 0
        for i = 1, 300 do
            local r = bump(i)
            sum = sum + r
            slot = bump(i)
            local a, b = bump_open(i)
            bump_none()
            sum = sum + a + b + count + read_slot()
        end
        return sum, count, slot
    ");
    // Per iteration, count goes up four times.
    let mut count = 0.0;
    let mut sum = 0.0;
    let mut slot = 0.0;
    for i in 1..=300 {
        let i = f64::from(i);
        count += 1.0;
        sum += i + count;
        count += 1.0;
        slot = i + count;
        count += 1.0;
        let (a, b) = (i, count);
        count += 1.0;
        sum += a + b + count + slot;
    }
    assert_numbers(&results, &[sum, count, slot]);
}

#[test]
fn native_callbacks_read_and_write_captured_locals() {
    let results = run(r#"
        local calls = 0
        local limit = 3
        local mark = "x"
        local function cmp(a, b) calls = calls + 1; return a < b end
        local t = {5, 3, 9, 1, 7}
        table.sort(t, cmp)
        local ordered = t[1] .. t[2] .. t[3] .. t[4] .. t[5]
        local replaced = 0
        local out = string.gsub("a b c d", "%a", function(ch)
            replaced = replaced + 1
            if replaced > limit then mark = mark .. ch end
            return ch:upper()
        end)
        return calls > 0 and 1 or 0, ordered == "13579" and 1 or 0, replaced, mark == "xd" and 1 or 0, out == "A B C D" and 1 or 0
    "#);
    assert_numbers(&results, &[1.0, 1.0, 4.0, 1.0, 1.0]);
}

#[test]
fn a_volatile_local_written_between_native_calls_reaches_the_callback() {
    let results = run(r"
        local v = 0
        local function show() return v end
        local acc = 0
        for i = 1, 5 do
            v = i * 10
            local r = {}
            table.sort({3, 2, 1}, function(a, b) acc = acc + show(); return a < b end)
        end
        return acc > 0 and 1 or 0, v
    ");
    assert_numbers(&results, &[1.0, 50.0]);
}

#[test]
fn debug_get_and_set_local_see_captured_locals() {
    let results = run_in(
        &[CompatibilityProfile::Lua54],
        r"
        local stable = 7
        local volatile = 1
        local function get() return stable, volatile end
        volatile = 2
        local name, value = debug.getlocal(1, 1)
        local n2, v2 = debug.getlocal(1, 2)
        debug.setlocal(1, 1, 70)
        debug.setlocal(1, 2, 20)
        local s, v = get()
        local direct = stable + volatile
        return value, v2, s, v, direct
    ",
    );
    assert_numbers(&results, &[7.0, 2.0, 70.0, 20.0, 90.0]);
}

#[test]
fn debug_upvalue_functions_see_captured_locals() {
    let results = run_in(
        &[CompatibilityProfile::Lua54],
        r"
        local a = 1
        local b = 2
        local function fa() return a end
        local function fb() return b end
        local _, got = debug.getupvalue(fa, 1)
        debug.setupvalue(fa, 1, 100)
        local after_set = a
        a = 101
        local via_closure = fa()
        debug.upvaluejoin(fa, 1, fb, 1)
        b = 5
        local joined = fa()
        return got, after_set, via_closure, joined, a
    ",
    );
    assert_numbers(&results, &[1.0, 100.0, 101.0, 5.0, 101.0]);
}

#[test]
fn an_error_inside_pcall_keeps_modifications_of_captured_locals() {
    let results = run(r"
        local state = 1
        local log = 0
        local function note() log = log + state end
        local ok = pcall(function()
            state = 2
            note()
            state = 3
            error('boom')
        end)
        note()
        return ok and 1 or 0, state, log
    ");
    assert_numbers(&results, &[0.0, 3.0, 5.0]);
}

#[test]
fn coroutines_keep_captured_locals_across_yield_and_resume() {
    let results = run(r"
        local n = 0
        local step = 5
        local function inc() n = n + step end
        local co = coroutine.wrap(function()
            for i = 1, 3 do
                inc()
                step = step + 1
                coroutine.yield(n)
            end
            return n
        end)
        local a, b, c, d = co(), co(), co(), co()
        return a, b, c, d, n, step
    ");
    assert_numbers(&results, &[5.0, 11.0, 18.0, 18.0, 18.0, 8.0]);
}

#[test]
fn hooks_read_captured_locals_of_the_running_function() {
    let results = run_in(
        &[CompatibilityProfile::Lua54],
        r"
        local fixed = 4
        local moving = 0
        local function read() return fixed + moving end
        local seen = {}
        debug.sethook(function(event, line)
            local _, f = debug.getlocal(2, 1)
            local _, m = debug.getlocal(2, 2)
            seen[#seen + 1] = f * 1000 + m
        end, 'l')
        moving = 1
        moving = 2
        debug.sethook()
        local last = seen[#seen]
        return #seen >= 2 and 1 or 0, read(), seen[1] // 1000
    ",
    );
    assert_numbers(&results, &[1.0, 6.0, 4.0]);
}

#[test]
fn count_hooks_see_a_captured_local_change() {
    let results = run_in(
        &[CompatibilityProfile::Lua54],
        r"
        local counter = 0
        local function bump() counter = counter + 1 end
        local max = 0
        debug.sethook(function()
            local _, c = debug.getlocal(2, 1)
            if c and c > max then max = c end
        end, '', 1)
        for i = 1, 20 do bump() end
        debug.sethook()
        return counter, max > 0 and 1 or 0
    ",
    );
    assert_numbers(&results, &[20.0, 1.0]);
}

#[test]
fn loops_that_recapture_a_local_each_iteration_keep_separate_cells() {
    let results = run(r"
        local fns = {}
        for i = 1, 3 do
            local v = i
            fns[i] = function() v = v + 10; return v end
            v = v * 2
        end
        return fns[1](), fns[2](), fns[3](), fns[1]()
    ");
    assert_numbers(&results, &[12.0, 14.0, 16.0, 22.0]);
}

#[test]
fn a_self_recursive_local_function_is_usable_right_after_its_definition() {
    let results = run(r"
        local function fact(n) if n <= 1 then return 1 end return n * fact(n - 1) end
        local r = fact(5)
        fact = function(n) return -n end
        return r, fact(3)
    ");
    assert_numbers(&results, &[120.0, -3.0]);
}

#[test]
fn a_coroutine_sees_the_resumers_writes_and_the_resumer_sees_the_coroutines() {
    let results = run(r"
        local shared = 0
        local fixed = 100
        local function get() return shared + fixed end
        local co = coroutine.create(function(a)
            local seen = {}
            for i = 1, 3 do
                shared = shared + 1
                seen[#seen + 1] = get()
                coroutine.yield(shared)
            end
            return shared
        end)
        local out = 0
        for i = 1, 3 do
            shared = shared * 10
            local ok, v = coroutine.resume(co)
            out = out + v
        end
        local ok, last = coroutine.resume(co)
        return out, last, shared
    ");
    // shared: 0 -> 1, 10 -> 11, 110 -> 111.
    assert_numbers(&results, &[1.0 + 11.0 + 111.0, 111.0, 111.0]);
}

#[test]
fn metamethods_pcall_and_tostring_update_captured_locals_seen_by_the_caller() {
    let results = run(r"
        local hits = 0
        local base = 5
        local function touch() hits = hits + base end
        local t = setmetatable({}, {
            __index = function(_, k) touch(); return k end,
            __add = function(a, b) touch(); return 1 end,
            __tostring = function() touch(); return 'obj' end,
        })
        local _ = t.x
        local after_index = hits
        local _ = t + 1
        local after_add = hits
        local _ = tostring(t)
        local after_tostring = hits
        local ok = pcall(function() touch(); error('x') end)
        return after_index, after_add, after_tostring, hits
    ");
    assert_numbers(&results, &[5.0, 10.0, 15.0, 20.0]);
}

/// A call that returns into a captured destination register. The callee
/// writes other captured locals, so the caller's cells are newer than its
/// registers when the results land. The fast return paths must sync the
/// caller before they write the result and flush it after, as the general
/// return path does. Expected texts are the output of stock Lua 5.1, 5.4,
/// 5.5 and `LuaJIT`, which agree.
mod captured_return {
    use std::process::Command;

    const PROFILES: [&str; 4] = ["lua51", "lua54", "lua55", "luajit21"];
    const MODES: [&str; 2] = ["interpreter", "tracing"];

    fn run(name: &str, body: &str, expected: &str) {
        // The body runs as a function so its `return` yields the text.
        run_chunk(
            name,
            &format!("print((function()\n{body}\nend)())\n"),
            expected,
        );
    }

    /// Runs `source` as the main chunk and expects `expected` plus a newline.
    fn run_chunk(name: &str, source: &str, expected: &str) {
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
        let path = dir.join(format!("captured_return_{name}.lua"));
        std::fs::write(&path, source).unwrap();
        for profile in PROFILES {
            for mode in MODES {
                for sync in ["0", "1"] {
                    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
                        .arg(format!("--compat={profile}"))
                        .arg(format!("--execution={mode}"))
                        .arg(&path)
                        .env("OXIMOND_JIT_SYNC", sync)
                        .output()
                        .unwrap();
                    assert!(
                        output.status.success(),
                        "{name} {profile} {mode} sync={sync}: {}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                    assert_eq!(
                        String::from_utf8_lossy(&output.stdout),
                        format!("{expected}\n"),
                        "{name} {profile} {mode} sync={sync}"
                    );
                }
            }
        }
    }

    /// The differential-test repro as given: a vararg main chunk, not a
    /// wrapped body.
    #[test]
    fn main_chunk_repro() {
        run_chunk(
            "main_chunk_repro",
            "local a, b = 5, 1\nlocal function f(p)\n  a = 4\n  return b + 3\nend\nb = f(nil)\nprint(b, a)\n",
            "4\t4",
        );
    }

    #[test]
    fn repro() {
        run(
            "repro",
            r#"
local a, b = 5, 1
local function f(p)
  a = 4
  return b + 3
end
b = f(nil)
return b .. "\t" .. a
"#,
            "4\t4",
        );
    }

    /// The callee writes a captured local that is not the destination; the
    /// caller reads both afterwards, through a closure and directly.
    #[test]
    fn other_captured_local_written_by_callee() {
        run(
            "other_captured",
            r#"
local a, b = 5, 1
local function f(p)
  a = a + 10
  return b + 3
end
local function read() return a, b end
b = f(nil)
local x, y = read()
return b .. "\t" .. a .. "\t" .. x .. "\t" .. y
"#,
            "4\t15\t15\t4",
        );
    }

    /// Multi-assignment from a call goes through temporaries and MOVEs, so
    /// this does not reach the multi-register fast-return flush. The VM tests
    /// in calls.rs cover that with hand-built chunks.
    #[test]
    fn multi_assignment_from_call_into_captured_locals() {
        run(
            "multi_result",
            r#"
local x, y, z = 1, 2, 3
local function f(p)
  z = z + 100
  return 10, 20
end
local function read() return x, y, z end
x, y = f(nil)
local rx, ry, rz = read()
return x .. "\t" .. y .. "\t" .. z .. "\t" .. rx .. "\t" .. ry .. "\t" .. rz
"#,
            "10\t20\t103\t10\t20\t103",
        );
    }

    #[test]
    fn false_arg() {
        run(
            "false_arg",
            r#"
local a, b = 5, 1
local function f(p)
  a = 4
  return b + 3
end
b = f(false)
return b .. "\t" .. a
"#,
            "4\t4",
        );
    }

    /// The compiler calls into a temporary and moves the result, so this
    /// shape never hit the bug. It guards the shapes that stay correct.
    #[test]
    fn no_arg() {
        run(
            "no_arg",
            r#"
local a, b = 5, 1
local function f()
  a = 4
  return b + 3
end
b = f()
return b .. "\t" .. a
"#,
            "4\t4",
        );
    }

    #[test]
    fn three_writes() {
        run(
            "three_writes",
            r#"
local a, c, d, b = 5, 6, 7, 1
local function f(p)
  a = a + 1
  c = c + 2
  d = d + 3
  return b + 3
end
b = f(nil)
return b .. "\t" .. a .. "\t" .. c .. "\t" .. d
"#,
            "4\t6\t8\t10",
        );
    }

    #[test]
    fn open_single() {
        run(
            "open_single",
            r#"
local a, b, c = 5, 1, 9
local function g() return 100, 200 end
local function f(p)
  a = 4
  return b + 3, g()
end
b = f(nil)
return b .. "\t" .. a .. "\t" .. c
"#,
            "4\t4\t9",
        );
    }

    /// Like `no_arg`: a guard, since two results land in temporaries.
    #[test]
    fn open_multi() {
        run(
            "open_multi",
            r#"
local a, b, c = 5, 1, 9
local function g() return 100, 200 end
local function f(p)
  a = 4
  return b + 3, g()
end
b, c = f(nil)
return b .. "\t" .. c .. "\t" .. a
"#,
            "4\t100\t4",
        );
    }

    #[test]
    fn declaration() {
        run(
            "declaration",
            r#"
local a, other = 5, 1
local b = 0
local function getb() return b end
local function f(p)
  other = other + 10
  a = 4
  return a + other
end
b = f(nil)
local d = f(nil)
return b .. "\t" .. d .. "\t" .. a .. "\t" .. other .. "\t" .. getb()
"#,
            "15\t25\t4\t21\t15",
        );
    }

    #[test]
    fn five_thousand_calls_in_a_loop() {
        run(
            "loop",
            r#"
local a, b = 0, 1
local function f(p)
  a = a + 1
  return b + 1
end
for i = 1, 5000 do
  b = f(i)
  b = b - 1
end
return b .. "\t" .. a
"#,
            "1\t5000",
        );
    }

    #[test]
    fn coroutine() {
        run(
            "coroutine",
            r#"
local co = coroutine.wrap(function()
  local a, b = 5, 1
  local function f(p)
    a = 4
    return b + 3
  end
  b = f(nil)
  coroutine.yield(b)
  b = f(nil)
  return b + a
end)
local x = co()
local y = co()
return x .. "\t" .. y
"#,
            "4\t11",
        );
    }
}
