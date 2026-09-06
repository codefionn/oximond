//! `pcall` and `xpcall` inside a hot loop: the trace runs the protected call
//! natively, as a protected frame, in every profile.
//!
//! A loop that calls `pcall(f, ...)` compiles with no abort, enters native
//! code, and gives the interpreter's results. An exit inside the protected
//! function rebuilds the interpreter's frames with the protected call
//! included, so an error raised after the exit is caught by the same `pcall`
//! or `xpcall`, with the interpreter's error value and handler result.

use oximond::runtime::Value;
use oximond::vm::TraceExitKind;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, LuaJitExtension};

const ALL: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// The profiles whose `xpcall` passes arguments to the protected function.
const XPCALL_ARGUMENTS: [CompatibilityProfile; 5] = [
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

/// Runs `source` in both modes of each profile and checks that the loop
/// traced and ran natively with no abort, and that the results agree.
fn assert_native(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {source} {stats:?}");
        assert!(stats.iterations > 1000, "{profile:?} {source} {stats:?}");
    }
}

/// Like [`assert_native`] for a loop that runs natively, leaves through a
/// guard once, and finishes in the interpreter. The results, errors caught by
/// the protected call included, are the interpreter's.
fn assert_exits(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        // A guard that fails after the function stored to a table leaves
        // with the store committed.
        assert!(
            stats.exits_of(TraceExitKind::GuardFail)
                + stats.exits_of(TraceExitKind::CommittedEffect)
                > 0,
            "{profile:?} {source} {stats:?}"
        );
    }
}

#[test]
fn pcall_of_a_lua_closure_returns_true_and_the_results() {
    assert_native(
        &ALL,
        "local function f(a, b) return a + b, a * b, a - b end local s = 0 \
         for i = 1, 100000 do \
           local ok, x, y, z = pcall(f, i, 2) \
           if ok then s = s + x + y + z end \
         end return s",
    );
}

#[test]
fn pcall_keeps_as_many_results_as_the_call_asks_for() {
    assert_native(
        &ALL,
        "local function f(a) return a, a + 1 end local s = 0 \
         for i = 1, 100000 do \
           local ok = pcall(f, i) \
           local ok2, x = pcall(f, i) \
           local ok3, p, q, r = pcall(f, i) \
           if ok and ok2 and ok3 and r == nil then s = s + x + p + q end \
         end return s",
    );
}

#[test]
fn pcall_of_a_function_without_results_gives_true() {
    assert_native(
        &ALL,
        "local t = {0} local function f(a) t[1] = t[1] + a end local n = 0 \
         for i = 1, 100000 do \
           local ok, x = pcall(f, i) \
           if ok and x == nil then n = n + 1 end \
         end return n, t[1]",
    );
}

#[test]
fn pcall_of_an_inlined_builtin() {
    assert_native(
        &ALL,
        "local s = 0 \
         for i = 1, 100000 do \
           local ok, x = pcall(math.floor, i / 3) \
           local ok2, y = pcall(math.abs, -i) \
           if ok and ok2 then s = s + x + y end \
         end return s",
    );
}

#[test]
fn pcall_of_a_builtin_that_calls_a_helper() {
    assert_native(
        &ALL,
        "local n = 0 \
         for i = 1, 100000 do \
           local ok, s = pcall(string.format, '%d:%s', i, 'x') \
           local ok2, t = pcall(tostring, i) \
           if ok and ok2 then n = n + #s + #t end \
         end return n",
    );
}

#[test]
fn pcall_of_select_assert_and_rawget() {
    assert_native(
        &ALL,
        "local t = { k = 3 } local n = 0 \
         for i = 1, 100000 do \
           local ok, c = pcall(select, '#', i, i) \
           local ok2, v = pcall(assert, i, 'm') \
           local ok3, w = pcall(rawget, t, 'k') \
           if ok and ok2 and ok3 then n = n + c + v + w end \
         end return n",
    );
}

#[test]
fn pcall_passes_a_vararg_forward() {
    assert_native(
        &ALL,
        "local function f(a, b) return a * 2, b end \
         local function w(...) return pcall(f, ...) end \
         local s = 0 \
         for i = 1, 100000 do \
           local ok, x, y = w(i, 3) \
           if ok then s = s + x + y end \
         end return s",
    );
}

#[test]
fn xpcall_passes_its_arguments() {
    assert_native(
        &XPCALL_ARGUMENTS,
        "local function f(a, b) return a + b, a - b end \
         local function h(e) return e end local s = 0 \
         for i = 1, 100000 do \
           local ok, x, y = xpcall(f, h, i, 2) \
           if ok then s = s + x + y end \
         end return s",
    );
}

#[test]
fn xpcall_of_lua_51_takes_no_arguments() {
    assert_native(
        &[CompatibilityProfile::Lua51],
        "local k = 0 local function f() k = k + 1 return k, 7 end \
         local function h(e) return e end local s = 0 \
         for i = 1, 100000 do \
           local ok, x, y = xpcall(f, h) \
           if ok then s = s + x + y end \
         end return s",
    );
}

#[test]
fn xpcall_of_lua_51_calls_a_native_without_arguments() {
    // `string.char()` gives '' where `string.char(65)` gives 'A'. Whether the
    // native is recorded or the trace aborts, the result is the interpreter's.
    let source = "local function h(e) return 'h' end local n = 0 \
         for i = 1, 100000 do \
           local ok, v = xpcall(string.char, h, 65) \
           if ok then n = n + #v end \
         end return n";
    let profile = CompatibilityProfile::Lua51;
    let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
    let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
    assert!(show(&expected).contains('0'), "{expected:?}");
    assert_eq!(show(&actual), show(&expected));
}

#[test]
fn nested_pcall_returns_through_both() {
    assert_native(
        &ALL,
        "local function g(a) return a + 1 end \
         local function f(a) return pcall(g, a) end \
         local s = 0 \
         for i = 1, 100000 do \
           local ok, ok2, x = pcall(f, i) \
           if ok and ok2 then s = s + x end \
         end return s",
    );
}

#[test]
fn pcall_in_a_return_gives_its_values_to_the_caller() {
    assert_native(
        &ALL,
        "local function f(a) return a, a + 1 end \
         local function w(a) return pcall(f, a) end \
         local s = 0 \
         for i = 1, 100000 do \
           local ok, x, y = w(i) \
           if ok then s = s + x + y end \
         end return s",
    );
}

#[test]
fn pcall_as_the_last_argument_gives_every_value() {
    assert_native(
        &ALL,
        "local function f(a) return a, a + 1 end \
         local function g(ok, x, y) if ok then return x + y end return 0 end \
         local s = 0 \
         for i = 1, 100000 do s = s + g(pcall(f, i)) end \
         return s",
    );
}

// Exits inside the protected function

/// A loop whose protected function stores to a table (so a guard in it
/// rebuilds the frames) and then does `body` once, at i == 90000. The errors
/// the loop catches are collected as text.
fn failing(body: &str) -> String {
    format!(
        "local t = {{0}} \
         local function f(i) \
           t[1] = i \
           if i == 90000 then {body} end \
           return i, i + 1 \
         end \
         local s, caught = 0, '' \
         for i = 1, 100000 do \
           local ok, a, b = pcall(f, i) \
           if ok then s = s + a + b \
           else caught = caught .. tostring(type(a) == 'string' and a or type(a)) .. '|' end \
         end return s, caught, t[1]"
    )
}

#[test]
fn pcall_catches_an_error_with_a_string() {
    assert_exits(&ALL, &failing("error('x')"));
}

#[test]
fn pcall_catches_an_error_with_a_table() {
    assert_exits(&ALL, &failing("error({})"));
}

#[test]
fn pcall_catches_an_error_at_level_two() {
    assert_exits(&ALL, &failing("error('x', 2)"));
}

#[test]
fn pcall_catches_an_error_at_level_zero() {
    assert_exits(&ALL, &failing("error('x', 0)"));
}

#[test]
fn pcall_catches_a_failed_assert() {
    assert_exits(&ALL, &failing("assert(false, 'm')"));
    assert_exits(&ALL, &failing("assert(false)"));
    assert_exits(&ALL, &failing("assert(nil, {})"));
}

#[test]
fn pcall_catches_a_runtime_error() {
    assert_exits(&ALL, &failing("local n = nil return n.x"));
    assert_exits(&ALL, &failing("return 1 + {}"));
}

#[test]
fn xpcall_catches_an_error_and_runs_its_handler() {
    for body in [
        "error('x')",
        "error({})",
        "error('x', 2)",
        "assert(false, 'm')",
        "local n = nil return n.x",
    ] {
        let source = format!(
            "local t = {{0}} \
             local function f(i, j) \
               t[1] = i \
               if i == 90000 then {body} end \
               return i, j \
             end \
             local function h(e) return type(e) == 'string' and 'h:' .. e or 'h:' .. type(e) end \
             local s, caught = 0, '' \
             for i = 1, 100000 do \
               local ok, a, b = xpcall(f, h, i, 2) \
               if ok then s = s + a + b else caught = caught .. a .. '|' end \
             end return s, caught, t[1]"
        );
        assert_exits(&XPCALL_ARGUMENTS, &source);
    }
}

#[test]
fn xpcall_of_lua_51_catches_an_error_and_runs_its_handler() {
    let source = "local t = {0} local n = 0 \
         local function f() \
           n = n + 1 t[1] = n \
           if n == 90000 then error('x') end \
           return n, 1 \
         end \
         local function h(e) return 'h:' .. tostring(e) end \
         local s, caught = 0, '' \
         for i = 1, 100000 do \
           local ok, a, b = xpcall(f, h) \
           if ok then s = s + a + b else caught = caught .. a .. '|' end \
         end return s, caught, t[1]";
    assert_exits(&[CompatibilityProfile::Lua51], source);
}

#[test]
fn a_guard_deep_in_nested_inlined_frames_resumes_inside_the_pcall() {
    // The guard fails in the innermost of three inlined frames, two of them
    // protected. The error that follows is caught by the inner pcall, and a
    // later one by the outer.
    for failure in ["error('inner')", "local n = nil return n.x"] {
        let source = format!(
            "local t = {{0}} \
             local function c(i) \
               t[1] = i \
               if i == 90000 then {failure} end \
               if i == 95000 then error('late') end \
               return i \
             end \
             local function b(i) local ok, v = pcall(c, i) if ok then return v end return -1 end \
             local function a(i) return pcall(b, i) end \
             local s, bad = 0, 0 \
             for i = 1, 100000 do \
               local ok, ok2, v = pcall(a, i) \
               if ok and ok2 and v ~= -1 then s = s + v else bad = bad + 1 end \
             end return s, bad, t[1]"
        );
        assert_exits(&ALL, &source);
    }
}

#[test]
fn a_guard_in_a_callee_of_the_protected_function_keeps_the_pcall_frame() {
    // After the exit the interpreter finishes `f` and calls `g`, which raises.
    // The pcall that is still on the interpreter's stack catches it.
    assert_exits(
        &ALL,
        "local t = {0} \
         local function g(i) if i == 90000 then error('g' .. i) end return i end \
         local function f(i) \
           t[1] = i \
           local x = g(i) \
           if i == 90000 then x = nil end \
           return x + 1 \
         end \
         local s, caught = 0, '' \
         for i = 1, 100000 do \
           local ok, v = pcall(f, i) \
           if ok then s = s + v else caught = caught .. v .. '|' end \
         end return s, caught",
    );
}

#[test]
fn closures_made_after_an_exit_keep_their_upvalues_when_the_pcall_catches() {
    // The exit leaves `f` in the interpreter, which makes the closure over
    // the rebuilt frame's local and then raises. The pcall that catches the
    // error closes the local, and the closure keeps its value.
    assert_exits(
        &ALL,
        "local t = {0} local keep \
         local function f(i) \
           t[1] = i \
           if i == 90000 then \
             local n = i \
             local function inc() n = n + 1 return n end \
             keep = inc \
             n = n + 10 \
             error('boom') \
           end \
           return i \
         end \
         local s, caught = 0, '' \
         for i = 1, 100000 do \
           local ok, v = pcall(f, i) \
           if ok then s = s + v else caught = caught .. v .. '|' end \
         end \
         return s, caught, keep(), keep()",
    );
}

#[test]
fn closures_made_after_an_exit_keep_their_upvalues_when_the_function_returns() {
    assert_exits(
        &ALL,
        "local t = {0} local keep \
         local function f(i) \
           t[1] = i \
           local n = i \
           if i == 90000 then \
             local function inc() n = n + 1 return n end \
             keep = inc \
             n = n + 10 \
           end \
           return n \
         end \
         local s = 0 \
         for i = 1, 100000 do \
           local ok, v = pcall(f, i) \
           s = s + v \
         end \
         return s, keep(), keep()",
    );
}

#[test]
fn a_different_function_under_pcall_runs_in_the_interpreter() {
    assert_exits(
        &ALL,
        "local function f(i) return i end \
         local function g(i) return i * 2 end \
         local s = 0 \
         for i = 1, 100000 do \
           local fn = f if i > 90000 then fn = g end \
           local ok, v = pcall(fn, i) \
           s = s + v \
         end return s",
    );
}

#[test]
fn xpcall_with_debug_traceback_runs_natively() {
    assert_native(
        &XPCALL_ARGUMENTS,
        "local function f(a, b) return a + b, a - b end \
         local s = 0 \
         for i = 1, 100000 do \
           local ok, x, y = xpcall(f, debug.traceback, i, 2) \
           if ok then s = s + x + y end \
         end return s",
    );
    assert_native(
        &[CompatibilityProfile::Lua51],
        "local k = 0 local function f() k = k + 1 return k, 7 end \
         local s = 0 \
         for i = 1, 100000 do \
           local ok, x, y = xpcall(f, debug.traceback) \
           if ok then s = s + x + y end \
         end return s",
    );
}

#[test]
fn xpcall_with_debug_traceback_reports_the_interpreters_traceback() {
    let source = "local t = {0} \
         local function f(i, j) \
           t[1] = i \
           if i == 90000 then error('boom') end \
           return i, j \
         end \
         local s, caught = 0, '' \
         for i = 1, 100000 do \
           local ok, a, b = xpcall(f, debug.traceback, i, 2) \
           if ok then s = s + a + b else caught = caught .. a .. '|' end \
         end return s, caught, t[1]";
    assert_exits(&XPCALL_ARGUMENTS, source);
    let source = "local t = {0} local n = 0 \
         local function f() \
           n = n + 1 t[1] = n \
           if n == 90000 then error('boom') end \
           return n, 1 \
         end \
         local s, caught = 0, '' \
         for i = 1, 100000 do \
           local ok, a, b = xpcall(f, debug.traceback) \
           if ok then s = s + a + b else caught = caught .. a .. '|' end \
         end return s, caught, t[1]";
    assert_exits(&[CompatibilityProfile::Lua51], source);
}

// Errors the trace catches

/// Runs `source` in both modes of each profile and checks that the loop
/// traced, that the error path compiled into a side trace, that it ran
/// natively and left the trace only a few times, and that the results agree.
fn assert_caught(profiles: &[CompatibilityProfile], source: &str) {
    assert_caught_within(profiles, source, 100);
}

/// [`assert_caught`] where the run may leave the trace up to `max_exits`
/// times.
fn assert_caught_within(profiles: &[CompatibilityProfile], source: &str, max_exits: u64) {
    assert_caught_with(profiles, source, max_exits, true);
}

/// A loop that catches an error value the trace made (a table), which no
/// side trace can start from: the error path stays in the root trace, so the
/// run needs no side install and no exit at all.
fn assert_caught_inline(profiles: &[CompatibilityProfile], source: &str) {
    assert_caught_with(profiles, source, 10, false);
}

fn assert_caught_with(profiles: &[CompatibilityProfile], source: &str, max_exits: u64, side: bool) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(
            !side || stats.side_installs > 0,
            "{profile:?} {source} {stats:?}"
        );
        assert!(stats.iterations > 50_000, "{profile:?} {source} {stats:?}");
        // The exits that grow the side trace, and a few more.
        let exits: u64 = stats.exits.iter().sum();
        assert!(exits < max_exits, "{profile:?} {source} {exits} {stats:?}");
        // The error value stays virtual: the run makes no table.
        assert!(
            side || stats.table_new_helper_calls == 0,
            "{profile:?} {source} {stats:?}"
        );
    }
}

/// A loop that calls `pcall(entry, i)` 100000 times, where `f` is `body` (a
/// function body) and `g` calls it. Every call fails. `read` runs for each
/// caught error `e` and keeps the loop from dropping it. `bad` counts the
/// values that differ from the one the interpreter gave the cold call before
/// the loop.
fn erroring(entry: &str, body: &str, read: &str) -> String {
    format!(
        "local function f(i)\n\
           {body}\n\
         end\n\
         local function g(i)\n\
           f(i)\n\
         end\n\
         local _, ref = pcall({entry}, 0)\n\
         local n, bad = 0, 0\n\
         for i = 1, 100000 do\n\
           local ok, e = pcall({entry}, i)\n\
           if ok then bad = bad + 1000000 else {read} end\n\
         end\n\
         return n, bad, type(ref), type(ref) == 'table' and ref.code or tostring(ref)"
    )
}

const READ_TEXT: &str = "n = n + #e if not rawequal(e, ref) then bad = bad + 1 end";
const READ_SAME: &str = "if rawequal(e, ref) then n = n + 1 else bad = bad + 1 end";

#[test]
fn pcall_catches_error_with_a_string_natively() {
    assert_caught(&ALL, &erroring("f", "error('boom')", READ_TEXT));
}

#[test]
fn pcall_catches_error_at_level_two_natively() {
    assert_caught(&ALL, &erroring("g", "error('boom', 2)", READ_TEXT));
    assert_caught(&ALL, &erroring("f", "error('boom', 2)", READ_TEXT));
}

#[test]
fn pcall_catches_error_at_level_zero_natively() {
    assert_caught(&ALL, &erroring("f", "error('boom', 0)", READ_TEXT));
}

/// A loop over `pcall(f, i)` where `f` is `body` and fails with a table
/// error value that the loop only passes on: it stays in the variable `e`
/// that outlives the loop, and the script checks it afterwards.
fn raising_table(body: &str, check: &str) -> String {
    format!(
        "local E = {{code = 1}}\n\
         local function f(i)\n\
           {body}\n\
         end\n\
         local function g(i)\n\
           f(i)\n\
         end\n\
         local n, bad, e = 0, 0\n\
         for i = 1, 100000 do\n\
           local ok\n\
           ok, e = pcall(g, i)\n\
           if ok then bad = bad + 1000000 else n = n + 1 end\n\
         end\n\
         return n, bad, {check}"
    )
}

#[test]
fn pcall_catches_error_with_a_table_natively() {
    // The table is made outside the loop; a constructor inside the protected
    // function is covered below.
    assert_caught(&ALL, &raising_table("error(E)", "rawequal(e, E), e.code"));
    assert_caught(
        &ALL,
        &raising_table("error(E, 2)", "rawequal(e, E), e.code"),
    );
    assert_caught(
        &ALL,
        &raising_table("error(E, 0)", "rawequal(e, E), e.code"),
    );
}

#[test]
fn pcall_catches_error_with_a_new_table_natively() {
    // The constructor runs inside the protected function. The table is
    // virtual in the trace and the caller reads its field without a table.
    let read = "n = n + e.code";
    assert_caught_inline(&ALL, &erroring("f", "error({code = 1})", read));
    assert_caught_inline(&ALL, &erroring("g", "error({code = 1}, 2)", read));
    assert_caught_inline(&ALL, &erroring("f", "error({code = 1}, 0)", read));
}

#[test]
fn pcall_catches_a_table_with_values_on_some_iterations() {
    // The table keeps values of the iteration and outlives it in `keep`, so
    // an exit has to build it.
    let source = "local function f(i)\n\
           if i % 7 == 0 then error({code = i, name = 'x'}) end\n\
           return i\n\
         end\n\
         local function g(i) return f(i) end\n\
         local keep\n\
         local n, m = 0, 0\n\
         for i = 1, 100000 do\n\
           local ok, e = pcall(g, i)\n\
           if ok then n = n + e else m = m + e.code keep = e end\n\
         end\n\
         return n, m, keep.code, keep.name";
    assert_same(&ALL, source);
}

#[test]
fn pcall_catches_error_with_nil_natively() {
    assert_caught(&ALL, &erroring("f", "error(nil)", READ_SAME));
    assert_caught(&ALL, &erroring("f", "error()", READ_SAME));
}

#[test]
fn pcall_catches_error_with_a_number_natively() {
    // A number passes through at level 0. At a level above that, the
    // profiles that turn it into text with a position fall back.
    assert_caught(&ALL, &erroring("f", "error(7, 0)", READ_SAME));
    assert_caught(&ALL, &erroring("f", "error(true, 0)", READ_SAME));
    assert_caught(
        &[
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        &erroring("f", "error(7)", READ_SAME),
    );
    assert_same(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::LuaJit21,
        ],
        &erroring("f", "error(7)", READ_SAME),
    );
}

/// The profiles whose `assert` raises a message that is no string as it is.
const RAISES_VALUES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// The profiles whose `assert` makes an argument error of a message that is
/// no string or number.
const ASSERT_STRING_ONLY: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::Lua52];

#[test]
fn pcall_catches_a_failed_assert_natively() {
    assert_caught(&ALL, &erroring("f", "assert(false)", READ_TEXT));
    assert_caught(&ALL, &erroring("g", "assert(nil, 'msg')", READ_TEXT));
    let table = erroring("f", "assert(false, E)", READ_SAME)
        .replace("local function f", "local E = {}\nlocal function f");
    assert_caught(&RAISES_VALUES, &table);
    assert_same(&ASSERT_STRING_ONLY, &table);
    assert_caught(&ALL, &erroring("f", "assert(i < 0, 'low')", READ_TEXT));
}

#[test]
fn pcall_catches_a_failed_assert_with_a_new_table_natively() {
    // The table is the error value, so the caller reads a field of it.
    let source = erroring("f", "assert(false, {code = 1})", "n = n + e.code");
    assert_caught_inline(&RAISES_VALUES, &source);
    // These profiles make an argument error of it, a string.
    assert_same(
        &ASSERT_STRING_ONLY,
        &erroring("f", "assert(false, {code = 1})", READ_TEXT),
    );
}

/// A loop over `pcall(f, i)` where `f` fails with a message that depends on
/// `i % 2`: `body` is its body.
fn computed(body: &str) -> String {
    format!(
        "local function f(i)\n\
           {body}\n\
         end\n\
         local _, r0 = pcall(f, 2)\n\
         local _, r1 = pcall(f, 1)\n\
         local n, bad = 0, 0\n\
         for i = 1, 100000 do\n\
           local ok, e = pcall(f, i)\n\
           if ok then bad = bad + 1000000\n\
           elseif rawequal(e, r0) or rawequal(e, r1) then n = n + 1\n\
           else bad = bad + 1 end\n\
         end\n\
         return n, bad, r0, r1"
    )
}

#[test]
fn pcall_catches_error_with_a_computed_message_natively() {
    assert_caught_within(&ALL, &computed("error('boom ' .. i % 2 .. 'x')"), 1000);
    assert_caught_within(&ALL, &computed("error('boom ' .. i % 2 .. 'x', 2)"), 1000);
    assert_caught_within(&ALL, &computed("assert(i < 0, 'low ' .. i % 2)"), 1000);
}

#[test]
fn a_computed_message_that_turns_into_no_utf8_matches_the_interpreter() {
    // The message is text while the trace records and bytes that are no
    // UTF-8 later. The interpreter turns them into replacement characters
    // when it prefixes the message, so the trace must leave for it.
    for body in ["error(t[1] .. 'x')", "assert(false, t[1] .. 'x')"] {
        let source = format!(
            "local strs = {{'ok'}}\n\
             local function f(t) {body} end\n\
             local e\n\
             for i = 1, 200000 do\n\
               if i == 150000 then strs[1] = '\\200' end\n\
               local _\n\
               _, e = pcall(f, strs)\n\
             end\n\
             return #e, e:byte(-2)"
        );
        assert_same(&ALL, &source);
    }
}

/// A loop over the protected call `hot` (and the cold call `cold` before it,
/// which gives the value the loop compares with), where `body` raises in
/// `f` and `h` is the message handler.
fn handled(cold: &str, hot: &str, body: &str, handler: &str) -> String {
    format!(
        "local E = {{code = 1}}\n\
         local function f(i)\n\
           {body}\n\
         end\n\
         local function h(e)\n\
           {handler}\n\
         end\n\
         local _, ref = {cold}\n\
         local n, bad = 0, 0\n\
         for i = 1, 100000 do\n\
           local ok, e = {hot}\n\
           if ok then bad = bad + 1000000\n\
           elseif rawequal(e, ref) then n = n + 1\n\
           else bad = bad + 1 end\n\
         end\n\
         return n, bad, ref"
    )
}

#[test]
fn xpcall_runs_a_lua_handler_on_the_caught_path_natively() {
    for body in [
        "error('boom')",
        "error('boom', 0)",
        "error('boom', 2)",
        "assert(false)",
        "assert(nil, 'msg')",
        "error(nil)",
        "error(E, 0)",
    ] {
        let handler = "if type(e) == 'string' then return 'h:' .. e end return type(e)";
        assert_caught(
            &XPCALL_ARGUMENTS,
            &handled("xpcall(f, h, 0)", "xpcall(f, h, i)", body, handler),
        );
        assert_caught(
            &[CompatibilityProfile::Lua51],
            &handled("xpcall(f, h)", "xpcall(f, h)", body, handler),
        );
    }
}

#[test]
fn xpcall_handler_results_follow_the_interpreter() {
    // Only the first value of the handler counts, and a handler without a
    // value gives nil (a message in Lua 5.5).
    for handler in [
        "return 'a', 'b'",
        "return 'one'",
        "return e",
        "return 7",
        "local x = 1",
    ] {
        let source = handled(
            "xpcall(f, h, 0)",
            "xpcall(f, h, i)",
            "error('boom', 0)",
            handler,
        );
        for profile in XPCALL_ARGUMENTS {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, _) = outcome(profile, ExecutionMode::Tracing, &source);
            assert!(expected.is_ok(), "{profile:?} {handler}: {expected:?}");
            assert_eq!(show(&actual), show(&expected), "{profile:?} {handler}");
        }
    }
}

#[test]
fn xpcall_with_debug_traceback_exits_before_the_error() {
    // The traceback needs the stack the error was raised on, so the trace
    // does not catch the error: the interpreter raises it. The traceback is
    // the interpreter's.
    let source = "local function f(i)\n\
           error('boom')\n\
         end\n\
         local n, e = 0\n\
         for i = 1, 50000 do\n\
           local ok\n\
           ok, e = xpcall(f, debug.traceback, i)\n\
           if not ok then n = n + 1 end\n\
         end\n\
         return n, e";
    for profile in XPCALL_ARGUMENTS {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
    }
}

#[test]
fn errors_on_some_iterations_keep_both_paths_native() {
    // The callee names `error` and `assert` as globals, and with `prefix` as
    // a local of the chunk, so its side recording reads them through the
    // callee's environment or a cell.
    let source = |call: &str, extra: &str, raise: &str, prefix: &str| {
        format!(
            "{prefix}\n\
             local function f(i)\n\
               if i % 7 == 0 then {raise} end\n\
               return i\n\
             end\n\
             local function h(e) return e end\n\
             local n, bad = 0, 0\n\
             for i = 1, 100000 do\n\
               local ok, v = {call}(f, {extra}i)\n\
               if ok then n = n + v else bad = bad + 1 end\n\
             end\n\
             return n, bad"
        )
    };
    for prefix in ["", "local error, assert = error, assert"] {
        for raise in [
            "error('boom')",
            "error('boom', 0)",
            "error('boom', 2)",
            "assert(false, 'boom')",
        ] {
            assert_caught_within(&ALL, &source("pcall", "", raise, prefix), 1000);
            assert_caught_within(
                &XPCALL_ARGUMENTS,
                &source("xpcall", "h, ", raise, prefix),
                1000,
            );
        }
    }
}

#[test]
fn two_error_sites_in_one_callee_keep_both_paths_native() {
    for prefix in ["", "local error = error"] {
        let source = format!(
            "{prefix}\n\
             local function f(i)\n\
               if i % 7 == 0 then error('a', 0) end\n\
               if i % 13 == 0 then error('b', 0) end\n\
               return i\n\
             end\n\
             local n, bad = 0, 0\n\
             for i = 1, 100000 do\n\
               local ok, v = pcall(f, i)\n\
               if ok then n = n + v else bad = bad + 1 end\n\
             end\n\
             return n, bad"
        );
        assert_caught_within(&ALL, &source, 1000);
    }
}

// Errors the trace leaves to the interpreter

/// Checks that tracing gives the interpreter's results in each profile.
fn assert_same(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
    }
}

const CLOSING: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55];

#[test]
fn a_close_variable_in_the_erroring_frame_closes_as_in_the_interpreter() {
    assert_same(
        &CLOSING,
        "local closed, n = 0, 0\n\
         local mt = {__close = function(_, e) closed = closed + 1 end}\n\
         local function f(i)\n\
           local x <close> = setmetatable({}, mt)\n\
           if i % 3 ~= 0 then error('boom') end\n\
           return i\n\
         end\n\
         for i = 1, 50000 do\n\
           local ok, e = pcall(f, i)\n\
           if not ok then n = n + 1 end\n\
         end\n\
         return closed, n",
    );
    // A to-be-closed variable of a frame the error discards, with a handler.
    assert_same(
        &CLOSING,
        "local log = {}\n\
         local mt = {__close = function(_, e) log[#log + 1] = tostring(e) end}\n\
         local function f(i)\n\
           local x <close> = setmetatable({}, mt)\n\
           error('boom' .. i % 2, 0)\n\
         end\n\
         local n = 0\n\
         for i = 1, 20000 do\n\
           local ok, e = xpcall(f, function(m) return m .. '!' end, i)\n\
           if not ok then n = n + #e end\n\
         end\n\
         return #log, n, log[1], log[#log]",
    );
}

#[test]
fn an_error_with_a_debug_hook_runs_the_hook_as_in_the_interpreter() {
    assert_same(
        &ALL,
        "local count, n = 0, 0\n\
         local function f(i)\n\
           error('boom')\n\
         end\n\
         debug.sethook(function() count = count + 1 end, 'l')\n\
         for i = 1, 30000 do\n\
           local ok, e = pcall(f, i)\n\
           if not ok then n = n + 1 end\n\
         end\n\
         debug.sethook()\n\
         return n, count",
    );
    assert_same(
        &ALL,
        "local count, n = 0, 0\n\
         local function f(i)\n\
           error('boom')\n\
         end\n\
         for i = 1, 30000 do\n\
           if i == 20000 then debug.sethook(function() count = count + 1 end, 'c') end\n\
           local ok, e = pcall(f, i)\n\
           if not ok then n = n + 1 end\n\
         end\n\
         debug.sethook()\n\
         return n, count",
    );
}

#[test]
fn a_closure_over_a_local_of_a_discarded_frame_keeps_its_value() {
    assert_same(
        &ALL,
        "local keep = {}\n\
         local function f(i)\n\
           local x = i * 2\n\
           keep[i % 3 + 1] = function() x = x + 1 return x end\n\
           error('boom')\n\
         end\n\
         local n = 0\n\
         for i = 1, 30000 do\n\
           local ok, e = pcall(f, i)\n\
           if not ok then n = n + 1 end\n\
         end\n\
         return n, keep[1](), keep[1](), keep[2](), keep[3]()",
    );
}

#[test]
fn errors_inside_a_metamethod_or_a_library_call_stay_with_the_interpreter() {
    assert_same(
        &ALL,
        "local t = setmetatable({}, {__index = function(t, k) error('idx' .. k) end})\n\
         local function f(i) return t[i % 2] end\n\
         local n, last = 0, ''\n\
         for i = 1, 30000 do\n\
           local ok, e = pcall(f, i)\n\
           if not ok then n = n + 1 last = e end\n\
         end\n\
         return n, last",
    );
    assert_same(
        &ALL,
        "local function f(i) return string.rep() end\n\
         local n, last = 0, ''\n\
         for i = 1, 30000 do\n\
           local ok, e = pcall(f, i)\n\
           if not ok then n = n + 1 last = e end\n\
         end\n\
         return n, last",
    );
}

#[test]
fn nested_protected_calls_catch_in_the_innermost() {
    assert_same(
        &ALL,
        "local function g(i)\n\
           if i % 3 == 0 then error('inner') end\n\
           return i\n\
         end\n\
         local function f(i)\n\
           local ok, v = pcall(g, i)\n\
           if not ok then v = 0 end\n\
           if i % 5 == 0 then error('outer') end\n\
           return v\n\
         end\n\
         local n, bad, s = 0, 0, 0\n\
         for i = 1, 60000 do\n\
           local ok, v = pcall(f, i)\n\
           if ok then s = s + v else n = n + 1 end\n\
         end\n\
         return n, s",
    );
    // The inner call is made by an inlined function: the trace carries on
    // after the caught error instead of leaving, so the loop stays native.
    let source = "local function g(i)\n\
           error('inner')\n\
         end\n\
         local function f(i)\n\
           local ok, v = pcall(g, i)\n\
           if ok then return 1 end\n\
           return 2\n\
         end\n\
         local n = 0\n\
         for i = 1, 100000 do\n\
           local ok, v = pcall(f, i)\n\
           if ok then n = n + v end\n\
         end\n\
         return n";
    assert_caught_inline(&ALL, source);
}

#[test]
fn a_pcall_made_by_an_inlined_function_keeps_its_error_path_native() {
    // The error value is a string the function reads, and the error happens
    // on some iterations only.
    let source = "local function g(i)\n\
           if i % 4 == 0 then error('bad ' .. i) end\n\
           return i\n\
         end\n\
         local function f(i)\n\
           local ok, v = pcall(g, i)\n\
           if ok then return v end\n\
           return #v\n\
         end\n\
         local n = 0\n\
         for i = 1, 100000 do\n\
           n = n + f(i)\n\
         end\n\
         return n";
    assert_same(&ALL, source);
    let constant = "local function g(i)\n\
           error('inner', 0)\n\
         end\n\
         local function f(i)\n\
           local ok, v = pcall(g, i)\n\
           return #v\n\
         end\n\
         local n = 0\n\
         for i = 1, 100000 do\n\
           n = n + f(i)\n\
         end\n\
         return n";
    assert_caught_inline(&ALL, constant);
}

#[test]
fn an_error_through_many_frames_is_caught_natively() {
    let wrapper = "local function g(i)\nf(i)\nend";
    let source = erroring("g", "error('deep')", READ_TEXT);
    assert!(source.contains(wrapper), "{source}");
    // Two wrappers between the loop and `error`: `g` calls `g0` calls `f`.
    let deep = source.replace(
        wrapper,
        "local function g0(i)\nf(i)\nend\nlocal function g(i)\ng0(i)\nend",
    );
    assert!(deep.contains("g0(i)"), "{deep}");
    assert_caught(&ALL, &deep);
}

// Side traces that link back with a constant error string

/// A loop of 300000 `pcall(f, i)` where `f` is `body`; `e` outlives the call
/// and `n` counts the lengths of the messages.
fn linking(body: &str) -> String {
    format!(
        "local function f(i) {body} end\n\
         local n, e = 0\n\
         for i = 1, 300000 do\n\
           local ok, x = pcall(f, i)\n\
           e = x\n\
           if ok then n = n + 1 else n = n + #e end\n\
         end\n\
         return n, e"
    )
}

#[test]
fn a_constant_error_string_links_back_when_the_register_is_a_loop_parameter() {
    assert_caught_within(&ALL, &linking("error('boom', 0)"), 300);
    assert_caught_within(
        &ALL,
        "local function f(i) if i % 7 == 0 then error('boom') end return i end\n\
         local n = 0\n\
         for i = 1, 300000 do\n\
           local ok, e\n\
           ok, e = pcall(f, i)\n\
           if ok then n = n + e else n = n + #e end\n\
         end\n\
         return n",
        300,
    );
}

#[test]
fn two_wrappers_of_one_error_site_share_a_side_trace_tree() {
    // The second wrapper's side pins a closure the root does not: the tree
    // takes the pin along with the cell alias the side found.
    assert_caught_within(
        &ALL,
        "local function f(i) error('boom', 2) end\n\
         local function g(i) f(i) end\n\
         local function g2(i) f(i) end\n\
         local n = 0\n\
         for i = 1, 100000 do\n\
           if i % 3 == 0 then local ok, e = pcall(g2, i) n = n + #e\n\
           else local ok, e = pcall(g, i) n = n + #e end\n\
         end\n\
         return n",
        1000,
    );
}

// Tail calls of error and assert

#[test]
fn a_tail_called_error_or_assert_is_caught_natively() {
    assert_caught(&ALL, &erroring("f", "return error('x')", READ_TEXT));
    assert_caught(&ALL, &erroring("f", "return error('x', 0)", READ_TEXT));
    assert_caught(&ALL, &erroring("g", "return error('x', 2)", READ_TEXT));
    assert_caught(&ALL, &erroring("f", "return assert(false, 'm')", READ_TEXT));
    assert_caught(&ALL, &erroring("f", "return assert(false)", READ_TEXT));
    assert_caught(&ALL, &erroring("g", "return assert(nil, 'm')", READ_TEXT));
}

// The message of a failed assert

/// The values `assert` raises for each form, as `type:text` (a table is its
/// type alone), in a loop that the trace compiles and in the interpreter.
fn asserting(call: &str) -> String {
    format!(
        "local function f(i) {call} end\n\
         local function show(v)\n\
           if type(v) == 'string' then return 'string:' .. v:gsub('^[^ ]*:%d+: ', 'pos ')\n\
           elseif type(v) == 'table' then return 'table'\n\
           else return type(v) .. ':' .. tostring(v) end\n\
         end\n\
         local seen, n = {{}}, 0\n\
         for i = 1, 100000 do\n\
           local ok, e = pcall(f, i)\n\
           if not ok then n = n + 1 seen[show(e)] = true end\n\
         end\n\
         local keys = {{}}\n\
         for k in pairs(seen) do keys[#keys + 1] = k end\n\
         table.sort(keys)\n\
         return n, table.concat(keys, '|')"
    )
}

#[test]
fn assert_raises_the_message_as_the_profile_does() {
    for call in [
        "assert(false, {1})",
        "assert(false, 42)",
        "assert(false, nil)",
        "assert(nil)",
        "assert(false, 'm')",
        "assert(false, true)",
        "return assert(false, 42)",
        "return assert(false, nil)",
        "return assert(false, 'm')",
    ] {
        assert_same(&ALL, &asserting(call));
    }
}

#[test]
fn a_caught_assert_with_any_message_is_native_where_the_trace_can_build_it() {
    for call in ["assert(false, 'm')", "assert(nil)"] {
        assert_caught(&ALL, &erroring("f", call, READ_TEXT));
    }
    let source = erroring("f", "assert(false, true)", READ_SAME);
    assert_caught(&RAISES_VALUES, &source);
    assert_same(&ASSERT_STRING_ONLY, &source);
    let passed = [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ];
    for call in ["assert(false, 42)", "assert(false, nil)"] {
        assert_caught(&passed, &erroring("f", call, READ_SAME));
    }
    // LuaJIT passes nil on and turns a number into text, which stays with the
    // interpreter; Lua 5.1 and 5.2 make an argument error of a table.
    assert_caught(
        &[CompatibilityProfile::LuaJit21],
        &erroring("f", "assert(false, nil)", READ_SAME),
    );
    assert_same(
        &[CompatibilityProfile::LuaJit21],
        &erroring("f", "assert(false, 42)", READ_SAME),
    );
    assert_caught(
        &ASSERT_STRING_ONLY,
        &erroring("f", "assert(false, nil)", READ_TEXT),
    );
}

// Stock interpreters

fn stock(binary: &str, profile: CompatibilityProfile) {
    use std::process::Command;
    let script = "local function show(...)\n\
           local ok, e = pcall(assert, ...)\n\
           io.write(tostring(ok), ' ', type(e), ' ', type(e) ~= 'table' and tostring(e) or 'T', '\\n')\n\
         end\n\
         local function inner(...)\n\
           local ok, e = pcall(function(...) assert(...) end, ...)\n\
           io.write(tostring(ok), ' ', type(e), ' ', type(e) ~= 'table' and (tostring(e):gsub('^[^ ]*:%d+: ', 'pos ')) or 'T', '\\n')\n\
         end\n\
         for _, f in ipairs{show, inner} do\n\
           f(false, {1}) f(false, 42) f(false, nil) f(nil) f(false, 'm') f(1, 2, 3)\n\
         end";
    let Ok(path) = std::env::temp_dir()
        .join(format!(
            "oximond-assert-{}-{binary}.lua",
            std::process::id()
        ))
        .into_os_string()
        .into_string()
    else {
        return;
    };
    std::fs::write(&path, script).unwrap();
    let output = Command::new(format!("/usr/bin/{binary}"))
        .arg(&path)
        .output();
    std::fs::remove_file(&path).ok();
    let Ok(output) = output else {
        eprintln!("no {binary} here");
        return;
    };
    let expected = String::from_utf8_lossy(&output.stdout).into_owned();
    let mut lua = Lua::builder().profile(profile).build().unwrap();
    let source = format!(
        "local out = {{}}\nlocal io = {{write = function(...) for _, v in ipairs{{...}} do out[#out + 1] = v end end}}\n{script}\nreturn table.concat(out)"
    );
    let actual = match lua.exec(&source).unwrap().first() {
        Some(Value::String(text)) => text.to_lossy().into_owned(),
        other => panic!("{other:?}"),
    };
    assert_eq!(actual, expected, "{profile:?} against {binary}");
}

#[test]
fn the_interpreters_assert_matches_the_stock_binaries() {
    stock("lua5.1", CompatibilityProfile::Lua51);
    stock("lua5.4", CompatibilityProfile::Lua54);
    stock("lua5.5", CompatibilityProfile::Lua55);
    stock("luajit", CompatibilityProfile::LuaJit21);
}
