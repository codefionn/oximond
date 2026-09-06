//! Native calls the fast interpreter loop makes in place: pure numeric
//! natives on plain numbers, `tostring` of a number, and the general scalar
//! and multi-result natives. Each program checks itself with `assert`, so its
//! expectations hold for every profile and execution mode. The programs cover
//! the paths that skip work for the common case: hooks, errors, metatables on
//! numbers, result counts, captured registers and open results must behave as
//! they do through the general call path.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(source: &str) {
    for profile in PROFILES {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            if let Err(error) = lua.exec(source) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

#[test]
fn numeric_natives_return_the_same_numbers_in_every_call_shape() {
    run(r"
        local max, min, sqrt, abs, floor, fmod = math.max, math.min, math.sqrt, math.abs, math.floor, math.fmod
        local sum, expected = 0, 0
        for i = 1, 3000 do
            -- Call with two arguments and one result, through a global field
            -- and through a local.
            sum = sum + math.max(i, 10) + max(i, 10)
            sum = sum - min(i, 2.5)
            expected = expected + 2 * (i > 10 and i or 10) - (i < 2.5 and i or 2.5)
            -- One argument: the callee and the argument sit in other registers.
            local root = sqrt(i * i)
            assert(root == i, root)
            assert(abs(-i - 0.5) == i + 0.5)
            assert(floor(i + 0.25) == i)
            assert(fmod(i, 7) == i % 7)
        end
        assert(sum == expected, sum)
        -- Direct values.
        assert(max(3, 7.5) == 7.5 and min(3, 7.5) == 3)
        assert(max(-1, -2) == -1 and min(-1, -2) == -2)
        assert(sqrt(16) == 4)
        -- The result is a float whatever the argument types are.
        assert(math.floor(2.7) == 2)
    ");
}

#[test]
fn numeric_natives_with_unusual_result_counts_and_uses() {
    run(r##"
        local max = math.max
        for i = 1, 2000 do
            -- Two result slots: the second is nil.
            local a, b = max(i, 3)
            assert(a == (i > 3 and i or 3) and b == nil)
            -- No result slot: the call still runs and changes nothing visible.
            max(i, 3)
            -- Result used as an argument, as an operand and as a table key.
            local t = { [max(i, 1)] = true }
            assert(t[i])
            assert(1 + max(i, 1) == i + 1)
            assert(select("#", max(i, 1)) == 1)
            -- More or fewer arguments than the pure form takes.
            assert(max(i, 1, i + 1) == i + 1)
            assert(max(i) == i)
        end
    "##);
}

#[test]
fn numeric_natives_on_other_argument_types_keep_their_behavior() {
    run(r#"
        local max, sqrt = math.max, math.sqrt
        for i = 1, 500 do
            -- A numeric string converts.
            assert(max("10", 2) == 10)
            assert(sqrt("16") == 4)
            -- Anything else raises the stock error, naming the native.
            local ok, message = pcall(function() local r = max({}, 1) return r end)
            assert(not ok and message:find("bad argument #1 to 'max'", 1, true), message)
            local ok, message = pcall(function() local r = max(1, "x") return r end)
            assert(not ok and message:find("bad argument #2 to 'max'", 1, true), message)
            local ok, message = pcall(function() local r = sqrt() return r end)
            assert(not ok and message:find("bad argument #1 to 'sqrt'", 1, true), message)
            local ok, message = pcall(function() local r = sqrt(nil) return r end)
            assert(not ok and message:find("bad argument #1 to 'sqrt'", 1, true), message)
        end
        -- NaN and infinities go through the same function.
        local nan = 0 / 0
        local function same(a, b) return a == b or (a ~= a and b ~= b) end
        assert(same(max(nan, 1), select(2, pcall(max, nan, 1))))
        assert(same(max(1, nan), select(2, pcall(max, 1, nan))))
        assert(max(math.huge, 1) == math.huge and max(-math.huge, 1) == 1)
    "#);
}

#[test]
fn numeric_natives_are_reported_to_call_hooks() {
    run(r#"
        local seen = 0
        local function hook()
            local info = debug.getinfo(2, "f")
            if info and info.func == math.max then seen = seen + 1 end
        end
        local max = math.max
        for i = 1, 200 do max(i, 1) end
        assert(seen == 0)
        debug.sethook(hook, "c")
        for i = 1, 200 do max(i, 1) end
        debug.sethook()
        assert(seen == 200, seen)
        -- A hook set from inside the loop sees the calls after it.
        seen = 0
        for i = 1, 200 do
            if i == 101 then debug.sethook(hook, "c") end
            max(i, 1)
        end
        debug.sethook()
        assert(seen == 100 or seen == 99, seen)
    "#);
}

#[test]
fn numeric_natives_written_into_captured_registers() {
    run(r"
        local x
        local function read() return x end
        for i = 1, 1000 do
            x = math.max(i, 1)
            assert(read() == i)
            local y = math.sqrt(i * i)
            local function capture() return y end
            assert(capture() == i)
        end
    ");
}

#[test]
fn tostring_of_numbers_matches_concatenation() {
    run(r#"
        local tostring = tostring
        for i = 0, 700 do
            assert(tostring(i) == i .. "")
            assert(tostring(i + 0.5) == (i + 0.5) .. "")
            assert(tostring(-i) == -i .. "")
            assert(tostring(i * 1.0) == i * 1.0 .. "")
        end
        for _, value in ipairs { 1e15, 2 ^ 53, 2 ^ 63, -2 ^ 63, 1e100, 1 / 3, math.huge, -math.huge, 0 / 0, -0.0 } do
            assert(tostring(value) == value .. "", tostring(value))
        end
        assert(tostring(123456) == "123456")
    "#);
}

#[test]
fn tostring_of_a_number_with_a_metatable_calls_the_metamethod() {
    run(r#"
        local tostring = tostring
        for i = 1, 300 do assert(tostring(i) == i .. "") end
        debug.setmetatable(0, { __tostring = function(n) return "num" end })
        for i = 1, 300 do assert(tostring(i) == "num", tostring(i)) end
        for i = 1, 300 do assert(tostring(i + 0.5) == "num") end
        debug.setmetatable(0, nil)
        for i = 1, 300 do assert(tostring(i) == i .. "") end
    "#);
}

#[test]
fn tostring_result_counts_arguments_and_errors() {
    run(r##"
        for i = 1, 400 do
            -- Two result slots: the second is nil.
            local a, b = tostring(i)
            assert(a == i .. "" and b == nil)
            -- Extra arguments are ignored.
            assert(tostring(i, i) == i .. "")
            -- Open results feed other calls.
            assert(select("#", tostring(i)) == 1)
            assert(string.rep(tostring(i % 10), 3) == string.rep(i % 10 .. "", 3))
            -- Other types take the general path.
            assert(tostring(nil) == "nil" and tostring(true) == "true")
            assert(tostring("s") == "s")
            assert(tostring({}):find("^table: "))
        end
        local ok, message = pcall(function() local r = tostring() return r end)
        assert(not ok and message:find("bad argument #1 to 'tostring'", 1, true), message)
    "##);
}

#[test]
fn tostring_in_every_call_form_and_argument_type() {
    run(r##"
        local tostring = tostring
        local function tail(i) return tostring(i) end
        for i = 1, 400 do
            local whole = math.floor(i)
            assert(tostring(whole) == whole .. "")
            assert(tail(i) == i .. "")
            assert(tail(i + 0.25) == (i + 0.25) .. "")
            assert(select("#", tostring(i), tostring(i)) == 2)
            assert(({ tostring(i) })[1] == i .. "")
            assert(tostring(tostring(i)) == i .. "")
            assert(tail("x") == "x" and tail(nil) == "nil")
        end
    "##);
}

#[test]
fn tostring_in_captured_registers_and_under_hooks() {
    run(r#"
        local s
        local function read() return s end
        for i = 1, 500 do
            s = tostring(i)
            assert(read() == i .. "")
        end
        -- With a call hook set, the results are unchanged and the hook runs.
        local events = 0
        debug.sethook(function() events = events + 1 end, "c")
        for i = 1, 100 do assert(tostring(i) == i .. "") end
        debug.sethook()
        assert(events >= 100, events)
    "#);
}

#[test]
fn multi_result_and_scalar_natives_store_every_result() {
    run(r##"
        for i = 1, 500 do
            local s, e = string.find("haystack", "stack", 1, true)
            assert(s == 4 and e == 8)
            local only = string.find("haystack", "stack", 1, true)
            assert(only == 4)
            local a, b, c = string.find("haystack", "stack", 1, true)
            assert(a == 4 and b == 8 and c == nil)
            assert(string.find("haystack", "zzz", 1, true) == nil)
            assert(#table.concat({ "left", "right" }, ":") == 10)
            assert(select("#", string.find("haystack", "stack", 1, true)) == 2)
            local ok, message = pcall(function() local r = string.rep() return r end)
            assert(not ok and message:find("bad argument #1 to 'rep'", 1, true), message)
        end
    "##);
}

/// User-space instructions `perf stat` counts for `source` in the
/// interpreter, or `None` where `perf` cannot count them.
fn instructions(name: &str, source: &str) -> Option<u64> {
    // Tests run in parallel, so every program gets its own file.
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    // Only an optimized binary has instruction counts worth bounding.
    if cfg!(debug_assertions) {
        return None;
    }
    let directory = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let file = directory.join(format!(
        "native_calls_{}_{id}_{name}.lua",
        std::process::id()
    ));
    std::fs::write(&file, source).unwrap();
    let output = std::process::Command::new("perf")
        .args(["stat", "-x,", "-e", "instructions:u"])
        .arg(env!("CARGO_BIN_EXE_oximond"))
        .arg("--compat=lua51")
        .arg(&file)
        .env_remove("LUA_INIT")
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&file);
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stderr)
        .ok()?
        .lines()
        .find(|line| line.contains("instructions"))?
        .split(',')
        .next()?
        .parse()
        .ok()
}

/// Instructions per loop iteration that a statement costs over `baseline`.
/// Run `cargo test --release -p oximond-cli --test native_calls` to check.
fn per_iteration(statement: &str, baseline: &str) -> Option<f64> {
    const ITERATIONS: u64 = 200_000;
    let program = |body: &str| {
        format!(
            "local s, q, r = 0, 0, 0\nlocal f, ts = math.max, tostring\nfor i = 1, {ITERATIONS} do {body} end\n"
        )
    };
    let with = instructions("with", &program(statement))?;
    let without = instructions("without", &program(baseline))?;
    Some((with as f64 - without as f64) / ITERATIONS as f64)
}

/// The fixed cost of a pure numeric native called through a local, over the
/// three register loads its operands take, stays below 190 instructions
/// (it was 225 when every native took the general call path).
#[test]
fn a_numeric_native_call_costs_little_over_its_operand_loads() {
    let Some(cost) = per_iteration("q = f(i, 10)", "s = f q = i r = 10") else {
        eprintln!("skipped: needs a release build and a working perf");
        return;
    };
    assert!(cost < 190.0, "{cost} instructions per call");
}

/// `tostring` of a small number, over loading its operands, stays below 300
/// instructions (it was 394 when it took the frame-switching call path, 320 before it ran from the loop's native call).
#[test]
fn tostring_of_a_number_costs_little_over_its_operand_loads() {
    let Some(cost) = per_iteration("s = ts(5)", "s = ts s = 5") else {
        eprintln!("skipped: needs a release build and a working perf");
        return;
    };
    assert!(cost < 300.0, "{cost} instructions per call");
}
