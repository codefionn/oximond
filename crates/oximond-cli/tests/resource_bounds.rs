//! Programs that stock Lua rejects early with a size or depth error must not
//! allocate gigabytes first. The upstream conformance suites probe these
//! limits and ran the machine out of memory before they were enforced.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn run(source: &str) -> Vec<Value> {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.exec(source).unwrap_or_else(|error| panic!("{error}"))
}

#[test]
fn unpack_rejects_more_results_than_the_value_stack_holds() {
    let source = r#"
        local ok, message = pcall(table.unpack, {}, 1, (1 << 31) - 1)
        return ok, string.find(message, "too many results", 1, true) ~= nil
    "#;
    assert_eq!(
        run(source),
        vec![Value::Boolean(false), Value::Boolean(true)]
    );
}

#[test]
fn string_rep_bounds_the_separator_even_for_an_empty_string() {
    let source = r#"
        local ok, message = pcall(string.rep, "", 1 << 30, "aa")
        return ok, string.find(message, "too large", 1, true) ~= nil
    "#;
    assert_eq!(
        run(source),
        vec![Value::Boolean(false), Value::Boolean(true)]
    );
}

#[test]
fn recursive_pcall_stops_at_the_call_depth_limit() {
    // Each level adds a position prefix to the propagated message, so an
    // unbounded depth grows memory quadratically.
    let source = r"
        local depth = 0
        local function loop()
            depth = depth + 1
            assert(pcall(loop))
        end
        local ok = pcall(loop)
        return ok, depth < 10000
    ";
    assert_eq!(
        run(source),
        vec![Value::Boolean(false), Value::Boolean(true)]
    );
}

/// Runs `source`, which must return `pcall`'s status and message, and checks
/// that it failed with `expected` instead of recursing without bound.
fn assert_recursion_fails(source: &str, expected: &str) {
    let results = run(source);
    assert_eq!(results.first(), Some(&Value::Boolean(false)), "{results:?}");
    let message = match results.get(1) {
        Some(Value::String(message)) => message.to_lossy().into_owned(),
        other => panic!("expected an error message, got {other:?}"),
    };
    assert!(message.contains(expected), "{message}");
}

#[test]
fn library_callbacks_stop_at_the_call_depth_limit() {
    // These re-enter Lua through a library function or metamethod rather
    // than a call instruction, which used to skip the frame limit.
    let cases = [
        r#"local function f(s) return (s:gsub(".", f)) end
           return pcall(f, "a")"#,
        "local t = setmetatable({}, {__tostring = function(x) return tostring(x) end})
         return pcall(tostring, t)",
        r#"local t = setmetatable({}, {__tostring = function(x) return string.format("%s", x) end})
           return pcall(tostring, t)"#,
        "local function c(a, b) table.sort({2, 1}, c) return a < b end
         return pcall(table.sort, {3, 2, 1}, c)",
        "local t = setmetatable({}, {})
         getmetatable(t).__index = function(t, k) return t[k] end
         return pcall(function() return t.x end)",
        "local function f() local x <close> = setmetatable({}, {__close = f}) end
         return pcall(f)",
    ];
    for source in cases {
        assert_recursion_fails(source, "stack overflow");
    }
}

#[test]
fn nested_coroutine_resumes_stop_like_lua_c_stack_limit() {
    // Each resume runs the interpreter on the native stack, so the nesting
    // limit is what keeps this from overflowing it.
    let source = "local f; f = function() return coroutine.wrap(f)() end
                  return pcall(f)";
    // An optimized build fits the 200 native levels in a 2 MiB thread; an
    // unoptimized one needs more than 8 MiB, as the CLI's interpreter
    // thread provides.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || assert_recursion_fails(source, "C stack overflow"))
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn garbage_after_building_a_large_structure_is_collected_young() {
    // Building the structure pauses minor collections, which is right while
    // every new object stays reachable. Once the program only churns
    // short-lived tables, the pause must end within a nursery or so: it used
    // to hold on for several whole-heap major cycles, and the heap grew to
    // about twice the live data, 620 MB here against 320 MB.
    let source = r"
        local function build(depth)
            if depth == 0 then return {} end
            return {build(depth - 1), build(depth - 1)}
        end
        local keep = {}
        for i = 1, 8 do keep[i] = build(16) end
        local peak = 0
        for i = 1, 3000000 do
            local x = {i, i + 1}
            if i % 50000 == 0 then
                local count = collectgarbage('count')
                if count > peak then peak = count end
            end
        end
        return peak / 1024, #keep
    ";
    let results = run(source);
    let Some(Value::Number(peak)) = results.first() else {
        panic!("expected the peak in MB, got {results:?}");
    };
    assert!(*peak < 450.0, "heap peaked at {peak} MB");
}

fn flat_table_churn_peak(store: bool) -> f64 {
    let store_line = if store {
        "if i % 1000 == 0 then keep[1] = {} end"
    } else {
        ""
    };
    let source = format!(
        r"
        local keep = {{}}
        for i = 1, 2000000 do keep[i] = {{i}} end
        local peak = 0
        for i = 1, 8000000 do
            local x = {{i, i + 1}}
            {store_line}
            if i % 100000 == 0 then
                local count = collectgarbage('count')
                if count > peak then peak = count end
            end
        end
        return peak / 1024
    "
    );
    let results = run(&source);
    let Some(Value::Number(peak)) = results.first() else {
        panic!("expected the peak in MB, got {results:?}");
    };
    *peak
}

#[test]
fn churn_after_building_a_flat_table_stays_bounded() {
    // The old table stays in the remembered set after the build, and the
    // trial must not charge its size against the young objects. Stock Lua
    // peaks near 330 MB here, the paused minor collections at 750 to 960 MB.
    let peak = flat_table_churn_peak(false);
    assert!(peak < 600.0, "heap peaked at {peak} MB");
}

#[test]
fn churn_with_stores_into_a_big_old_table_stays_bounded() {
    let peak = flat_table_churn_peak(true);
    assert!(peak < 600.0, "heap peaked at {peak} MB");
}
