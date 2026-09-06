//! Protected errors retain their Lua values and coroutine resources until close.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn check(profiles: &[CompatibilityProfile], source: &str, expected: &[Value]) {
    for &profile in profiles {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            assert_eq!(lua.exec(source).unwrap(), expected, "{profile:?} {mode:?}");
        }
    }
}

const PUC: &[CompatibilityProfile] = &[
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];
const CLOSE: &[CompatibilityProfile] = &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55];

#[test]
fn handler_can_recover_from_its_own_error() {
    let source = r#"
        local marker = {}
        local attempts = 0
        local ok, result = xpcall(function() error("first", 0) end, function(e)
            attempts = attempts + 1
            if attempts == 1 then error(marker, 0) end
            return e == marker and "recovered" or "other"
        end)
        return ok, result == "recovered", attempts == 2
    "#;
    check(
        PUC,
        source,
        &[
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
        ],
    );
    check(
        &[CompatibilityProfile::LuaJit21],
        source,
        &[
            Value::Boolean(false),
            Value::Boolean(false),
            Value::Boolean(false),
        ],
    );
}

#[test]
fn repeatedly_failing_handler_is_bounded() {
    check(
        PUC,
        r#"
        local attempts = 0
        local ok, result = xpcall(function() error("first", 0) end, function(e)
            attempts = attempts + 1
            error(e, 0)
        end)
        return ok, result == "error in error handling", attempts > 1 and attempts < 10000
    "#,
        &[
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
        ],
    );
}

#[test]
fn failed_coroutine_retains_resources_and_error_until_close() {
    check(
        CLOSE,
        r#"
        local log, marker = {}, {}
        local function resource(name)
            return setmetatable({}, {__close = function(_, e)
                log[#log + 1] = name
                assert(e == marker)
            end})
        end
        local co = coroutine.create(function()
            local first <close> = resource("first")
            local function fail()
                local second <close> = resource("second")
                error(marker, 0)
            end
            fail()
        end)
        local ok, e = coroutine.resume(co)
        local deferred = #log == 0 and coroutine.status(co) == "dead"
        collectgarbage("collect")
        local resumed = coroutine.resume(co)
        local closed, close_error = coroutine.close(co)
        local ordered = table.concat(log, ",") == "second,first"
        local closed_again = coroutine.close(co)
        return ok, e == marker, deferred, resumed, closed, close_error == marker,
               ordered, closed_again
    "#,
        &[
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
        ],
    );
}

#[test]
fn coroutine_close_error_replaces_failure_and_inner_pcall_closes_immediately() {
    check(
        CLOSE,
        r"
        local first, replacement, log = {}, {}, {}
        local co = coroutine.create(function()
            local outer <close> = setmetatable({}, {__close = function(_, e)
                log[#log + 1] = e == replacement
            end})
            local inner <close> = setmetatable({}, {__close = function(_, e)
                log[#log + 1] = e == first
                error(replacement, 0)
            end})
            local ok = pcall(function()
                local caught <close> = setmetatable({}, {__close = function(_, e)
                    log[#log + 1] = e == first
                end})
                error(first, 0)
            end)
            assert(not ok and #log == 1)
            error(first, 0)
        end)
        local ok, e = coroutine.resume(co)
        local deferred = #log == 1
        local closed, close_error = coroutine.close(co)
        return ok, e == first, deferred, closed, close_error == replacement,
               #log == 3 and log[1] and log[2] and log[3]
    ",
        &[
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
        ],
    );
}

#[test]
fn close_and_wrap_publish_captured_local_writes() {
    check(
        CLOSE,
        r"
        local count = 0
        local marker, replacement = {}, {}
        local function fail()
            local item <close> = setmetatable({}, {__close = function(_, e)
                assert(e == marker)
                count = count + 1
                error(replacement, 0)
            end})
            error(marker, 0)
        end
        local co = coroutine.create(fail)
        local ok, e = coroutine.resume(co)
        local deferred = count == 0
        local closed, close_error = coroutine.close(co)
        local published = count == 1
        local wrapped, wrap_error = pcall(coroutine.wrap(fail))
        return ok, e == marker, deferred, closed, close_error == replacement,
               published, wrapped, wrap_error == replacement, count == 2
    ",
        &[
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
        ],
    );
}

#[test]
fn failed_coroutine_locals_remain_editable_until_close() {
    check(
        CLOSE,
        r#"
        local marker, observed = {}, nil
        local co = coroutine.create(function()
            local x = marker
            local item <close> = setmetatable({}, {__close = function() observed = x end})
            error("failed", 0)
        end)
        coroutine.resume(co)
        local name, value = debug.getlocal(co, 1, 1)
        local replacement = {}
        local edited = debug.setlocal(co, 1, 1, replacement)
        collectgarbage("collect")
        coroutine.close(co)
        return name == "x", value == marker, edited == "x", observed == replacement,
               debug.getinfo(co, 1) == nil
    "#,
        &[
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
        ],
    );
}

#[test]
fn failed_coroutine_close_refuses_yield_and_preserves_later_resumes() {
    check(
        CLOSE,
        r#"
        local co = coroutine.create(function()
            local item <close> = setmetatable({}, {__close = function() coroutine.yield(3) end})
            error("bad", 0)
        end)
        coroutine.resume(co)
        local ok, message = coroutine.close(co)
        local other = coroutine.create(function() coroutine.yield("fresh") return "done" end)
        local resumed, value = coroutine.resume(other)
        local finished, result = coroutine.resume(other)
        return ok, message == "attempt to yield across a C-call boundary",
               coroutine.status(co) == "dead", resumed and value == "fresh",
               finished and result == "done"
    "#,
        &[
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
        ],
    );
}

#[test]
fn nil_errors_reach_handlers_before_lua55_supplies_a_message() {
    let source = r#"
        local function fail() error(nil, 0) end
        local ok, e = pcall(fail)
        local native_ok, native_error = pcall(error, nil, 0)
        local handled, kind = xpcall(fail, function(e) return type(e) end)
        local nil_handled, nil_result = xpcall(fail, function(e) return nil end)
        local co = coroutine.create(fail)
        local resumed, resume_error = coroutine.resume(co)
        local explicit_ok, explicit_error = pcall(function() error("nil", 0) end)
        local modern = _VERSION == "Lua 5.5"
        local expected = modern and "<no error object>" or nil
        return ok, e == expected, native_ok, native_error == expected,
               handled, kind == "nil", nil_handled, nil_result == expected,
               resumed, resume_error == expected, explicit_ok, explicit_error == "nil"
    "#;
    let mut profiles = PUC.to_vec();
    profiles.push(CompatibilityProfile::LuaJit21);
    check(
        &profiles,
        source,
        &[
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
        ],
    );
}

#[test]
fn nil_failures_supply_the_profile_error_to_close_handlers() {
    check(
        CLOSE,
        r#"
        local modern = _VERSION == "Lua 5.5"
        local expected = modern and "<no error object>" or nil
        local observed, count
        local function fail()
            local item <close> = setmetatable({}, {__close = function(_, e)
                count = (count or 0) + 1
                observed = e
            end})
            error(nil, 0)
        end
        local ok, e = pcall(fail)
        local protected = count == 1 and observed == expected and e == expected
        local co = coroutine.create(fail)
        local resumed, resume_error = coroutine.resume(co)
        local deferred = count == 1 and resume_error == expected
        local closed, close_error = coroutine.close(co)
        local after_close = count == 2 and observed == expected and close_error == expected
        local replaced, replacement = pcall(function()
            local item <close> = setmetatable({}, {__close = function() error(nil, 0) end})
            error("first", 0)
        end)
        return ok, protected, resumed, deferred, closed, after_close,
               replaced, replacement == expected
    "#,
        &[
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
        ],
    );
}

#[test]
fn repeatedly_failing_handler_closes_every_attempt_after_collection() {
    check(
        CLOSE,
        r#"
        local attempts, closed = 0, 0
        local ok, result = xpcall(function() error("first", 0) end, function(e)
            attempts = attempts + 1
            local item <close> = setmetatable({}, {__close = function()
                closed = closed + 1
            end})
            collectgarbage("collect")
            error(e, 0)
        end)
        return ok, result == "error in error handling",
               attempts > 1 and attempts < 10000, closed == attempts
    "#,
        &[
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
        ],
    );
}
