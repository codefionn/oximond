//! `__index` and `__newindex` handlers that are plain Lua functions, entered
//! by the interpreter's fast loop without leaving it. Every program runs on
//! the fast loop, in tracing mode, and on the dispatch loop alone (profiling
//! keeps the fast loop out), and all three must agree.

use oximond::compat::{CompatibilityProfile, ExecutionMode};
use oximond::compiler::{CompilerConfig, compile_with_config};
use oximond::runtime::Value;
use oximond::stdlib::install_base_with_profile;
use oximond::vm::Vm;

/// Runs `source` and returns its first result as text, or the error message.
fn run(profile: CompatibilityProfile, mode: ExecutionMode, general: bool, source: &str) -> String {
    let chunk = compile_with_config(source, CompilerConfig::new(profile)).expect("compile");
    let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
    vm.use_synchronous_jit_compilation();
    vm.set_interpreter_profiling(general);
    install_base_with_profile(vm.runtime_mut(), profile).expect("install base");
    match vm.execute(&chunk) {
        Ok(values) => match values.first() {
            Some(Value::String(text)) => String::from_utf8_lossy(text.as_bytes()).into_owned(),
            other => format!("{other:?}"),
        },
        Err(error) => format!("error: {error}"),
    }
}

/// The result on the fast loop, after checking that tracing mode and the
/// dispatch loop alone produce the same text.
fn agreed(profile: CompatibilityProfile, source: &str) -> String {
    let fast = run(profile, ExecutionMode::Interpreter, false, source);
    let tracing = run(profile, ExecutionMode::Tracing, false, source);
    let general = run(profile, ExecutionMode::Interpreter, true, source);
    assert_eq!(
        fast, general,
        "{profile:?}: fast loop against dispatch loop"
    );
    assert_eq!(
        tracing, general,
        "{profile:?}: tracing against dispatch loop"
    );
    fast
}

const PROFILES: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54];

#[test]
fn hot_reads_and_writes_reach_the_handlers_with_their_operands() {
    let source = r#"
        local calls = {index = 0, newindex = 0}
        local store = {}
        local obj = setmetatable({}, {
            __index = function(t, k)
                calls.index = calls.index + 1
                if k == "none" then return nil end
                return #k
            end,
            __newindex = function(t, k, v)
                calls.newindex = calls.newindex + 1
                store[k] = v
            end,
        })
        local keys = {"a", "bb", "ccc", "none"}
        local total = 0
        for i = 1, 3000 do
            local k = keys[i % 4 + 1]
            local v = obj[k]
            if v then total = total + v end
            obj[k] = i
        end
        local named = 0
        for i = 1, 500 do named = named + obj.field + obj.other end
        return table.concat({total, calls.index, calls.newindex, store.a,
                             store.bb, store.ccc, store.none, named}, ",")
    "#;
    for profile in PROFILES {
        assert_eq!(
            agreed(profile, source),
            "4500,4000,3000,3000,2997,2998,2999,5000",
            "{profile:?}"
        );
    }
}

#[test]
fn an_index_handler_keeps_only_its_first_result() {
    let source = r#"
        local mt = {}
        local obj = setmetatable({}, mt)
        mt.__index = function(t, k) return 1, 2, 3 end
        local log = {}
        for i = 1, 400 do
            local a, b = obj.x, obj[i]
            if i == 100 then mt.__index = function() end end
            if i == 200 then mt.__index = {x = "tab", y = "y"} end
            if i == 300 then mt.__index = nil end
            if i % 50 == 1 or i == 100 then
                log[#log + 1] = i .. ":" .. tostring(a) .. "/" .. tostring(b)
            end
        end
        return table.concat(log, " ")
    "#;
    for profile in PROFILES {
        assert_eq!(
            agreed(profile, source),
            "1:1/1 51:1/1 100:1/1 101:nil/nil 151:nil/nil 201:tab/nil \
             251:tab/nil 301:nil/nil 351:nil/nil",
            "{profile:?}"
        );
    }
}

#[test]
fn a_handler_replaced_between_iterations_runs_the_new_one() {
    let source = r#"
        local mt = {}
        local obj = setmetatable({}, mt)
        local seen = {}
        mt.__newindex = function(t, k, v) seen[#seen + 1] = "first" end
        mt.__index = function(t, k) return "first" end
        local last
        for i = 1, 300 do
            last = obj.k
            obj.k = i
            if i == 100 then
                mt.__newindex = function(t, k, v) seen[#seen + 1] = "second" end
                mt.__index = function(t, k) return "second" end
            end
            if i == 200 then
                mt.__newindex = nil
                mt.__index = nil
            end
        end
        return table.concat({#seen, seen[1], seen[100], seen[101], tostring(last),
                             tostring(rawget(obj, "k"))}, ",")
    "#;
    for profile in PROFILES {
        assert_eq!(
            agreed(profile, source),
            "200,first,first,second,299,300",
            "{profile:?}"
        );
    }
}

#[test]
fn newindex_chains_through_tables_and_rawset_inside_handlers() {
    let source = r#"
        local final = {}
        local middle = setmetatable({}, {
            __newindex = function(t, k, v) rawset(final, k, v * 2) end,
        })
        local top = setmetatable({}, {__newindex = middle})
        local raw = setmetatable({}, {
            __newindex = function(t, k, v) rawset(t, k, v + 1) end,
        })
        for i = 1, 300 do
            top["k" .. (i % 5)] = i
            raw[i] = i
        end
        return table.concat({final.k0, final.k1, final.k4, tostring(rawget(top, "k0")),
                             raw[1], raw[300], rawget(raw, 2)}, ",")
    "#;
    for profile in PROFILES {
        assert_eq!(
            agreed(profile, source),
            "600,592,598,nil,2,301,3",
            "{profile:?}"
        );
    }
}

#[test]
fn errors_in_handlers_keep_their_messages_positions_and_tracebacks() {
    let source = r#"
        local obj = setmetatable({}, {
            __index = function(t, k)
                if k == "bad" then error("index boom") end
                return 1
            end,
            __newindex = function(t, k, v)
                if k ~= "v" then return end
                if v == 99 then error({code = 7}) end
                if v == 98 then error("set boom", 2) end
                if v == 97 then local x = nil; x.y = 1 end
            end,
        })
        local n = 0
        for i = 1, 500 do n = n + obj.ok; obj.ok = i end
        local ok1, e1 = pcall(function() return obj.bad end)
        local ok2, e2 = pcall(function() obj.v = 99 end)
        local ok3, e3 = pcall(function() obj.v = 98 end)
        local ok4, e4 = pcall(function() obj.v = 97 end)
        local ok5, trace = xpcall(function() return obj.bad end, debug.traceback)
        local ok6, trace2 = xpcall(function() obj.v = 97 end, debug.traceback)
        return table.concat({n, tostring(ok1), e1, tostring(ok2), type(e2) == "table" and e2.code or "?",
                             tostring(ok3), e3, tostring(ok4), e4, trace, trace2}, "|")
    "#;
    for profile in PROFILES {
        let text = agreed(profile, source);
        assert!(text.starts_with("500|false|"), "{profile:?}: {text}");
        assert!(text.contains("index boom"), "{profile:?}: {text}");
        assert!(text.contains("set boom"), "{profile:?}: {text}");
        assert!(text.contains("stack traceback"), "{profile:?}: {text}");
    }
}

#[test]
fn hooks_see_the_same_handler_calls_and_returns() {
    let source = r#"
        local obj = setmetatable({}, {
            __index = function(t, k) return 5 end,
            __newindex = function(t, k, v) end,
        })
        local events = {}
        local function hook(event)
            local info = debug.getinfo(2, "nS")
            events[#events + 1] = event .. ":" .. tostring(info.name) .. ":" .. info.what
        end
        local s = 0
        for i = 1, 200 do s = s + obj.a; obj.b = i; s = s + obj[i] end
        debug.sethook(hook, "cr")
        for i = 1, 3 do s = s + obj.a; obj.b = i end
        debug.sethook()
        local lines = {}
        debug.sethook(function(event, line) lines[#lines + 1] = line end, "l")
        local x = obj.a
        obj.c = x
        debug.sethook()
        return s .. " " .. table.concat(events, ",") .. " " .. #lines
    "#;
    for profile in PROFILES {
        let text = agreed(profile, source);
        assert!(text.starts_with("2015 "), "{profile:?}: {text}");
        assert!(text.contains("call:"), "{profile:?}: {text}");
        assert!(text.contains("return:"), "{profile:?}: {text}");
    }
}

#[test]
fn a_hook_installed_inside_a_handler_sees_the_rest_of_the_loop() {
    let source = r#"
        local events = {}
        local function hook(event) events[#events + 1] = event end
        local n = 0
        local obj = setmetatable({}, {
            __index = function(t, k)
                n = n + 1
                if n == 100 then debug.sethook(hook, "cr") end
                if n == 103 then debug.sethook() end
                return n
            end,
        })
        local s = 0
        for i = 1, 200 do s = s + obj.v end
        return s .. " " .. #events .. " " .. table.concat(events, ",", 1, 6)
    "#;
    for profile in PROFILES {
        agreed(profile, source);
    }
}

#[test]
fn a_coroutine_yielding_inside_a_handler_resumes_into_the_right_register() {
    let source = r##"
        local obj = setmetatable({}, {
            __index = function(t, k) return coroutine.yield(k) end,
            __newindex = function(t, k, v)
                coroutine.yield("set " .. k)
                rawset(t, k, v)
            end,
        })
        local co = coroutine.wrap(function()
            local acc = {}
            for i = 1, 150 do
                local a, b = obj.x, obj.y
                acc[#acc + 1] = a .. b
            end
            obj.z = 1
            return table.concat(acc, ",", 1, 3) .. "#" .. #acc .. rawget(obj, "z")
        end)
        local r = co()
        local n = 0
        while true do
            n = n + 1
            if r == "x" then r = co("A" .. n)
            elseif r == "y" then r = co("B" .. n)
            elseif r == "set z" then r = co()
            else break end
        end
        return r .. "/" .. n
    "##;
    let text = agreed(CompatibilityProfile::Lua54, source);
    assert_eq!(text, "A1B2,A3B4,A5B6#1501/302");
}

#[test]
fn handlers_that_call_other_handlers_nest_and_unwind() {
    let source = r#"
        local depth = 0
        local deepest = 0
        local inner = setmetatable({}, {
            __index = function(t, k)
                depth = depth + 1
                if depth > deepest then deepest = depth end
                depth = depth - 1
                return k
            end,
            __newindex = function(t, k, v) rawset(t, k, v) end,
        })
        local outer = setmetatable({}, {
            __index = function(t, k)
                depth = depth + 1
                if depth > deepest then deepest = depth end
                local v = inner[k]
                inner[k .. "!"] = v
                depth = depth - 1
                return v .. "?"
            end,
        })
        local s = 0
        for i = 1, 300 do s = s + #outer["key" .. i % 3] end
        return s .. " " .. deepest .. " " .. tostring(rawget(inner, "key1!"))
    "#;
    for profile in PROFILES {
        assert_eq!(agreed(profile, source), "1500 2 key1", "{profile:?}");
    }
}

#[test]
fn a_handler_with_varargs_or_no_parameters_gets_the_same_operands() {
    let source = r##"
        local seen = {}
        local varargs = setmetatable({}, {
            __index = function(...) return select("#", ...) end,
            __newindex = function(...) seen[#seen + 1] = select("#", ...) end,
        })
        local bare = setmetatable({}, {
            __index = function() return "bare" end,
            __newindex = function() seen[#seen + 1] = "bare" end,
        })
        local s = 0
        for i = 1, 200 do
            s = s + varargs.a
            varargs.b = i
            bare.c = i
            if bare.d ~= "bare" then s = -1 end
        end
        return s .. " " .. #seen .. " " .. seen[1] .. " " .. seen[2]
    "##;
    for profile in PROFILES {
        assert_eq!(agreed(profile, source), "400 400 3 bare", "{profile:?}");
    }
}

#[test]
fn protected_calls_close_variables_and_garbage_collection_around_handlers() {
    let source = r#"
        local made = 0
        local obj = setmetatable({}, {
            __index = function(t, k)
                made = made + 1
                local garbage = {}
                for i = 1, 20 do garbage[i] = {i} end
                if made % 50 == 0 then collectgarbage() end
                if made % 77 == 0 then error("boom " .. made) end
                return #garbage
            end,
            __newindex = function(t, k, v)
                local junk = {v, k}
                if v % 60 == 0 then collectgarbage() end
                if v % 101 == 0 then error("set " .. v) end
            end,
        })
        local ok, bad, s = 0, 0, 0
        for i = 1, 400 do
            local good, value = pcall(function() return obj.k end)
            if good then ok = ok + 1; s = s + value else bad = bad + 1 end
            pcall(function() obj.k = i end)
        end
        return table.concat({ok, bad, s, made}, ",")
    "#;
    for profile in PROFILES {
        assert_eq!(agreed(profile, source), "395,5,7900,400", "{profile:?}");
    }
}

#[test]
fn stores_that_settle_without_the_handler_keep_every_cache_right() {
    // Hashed tables, written through dynamic string keys: a present key takes
    // a scalar in place, an absent one (or a stored nil) goes to `__newindex`,
    // and a table that reads are cached through is still watched.
    let source = r#"
        local keys = {}
        for i = 1, 300 do keys[i] = "k" .. i end
        local plain, base = {}, {}
        for i = 1, 300 do plain[keys[i]] = i; base[keys[i]] = {} end
        local obj = setmetatable({}, {__index = base})
        for round = 1, 2 do
            for i = 1, 300 do local _ = obj[keys[i]] end
        end
        for i = 1, 300 do base[keys[i]] = i end
        for i = 1, 300 do plain[keys[i]] = plain[keys[i]] + 1 end
        local ok = true
        for i = 1, 300 do
            if obj[keys[i]] ~= i or plain[keys[i]] ~= i + 1 then ok = false end
        end
        local calls = 0
        local p = setmetatable({}, {__newindex = function(t, k, v)
            calls = calls + 1
            rawset(t, k, v)
        end})
        for i = 1, 300 do p[keys[i]] = i end
        for i = 1, 300 do p[keys[i]] = i * 2 end
        p[keys[1]] = nil
        p[keys[1]] = 7
        return table.concat({tostring(ok), calls, p[keys[1]], p[keys[300]]}, ",")
    "#;
    for profile in PROFILES {
        assert_eq!(agreed(profile, source), "true,301,7,600", "{profile:?}");
    }
}

#[test]
fn method_calls_through_an_index_handler_pass_self_and_follow_replacements() {
    let source = r#"
        local mt = {}
        local obj = setmetatable({name = "o"}, mt)
        mt.__index = function(t, k)
            return function(self, x) return self.name .. k .. x end
        end
        local log = {}
        for i = 1, 300 do
            local r = obj:m(i)
            if i == 100 then
                mt.__index = function(t, k)
                    return function(self, x) return "new" .. x end
                end
            end
            if i == 200 then mt.__index = {m = function(self, x) return "tab" .. x end} end
            if i == 1 or i == 100 or i == 101 or i == 200 or i == 201 then
                log[#log + 1] = r
            end
        end
        return table.concat(log, " ")
    "#;
    for profile in PROFILES {
        assert_eq!(
            agreed(profile, source),
            "om1 om100 new101 new200 tab201",
            "{profile:?}"
        );
    }
}

#[test]
fn a_missing_method_behind_an_index_handler_keeps_its_error() {
    let source = r"
        local obj = setmetatable({}, {__index = function(t, k) return nil end})
        local n = 0
        for i = 1, 300 do n = n + 1 end
        local ok, err = pcall(function()
            for i = 1, 300 do obj:m(i) end
        end)
        return err
    ";
    let expected = [
        (
            CompatibilityProfile::Lua51,
            "?:6: attempt to call method 'm' (a nil value)",
        ),
        (
            CompatibilityProfile::Lua54,
            "?:6: attempt to call a nil value (method 'm')",
        ),
    ];
    for (profile, text) in expected {
        let message = agreed(profile, source);
        assert!(message.ends_with(text), "{profile:?}: {message}");
    }
}

#[test]
fn an_upvalue_table_read_enters_its_index_handler() {
    let source = r#"
        local calls = 0
        local _ENV = setmetatable({}, {__index = function(t, k)
            calls = calls + 1
            return k .. calls
        end})
        local function read() return alpha end
        local last
        for i = 1, 300 do last = read() end
        return last .. "," .. calls
    "#;
    assert_eq!(agreed(CompatibilityProfile::Lua54, source), "alpha300,300");
}

#[test]
fn a_method_handler_is_called_with_the_table_the_walk_reached() {
    let source = r#"
        local seen = {}
        local inner = setmetatable({tag = "inner"}, {__index = function(t, k)
            seen[#seen + 1] = t.tag
            return function(self, x) return x + 1 end
        end})
        local direct = setmetatable({tag = "direct"}, {__index = function(t, k)
            seen[#seen + 1] = t.tag
            return function(self, x) return x + 2 end
        end})
        local viaHop = setmetatable({tag = "outer"}, {__index = inner})
        local sum = 0
        for i = 1, 200 do
            sum = sum + viaHop:m(i) + direct:m(i)
            if i == 100 then
                -- a new metatable for the same receiver
                setmetatable(direct, {__index = function(t, k)
                    return function(self, x) return x * 0 end
                end})
            end
        end
        return sum .. "," .. seen[1] .. "," .. seen[2] .. "," .. #seen
    "#;
    // `viaHop` adds i + 1 every round, `direct` adds i + 2 for the first 100
    // rounds and 0 after its metatable changes. The handlers run 200 times
    // on `inner` and 100 times on `direct` before the swap.
    let expected = (1..=200).map(|i| i + 1).sum::<i64>() + (1..=100).map(|i| i + 2).sum::<i64>();
    for profile in PROFILES {
        assert_eq!(
            agreed(profile, source),
            format!("{expected},inner,direct,300"),
            "{profile:?}"
        );
    }
}

#[test]
fn a_string_key_the_handler_was_given_does_not_leak_into_the_next_call() {
    let source = r#"
        local obj = setmetatable({}, {__index = function(t, k)
            return function(self, x, y) return x .. y end
        end})
        local out
        for i = 1, 300 do out = obj:go("a", "b" .. (i % 3)) end
        return out
    "#;
    for profile in PROFILES {
        assert_eq!(agreed(profile, source), "ab0", "{profile:?}");
    }
}

#[test]
fn a_handler_replaced_every_iteration_from_one_store_site_runs_the_new_one() {
    // The same store site overwrites a closure with a closure each round, so
    // a cached handler must be dropped by an overwrite, not only by a first
    // store.
    let method = r"
        local h1 = function(t, k) return function(s, x) return 1 end end
        local h2 = function(t, k) return function(s, x) return 2 end end
        local mt = {__index = h1}
        local obj = setmetatable({}, mt)
        local function run()
            local s = 0
            for i = 1, 300 do
                mt.__index = (i % 2 == 0) and h1 or h2
                s = s + obj:m(i)
            end
            return s
        end
        return tostring(run())
    ";
    let upvalue = r"
        local h1 = function(t, k) return 1 end
        local h2 = function(t, k) return 2 end
        local mt = {__index = h1}
        local T = setmetatable({}, mt)
        local function run()
            local s = 0
            for i = 1, 300 do
                mt.__index = (i % 2 == 0) and h1 or h2
                s = s + T.field
            end
            return s
        end
        return tostring(run())
    ";
    for profile in PROFILES {
        assert_eq!(agreed(profile, method), "450", "{profile:?} self");
        assert_eq!(agreed(profile, upvalue), "450", "{profile:?} upvalue");
    }
}
