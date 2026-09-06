//! The length operator on the fast path of the interpreter: a string or a
//! table with no `__len` to run is answered in place, and anything else takes
//! the general path with its handlers and its errors. The answers are the
//! language's in every profile, in the interpreter and under the tracer.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

const MODES: [ExecutionMode; 2] = [ExecutionMode::Interpreter, ExecutionMode::Tracing];

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> Result<Vec<Value>, String> {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.exec(source).map_err(|error| format!("{error:?}"))
}

/// The number `n` in the profile's number type: an integer from Lua 5.3 on.
fn count(profile: CompatibilityProfile, n: i64) -> Value {
    #[allow(clippy::cast_precision_loss)]
    match profile {
        CompatibilityProfile::Lua53 | CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55 => {
            Value::Integer(n)
        }
        _ => Value::Number(n as f64),
    }
}

#[test]
fn the_length_of_tables_and_strings_has_the_profiles_number_type() {
    let source = "local t = {10, 20, 30}
        local empty = {}
        local hash = {x = 1}
        local holes = {1, 2, nil, 4}
        local s = 'hello'
        return #t, #empty, #hash, #s, #'', #(t)";
    for profile in PROFILES {
        for mode in MODES {
            let results = run(profile, mode, source).unwrap();
            assert_eq!(
                results,
                [3, 0, 0, 5, 0, 3].map(|n| count(profile, n)).to_vec(),
                "{profile:?}/{mode:?}"
            );
        }
    }
    // A table with a hole has a border; the answer is one of the two, as
    // before the fast path.
    for profile in PROFILES {
        for mode in MODES {
            let results = run(profile, mode, "local t = {1, 2, nil, 4} return #t").unwrap();
            assert!(
                results == vec![count(profile, 4)] || results == vec![count(profile, 2)],
                "{profile:?}/{mode:?} {results:?}"
            );
        }
    }
}

#[test]
fn a_table_that_grows_and_shrinks_between_lengths() {
    let source = "local t = {}
        local out = {}
        for i = 1, 5 do t[#t + 1] = i * i; out[#out + 1] = #t end
        for i = 1, 3 do t[#t] = nil; out[#out + 1] = #t end
        return out[1], out[5], out[6], out[8]";
    for profile in PROFILES {
        for mode in MODES {
            let results = run(profile, mode, source).unwrap();
            assert_eq!(
                results,
                [1, 5, 4, 2].map(|n| count(profile, n)).to_vec(),
                "{profile:?}/{mode:?}"
            );
        }
    }
}

#[test]
fn len_handlers_run_from_lua52_on_and_never_in_lua51() {
    let source = "local t = setmetatable({1, 2, 3}, {__len = function() return 42 end})
        return #t";
    for profile in PROFILES {
        for mode in MODES {
            let results = run(profile, mode, source).unwrap();
            let expected = match profile {
                CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21 => 3,
                _ => 42,
            };
            assert_eq!(
                results,
                vec![count(profile, expected)],
                "{profile:?}/{mode:?}"
            );
        }
    }
}

#[test]
fn a_len_handler_added_in_the_middle_of_a_loop_is_called_from_then_on() {
    let source = "local t = setmetatable({1, 2, 3}, {})
        local sum = 0
        for i = 1, 20 do
            if i == 10 then getmetatable(t).__len = function() return 100 end end
            sum = sum + #t
        end
        return sum";
    for profile in PROFILES {
        for mode in MODES {
            let results = run(profile, mode, source).unwrap();
            let expected = match profile {
                CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21 => 60,
                _ => 9 * 3 + 11 * 100,
            };
            assert_eq!(
                results,
                vec![count(profile, expected)],
                "{profile:?}/{mode:?}"
            );
        }
    }
}

#[test]
fn the_length_of_other_values_is_an_error_with_its_message() {
    for profile in PROFILES {
        for mode in MODES {
            for source in [
                "local n = nil return #n",
                "local n = 5 return #n",
                "return #true",
            ] {
                let error = run(profile, mode, source).expect_err(source);
                assert!(
                    error.contains("length"),
                    "{profile:?}/{mode:?} {source}: {error}"
                );
            }
        }
    }
}

#[test]
fn a_loop_of_lengths_agrees_between_the_interpreter_and_the_tracer() {
    let source = "local list = {}
        local names = {'a', 'bb', 'ccc'}
        local total = 0
        for round = 1, 3000 do
            list[#list + 1] = round
            total = total + #list + #names[round % 3 + 1]
            if #list > 50 then list = {} end
        end
        return total, #list";
    for profile in PROFILES {
        let expected = run(profile, ExecutionMode::Interpreter, source).unwrap();
        let traced = run(profile, ExecutionMode::Tracing, source).unwrap();
        assert_eq!(traced, expected, "{profile:?}");
    }
}
