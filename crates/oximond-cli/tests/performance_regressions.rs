use oximond::compat::CompatibilityProfile;
use oximond::compat::ExecutionMode;
use oximond::compiler::{CompilerConfig, compile, compile_with_config};
use oximond::stdlib::{install_base, install_base_with_profile};
use oximond::vm::Vm;

fn run_in_interpreter_and_tracing(source: &str) {
    let chunk = compile(source).expect("compile regression program");
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_execution_mode(mode);
        vm.use_synchronous_jit_compilation();
        install_base(vm.runtime_mut()).expect("install standard library");
        vm.execute(&chunk).expect("run regression program");
    }
}

#[test]
fn captured_two_argument_callee_falls_back_for_other_values() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local function adjust(x, y)
            if x < 0 then return y end
            return x + y
        end
        local function wrapper(x, y)
            return adjust(x, y) + adjust(x - 1, y + 2)
        end
        local total = 0
        for i = 1, 1000 do total = total + wrapper(i - 500, 3) end
        local string_arg = adjust(7, '2')
        local extra_arg = adjust(4, 2, 9)
        local missing_ok = pcall(adjust, 4)
        local calls = 0
        debug.sethook(function(event)
            if event == 'call' then calls = calls + 1 end
        end, 'c')
        local hooked = wrapper(6, 3)
        debug.sethook()
        return total, string_arg, extra_arg, missing_ok, hooked, calls
        ",
        CompilerConfig::new(profile),
    )
    .unwrap();
    let mut interpreter = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    install_base_with_profile(interpreter.runtime_mut(), profile).unwrap();
    let expected = interpreter.execute(&chunk).unwrap();
    assert_eq!(
        expected[1..5],
        [
            oximond::runtime::Value::Number(9.0),
            oximond::runtime::Value::Number(6.0),
            oximond::runtime::Value::Boolean(false),
            oximond::runtime::Value::Number(19.0),
        ]
    );
    assert!(matches!(expected[5], oximond::runtime::Value::Number(count) if count >= 3.0));
    let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    tracing.use_synchronous_jit_compilation();
    install_base_with_profile(tracing.runtime_mut(), profile).unwrap();
    assert_eq!(tracing.execute(&chunk).unwrap(), expected);
}

#[test]
fn captured_scalar_callee_preserves_nil_booleans_and_profile_semantics() {
    for profile in CompatibilityProfile::ALL {
        let chunk = compile_with_config(
            r"
            local function select_value(flag, value)
                if flag then return value end
                return nil
            end
            local function wrapper(flag, value)
                local result = select_value(flag, value)
                return result
            end
            local seen_true, seen_false, seen_nil = 0, 0, 0
            for i = 1, 600 do
                local value = wrapper(i % 3 ~= 0, i % 2 == 0)
                if value == true then seen_true = seen_true + 1
                elseif value == false then seen_false = seen_false + 1
                else seen_nil = seen_nil + 1 end
            end
            local missing = wrapper(true)
            local extra = wrapper(true, false, 42)
            local text = wrapper(true, 'kept')
            select_value = function() return 'rebound' end
            local rebound = wrapper(true, false)
            return seen_true, seen_false, seen_nil, missing, extra, text, rebound
            ",
            CompilerConfig::new(profile),
        )
        .unwrap();
        let mut interpreter =
            Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
        let expected = interpreter.execute(&chunk).unwrap();
        let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        tracing.use_synchronous_jit_compilation();
        assert_eq!(tracing.execute(&chunk).unwrap(), expected, "{profile}");
    }
}

#[test]
fn scalar_self_recursion_returns_boolean_and_nil() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local function descend(n, value)
            if n <= 0 then return value end
            local result = descend(n - 1, value)
            return result
        end
        local yes, absent
        for _ = 1, 200 do
            yes = descend(8, true)
            absent = descend(8, nil)
        end
        return yes, absent
        ",
        CompilerConfig::new(profile),
    )
    .unwrap();
    let mut interpreter = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    let expected = interpreter.execute(&chunk).unwrap();
    let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    tracing.use_synchronous_jit_compilation();
    assert_eq!(tracing.execute(&chunk).unwrap(), expected);
}

#[test]
fn bound_calls_pass_opaque_values_without_losing_identity() {
    let source = r"
        local function identity(value) return value end
        local function forward(value)
            local result = identity(value)
            return result
        end
        local table_value = {marker = 42}
        local closure_value = function() return 17 end
        local thread_value = coroutine.create(function() end)
        local string_value = 'opaque value'
        local values = {table_value, closure_value, thread_value, string_value, host_userdata}
        local good = 0
        for i = 1, 600 do
            local original = values[(i - 1) % #values + 1]
            local returned = forward(original)
            if returned == original then good = good + 1 end
            if i % 25 == 0 then collectgarbage('collect') end
        end
        local fresh = 0
        for i = 1, 100 do
            local returned = forward({marker = i})
            collectgarbage('collect')
            if returned.marker == i then fresh = fresh + 1 end
        end
        local returned_userdata = forward(host_userdata)
        values = nil
        host_userdata = nil
        collectgarbage('collect')
        return good, fresh, forward(table_value).marker, forward(closure_value)(),
            type(returned_userdata), type(forward(thread_value))
    ";
    let chunk =
        compile_with_config(source, CompilerConfig::new(CompatibilityProfile::Lua51)).unwrap();
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_profile_and_execution_mode(CompatibilityProfile::Lua51, mode);
        vm.use_synchronous_jit_compilation();
        install_base_with_profile(vm.runtime_mut(), CompatibilityProfile::Lua51).unwrap();
        let userdata = vm.runtime_mut().create_userdata().unwrap();
        vm.runtime_mut()
            .set_global("host_userdata", oximond::runtime::Value::Userdata(userdata));
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![
                oximond::runtime::Value::Number(600.0),
                oximond::runtime::Value::Number(100.0),
                oximond::runtime::Value::Number(42.0),
                oximond::runtime::Value::Number(17.0),
                oximond::runtime::Value::string("userdata"),
                oximond::runtime::Value::string("thread"),
            ],
            "{mode:?}"
        );
    }
}

#[test]
fn recursion_returns_live_tables_across_gc() {
    let source = r"
        local function descend(depth, value)
            if depth <= 0 then return value end
            local result = descend(depth - 1, value)
            return result
        end
        local good = 0
        for i = 1, 200 do
            local returned = descend(6, {marker = i})
            collectgarbage('collect')
            if returned.marker == i then good = good + 1 end
        end
        return good
    ";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        let chunk = compile_with_config(source, CompilerConfig::new(profile)).unwrap();
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        vm.use_synchronous_jit_compilation();
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![if profile == CompatibilityProfile::Lua51 {
                oximond::runtime::Value::Number(200.0)
            } else {
                oximond::runtime::Value::Integer(200)
            }],
            "{profile}"
        );
    }
}

#[test]
fn lua_calls_return_string_constants() {
    let source = r"
        local function constant() return 'native result' end
        local function bound()
            local result = constant()
            return result
        end
        local function descend(depth)
            if depth <= 0 then return 'recursive result' end
            local result = descend(depth - 1)
            return result
        end
        for i = 1, 300 do
            local first = bound()
            local second = descend(3)
            assert(first == 'native result' and second == 'recursive result')
        end
        return bound(), descend(3)
    ";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        let chunk = compile_with_config(source, CompilerConfig::new(profile)).unwrap();
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        vm.use_synchronous_jit_compilation();
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![
                oximond::runtime::Value::string("native result"),
                oximond::runtime::Value::string("recursive result"),
            ],
            "{profile}"
        );
    }
}

#[test]
fn lua_calls_apply_truthiness_to_opaque_values() {
    let source = r"
        local function invert(value) return not value end
        local function bound(value)
            local result = invert(value)
            return result
        end
        local table_value = {}
        local good = 0
        for i = 1, 300 do
            local value = i % 2 == 0 and table_value or false
            if bound(value) == (value == false) then good = good + 1 end
        end
        return good, bound(nil), bound(false), bound(0), bound('text')
    ";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        let chunk = compile_with_config(source, CompilerConfig::new(profile)).unwrap();
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        vm.use_synchronous_jit_compilation();
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![
                if profile == CompatibilityProfile::Lua51 {
                    oximond::runtime::Value::Number(300.0)
                } else {
                    oximond::runtime::Value::Integer(300)
                },
                oximond::runtime::Value::Boolean(true),
                oximond::runtime::Value::Boolean(true),
                oximond::runtime::Value::Boolean(false),
                oximond::runtime::Value::Boolean(false),
            ],
            "{profile}"
        );
    }
}

#[test]
fn native_committed_call_roots_opaque_temporaries_before_collection() {
    let chunk = compile(
        r"
        local function run(i)
            local a = i + 1
            local b = a * 2
            local c = b + 3
            local d = c * 4
            local e = d - 1
            local value = {marker = e}
            collectgarbage('collect')
            return value.marker
        end
        local good = 0
        for i = 1, 600 do
            if run(i) == ((i + 1) * 2 + 3) * 4 - 1 then
                good = good + 1
            end
        end
        return good
        ",
    )
    .unwrap();
    let mut interpreter = Vm::with_execution_mode(ExecutionMode::Interpreter);
    install_base(interpreter.runtime_mut()).unwrap();
    let expected = interpreter.execute(&chunk).unwrap();
    assert_eq!(expected, vec![oximond::runtime::Value::Number(600.0)]);
    let mut tracing = Vm::with_execution_mode(ExecutionMode::Tracing);
    tracing.use_synchronous_jit_compilation();
    install_base(tracing.runtime_mut()).unwrap();
    for _ in 0..32 {
        assert_eq!(tracing.execute(&chunk).unwrap(), expected);
    }
}

#[test]
fn native_vm_call_exit_runs_erroring_callee_once() {
    let chunk = compile(
        r"
        local calls = 0
        local function fail(value)
            calls = calls + 1
            collectgarbage('collect')
            error(value.marker)
        end
        local function run(i)
            local a = i + 1
            local b = a * 2
            local c = b + 3
            local d = c * 4
            local e = d - 1
            -- Unused arithmetic keeps the acyclic prefix above the
            -- minimum size worth compiling.
            local f = e + 5
            local g = f * 3
            local h = g - 7
            local k = h * 2
            local value = {marker = e}
            fail(value)
        end
        local good = 0
        for i = 1, 300 do
            local ok, message = pcall(run, i)
            -- Lua 5.1 adds a position to a number message, so compare the
            -- trailing digits.
            if not ok and tonumber(tostring(message):match('%d+$')) == ((i + 1) * 2 + 3) * 4 - 1 then
                good = good + 1
            end
        end
        return good, calls
        ",
    )
    .unwrap();
    let mut interpreter = Vm::with_execution_mode(ExecutionMode::Interpreter);
    install_base(interpreter.runtime_mut()).unwrap();
    let expected = interpreter.execute(&chunk).unwrap();
    assert_eq!(expected, vec![oximond::runtime::Value::Number(300.0); 2]);
    let mut tracing = Vm::with_execution_mode(ExecutionMode::Tracing);
    tracing.use_synchronous_jit_compilation();
    install_base(tracing.runtime_mut()).unwrap();
    assert_eq!(tracing.execute(&chunk).unwrap(), expected);
}

#[test]
fn native_committed_call_preserves_yielded_opaque_values() {
    let chunk = compile(
        r"
        local function run()
            for i = 1, 300 do
                local a = i + 1
                local b = a * 2
                local c = b + 3
                local d = c * 4
                local e = d - 1
                local value = {marker = e}
                coroutine.yield(value)
            end
        end
        local thread = coroutine.create(run)
        local good, checksum = 0, 0
        for i = 1, 300 do
            local ok, value = coroutine.resume(thread)
            collectgarbage('collect')
            -- Enough arithmetic per call that the trace engine prefers native code
            -- for this loop over interpreting a call-bound body.
            local p = i * 3 + 1
            local q = p * p - i
            local r = q % 7 + p * 2
            local s = r * 5 - q % 3
            local t = s + r * p - q
            local u = t % 11 + s * 2
            checksum = checksum + u % 13 + t % 5 + r
            if ok and value.marker == ((i + 1) * 2 + 3) * 4 - 1 then
                good = good + 1
            end
        end
        return good
        ",
    )
    .unwrap();
    let mut interpreter = Vm::with_execution_mode(ExecutionMode::Interpreter);
    install_base(interpreter.runtime_mut()).unwrap();
    let expected = interpreter.execute(&chunk).unwrap();
    assert_eq!(expected, vec![oximond::runtime::Value::Number(300.0)]);
    let mut tracing = Vm::with_execution_mode(ExecutionMode::Tracing);
    tracing.use_synchronous_jit_compilation();
    install_base(tracing.runtime_mut()).unwrap();
    for _ in 0..32 {
        assert_eq!(tracing.execute(&chunk).unwrap(), expected);
    }
}

#[test]
fn bound_calls_preserve_integer_arithmetic_in_modern_profiles() {
    let source = r"
        local function increment(value) return value + 1 end
        local function wrap(value)
            local result = increment(value)
            return result
        end
        local function below_limit(value) return value < 9007199254740993 end
        local function compare(value)
            local result = below_limit(value)
            return result
        end
        local function half(value) return value / 2 end
        local function divide(value)
            local result = half(value)
            return result
        end
        local count = 0
        for i = 1, 600 do
            local incremented = wrap(i)
            local smaller = compare(i)
            local divided = divide(i)
            if incremented > 0 and smaller and divided > 0 then count = count + 1 end
        end
        return count, wrap(9223372036854775807), wrap(9007199254740992),
            wrap(1.5), compare(9007199254740992),
            compare(9007199254740993), compare(9007199254740992.0), divide(3)
    ";
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let chunk = compile_with_config(source, CompilerConfig::new(profile)).unwrap();
        let mut interpreter =
            Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
        let expected = interpreter.execute(&chunk).unwrap();
        assert_eq!(
            expected,
            vec![
                oximond::runtime::Value::Integer(600),
                oximond::runtime::Value::Integer(i64::MIN),
                oximond::runtime::Value::Integer(9_007_199_254_740_993),
                oximond::runtime::Value::Number(2.5),
                oximond::runtime::Value::Boolean(true),
                oximond::runtime::Value::Boolean(false),
                oximond::runtime::Value::Boolean(true),
                oximond::runtime::Value::Number(1.5),
            ],
            "{profile}"
        );
        let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        tracing.use_synchronous_jit_compilation();
        assert_eq!(tracing.execute(&chunk).unwrap(), expected, "{profile}");
    }
}

#[test]
fn two_captured_callees_guard_each_rebinding() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local function add_one(n) return n + 1 end
        local function double(n) return n * 2 end
        local function combine(n)
            local first = add_one(n)
            local second = double(n)
            return first + second
        end
        local total = 0
        for i = 1, 800 do total = total + combine(i) end
        add_one = function(n) return n + 10 end
        local first_rebound = combine(5)
        double = function(n) return n * 3 end
        local both_rebound = combine(5)
        return total, first_rebound, both_rebound
        ",
        CompilerConfig::new(profile),
    )
    .unwrap();
    let mut interpreter = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    let expected = interpreter.execute(&chunk).unwrap();
    let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    tracing.use_synchronous_jit_compilation();
    assert_eq!(tracing.execute(&chunk).unwrap(), expected);
}

#[test]
fn two_captured_scalar_callees_across_profiles() {
    for profile in CompatibilityProfile::ALL {
        let chunk = compile_with_config(
            r"
            local function first(value) return value end
            local function second(value) return value end
            local function choose(flag)
                local left = first(flag)
                local right = second(nil)
                if left then return left end
                return right
            end
            local yes, absent = 0, 0
            for i = 1, 500 do
                local result = choose(i % 2 == 0)
                if result then yes = yes + 1 else absent = absent + 1 end
            end
            return yes, absent
            ",
            CompilerConfig::new(profile),
        )
        .unwrap();
        let mut interpreter =
            Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
        let expected = interpreter.execute(&chunk).unwrap();
        let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        tracing.use_synchronous_jit_compilation();
        assert_eq!(tracing.execute(&chunk).unwrap(), expected, "{profile}");
    }
}

#[test]
fn two_argument_self_recursion_guards_rebinding() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local function sum(n, acc)
            if n <= 0 then return acc end
            return sum(n - 1, acc + n) + 0
        end
        local warm = sum(40, 2)
        local old = sum
        sum = function(n, acc) return n * acc end
        local rebound = old(3, 4)
        return warm, rebound
        ",
        CompilerConfig::new(profile),
    )
    .unwrap();
    let mut interpreter = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    let expected = interpreter.execute(&chunk).unwrap();
    assert_eq!(
        expected,
        vec![
            oximond::runtime::Value::Number(822.0),
            oximond::runtime::Value::Number(14.0),
        ]
    );
    let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    tracing.use_synchronous_jit_compilation();
    assert_eq!(tracing.execute(&chunk).unwrap(), expected);
}

#[test]
fn bound_calls_guard_rebound_targets_and_hooks() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local function adjust(n) return n + 1 end
        local function wrapper(n) return adjust(n) + adjust(n - 1) end
        local total = 0
        for i = 1, 400 do total = total + wrapper(i) end
        adjust = function(n) return n * 10 end
        local rebound = wrapper(5)
        local calls = 0
        debug.sethook(function(event)
            if event == 'call' then calls = calls + 1 end
        end, 'c')
        local hooked = wrapper(6)
        debug.sethook()
        return total, rebound, hooked, calls
        ",
        CompilerConfig::new(profile),
    )
    .unwrap();
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
        vm.use_synchronous_jit_compilation();
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        let values = vm.execute(&chunk).unwrap();
        assert_eq!(
            values[..3],
            [
                oximond::runtime::Value::Number(160_800.0),
                oximond::runtime::Value::Number(90.0),
                oximond::runtime::Value::Number(110.0),
            ]
        );
        assert!(matches!(values[3], oximond::runtime::Value::Number(count) if count >= 3.0));
    }
}

#[test]
fn recursive_native_calls_guard_rebound_upvalues_and_hooks() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local function fib(n)
            if n < 2 then return n end
            return fib(n - 1) + fib(n - 2)
        end
        local old = fib
        local warm = fib(12)
        fib = function() return 7 end
        local rebound = old(5)
        fib = old
        local calls = 0
        local function hook(event)
            if event == 'call' then calls = calls + 1 end
        end
        debug.sethook(hook, 'c')
        local hooked = fib(8)
        debug.sethook()
        return warm, rebound, hooked, calls
        ",
        CompilerConfig::new(profile),
    )
    .unwrap();
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
        vm.use_synchronous_jit_compilation();
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        let values = vm.execute(&chunk).unwrap();
        assert_eq!(
            values[..3],
            [
                oximond::runtime::Value::Number(144.0),
                oximond::runtime::Value::Number(14.0),
                oximond::runtime::Value::Number(21.0),
            ]
        );
        assert!(matches!(values[3], oximond::runtime::Value::Number(count) if count > 20.0));
    }
}

#[test]
fn temporary_table_fragments_do_not_allocate_in_either_execution_mode() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local total = 0
        for i = 1, 20000 do
            local value = {i, i + 1, i + 2, label = 'node'}
            total = total + value[2]
        end
        return total
        ",
        CompilerConfig::new(profile),
    )
    .unwrap();
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
        vm.use_synchronous_jit_compilation();
        let before = vm.runtime().allocation_serial();
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![oximond::runtime::Value::Number(200_030_000.0)]
        );
        let allocations = vm.runtime().allocation_serial() - before;
        assert!(
            allocations < 10,
            "{mode:?} allocated {allocations} times for temporary tables"
        );
    }
}

#[test]
fn temporary_table_fragments_preserve_escapes_and_numeric_fallbacks() {
    run_in_profiled_interpreter_and_tracing(
        r"
        local function run(input)
            local total = 0
            for i = 1, 100 do
                local value = {input, label = 'node'}
                local alias = value
                value[1] = value[1] + 1
                total = total + alias[1]
            end
            return total
        end
        assert(run(4) == 500)
        assert(run('4') == 500)
        local object = setmetatable({}, {__add = function() return 7 end})
        assert(run(object) == 700)
        local saved = {}
        for i = 1, 100 do
            local value = {i}
            saved[i] = value
        end
        assert(saved[1] ~= saved[2] and saved[40][1] == 40)
        local readers = {}
        for i = 1, 100 do
            local value = {i}
            readers[i] = function() return value[1] end
        end
        collectgarbage('collect')
        assert(readers[1]() == 1 and readers[100]() == 100)
        local function fail(key)
            local value = {}
            value[key] = 1
        end
        assert(not pcall(fail, nil))
        assert(not pcall(fail, 0 / 0))
    ",
        CompatibilityProfile::Lua51,
    );
}

#[test]
fn temporary_tables_remain_visible_to_instruction_hooks() {
    run_in_profiled_interpreter_and_tracing(
        r"
        local seen = 0
        local function hook()
            for index = 1, 30 do
                local name, value = debug.getlocal(2, index)
                if name == 'temporary' and type(value) == 'table' then
                    seen = seen + 1
                end
            end
        end
        debug.sethook(hook, '', 1)
        local total = 0
        for i = 1, 20 do
            local temporary = {i}
            total = total + temporary[1]
        end
        debug.sethook()
        assert(total == 210 and seen > 0)
    ",
        CompatibilityProfile::Lua51,
    );
}

#[test]
fn single_argument_calls_keep_numeric_and_fallback_semantics() {
    run_in_interpreter_and_tracing(
        r"
        local function twice(value) return value * 2 end
        local function branch(value)
            if value < 0 then return -value end
            return value + 1
        end
        local function first(...) local value = ...; return value + 3 end
        local captured = 4
        local function add(value) return value + captured end
        local object = setmetatable({}, {__mul = function() return 17 end})
        for i = 1, 2000 do
            assert(twice(i) == i * 2)
            assert(twice('6') == 12)
            assert(twice(object) == 17)
            assert(branch(-i) == i)
            assert(branch(i) == i + 1)
            assert(first(i) == i + 3)
            assert(add(i) == i + 4)
        end
        captured = '8'
        assert(add(2) == 10)
        local function needs_two(first, second) return first + second end
        assert(not pcall(function() return needs_two(1) end))
    ",
    );
}

#[test]
fn callback_continuations_keep_values_alive_across_gc_and_errors() {
    run_in_interpreter_and_tracing(
        r"
        local values = {5, 1, 4, 2, 3}
        table.sort(values, function(left, right)
            collectgarbage('collect')
            return left < right
        end)
        assert(table.concat(values, ',') == '1,2,3,4,5')
        local ok, message = pcall(function()
            table.sort({3, 2, 1}, function()
                collectgarbage('collect')
                error('sort failure')
            end)
        end)
        assert(not ok and string.find(message, 'sort failure', 1, true))

        local replacements = {a = 'one', b = 'two', c = 'three'}
        local replaced, count = string.gsub('abc', '.', function(character)
            collectgarbage('collect')
            return replacements[character]
        end)
        assert(replaced == 'onetwothree' and count == 3)
        ok, message = pcall(function()
            string.gsub('abc', '.', function()
                collectgarbage('collect')
                error('replacement failure')
            end)
        end)
        assert(not ok and string.find(message, 'replacement failure', 1, true))

        local object = setmetatable({}, {
            __index = function(_, key)
                collectgarbage('collect')
                return replacements[key]
            end,
            __add = function()
                collectgarbage('collect')
                return 42
            end,
            __concat = function()
                collectgarbage('collect')
                return 'joined'
            end,
            __tostring = function()
                collectgarbage('collect')
                return 'object'
            end,
        })
        assert(object.b == 'two')
        assert(object + object == 42)
        assert('prefix:' .. object .. object == 'prefix:joined')
        assert(tostring(object) == 'object')
    ",
    );
}

#[test]
fn many_capture_closures_keep_shared_cells_during_churn() {
    run_in_interpreter_and_tracing(
        r"
        local function factory(first, second, third)
            local function add(delta)
                first = first + delta
                second = second + delta * 2
                third = third + delta * 3
                return first + second + third
            end
            return add, function() return first, second, third end
        end
        local total = 0
        for i = 1, 2000 do
            local add, read = factory(i, i * 2, i * 3)
            total = total + add(1)
            local first, second, third = read()
            assert(first == i + 1 and second == i * 2 + 2 and third == i * 3 + 3)
            if i % 200 == 0 then collectgarbage('collect') end
            assert(add(1) == i * 6 + 12)
        end
        assert(total == 12018000)
    ",
    );
}

#[test]
fn suspended_closures_rebind_single_and_shared_capture_sets() {
    let source = r"
        local function factory(value)
            return function(pause) pause(); return value end,
                   function() return value end
        end
        local function multi_factory(value, bonus)
            return function(pause) pause(); return value + bonus end
        end
        local function upvalue_index(fn, wanted)
            for index = 1, 20 do
                local name = debug.getupvalue(fn, index)
                if name == wanted then return index end
            end
            error('missing upvalue')
        end
        local suspended, old_reader = factory(10)
        local unused, replacement = factory(30)
        local multiple = multi_factory(20, 5)
        local first = coroutine.create(suspended)
        local second = coroutine.create(multiple)
        assert(coroutine.resume(first, coroutine.yield))
        assert(coroutine.resume(second, coroutine.yield))
        local replacement_index = upvalue_index(replacement, 'value')
        debug.upvaluejoin(suspended, upvalue_index(suspended, 'value'),
                          replacement, replacement_index)
        debug.upvaluejoin(multiple, upvalue_index(multiple, 'value'),
                          replacement, replacement_index)
        debug.setupvalue(replacement, replacement_index, 40)
        collectgarbage('collect')
        local ok, value = coroutine.resume(first)
        assert(ok and value == 40)
        ok, value = coroutine.resume(second)
        assert(ok and value == 45)
        assert(old_reader() == 10)
    ";
    run_in_profiled_interpreter_and_tracing(source, CompatibilityProfile::Lua55);
}

#[test]
fn captured_numeric_loop_stays_native_after_warmup() {
    let chunk = compile(
        r"
        local function make_counter()
            local value = 0
            return function(delta, extra)
                value = value + delta + extra
                return value
            end
        end
        local counter = make_counter()
        local total = 0
        for i = 1, 20000 do total = total + counter(1, 0) end
        return total
        ",
    )
    .unwrap();
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    // Each run's loop alone is too short to repay a compile; the test is
    // about the trace staying native across runs.
    vm.ignore_remaining_work();
    install_base(vm.runtime_mut()).unwrap();
    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![oximond::runtime::Value::Number(200_010_000.0)]
        );
    }
    // A trace inlines the call.
    let native_backedges = |statistics: &oximond::vm::JitStatistics| statistics.trace.iterations;
    let warmup = vm.jit_statistics();
    assert!(native_backedges(&warmup) > 0, "{warmup:?}");
    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![oximond::runtime::Value::Number(200_010_000.0)]
        );
    }
    let statistics = vm.jit_statistics();
    assert!(
        native_backedges(&statistics) - native_backedges(&warmup) > 150_000,
        "{statistics:?}"
    );
}

#[test]
fn captured_numeric_calls_preserve_active_and_shared_cells() {
    let source = r"
        local value = 0
        local function add(delta)
            value = value + delta
            return value
        end
        local total = 0
        for i = 1, 20000 do
            value = value + 10
            total = total + add(1)
        end
        assert(value == 220000 and total == 2200110000)

        local function pair()
            local shared = 0
            local function read() return shared end
            local function first(delta)
                shared = shared + delta
                return shared
            end
            local function second(delta)
                shared = shared + delta * 2
                return shared
            end
            return first, second, read
        end
        local first, second, read = pair()
        total = 0
        local observed = 0
        for i = 1, 20000 do
            total = total + first(1)
            observed = observed + read()
            total = total + second(1)
        end
        assert(first(0) == 60000 and total == 1200020000)
        assert(read() == 60000 and observed == 599990000)
    ";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        run_in_profiled_interpreter_and_tracing(source, profile);
    }
}

#[test]
fn fixed_tail_arguments_preserve_owned_values_nil_slots_and_fallbacks() {
    let source = r##"
        local token = {}
        local text = "owned-" .. tostring(17)
        local function rotate(n, a, b, c)
            if n == 0 then return a, b, c end
            return rotate(n - 1, b, c, a)
        end
        local a, b, c = rotate(3000, text, token, nil)
        assert(a == text and b == token and c == nil)
        assert(select("#", rotate(3000, text, token, nil)) == 3)
        local function values(...) return ... end
        local function zero() return values() end
        local function four() return values(text, nil, token, false) end
        local function five() return values(text, nil, token, false, 5) end
        assert(select("#", zero()) == 0)
        assert(select("#", four()) == 4)
        assert(select("#", five()) == 5)
        a, b, c = five()
        assert(a == text and b == nil and c == token)
        local callable
        callable = setmetatable({}, {
            __call = function(self, first, second)
                assert(self == callable)
                return first, second
            end
        })
        local function invoke() return callable(text, token) end
        a, b = invoke()
        assert(a == text and b == token)
        local function native() return type(token) end
        assert(native() == "table")
        collectgarbage("collect")
        assert(rotate(3000, text, token, nil) == text)
    "##;
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        run_in_profiled_interpreter_and_tracing(source, profile);
    }
}

#[test]
fn inline_plan_refusals_do_not_leak_between_live_callees() {
    run_in_interpreter_and_tracing(
        r"
        local function branch(value)
            if value < 0 then return -value end
            return value * 2
        end
        local function direct(value) return value + 1 end
        local function invoke(function_value, value) return function_value(value) end
        for i = 1, 10000 do
            assert(invoke(branch, i) == i * 2)
            assert(invoke(direct, i) == i + 1)
        end
        assert(invoke(branch, -3) == 3)
        local function recurse(value)
            if value == 0 then return 0 end
            return 1 + recurse(value - 1)
        end
        assert(invoke(recurse, 100) == 100)
        local ok, message = pcall(invoke, branch, {})
        assert(not ok and message ~= nil)
        assert(invoke(direct, 9) == 10)
    ",
    );
}

#[test]
fn cached_closure_plans_read_rebound_cells_and_distinct_instances() {
    let source = r#"
        local function counter(value)
            return function(delta)
                value = value + delta
                return value
            end
        end
        local first, second = counter(0), counter(100000)
        for i = 1, 20000 do assert(first(1) == i) end
        for i = 1, 20000 do assert(second(2) == 100000 + i * 2) end
        assert(first(1) == 20001)
        assert(debug.setupvalue(first, 1, 7) ~= nil)
        assert(first(3) == 10)
        assert(second(0) == 140000)
        assert(debug.setupvalue(first, 1, "not a number") ~= nil)
        assert(not pcall(first, 1))
        debug.setupvalue(first, 1, 20)
        assert(first(2) == 22)
        if debug.upvaluejoin then
            debug.upvaluejoin(first, 1, second, 1)
            assert(first(3) == 140003 and second(1) == 140004)
        end
        for i = 1, 1000 do
            local temporary = counter(i)
            assert(temporary(1) == i + 1)
        end
        collectgarbage("collect")
        local replacement = counter(8)
        assert(replacement(2) == 10)
    "#;
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        run_in_profiled_interpreter_and_tracing(source, profile);
    }
}

#[test]
fn register_move_runs_preserve_parallel_assignment_and_captured_values() {
    run_in_interpreter_and_tracing(
        r#"
        local a, b, c = 1, 2, 3
        for i = 1, 30000 do a, b, c = b, c, a end
        assert(a == 1 and b == 2 and c == 3)
        local function captured()
            local x, y = "left", "right"
            local function read() return x, y end
            for i = 1, 1001 do x, y = y, x end
            local first, second = read()
            assert(first == "right" and second == "left")
        end
        captured()
        local events = 0
        debug.sethook(function() events = events + 1 end, "", 1)
        local x, y, z = a, b, c
        x, y, z = z, x, y
        debug.sethook()
        assert(events > 0 and x == 3 and y == 1 and z == 2)
    "#,
    );
}

#[test]
fn boolean_table_loops_complete_backedges_in_native_code() {
    let chunk = compile(
        r"
        local flags = {}
        local count = 0
        for run = 1, 3 do
        for i = 1, 20000 do flags[i] = true end
        count = 0
        for i = 1, 20000 do
            if flags[i] then count = count + 1 end
            flags[i] = false
        end
        assert(count == 20000)
        assert(flags[1] == false and flags[20000] == false)
        end
        return count
    ",
    )
    .expect("compile Boolean table loops");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).expect("install standard library");
    vm.execute(&chunk).expect("execute Boolean table loops");
    let statistics = vm.jit_statistics();
    assert!(statistics.trace.iterations > 20000, "{statistics:?}");
}

#[test]
fn boolean_table_specializations_preserve_type_changes_and_metamethods() {
    run_in_interpreter_and_tracing(
        r#"
        local flags = {}
        for i = 1, 2000 do flags[i] = true end
        local function count()
            local total = 0
            for i = 1, 2000 do
                if flags[i] then total = total + 1 end
            end
            return total
        end
        for run = 1, 10 do assert(count() == 2000) end
        flags[500] = false
        flags[1000] = nil
        flags[1500] = 0
        flags[1600] = ""
        assert(count() == 1998)
        local reads, writes = 0, 0
        setmetatable(flags, {
            __index = function() reads = reads + 1; return true end,
            __newindex = function(t, key, value)
                writes = writes + 1
                rawset(t, key, value)
            end
        })
        assert(count() == 1999 and reads == 1)
        for i = 1, 2000 do flags[i] = false end
        assert(writes == 1 and count() == 0)
        setmetatable(flags, nil)
        for i = 1, 2000 do flags[i] = i end
        assert(count() == 2000)
        for i = 1, 2000 do flags[i] = false end
        assert(count() == 0)
        local sparse = {}
        for i = 1, 2000 do sparse[i * 1000] = true end
        assert(sparse[1000000] == true and sparse[1] == nil)
    "#,
    );
}

#[test]
fn sparse_sieve_writes_keep_the_prime_count() {
    let chunk = compile(
        r"
        local result = 0
        for run = 1, 3 do
            local limit = 10000
            local composite = {}
            for i = 2, limit do
                if not composite[i] then
                    result = result + 1
                    local multiple = i * i
                    while multiple <= limit do
                        composite[multiple] = true
                        multiple = multiple + i
                    end
                end
            end
        end
        assert(result == 3687)
    ",
    )
    .expect("compile sparse sieve");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).expect("install standard library");
    vm.execute(&chunk).expect("execute sparse sieve");
}

fn run_in_profiled_interpreter_and_tracing(source: &str, profile: CompatibilityProfile) {
    let chunk = compile_with_config(source, CompilerConfig::new(profile))
        .expect("compile profiled regression program");
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
        vm.use_synchronous_jit_compilation();
        install_base_with_profile(vm.runtime_mut(), profile)
            .expect("install profiled standard library");
        vm.execute(&chunk).expect("run profiled regression program");
    }
}

#[test]
fn loop_tail_upvalue_load_keeps_the_live_loop_index() {
    // The fill loop's region extends past its exit into the `_ENV` load for
    // `ipairs`, whose destination reuses the numeric `for` index register.
    // Materializing that upvalue at entry used to overwrite the live index,
    // so the interpreter resumed the loop on a table.
    let source = r"
        local v = {}
        for i = 1, 20000 do v[i] = i end
        local f, t = ipairs(v)
        local s = 0
        for i, x in f, t, 0.0 do s = s + x end
        assert(s == 200010000)
        local w = {}
        for i = 1, 20000 do w[i] = i end
        s = 0
        for i, x in ipairs(w) do s = s + x end
        assert(s == 200010000)
    ";
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua55,
    ] {
        run_in_profiled_interpreter_and_tracing(source, profile);
    }
}

#[test]
fn ipairs_fixed_calls_use_the_allocation_free_iterator_path() {
    let source = r"
        local values = {}
        for i = 1, 20000 do values[i] = i end
        local sum = 0
        for i, value in ipairs(values) do sum = sum + value end
        assert(sum == 200010000)
    ";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        let chunk = compile_with_config(source, CompilerConfig::new(profile)).unwrap();
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
            vm.use_synchronous_jit_compilation();
            install_base_with_profile(vm.runtime_mut(), profile).unwrap();
            vm.execute(&chunk).unwrap();
            let statistics = vm.jit_statistics();
            assert!(
                statistics.interpreter_ipairs_steps + statistics.trace.iterations >= 20000,
                "{statistics:?}"
            );
        }
    }
}

#[test]
fn numeric_ipairs_loops_complete_backedges_in_native_code() {
    let chunk = compile(
        r"
        local values = {}
        for i = 1, 20000 do values[i] = i end
        local sum = 0
        for run = 1, 10 do
            for index, value in ipairs(values) do sum = sum + value end
        end
        assert(sum == 2000100000)
    ",
    )
    .unwrap();
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).unwrap();
    vm.execute(&chunk).unwrap();
    let statistics = vm.jit_statistics();
    assert!(statistics.trace.iterations > 100_000, "{statistics:?}");
}

#[test]
fn fused_ipairs_steps_observe_mutation_during_traversal() {
    // Native `ipairs` steps read the generic-for hidden registers directly
    // and only materialize the call window when they fall back. Each round
    // mutates the table mid-traversal so steps refuse at different points.
    let source = r#"
        local total, seen = 0, 0
        for round = 1, 40 do
            local t = {}
            for i = 1, 300 do t[i] = i end
            for i, v in ipairs(t) do
                seen = seen + 1
                if type(v) == "string" then total = total + #v else total = total + v end
                if i == 50 + round then t[i + 1] = "x" .. i end
                if i == 200 then t[#t + 1] = 1 end
                if i == 250 + round then t[i + 2] = nil end
            end
        end
        assert(seen == 10860, seen)
        assert(total == 1479600, total)
    "#;
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        run_in_profiled_interpreter_and_tracing(source, profile);
    }
    // Lua 5.1 `ipairs` reads raw: a hole ends the traversal even when
    // `__index` would supply a value.
    let raw = r"
        local t = setmetatable({1, 2, nil, 4}, {__index = function(_, k) return k * 10 end})
        local count = 0
        for i = 1, 200 do
            for _, v in ipairs(t) do count = count + v end
        end
        assert(count == 600, count)
    ";
    run_in_profiled_interpreter_and_tracing(raw, CompatibilityProfile::Lua51);
}

#[test]
fn native_ipairs_falls_back_on_holes_type_changes_and_iterator_rebinding() {
    run_in_interpreter_and_tracing(
        r#"
        local values = {}
        for i = 1, 20000 do values[i] = i end
        local function sum(t)
            local total = 0
            for index, value in ipairs(t) do total = total + value end
            return total
        end
        for run = 1, 4 do assert(sum(values) == 200010000) end
        values[10000] = nil
        assert(sum(values) == 49995000)
        values[10000] = false
        assert(not pcall(sum, values))
        values[10000] = 10000
        assert(sum(values) == 200010000)
        local original = ipairs
        ipairs = function(t)
            return function(state, control)
                if control == 0 then return 1, 7 end
            end, t, 0
        end
        assert(sum(values) == 7)
        ipairs = original
        assert(sum(values) == 200010000)
        assert(sum({2, 3, 4}) == 9)
        setmetatable(values, {})
        assert(sum(values) == 200010000)
        local calls = 0
        debug.sethook(function() calls = calls + 1 end, "c")
        assert(sum({1, 2}) == 3)
        debug.sethook()
        assert(calls > 0)
    "#,
    );
}

#[test]
fn native_ipairs_moves_preserve_receivers_across_mutations_and_exits() {
    let source = r#"
        local first, second = {}, {}
        for i = 1, 4000 do first[i] = i; second[i] = i * 2 end
        local holder = { values = first, label = "first" }
        local function sum()
            local total = 0
            for index, value in ipairs(holder.values) do
                total = total + math.abs(value)
                if index == 2000 then
                    holder.values = second
                    holder.label = "second"
                end
            end
            return total, holder.label
        end
        for run = 1, 10 do
            holder.values = first
            holder.label = "first"
            local total, label = sum()
            assert(total == 8002000 and label == "second")
            assert(sum() == 16004000)
        end
        local iterator = ipairs(first)
        local function traverse(next_value, state)
            local total = 0
            for index, value in next_value, state, 0 do
                total = total + value
                if index == 2000 then state[2001] = nil end
            end
            return total
        end
        for run = 1, 10 do
            first[2001] = 2001
            assert(traverse(iterator, first) == 2001000)
        end
        local replacement = function(state, index)
            if index == 0 then return 1, 13 end
        end
        assert(traverse(replacement, second) == 13)
        assert(second[2001] == 4002)
    "#;
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        run_in_profiled_interpreter_and_tracing(source, profile);
    }
}

#[test]
fn ipairs_fixed_calls_preserve_values_result_counts_and_hooks() {
    let source = r##"
        local values = {false, "text", {}, 0, nil, "after hole"}
        local count = 0
        for index, value in ipairs(values) do
            count = count + 1
            assert(value == values[index])
        end
        assert(count == 4)
        local iterator, state, control = ipairs(values)
        local key, value = iterator(state, control)
        assert(key == 1 and value == false)
        key, value = iterator(state, 4)
        assert(key == nil and value == nil)
        assert(select("#", iterator(state, 4)) == 1)
        local a, b, c = iterator(state, 1)
        assert(a == 2 and b == "text" and c == nil)
        assert(not pcall(iterator, false, 0))
        assert(not pcall(iterator, state, {}))
        local replacement = function(t, i)
            if i == 0 then return 1, "replacement" end
        end
        for index, value in replacement, values, 0 do
            assert(index == 1 and value == "replacement")
        end
        setmetatable(values, {})
        key, value = iterator(values, 0)
        assert(key == 1 and value == false)
        local events = 0
        debug.sethook(function() events = events + 1 end, "c")
        key, value = iterator(values, 1)
        debug.sethook()
        assert(key == 2 and value == "text" and events > 0)
        setmetatable(values, nil)
        for index, value in ipairs(values) do
            if index == 1 then values[2] = nil end
        end
        key, value = iterator(values, 1)
        assert(key == nil and value == nil)
    "##;
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        run_in_profiled_interpreter_and_tracing(source, profile);
    }
}

#[test]
fn scalar_natives_preserve_results_rebinding_errors_and_hooks() {
    let source = r##"
        local original = math.sqrt
        local function check()
            local a, b, c = math.sqrt("9", "ignored")
            assert(a == 3 and b == nil and c == nil)
            local part, extra = string.sub("abc", 2, 2)
            assert(part == "b" and extra == nil)
            assert(select("#", string.sub("abc", 4)) == 1)
            assert(select("#", string.len("abc")) == 1)
            assert(select("#", math.sin(0)) == 1)
            local function open() return math.sqrt(16) end
            assert(select("#", open()) == 1 and open() == 4)
            local ok, message = pcall(function()
                local value = math.sqrt({})
                return value
            end)
            assert(not ok and type(message) == "string")
        end
        check()
        local events = 0
        debug.sethook(function() events = events + 1 end, "l")
        check()
        debug.sethook()
        assert(events > 0)
        math.sqrt = function() return 7, 8 end
        local a, b = math.sqrt(9)
        assert(a == 7 and b == 8)
        math.sqrt = original
        assert(math.sqrt(9) == 3)
        if math.maxinteger then
            assert(math.floor(math.maxinteger) == math.maxinteger)
        end
    "##;
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        run_in_profiled_interpreter_and_tracing(source, profile);
    }
}

#[test]
fn tracing_math_unary_calls_keep_dynamic_lua_semantics() {
    let chunk = compile(
        r#"
        local function hot(count)
            local total = 0
            for i = 1, count do total = total + math.sin(i) end
            return total
        end

        local first = hot(20000)
        assert(type(first) == "number" and first ~= 0)
        assert(math.sin("0") == 0)
        assert(not pcall(function() return math.sin({}) end))

        local original = math.sin
        math.sin = function(value) return value * 2 end
        assert(hot(4) == 20)
        math.sin = original

        local sandbox = { math = { sin = function(value) return value + 3 end } }
        local function from_environment() return math.sin(2) end
        setfenv(from_environment, sandbox)
        assert(from_environment() == 5)

        local events = 0
        debug.sethook(function() events = events + 1 end, "l")
        assert(type(hot(3)) == "number")
        debug.sethook()
        assert(events > 0)
        return first
        "#,
    )
    .expect("compile native unary regression program");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).expect("install standard library");
    for _ in 0..8 {
        let values = vm
            .execute(&chunk)
            .expect("run native unary regression program");
        assert!(
            matches!(values.as_slice(), [oximond::runtime::Value::Number(value)] if value.is_finite())
        );
    }
    let statistics = vm.jit_statistics();
    // The trace calls `math.sin` itself, so every call stays in native code.
    assert!(statistics.trace.iterations > 20000, "{statistics:?}");
}

#[test]
fn inline_numeric_calls_preserve_fixed_and_open_result_counts() {
    run_in_interpreter_and_tracing(
        r##"
        local function add(a, b) return a + b end
        for i = 1, 200 do
            local first, second, third = add(i, 2)
            assert(first == i + 2 and second == nil and third == nil)
            assert(select("#", add(i, 2)) == 1)
            local function forward() return add(i, 2) end
            assert(select("#", forward()) == 1 and forward() == i + 2)
        end
        "##,
    );
}

#[test]
fn interpreter_stages_raw_table_method_calls_and_keeps_lua_fallbacks_visible() {
    let chunk = compile(
        r"
        local function step(self, delta)
            self.value = self.value + delta
            return self.value
        end
        local item = { value = 1 }
        for i = 1, 32 do assert(step(item, i) == item.value) end
        return item.value
        ",
    )
    .expect("compile raw table method regression program");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Interpreter);
    install_base(vm.runtime_mut()).expect("install standard library");
    assert_eq!(
        vm.execute(&chunk)
            .expect("run raw table method regression program"),
        vec![oximond::runtime::Value::Number(529.0)]
    );
    assert!(vm.jit_statistics().interpreter_inline_table_calls > 0);

    run_in_interpreter_and_tracing(
        r#"
        local function step(self, delta)
            self.value = self.value + delta
            return self.value
        end
        local index_hits, add_hits, newindex_hits = 0, 0, 0
        local missing = setmetatable({}, {
            __index = function() index_hits = index_hits + 1; return 4 end,
        })
        assert(step(missing, 2) == 6 and index_hits == 1)
        local additive_value = setmetatable({}, {
            __add = function() add_hits = add_hits + 1; return 9 end,
        })
        local additive = { value = additive_value }
        assert(step(additive, 2) == 9 and add_hits == 1)
        local writable = setmetatable({ value = 1 }, {
            __index = function() return 0 end,
            __newindex = function(tbl, key, value) newindex_hits = newindex_hits + 1; rawset(tbl, "value", value) end,
        })
        writable.value = nil
        assert(step(writable, 2) == 2 and newindex_hits == 1)
        local hooks = 0
        debug.sethook(function() hooks = hooks + 1 end, "c")
        assert(step({ value = 1 }, 2) == 3)
        debug.sethook()
        assert(hooks > 0)
        local replacement = function(self, delta)
            self.value = self.value + delta * 2
            return self.value
        end
        step = replacement
        assert(step({ value = 1 }, 2) == 5)
        "#,
    );
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        run_in_profiled_interpreter_and_tracing(
            r"
            local function step(self, delta)
                self.value = self.value + delta
                return self.value
            end
            local item = { value = 1 }
            assert(step(item, 2) == 3)
            assert(step(item, 0.5) == 3.5)
            ",
            profile,
        );
    }
    run_in_profiled_interpreter_and_tracing(
        r"
        local function fail_after_write(self)
            self.value = 7
            return self.value // 0
        end
        local error_item = { value = 1 }
        assert(not pcall(fail_after_write, error_item) and error_item.value == 7)
        ",
        CompatibilityProfile::Lua55,
    );
}

#[test]
fn tracing_vararg_loop_reaches_native_code() {
    let chunk = compile(
        r"
        local function blend(a, b, c, d, ...)
            return a + b + c + d + (...)
        end
        local total = 0
        for i = 1, 16384 do
            total = total + blend(i, 1, 2, 3, 4)
        end
        return total
        ",
    )
    .expect("compile vararg inline regression program");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    // The loop runs once per execution of a main chunk, with too little
    // work to repay a compile; this checks what its trace does.
    vm.ignore_remaining_work();
    install_base(vm.runtime_mut()).expect("install standard library");
    let expected = 134_389_760.0;
    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk)
                .expect("run vararg inline regression program"),
            vec![oximond::runtime::Value::Number(expected)]
        );
    }
    let warmup = vm.jit_statistics();
    assert!(warmup.trace.iterations > 0, "{warmup:?}");
    // Compilation runs on a background worker. Check steady execution after
    // installation instead of bounding how fast that worker gets scheduled.
    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![oximond::runtime::Value::Number(expected)]
        );
    }
    let statistics = vm.jit_statistics();
    assert!(
        statistics.trace.iterations > warmup.trace.iterations,
        "{statistics:?}"
    );
    assert!(
        statistics.interpreter_inline_calls - warmup.interpreter_inline_calls < 4_096,
        "{statistics:?}"
    );
}

#[test]
fn tracing_nested_hash_reads_keep_the_hot_loop_native() {
    let chunk = compile(
        r#"
        local keys = { "left", "right" }
        local values = { left = 3, right = 5 }
        local total = 0
        for i = 1, 8192 do
            local key = keys[(i % 2) + 1]
            total = total + values[key]
        end
        return total
        "#,
    )
    .expect("compile nested hash read regression program");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).expect("install standard library");
    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk)
                .expect("run nested hash read regression program"),
            vec![oximond::runtime::Value::Number(32_768.0)]
        );
    }
}

#[test]
fn arithmetic_inner_loop_runs_natively_across_effectful_outer_calls() {
    let profile = CompatibilityProfile::Lua51;
    let source = include_str!("../../../benchmarks/lua/control/nested_calls.lua")
        .replace("1000000 * scale", "20000 * scale")
        .replace("print(total)", "return total");
    let chunk = compile_with_config(&source, CompilerConfig::new(profile)).unwrap();
    let mut interpreter = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    install_base_with_profile(interpreter.runtime_mut(), profile).unwrap();
    let expected = interpreter.execute(&chunk).unwrap();
    let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    tracing.use_synchronous_jit_compilation();
    install_base_with_profile(tracing.runtime_mut(), profile).unwrap();
    for _ in 0..4 {
        assert_eq!(tracing.execute(&chunk).unwrap(), expected);
    }
    let statistics = tracing.jit_statistics();
    assert!(statistics.trace.iterations > 0, "{statistics:?}");
    assert!(statistics.trace.traces_compiled > 0, "{statistics:?}");
}

#[test]
fn inline_table_deletion_preserves_index_metamethods() {
    run_in_interpreter_and_tracing(
        r#"
        local reads = 0
        local t = setmetatable({x = 1}, {
            __index = function() reads = reads + 1; return 99 end
        })
        local function remove(receiver)
            receiver.x = nil
            return receiver.x
        end
        for i = 1, 200 do
            rawset(t, "x", i)
            assert(remove(t) == 99)
            assert(rawget(t, "x") == nil)
        end
        assert(reads == 200)
        "#,
    );
}

#[test]
fn repeated_calls_keep_argument_slots_nil_filled_across_frame_sizes() {
    run_in_interpreter_and_tracing(
        r"
        local function narrow(first, second, third, fourth)
            assert(first == 7 and second == nil and third == nil and fourth == nil)
        end
        local function wide(first, second, third, fourth, fifth, sixth, seventh, eighth)
            assert(first == 9 and eighth == nil)
            assert(second == nil and third == nil and fourth == nil)
            assert(fifth == nil and sixth == nil and seventh == nil)
        end
        for i = 1, 200 do
            narrow(7)
            wide(9)
        end
        ",
    );
}

#[test]
fn returned_captures_and_error_unwinds_survive_aggressive_frame_reuse() {
    run_in_interpreter_and_tracing(
        r#"
        local function make_keeper(value)
            local padding1, padding2, padding3, padding4 = value, value, value, value
            return function() return value end
        end
        local keeper = make_keeper("kept")
        local function fail()
            local a, b, c, d, e, f, g, h = 1, 2, 3, 4, 5, 6, 7, 8
            error("expected")
        end
        local function reuse(value)
            local a, b, c, d, e, f, g, h = value, value, value, value, value, value, value, value
            return a
        end
        for i = 1, 200 do
            assert(not pcall(fail))
            assert(reuse(i) == i)
        end
        collectgarbage("collect")
        assert(keeper() == "kept")
        "#,
    );
}

#[test]
fn vararg_forwarding_keeps_nil_positions_after_many_calls() {
    run_in_interpreter_and_tracing(
        r#"
        local function forward(prefix, ...)
            return prefix, ...
        end
        for i = 1, 200 do
            local first, second, third, fourth = forward("first", nil, "third", nil)
            assert(first == "first" and second == nil)
            assert(third == "third" and fourth == nil)
        end
        "#,
    );
}

#[test]
fn legacy_vararg_table_survives_repeated_fixed_arity_closure_calls() {
    run_in_profiled_interpreter_and_tracing(
        r#"
        local function inspect(first, second, third, ...)
            assert(first == "first" and second == nil and third == "third")
            assert(arg.n == 2 and arg[1] == nil and arg[2] == "tail")
            return first, second, third, arg[1], arg[2]
        end
        for i = 1, 200 do
            local a, b, c, d, e = inspect("first", nil, "third", nil, "tail")
            assert(a == "first" and b == nil and c == "third")
            assert(d == nil and e == "tail")
        end
        "#,
        CompatibilityProfile::Lua51,
    );
}

#[test]
fn forwarded_numeric_varargs_elide_the_tail_callee_frame() {
    let chunk = compile(
        r"local function sum(a,b,c,d,e) return a+b+c+d+e end
        local function forward(...) return sum(...) end
        local total = 0
        for i = 1, 2000 do total = total + forward(i,2,3,4,5) end
        assert(total == 2029000)
        local ok = pcall(function() return forward(1,2) end)
        assert(not ok)",
    )
    .unwrap();
    let mut interpreter = Vm::with_execution_mode(ExecutionMode::Interpreter);
    install_base(interpreter.runtime_mut()).unwrap();
    interpreter.execute(&chunk).unwrap();

    let mut tracing = Vm::with_execution_mode(ExecutionMode::Tracing);
    tracing.use_synchronous_jit_compilation();
    // The loop runs once per execution of a main chunk, with too little
    // work to repay a compile; this checks what its trace does.
    tracing.ignore_remaining_work();
    install_base(tracing.runtime_mut()).unwrap();
    for _ in 0..8 {
        tracing.execute(&chunk).unwrap();
    }
    let warmup = tracing.jit_statistics();
    for _ in 0..8 {
        tracing.execute(&chunk).unwrap();
    }
    let statistics = tracing.jit_statistics();
    assert!(
        statistics.trace.iterations > warmup.trace.iterations,
        "{statistics:?}"
    );
    assert!(
        statistics.interpreter_inline_calls - warmup.interpreter_inline_calls < 4_000,
        "{statistics:?}"
    );
}

#[test]
fn forwarded_numeric_varargs_respecialize_after_target_rebinding() {
    run_in_interpreter_and_tracing(
        r"
        local function sum(a, b, c, d, e) return a + b + c + d + e end
        local function forward(...) return sum(...) end
        local function run()
            local total = 0
            for i = 1, 16000 do
                total = total + forward(i, 2, 3, 4, 5)
            end
            return total
        end
        assert(run() == 128232000)
        sum = function(a) return a end
        assert(run() == 128008000)
        ",
    );
}

#[test]
fn fixed_tail_replacement_preserves_arguments_and_environments() {
    run_in_profiled_interpreter_and_tracing(
        r"
        local marker = {}
        local function target(a, b, c, d)
            assert(a == marker and b == 'payload' and c == nil and d == nil)
            return answer
        end
        setfenv(target, {assert = assert, answer = 73})
        local function forward(...)
            return target(marker, 'payload')
        end
        assert(forward(1, 2, 3) == 73)
        local function pair(a, b)
            return a, b
        end
        local function swap(a, b)
            return pair(b, a, marker, 'discarded')
        end
        local first, second = swap('left', marker)
        assert(first == marker and second == 'left')
        local saved
        local function captured(a)
            saved = function() return a end
            return pair('new', marker)
        end
        local a, b = captured('old')
        collectgarbage('collect')
        assert(a == 'new' and b == marker and saved() == 'old')
        ",
        CompatibilityProfile::Lua51,
    );
}

#[test]
fn tail_calls_keep_hooks_close_order_and_captured_cells_observable() {
    run_in_interpreter_and_tracing(
        r#"
        local calls, returns = 0, 0
        local function target(value) return value + 1 end
        local function forward(value) return target(value) end
        debug.sethook(function(event)
            if event == "call" then calls = calls + 1 end
            if event == "return" then returns = returns + 1 end
        end, "cr")
        assert(forward(4) == 5)
        debug.sethook()
        assert(calls >= 2 and returns >= 2)

        local total = 0
        local function add(value)
            total = total + value
            return total
        end
        local function forward_add(value) return add(value) end
        assert(forward_add(2) == 2)
        assert(forward_add(3) == 5)
        "#,
    );

    run_in_profiled_interpreter_and_tracing(
        r"
        local value = 1
        local function read_value() return value end
        local function close_then_tail()
            local guard <close> = setmetatable({}, {
                __close = function() value = 9 end,
            })
            return read_value()
        end
        assert(close_then_tail() == 9)

        local counter = 0
        local function mutate_closed_cell(value)
            counter = counter + value
            return counter
        end
        local function tail_mutate(value) return mutate_closed_cell(value) end
        assert(tail_mutate(2) == 2 and tail_mutate(3) == 5)
        ",
        CompatibilityProfile::Lua55,
    );
}

#[test]
fn open_tail_varargs_preserve_nil_positions_and_result_counts() {
    run_in_interpreter_and_tracing(
        r##"
        local function target(...)
            return nil, ...
        end
        local function forward(...)
            return target(...)
        end
        for i = 1, 200 do
            local a, b, c, d, e = forward("first", nil, "third")
            assert(a == nil and b == "first" and c == nil)
            assert(d == "third" and e == nil)
            assert(select("#", forward("first", nil, "third")) == 4)
        end
        "##,
    );
}

#[test]
fn suspended_debug_local_edits_reach_captured_cells_before_resume() {
    run_in_interpreter_and_tracing(
        r#"
local thread = coroutine.create(function()
    local value = 1
    local function read() return value end
    coroutine.yield(read)
    return value, read()
end)
local ok, read = coroutine.resume(thread)
assert(ok)
local changed = false
for index = 1, 20 do
    local name = debug.getlocal(thread, 1, index)
    if name == "value" then
        assert(debug.setlocal(thread, 1, index, 9) == "value")
        changed = true
        break
    end
end
assert(changed)
assert(read() == 9)
local resumed, local_value, captured_value = coroutine.resume(thread)
assert(resumed and local_value == 9 and captured_value == 9)
"#,
    );
}

#[test]
fn raw_table_method_loop_stays_native_after_warmup() {
    let chunk = compile(
        r"
        local item = { value = 0 }
        function item:add(delta)
            self.value = self.value + delta
            return self.value
        end
        local total = 0
        for i = 1, 20000 do total = total + item:add(1) end
        return total
    ",
    )
    .unwrap();
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    // Each run's loop alone is too short to repay a compile; the test is
    // about the trace staying native across runs.
    vm.ignore_remaining_work();
    install_base(vm.runtime_mut()).unwrap();
    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![oximond::runtime::Value::Number(200_010_000.0)]
        );
    }
    let warmup = vm.jit_statistics();
    assert!(warmup.trace.iterations > 0, "{warmup:?}");
    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![oximond::runtime::Value::Number(200_010_000.0)]
        );
    }
    let statistics = vm.jit_statistics();
    assert!(
        statistics.interpreter_inline_table_calls - warmup.interpreter_inline_table_calls < 4000,
        "{statistics:?}"
    );
}

#[test]
fn native_table_method_refusals_do_not_repeat_visible_writes() {
    run_in_interpreter_and_tracing(
        r"
        local item = { value = 0 }
        function item:add(delta)
            self.value = self.value + delta
            return self.value
        end
        local total = 0
        for i = 1, 4000 do
            if i == 2001 then
                item.add = function(self, delta)
                    self.value = self.value + delta * 10
                    return self.value
                end
            end
            total = total + item:add(1)
        end
        assert(item.value == 22000)

        local object = { value = 0 }
        function object:step(delta)
            self.value = self.value + delta
            return self
        end
        for i = 1, 4000 do
            local returned = object:step(1)
            assert(returned == object)
        end
        assert(object.value == 4000)

        local reads, writes = 0, 0
        local missing = setmetatable({}, {
            __index = function(self, key) reads = reads + 1; return 3 end,
            __newindex = function(self, key, value)
                writes = writes + 1
                rawset(self, key, value)
            end,
        })
        local function step(self, delta)
            self.value = self.value + delta
            return self.value
        end
        assert(step(missing, 1) == 4)
        assert(reads == 1 and writes == 1)
    ",
    );
}

#[test]
fn array_appends_and_overwrites_keep_length_order_and_slot_types() {
    run_in_interpreter_and_tracing(
        r#"
        local t = {}
        for i = 1, 300 do t[i] = i * 0.5 end
        assert(#t == 300 and t[300] == 150)
        t[10] = true
        t[11] = "owned"
        for i = 1, 300 do t[i] = i end
        assert(#t == 300 and t[11] == 11 and t[10] == 10)
        t[302] = 1
        assert(#t == 300)
        t[301] = 1
        assert(#t == 302)
        t[5] = nil
        t[5] = 7
        local n, last = 0, 0
        for k, v in pairs(t) do n = n + 1; assert(k > last); last = k end
        assert(n == 302)
        local u = setmetatable({}, { __newindex = function(self, k, v) rawset(self, k, v + 1) end })
        for i = 1, 5 do u[i] = i end
        assert(u[1] == 2 and #u == 5)
        local r = {}
        for i = 1, 8 do r[i] = i end
        r.x = 1
        for i = 9, 16 do r[i] = i end
        assert(#r == 16 and r.x == 1)
        local p = {}
        p[1.5] = 1; p[2^53] = 2; p[1] = 3
        assert(p[1.5] == 1 and p[1] == 3 and #p == 1)
    "#,
    );
}

/// Runs `source` in the interpreter and in tracing mode, checks that both
/// return the same values, and returns the tracing VM's statistics.
fn tracing_statistics_matching_interpreter(source: &str) -> oximond::vm::JitStatistics {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(source, CompilerConfig::new(profile)).expect("compile");
    let mut results = Vec::new();
    let mut statistics = None;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
        vm.use_synchronous_jit_compilation();
        install_base_with_profile(vm.runtime_mut(), profile).expect("install library");
        results.push(vm.execute(&chunk).expect("run"));
        statistics = Some(vm.jit_statistics());
    }
    assert_eq!(results[0], results[1]);
    statistics.unwrap()
}

#[test]
fn upvalue_swapped_from_native_to_lua_closure_keeps_correct_results() {
    tracing_statistics_matching_interpreter(
        r"
        local helper = math.floor
        local function step(x) return helper(x) + 1 end
        local function swap_to_lua() helper = function(x) return x * 2 end end
        local sum = 0
        for i = 1, 30000 do
            if i == 100 then swap_to_lua() end
            sum = (sum + step(i + 0.5)) % 1000003
        end
        return sum
        ",
    );
}

#[test]
fn refused_callees_called_from_captured_frames_see_current_captures() {
    // The chunk's `counter` lives in a register and a cell. Once the
    // callees are refused for native code, tracing calls push them straight
    // from the fast loop, and a call from this frame must still write the
    // register back to the cell first. The callees read and write it, and
    // return one, two, and three results.
    tracing_statistics_matching_interpreter(
        r"
        local counter = 0
        local function bump(n) counter = counter + n return counter end
        local function peek(a) return counter * 2 + a, a end
        local function rest(t, k) t[k] = counter return t[k], counter, k end
        local t = {}
        local sum = 0
        for i = 1, 6000 do
            counter = counter + 1
            local x = bump(0)
            local a, b = peek(i)
            local p, q, r = rest(t, i % 7)
            sum = (sum + x + a + b + p + q + r + bump(1)) % 1000003
        end
        return sum, counter
        ",
    );
}

#[test]
fn loops_the_tracer_gave_up_do_not_leave_the_fast_loop_for_lookups() {
    // Neither loop records (a native without a numeric form takes the open
    // results of the calls and a table store follows it), and the outer one
    // calls the inner one's function now and then, so the backedge skip
    // alternates between two rejected headers. Counting the headers and
    // re-arming the skip both stay in the fast loop.
    let statistics = tracing_statistics_matching_interpreter(
        r#"
        local len, tostring, sub = string.len, tostring, string.sub
        local function low(x) return len(tostring(x % 4294967296)) end
        local function split(a, v)
            a[1] = len(tostring(v))
            return low(a[1] * 3 + 1)
        end
        local function digits(s)
            local t = 0
            for i = 1, #s do t = t * 10 + len(sub(s, i, i)) end
            return t
        end
        local a, sum = { 0 }, 0
        for i = 1, 24000 do
            sum = (sum + split(a, i * 7919 + sum)) % 1000000007
            if i % 32 == 0 then sum = (sum + digits("3948575739")) % 1000000007 end
        end
        return sum
        "#,
    );
    assert_eq!(statistics.trace.traces_recorded, 0, "{statistics:?}");
    assert!(statistics.fast_backedge_lookups <= 100, "{statistics:?}");
}

#[test]
fn library_calls_with_open_results_and_tail_calls_trace_the_hot_loop() {
    // Calls whose results feed another call or a return, tail calls inside
    // inlined callees, and a loop over the bytes of a string: none of it may
    // keep the loops out of native code. Every iteration runs in a trace,
    // and no recording aborts on the hot loops.
    let statistics = tracing_statistics_matching_interpreter(
        r#"
        local floor, abs, max, min = math.floor, math.abs, math.max, math.min
        local byte, sub = string.byte, string.sub
        local tonumber = tonumber
        local function low_word(x) return floor(x % 4294967296) end
        local function split(a, value)
            a[1] = tonumber(floor(value / 65536))
            a[2] = abs(floor(value % 65536))
            return low_word(a[1] * 3 + a[2])
        end
        local function digits(str)
            local total = 0
            for i = 1, #str do
                total = total * 10 + max(min(byte(sub(str, i, i)) - 48, 9), 0)
            end
            return total
        end
        local a, sum, text = { 0, 0 }, 0, "3948575739"
        for i = 1, 24000 do
            sum = (sum + split(a, i * 7919 + sum)) % 1000000007
            if i % 32 == 0 then sum = (sum + digits(text)) % 1000000007 end
        end
        return sum
        "#,
    );
    let trace = &statistics.trace;
    assert!(trace.traces_compiled >= 2, "{statistics:?}");
    // The loop over `split` runs natively, less the iterations the
    // interpreter runs before its trace is installed. The 750 calls of
    // `digits` run in a side of that loop, which unrolls the ten bytes, so
    // they add no iterations and no entries: the loop is entered a few dozen
    // times, not once per call of `digits`.
    assert!(trace.iterations > 23_500, "{statistics:?}");
    assert!(trace.side_installs >= 1, "{statistics:?}");
    assert!(trace.native_entries < 200, "{statistics:?}");
    assert_eq!(
        trace.recording_aborts.iter().sum::<u64>(),
        0,
        "{statistics:?}"
    );
    assert!(statistics.interpreter_inline_calls == 0, "{statistics:?}");
}

#[test]
fn tail_called_closures_reached_through_upvalues_and_fields_trace_the_hot_loop() {
    let statistics = tracing_statistics_matching_interpreter(
        r"
        local floor, fmod = math.floor, math.fmod
        local function within(x, bound) return x % bound end
        local function lcg(seed)
            local state = { seed, 0 }
            return function(bits)
                state[1] = (state[1] * 69069 + 12345) % 4294967296
                state[2] = (state[2] * 31 + state[1]) % 4294967296
                return floor(state[1] / 2 ^ (32 - bits))
            end
        end
        local function random(seed)
            local fn = lcg(seed)
            return {
                next_integer = function(self) return fn(32) end,
                next_within = function(self, bound) return within(fn(31), bound) end,
            }
        end
        local rng, other = random(304853), random(304853)
        local sum = 0
        for i = 1, 20000 do
            local a, b = rng:next_integer(), other:next_within(5000)
            sum = fmod(sum * 3 + a + b, 1000000007)
        end
        return sum
        ",
    );
    let trace = &statistics.trace;
    assert!(trace.traces_compiled >= 1, "{statistics:?}");
    assert!(trace.iterations > 19_000, "{statistics:?}");
    assert_eq!(
        trace.recording_aborts.iter().sum::<u64>(),
        0,
        "{statistics:?}"
    );
    assert!(trace.inlined_calls >= 5, "{statistics:?}");
}

#[test]
fn a_sieve_runs_its_loops_natively_with_few_recordings() {
    // The shape of `benchmarks/lua/algorithm/sieve.lua`. The inner loop
    // stores keys the table does not hold yet (holes, and the slots past the
    // end on the first pass), which a trace stores through the VM as raw
    // sets. The outer loop reads `composite[i]`, nil for a prime: it traces
    // with a side for the nil arm, and leaves for the inner loop's own trace
    // only at the primes whose inner loop runs. Recordings that fail at the
    // same store, and traces compiled and retired, used to leave both loops
    // interpreted.
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local result, iterations = 0, 0
        for _ = 1, 5 do
            local limit = 20000
            local composite = {}
            for i = 2, limit do
                iterations = iterations + 1
                if not composite[i] then
                    result = result + 1
                    local multiple = i * i
                    while multiple <= limit do
                        composite[multiple] = true
                        multiple = multiple + i
                        iterations = iterations + 1
                    end
                end
            end
        end
        return result, iterations
        ",
        CompilerConfig::new(profile),
    )
    .expect("compile");
    let mut results = Vec::new();
    let mut statistics = None;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
        vm.use_synchronous_jit_compilation();
        install_base_with_profile(vm.runtime_mut(), profile).expect("install library");
        results.push(vm.execute(&chunk).expect("run"));
        statistics = Some(vm.jit_statistics());
    }
    assert_eq!(results[0], results[1]);
    // 2262 primes below 20000, five times.
    assert_eq!(results[1][0], oximond::runtime::Value::Number(11310.0));
    let oximond::runtime::Value::Number(iterations) = results[1][1] else {
        panic!("{:?}", results[1]);
    };
    let statistics = statistics.unwrap();
    let trace = &statistics.trace;
    // Native code runs nearly every iteration of both loops.
    #[allow(clippy::cast_precision_loss)]
    let native = trace.iterations as f64;
    assert!(
        native >= 0.9 * iterations,
        "{native} of {iterations} iterations native: {statistics:?}"
    );
    // A trace for each loop, and a side for the hole reads: a handful of
    // recordings, none failing at a table access, and nothing compiled and
    // thrown away.
    assert!(trace.recording_starts <= 8, "{statistics:?}");
    assert_eq!(
        trace.aborts(oximond::vm::TraceAbortReason::TableMiss),
        0,
        "{statistics:?}"
    );
    assert!(trace.traces_compiled <= 4, "{statistics:?}");
    assert!(trace.native_entries <= 1_000, "{statistics:?}");
}

#[test]
fn outer_loop_around_a_traced_inner_loop_does_not_leave_the_fast_loop_per_iteration() {
    // The inner loop runs as a trace, and so does the outer one, with a side
    // for the holes it reads (see `a_sieve_runs_its_loops_natively_with_few_recordings`).
    // While a loop waits for its trace, its consults only count, and the fast
    // loop settles them itself.
    let statistics = tracing_statistics_matching_interpreter(
        r"
        local result = 0
        for _ = 1, 3 do
            local limit = 20000
            local composite = {}
            for i = 2, limit do
                if not composite[i] then
                    result = result + 1
                    local multiple = i * i
                    while multiple <= limit do
                        composite[multiple] = true
                        multiple = multiple + i
                    end
                end
            end
        end
        return result
        ",
    );
    assert!(statistics.trace.traces_compiled > 0, "{statistics:?}");
    assert!(statistics.fast_backedge_lookups <= 2_000, "{statistics:?}");
    // Settling the consults in the fast loop must still let the traces in.
    assert!(statistics.trace.iterations >= 150, "{statistics:?}");
}

#[test]
fn a_trace_installed_while_the_fast_loop_settles_compile_waits_is_entered() {
    // Background compilation: the header waits for its compile while the
    // fast loop settles each consult itself. The install lands during one of
    // those, and the loop must leave for the full lookup and enter it.
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local sum = 0
        for i = 1, 3000000 do
            sum = (sum + i * 3) % 1000003
        end
        return sum
        ",
        CompilerConfig::new(profile),
    )
    .expect("compile");
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    install_base_with_profile(vm.runtime_mut(), profile).expect("install library");
    let result = vm.execute(&chunk).expect("run");
    let mut interpreter = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    install_base_with_profile(interpreter.runtime_mut(), profile).expect("install library");
    assert_eq!(result, interpreter.execute(&chunk).expect("run"));
    let statistics = vm.jit_statistics();
    assert!(statistics.trace.traces_compiled > 0, "{statistics:?}");
    assert!(statistics.trace.native_entries > 0, "{statistics:?}");
    // The compile finishes within a small part of the loop even on a loaded
    // machine; a swallowed install leaves every iteration interpreted.
    assert!(statistics.trace.iterations > 300_000, "{statistics:?}");
    assert!(statistics.fast_backedge_lookups <= 100, "{statistics:?}");
}

const STRING_READ_CACHE_PROGRAM: &str = r#"
    local out = {}
    local function say(value) out[#out + 1] = tostring(value) end
    local function wide(extra)
        local t = { b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8, i = 9, j = 10 }
        for key, value in pairs(extra) do t[key] = value end
        return t
    end
    local function field(t) return t.alpha end
    local t = wide({ alpha = 1 })
    for _ = 1, 40 do field(t) end
    say(field(t))
    rawset(t, "alpha", 11)
    say(field(t))
    t.alpha = nil
    say(field(t))
    t.alpha = 12
    say(field(t))
    for i = 1, 500 do t["pad" .. i] = i end
    say(field(t))
    for i = 1, 500 do t["pad" .. i] = nil end
    say(field(t))
    t.alpha = nil
    for i = 1, 500 do t["pad" .. i] = i end
    say(field(t))
    t.alpha = 13
    say(field(t))
    setmetatable(t, { __index = function() return "meta" end })
    say(field(t))
    t.alpha = nil
    say(field(t))
    setmetatable(t, nil)
    say(field(t))
    t.alpha = 14
    setmetatable(t, { __index = { alpha = "base" } })
    say(field(t))
    local other = wide({ alpha = "other" })
    for _ = 1, 3 do
        say(field(t))
        say(field(other))
    end
    t = nil
    collectgarbage()
    local fresh = wide({ alpha = "fresh" })
    say(field(fresh))

    gfield = "g1"
    local function global() return gfield end
    for _ = 1, 40 do global() end
    say(global())
    gfield = "g2"
    say(global())
    rawset(_G, "gfield", "g3")
    say(global())
    gfield = nil
    say(global())
    gfield = "g4"
    for i = 1, 500 do _G["gpad" .. i] = i end
    say(global())
    local replaced = wide({ gfield = "env" })
    setfenv(global, replaced)
    say(global())
    replaced.gfield = "env2"
    say(global())
    setfenv(global, wide({ gfield = "env3" }))
    say(global())
    setfenv(global, _G)
    say(global())
    return table.concat(out, ",")
"#;

const STRING_READ_CACHE_EXPECTED: &str = "1,11,nil,12,12,12,nil,13,13,meta,nil,14,14,other,14,other,14,other,fresh,g1,g2,g3,nil,g4,env,env2,env3,g4";

fn string_read_cache_result(mode: ExecutionMode) -> (String, u64) {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(STRING_READ_CACHE_PROGRAM, CompilerConfig::new(profile))
        .expect("compile cache program");
    let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
    vm.use_synchronous_jit_compilation();
    install_base_with_profile(vm.runtime_mut(), profile).expect("install standard library");
    let result = vm.execute(&chunk).expect("run cache program");
    let [oximond::runtime::Value::String(text)] = result.as_slice() else {
        panic!("unexpected result {result:?}");
    };
    (
        String::from_utf8_lossy(text.as_bytes()).into_owned(),
        vm.string_read_cache().hits(),
    )
}

#[test]
fn string_read_cache_follows_every_table_change() {
    // The program's constant-name reads are served by per-site caches
    // (`field_inline_caches.rs`), which these table changes exercise too, and
    // by the hashed-read cache where a site declines. Either way the values
    // must follow every change.
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let (text, _) = string_read_cache_result(mode);
        assert_eq!(text, STRING_READ_CACHE_EXPECTED, "{mode:?}");
    }
}

#[test]
fn constant_global_and_field_reads_are_served_by_site_caches() {
    let profile = CompatibilityProfile::Lua51;
    // The loop is in a function: the VM runs a copy of the main chunk, but
    // shares its functions' prototypes, whose sites can be inspected.
    let chunk = compile_with_config(
        r#"
        local function run()
            local total = 0
            for i = 1, 5000 do
                total = total + math.max(i, 10) + #string.rep("a", 2)
            end
            return total
        end
        return run()
        "#,
        CompilerConfig::new(profile),
    )
    .unwrap();
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    install_base_with_profile(vm.runtime_mut(), profile).unwrap();
    vm.execute(&chunk).unwrap();
    let cache = vm.string_read_cache();
    // Each iteration reads two globals and two fields. Their sites answer
    // all but the first reads, so the hashed-read cache stays out of it.
    assert!(cache.hits() < 100, "{} hits", cache.hits());
    assert!(cache.installs() <= 16, "{} installs", cache.installs());
    let sites = &chunk.prototypes[0];
    let filled = (0..sites.code.len())
        .filter(|&pc| {
            matches!(
                sites.code[pc].opcode(),
                Ok(oximond::bytecode::OpCode::GetGlobal | oximond::bytecode::OpCode::GetField)
            ) && sites.field_site(pc).is_some_and(|site| site.load().0 != 0)
        })
        .count();
    assert_eq!(filled, 4, "filled read sites");
}

/// Loops over 64-bit integer cdata (`ull` literals, mixed number arithmetic,
/// `bit` calls, `tonumber`) trace: the recorder used to abort at the first
/// cdata constant, so every iteration ran in the interpreter.
#[test]
fn int64_cdata_loops_compile_and_run_in_native_traces() {
    let profile = CompatibilityProfile::LuaJit21;
    let chunk = compile_with_config(
        r"
        local band, rshift = bit.band, bit.rshift
        local halves = { 0, 0 }
        local acc = 0ull
        for i = 1, 3000 do
            halves[1] = (i * 7919) % 1048576
            halves[2] = i % 1024
            local along = 0x100000ull * halves[2] + halves[1]
            local value = band(along * 3ull + 12345, 0xffffffffull)
            acc = (acc + value + rshift(along, 4)) % 1000000007ull
            halves[1] = tonumber(band(acc, 0xffffull))
        end
        return tonumber(acc), halves[1]
        ",
        CompilerConfig::new(profile),
    )
    .unwrap();
    let run = |mode| {
        let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
        vm.use_synchronous_jit_compilation();
        // The loop runs once per execution of a main chunk, with too little
        // work to repay a compile; this checks what its trace does.
        vm.ignore_remaining_work();
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        let results = vm.execute(&chunk).unwrap();
        (results, vm.jit_statistics().trace)
    };
    let (expected, _) = run(ExecutionMode::Interpreter);
    let (results, stats) = run(ExecutionMode::Tracing);
    assert_eq!(results, expected);
    assert_eq!(stats.recording_aborts.iter().sum::<u64>(), 0, "{stats:?}");
    assert!(stats.traces_compiled >= 1, "{stats:?}");
    assert!(stats.iterations >= 2_500, "{stats:?}");
}

/// Runs `source` in tracing mode with the background compile worker and
/// returns its results and statistics.
fn async_tracing_run(source: &str) -> (Vec<oximond::runtime::Value>, oximond::vm::JitStatistics) {
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(source, CompilerConfig::new(profile)).expect("compile");
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    install_base_with_profile(vm.runtime_mut(), profile).expect("install library");
    let results = vm.execute(&chunk).expect("run");
    (results, vm.jit_statistics())
}

#[test]
fn a_rejected_loop_earns_longer_backedge_skips() {
    // A call of `os.getenv` in the loop body never traces, so the loop stays
    // rejected for its 600000 iterations. (A concatenation was the example
    // until traces called the VM for it, `keys[i]:upper()` until they called
    // the string library, and a plain read by a key taken from another table
    // until it traced, see `dynamic_string_key_reads_run_natively_in_every_profile`.)
    // Each skip that ran undisturbed doubles the next, so the fast loop is
    // left a thousand times, not a hundred thousand. Before the skips start
    // the loop spends its 8192 unskipped lookups (`UNSKIPPED_LOOKUPS`) on the
    // loop around it, which is still counting toward a root that could end
    // in front of it.
    let statistics = tracing_statistics_matching_interpreter(
        r#"
        local values, keys = {}, {}
        for i = 1, 2000 do
            local key = "key_" .. i
            keys[i] = key
            values[key] = i % 107
        end
        local total = 0
        for _ = 1, 300 do
            for i = 1, 2000 do
                total = total + values[keys[i]] + (os.getenv("OXIMOND_NO_SUCH_VARIABLE") and 1 or 0)
            end
        end
        return total
        "#,
    );
    assert_eq!(statistics.trace.traces_recorded, 0, "{statistics:?}");
    assert!(
        statistics.fast_backedge_resumes <= 8_192 + 3_000,
        "{statistics:?}"
    );
}

#[test]
fn a_rejected_loop_beside_a_running_trace_skips_its_lookups() {
    // The first loop of `step` traces. The second never does (a call of
    // `os.getenv`), and runs five trips per call, so it is rejected and every
    // call of `step` finds a trace installed. Its backedges used to leave
    // the fast loop one by one while a trace was installed. They now have
    // skip windows of their own, which delay nothing else: the trace is
    // still entered on every call.
    let statistics = tracing_statistics_matching_interpreter(
        r#"
        local function step(round)
            local sum = 0
            for i = 1, 100 do sum = sum + i % 7 end
            for i = 1, 5 do
                sum = sum + i + (os.getenv("OXIMOND_NO_SUCH_VARIABLE") and 1 or 0)
            end
            return sum + round
        end
        local total = 0
        for round = 1, 30000 do total = (total + step(round)) % 1000003 end
        return total
        "#,
    );
    assert!(statistics.trace.traces_compiled >= 1, "{statistics:?}");
    // The first loop is entered on (nearly) every call of `step`.
    assert!(statistics.trace.native_entries >= 25_000, "{statistics:?}");
    // Five rejected backedges per call would leave the fast loop 150000
    // times. With windows that double up to 4096 backedges, a few hundred
    // lookups remain (the entries of the first loop count among them).
    assert!(statistics.fast_backedge_resumes <= 40_000, "{statistics:?}");
}

#[test]
fn a_loop_waiting_for_its_compile_stays_in_the_fast_loop() {
    // The skip a header waiting for its compile arms (`COMPILE_WAIT_SKIP`).
    const COMPILE_WAIT_SKIP: u64 = 32;
    // The loop traces, and the worker takes a while to compile it. Until the
    // trace installs, the header has nothing to count or enter, so the
    // backedges skip in windows instead of leaving the fast loop each time.
    let (results, statistics) = async_tracing_run(
        r"
        local s = 0
        for i = 1, 3000000 do s = s + i % 7 end
        return s
        ",
    );
    assert_eq!(results, vec![oximond::runtime::Value::Number(8_999_997.0)]);
    assert_eq!(statistics.trace.traces_compiled, 1, "{statistics:?}");
    // The backedges that run before the trace installs depend on the compile
    // time, so bound the resumes by the iterations that ran interpreted: one
    // per skip window, plus slack for the windows before the first arming.
    let interpreted = 3_000_000 - statistics.trace.iterations;
    let bound = interpreted / COMPILE_WAIT_SKIP + 200;
    assert!(statistics.fast_backedge_resumes <= bound, "{statistics:?}");
}

#[test]
fn a_trace_that_never_pays_is_not_compiled_again_for_the_same_recording() {
    // The outer loop of a sieve records the path that skips a composite
    // number, and every prime leaves the trace after a few iterations. The
    // trace retires without one productive window, and the recording that
    // follows is the same trace: the site is rejected without compiling it
    // again.
    let statistics = tracing_statistics_matching_interpreter(
        r"
        local result = 0
        for _ = 1, 5 do
            local limit = 50000
            local composite = {}
            for i = 2, limit do
                if not composite[i] then
                    result = result + 1
                    local multiple = i * i
                    while multiple <= limit do
                        composite[multiple] = true
                        multiple = multiple + i
                    end
                end
            end
        end
        return result
        ",
    );
    assert!(statistics.trace.traces_compiled <= 2, "{statistics:?}");
}

const OUTER_WAITING_ON_INNER: &str = r#"
    local floor, max, min, byte, sub = math.floor, math.max, math.min, string.byte, string.sub
    local function low_word(x) return floor(x % 4294967296) end
    local function split(a, value)
        a[1] = floor(value / 65536)
        a[2] = floor(value % 65536)
        return low_word(a[1] * 3 + a[2])
    end
    local function digits(str)
        local total = 0
        for i = 1, #str do
            total = total * 10 + max(min(byte(sub(str, i, i)) - 48, 9), 0)
        end
        return total
    end
    local a, sum, text = { 0, 0 }, 0, "3948575739"
    for i = 1, 72000 do
        sum = (sum + split(a, i * 7919 + sum)) % 1000000007
        if i % 32 == 0 then sum = (sum + digits(text)) % 1000000007 end
    end
    return sum
"#;

#[test]
fn an_outer_trace_waiting_on_an_inner_compile_is_not_given_up() {
    // The outer loop calls a helper (a trace of its own, compiled by the
    // worker) and takes a periodic branch into another callee with an inner
    // loop. The outer trace can retire before its first productive window
    // while the inner loop's compile is still in flight, and recording it
    // again gives the same trace. That repeat must still compile: whether a
    // trace pays depends on its siblings, and giving up on it leaves the
    // loop interpreted. Compile timing varies, so run it many times.
    const COUNT: u64 = 24_000 * 3;
    for _ in 0..30 {
        let (results, statistics) = async_tracing_run(OUTER_WAITING_ON_INNER);
        assert_eq!(results.len(), 1);
        assert!(
            statistics.trace.iterations >= COUNT / 2,
            "the loop ran mostly interpreted: {statistics:?}"
        );
    }
}

#[test]
fn a_trace_installed_while_another_loop_compiles_keeps_its_entries() {
    // The same program, with each compile installing a fixed number of polls
    // after the previous one: the outer trace installs first and runs while
    // the inner loop's compile is still in flight. The header that waits for
    // its compile must not arm the global backedge skip then. Armed in the
    // inner loop, it ran on into the outer loop's backedges, so the outer
    // trace was entered late in every 32-iteration period, failed its
    // productivity window and retired, again and again until its site was
    // rejected. Each of these delays retired a trace before the fix.
    const COUNT: u64 = 24_000 * 3;
    let profile = CompatibilityProfile::Lua51;
    let chunk =
        compile_with_config(OUTER_WAITING_ON_INNER, CompilerConfig::new(profile)).expect("compile");
    for delay in [175, 250, 300, 375, 459, 501, 571, 641, 655] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        vm.use_delayed_synchronous_jit_compilation(delay);
        install_base_with_profile(vm.runtime_mut(), profile).expect("install library");
        let results = vm.execute(&chunk).expect("run");
        assert_eq!(results.len(), 1);
        let statistics = vm.jit_statistics();
        // One trace per loop, never retired and recorded again.
        assert_eq!(
            statistics.trace.traces_recorded, 2,
            "delay {delay}: {statistics:?}"
        );
        assert!(
            statistics.trace.iterations >= COUNT / 2,
            "delay {delay}: the loop ran mostly interpreted: {statistics:?}"
        );
    }
}

#[test]
fn a_rejected_loop_keeps_its_skips_while_an_unentered_trace_is_installed() {
    // The outer sieve trace installs while the inner loop is rejected, and
    // it is entered rarely. A trace that is installed but not entered must
    // not turn the inner loop's skips off, or every inner backedge leaves
    // the fast loop (about one resume per prime, 70000 and more).
    //
    // The state the trace reaches depends on when its compile lands. The
    // resumes before it lands grow with that time (about 2.5 per safepoint
    // poll of the compile's delay), so a run with the worker thread, whose
    // compile takes as long as the machine's load makes it, counted the
    // load: 6000 to 50000 resumes on an idle machine, more under the other
    // tests of this file. Each delay below installs the trace at a fixed
    // poll instead, from at once to well after the loop has run for a
    // while, and the bound is the same for all of them.
    const PRIMES: f64 = 76_995.0;
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(
        r"
        local result = 0
        for _ = 1, 15 do
            local limit = 50000
            local composite = {}
            for i = 2, limit do
                if not composite[i] then
                    result = result + 1
                    local multiple = i * i
                    while multiple <= limit do
                        composite[multiple] = true
                        multiple = multiple + i
                    end
                end
            end
        end
        return result
        ",
        CompilerConfig::new(profile),
    )
    .expect("compile");
    for delay in [0, 100, 250, 500, 1_000, 2_000, 4_000, 6_000] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        vm.use_delayed_synchronous_jit_compilation(delay);
        install_base_with_profile(vm.runtime_mut(), profile).expect("install library");
        let results = vm.execute(&chunk).expect("run");
        assert_eq!(results, vec![oximond::runtime::Value::Number(PRIMES)]);
        let statistics = vm.jit_statistics();
        assert!(
            statistics.fast_backedge_resumes <= 20_000,
            "delay {delay}: {statistics:?}"
        );
    }
}

#[test]
fn a_rejected_loop_still_collects_garbage_inside_skip_windows() {
    // The backedge skip leaves the collector poll to its end and to the
    // allocations of the loop. A rejected loop that makes garbage on every
    // iteration must not outgrow the interpreter's heap.
    let source = r#"
        local keys, values = {}, {}
        for i = 1, 64 do keys[i] = "k" .. i; values[keys[i]] = i end
        local total = 0
        for round = 1, 400 do
            for i = 1, 64 do
                local scratch = { i, round, keys[i] .. round }
                total = total + values[keys[i]] + #scratch
            end
        end
        return total, collectgarbage("count")
    "#;
    let profile = CompatibilityProfile::Lua51;
    let chunk = compile_with_config(source, CompilerConfig::new(profile)).expect("compile");
    let (mut counts, mut totals) = (Vec::new(), Vec::new());
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
        vm.use_synchronous_jit_compilation();
        install_base_with_profile(vm.runtime_mut(), profile).expect("install library");
        let results = vm.execute(&chunk).expect("run");
        let oximond::runtime::Value::Number(kilobytes) = results[1] else {
            panic!("count is a number: {results:?}");
        };
        totals.push(results[0].clone());
        counts.push(kilobytes);
    }
    assert_eq!(totals[0], totals[1]);
    assert!(counts[1] < counts[0] * 4.0 + 512.0, "{counts:?}");
}

#[test]
fn dynamic_string_key_reads_run_natively_in_every_profile() {
    // The shape of `benchmarks/lua/table/hash_read.lua`: `values[keys[i]]`
    // over 2000 distinct strings. The key is read out of an array, so it is a
    // runtime value; the recording used to abort at it, leaving the loop to
    // the interpreter. Now the lookup is one native call per iteration: the
    // trace compiles, runs nearly every iteration, and the interpreter
    // dispatches a small fraction of what it does alone.
    let source = r"
        local values, keys = {}, {}
        for i = 1, 2000 do
            local key = 'key_' .. i
            keys[i] = key
            values[key] = i % 107
        end
        local total = 0
        for _ = 1, 30 do
            for i = 1, 2000 do
                total = total + values[keys[i]]
            end
        end
        return total
    ";
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
        CompatibilityProfile::LuaJit21,
    ] {
        let chunk = compile_with_config(source, CompilerConfig::new(profile)).expect("compile");
        let mut interpreter =
            Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
        interpreter.set_interpreter_profiling(true);
        install_base_with_profile(interpreter.runtime_mut(), profile).expect("install library");
        let expected = interpreter.execute(&chunk).expect("run");
        let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        tracing.use_synchronous_jit_compilation();
        tracing.set_interpreter_profiling(true);
        install_base_with_profile(tracing.runtime_mut(), profile).expect("install library");
        assert_eq!(
            tracing.execute(&chunk).expect("run"),
            expected,
            "{profile:?}"
        );
        let statistics = tracing.jit_statistics();
        let trace = &statistics.trace;
        // The hot loop compiles into one trace that runs nearly all of the
        // 60000 lookups, and does not keep aborting at the key.
        assert!(trace.traces_compiled >= 1, "{profile:?}: {statistics:?}");
        assert!(trace.native_entries >= 1, "{profile:?}: {statistics:?}");
        assert!(trace.iterations >= 50_000, "{profile:?}: {statistics:?}");
        assert!(trace.traces_compiled <= 3, "{profile:?}: {statistics:?}");
        assert!(
            trace.aborts(oximond::vm::TraceAbortReason::UnsupportedOperand) <= 3,
            "{profile:?}: only the setup loop's store may abort: {statistics:?}"
        );
        // The lookups probe the table's string index in native code. The
        // runtime answers only the first entry's (a table this big builds
        // its index when the trace asks a second time): at most 2000 of the
        // 60000 lookups, and none once the loop is warm.
        assert!(
            trace.hash_helper_calls <= 2_000,
            "{profile:?}: the hot loop still calls trace_hash_get: {statistics:?}"
        );
        assert!(
            tracing.interpreter_counters().steps.get() * 2
                < interpreter.interpreter_counters().steps.get(),
            "{profile:?}: native entries should replace interpreter dispatches"
        );
    }
}

/// `allocation/table_churn`: the constructor in the loop never allocates a
/// table once the trace runs. The count is by counters, not time: tripling
/// the iterations adds no real table and adds the constructors as phantom
/// charges, the checksum is the interpreter's, and the trace runs the loop.
#[test]
fn table_churn_traces_without_allocating_tables() {
    fn churn(profile: CompatibilityProfile, mode: ExecutionMode, count: usize) -> (Vm, String) {
        let source = format!(
            "local total = 0
             for i = 1, {count} do
                 local value = {{i, i + 1, i + 2, label = 'node'}}
                 total = total + value[2]
             end
             return total"
        );
        let chunk = compile_with_config(&source, CompilerConfig::new(profile))
            .expect("compile table churn");
        let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
        vm.use_synchronous_jit_compilation();
        vm.ignore_remaining_work();
        install_base_with_profile(vm.runtime_mut(), profile).expect("install library");
        let result = format!("{:?}", vm.execute(&chunk).expect("run table churn"));
        (vm, result)
    }
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
        CompatibilityProfile::LuaJit21,
    ] {
        let (_, expected) = churn(profile, ExecutionMode::Interpreter, 60_000);
        let (small, _) = churn(profile, ExecutionMode::Tracing, 20_000);
        let (large, result) = churn(profile, ExecutionMode::Tracing, 60_000);
        assert_eq!(result, expected, "{profile:?}");
        let trace = &large.jit_statistics().trace;
        assert!(trace.traces_compiled >= 1, "{profile:?}: {trace:?}");
        assert!(trace.iterations >= 55_000, "{profile:?}: {trace:?}");
        assert!(
            large.runtime().tables_created() <= small.runtime().tables_created() + 4,
            "{profile:?}: {} real tables at 20000 iterations, {} at 60000",
            small.runtime().tables_created(),
            large.runtime().tables_created()
        );
        let phantom =
            large.runtime().elided_tables_total().0 - small.runtime().elided_tables_total().0;
        assert!(phantom >= 39_000, "{profile:?}: {phantom} phantom charges");
    }
}
