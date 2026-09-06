use oximond::compat::ExecutionMode;
use oximond::compiler::compile;
use oximond::stdlib::install_base;
use oximond::vm::Vm;

#[test]
fn short_numeric_helpers_match_the_interpreter() {
    let chunk = compile(
        r"
        local function a(i, j)
            local ij = i + j
            return 1 / (ij * (ij + 1) / 2 + i + 1)
        end

        local total = 0
        for i = 1, 2048 do
            total = total + a(i, i + 1)
        end
        return total
        ",
    )
    .expect("compile short-helper regression program");
    let mut interpreter = Vm::new();
    install_base(interpreter.runtime_mut()).expect("install standard library");
    let expected = interpreter.execute(&chunk).expect("interpret short helper");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).expect("install standard library");

    assert_eq!(
        vm.execute(&chunk)
            .expect("execute short-helper regression program"),
        expected
    );
}

#[test]
#[ignore = "the trace recorder refuses string operands (UnsupportedOperand), so this loop stays interpreted"]
fn warmed_string_array_writes_stay_in_native_loop() {
    let chunk = compile(
        r#"
        local values = { "old" }
        local replacement = "new"
        for i = 1, 100000 do
            values[1] = replacement
        end
        return values[1]
        "#,
    )
    .expect("compile array write program");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).expect("install standard library");
    vm.set_interpreter_profiling(true);

    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk).expect("warm array write program"),
            vec![oximond::runtime::Value::from("new")]
        );
    }
    let interpreted_before = vm.interpreter_counters().steps.get();
    let native_before = vm.jit_statistics().trace.iterations;
    assert_eq!(
        vm.execute(&chunk).expect("run warmed array write program"),
        vec![oximond::runtime::Value::from("new")]
    );
    let interpreted = vm.interpreter_counters().steps.get() - interpreted_before;
    let native_iterations = vm.jit_statistics().trace.iterations - native_before;
    assert!(
        native_iterations > 0,
        "warmed loop never entered native code; interpreted {interpreted} instructions; statistics: {:?}",
        vm.jit_statistics()
    );
    assert!(
        interpreted < 500,
        "warmed array write still interpreted {interpreted} instructions; native iterations: {native_iterations}"
    );
}

#[test]
#[ignore = "the trace recorder refuses string operands (UnsupportedOperand), so this loop stays interpreted"]
fn warmed_string_length_uses_native_loop_and_counts_bytes() {
    let chunk = compile(
        r#"
        local text = "é"
        local total = 0
        for i = 1, 100000 do
            total = total + #text
        end
        return total
        "#,
    )
    .expect("compile string length program");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).expect("install standard library");
    vm.set_interpreter_profiling(true);

    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk).expect("warm string length program"),
            vec![oximond::runtime::Value::Number(200_000.0)]
        );
    }
    let interpreted_before = vm.interpreter_counters().steps.get();
    let native_before = vm.jit_statistics().trace.iterations;
    assert_eq!(
        vm.execute(&chunk)
            .expect("run warmed string length program"),
        vec![oximond::runtime::Value::Number(200_000.0)]
    );
    let interpreted = vm.interpreter_counters().steps.get() - interpreted_before;
    let native_iterations = vm.jit_statistics().trace.iterations - native_before;
    assert!(
        native_iterations > 0,
        "warmed string length loop never entered native code; interpreted {interpreted} instructions"
    );
    assert!(
        interpreted < 500,
        "warmed string length loop interpreted {interpreted} instructions; native iterations: {native_iterations}"
    );
}

#[test]
#[ignore = "the trace recorder refuses string operands (UnsupportedOperand), so this loop stays interpreted"]
fn opaque_post_loop_comparison_does_not_discard_native_loop() {
    let chunk = compile(
        r#"
        local values = { "old" }
        local replacement = "new"
        for i = 1, 100000 do
            values[1] = replacement
        end
        assert(values[1] == "new")
        return values[1]
        "#,
    )
    .expect("compile loop with opaque post-loop comparison");
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).expect("install standard library");
    vm.set_interpreter_profiling(true);

    for _ in 0..8 {
        assert_eq!(
            vm.execute(&chunk).expect("warm loop with comparison"),
            vec![oximond::runtime::Value::from("new")]
        );
    }
    let interpreted_before = vm.interpreter_counters().steps.get();
    let native_before = vm.jit_statistics().trace.iterations;
    assert_eq!(
        vm.execute(&chunk).expect("run warmed loop with comparison"),
        vec![oximond::runtime::Value::from("new")]
    );
    let interpreted = vm.interpreter_counters().steps.get() - interpreted_before;
    let native_iterations = vm.jit_statistics().trace.iterations - native_before;
    assert!(
        native_iterations > 0,
        "post-loop comparison prevented native entry; interpreted {interpreted} instructions; statistics: {:?}",
        vm.jit_statistics()
    );
    assert!(
        interpreted < 500,
        "post-loop comparison caused {interpreted} interpreted instructions; native iterations: {native_iterations}"
    );
}
