use oximond::compat::ExecutionMode;
use oximond::compiler::compile;
use oximond::stdlib::install_base;
use oximond::vm::Vm;

fn run(source: &str, mode: ExecutionMode) -> Vec<oximond::runtime::Value> {
    let chunk = compile(source).expect("compile tail graph");
    let mut vm = Vm::with_execution_mode(mode);
    vm.use_synchronous_jit_compilation();
    install_base(vm.runtime_mut()).expect("install standard library");
    vm.execute(&chunk).expect("run tail graph")
}

#[test]
fn mutual_tail_graph_rechecks_rebound_global_after_collection() {
    let source = r"
        local function even(n)
            if n == 0 then return 0 end
            return odd(n - 1)
        end
        function odd(n)
            if n == 0 then return 1 end
            return even(n - 1)
        end
        local warm = even(4000)
        odd = function(_) return 77 end
        collectgarbage('collect')
        return warm, even(8)
    ";
    let expected = run(source, ExecutionMode::Interpreter);
    assert_eq!(run(source, ExecutionMode::Tracing), expected);
}

#[test]
fn mutual_tail_graph_respects_instruction_hooks_after_warmup() {
    let source = r"
        local function even(n)
            if n == 0 then return 0 end
            return odd(n - 1)
        end
        function odd(n)
            if n == 0 then return 1 end
            return even(n - 1)
        end
        local warm = even(4000)
        local events = 0
        debug.sethook(function() events = events + 1 end, '', 1)
        local hooked = even(80)
        debug.sethook()
        return warm, hooked, events
    ";
    let expected = run(source, ExecutionMode::Interpreter);
    assert_eq!(run(source, ExecutionMode::Tracing), expected);
}
