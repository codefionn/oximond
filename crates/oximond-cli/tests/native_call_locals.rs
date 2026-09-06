use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value, Vm, VmError};

fn edit_outer(vm: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    for index in 1..30 {
        if let Some((name, _)) = vm.host_get_local(2, index)?
            && name.as_bytes() == b"held"
        {
            vm.host_set_local(2, index, Value::Number(41.0))?;
            return Ok(Vec::new());
        }
    }
    panic!("outer local was not found");
}

#[test]
fn generic_native_calls_preserve_uncaptured_debugger_local_writes() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        for host in [false, true] {
            let mut lua = Lua::builder()
                .profile(CompatibilityProfile::Lua51)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            let callback = lua
                .vm_mut()
                .register_host_function(edit_outer, 0, Vec::new())
                .unwrap();
            lua.vm_mut()
                .runtime_mut()
                .set_global("edit_outer", callback);
            let edit = if host {
                "edit_outer()"
            } else {
                r#"for index = 1, 30 do
                    local name = debug.getlocal(2, index)
                    if name == "held" then debug.setlocal(2, index, 41); break end
                end"#
            };
            let source = format!(
                r"local function callee(i) {edit}; if i % 2 == 0 then return 1 else return 1 end end
                local held = 0
                local total = 0
                for i = 1, 20000 do
                    held = 0
                    local result = callee(i)
                    total = total + held + result
                end
                return held, total"
            );
            assert_eq!(
                lua.exec(&source).unwrap(),
                vec![Value::Number(41.0), Value::Number(840_000.0)]
            );
        }
    }
}
