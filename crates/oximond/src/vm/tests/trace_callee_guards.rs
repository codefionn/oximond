//! Callee guards of calls the recording already guarded
//! (`Recorder::callee_guarded`): a helper in an upvalue that an inlined
//! frame calls many times is compared with its pinned closure once, and a
//! callee read out of a table field again for every call keeps a guard per
//! call.

use super::*;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::{TraceIr, TraceOp, TraceScalar, ValueId};

/// The root trace the recorder queued for `source`, which it does not compile.
fn recorded(profile: CompatibilityProfile, source: &str) -> TraceIr {
    let mut lua = crate::Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Tracing)
        .build()
        .unwrap();
    let vm = lua.vm_mut();
    vm.jit.trace.set_native_compilation(false);
    lua.exec(source).unwrap();
    let queued = lua.vm().jit.trace.queued();
    queued.front().expect("a recording").ir.clone()
}

/// The equality compares of the trace against a pinned closure, the shape a
/// callee guard has.
fn closure_compares(ir: &TraceIr) -> usize {
    let constants: Vec<ValueId> = ir
        .body
        .iter()
        .chain(&ir.preheader)
        .filter(|instruction| matches!(instruction.op, TraceOp::Constant(TraceScalar::Closure(_))))
        .filter_map(|instruction| instruction.result)
        .collect();
    ir.body
        .iter()
        .chain(&ir.preheader)
        .filter(|instruction| {
            matches!(
                instruction.op,
                TraceOp::Compare { comparison: Comparison::Equal, right, .. }
                    if constants.contains(&right)
            )
        })
        .count()
}

/// `step` calls the upvalue `mix` six times; the loop calls `step`.
const SIX_CALLS: &str = "
local function mix(a, b) return (a * 31 + b) % 65521 end
local function step(x, i)
  x = mix(x, i) x = mix(x, i + 1) x = mix(x, i + 2)
  x = mix(x, i + 3) x = mix(x, i + 4) x = mix(x, i + 5)
  return x
end
local function run(n)
  local x = 1
  for i = 1, n do x = step(x, i) end
  return x
end
return run(3000)
";

#[test]
fn an_upvalue_helper_called_six_times_takes_one_callee_guard() {
    for profile in [
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua51,
        CompatibilityProfile::LuaJit21,
    ] {
        let ir = recorded(profile, SIX_CALLS);
        ir.verify().unwrap();
        // `step` in the root frame, and `mix` once.
        assert_eq!(closure_compares(&ir), 2, "{profile:?}");
    }
}

#[test]
fn a_callee_read_out_of_a_field_keeps_a_guard_per_call() {
    let ir = recorded(
        CompatibilityProfile::Lua54,
        "
local t = {}
function t.mix(a, b) return (a * 31 + b) % 65521 end
local function step(x, i)
  x = t.mix(x, i) x = t.mix(x, i + 1) x = t.mix(x, i + 2)
  return x
end
local function run(n)
  local x = 1
  for i = 1, n do x = step(x, i) end
  return x
end
return run(3000)
",
    );
    ir.verify().unwrap();
    assert_eq!(closure_compares(&ir), 1 + 3);
}
