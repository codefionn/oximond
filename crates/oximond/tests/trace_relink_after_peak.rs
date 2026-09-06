//! A side compiled apart whose guard turns hot after its tree reached the peak
//! tier is compiled back into the tree. The peak recompile is spent by then,
//! and the relink must not wait for a tree that never asked for it.

use oximond::vm::TraceExitKind;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const SOURCE: &str = "
local a, b = 0, 0
for i = 1, 3500000 do
  local m = 1e9
  if i >= 2600000 then m = 4 elseif i >= 2300000 then m = 50000 end
  a = a + i
  if i % m == 0 then b = b + 1 end
end
return a, b
";

#[test]
fn a_side_that_turns_hot_after_the_peak_recompile_is_linked_back() {
    let profile = CompatibilityProfile::Lua51;
    let run = |mode| {
        let mut lua = Lua::builder()
            .profile(profile)
            .execution_mode(mode)
            .build()
            .unwrap();
        lua.vm_mut().use_synchronous_jit_compilation();
        // The subject is a side compiled apart: a tree this small would
        // otherwise take the side in before the peak recompile.
        lua.vm_mut().keep_jit_sides_apart();
        let results = lua
            .exec(SOURCE)
            .map_err(|error| format!("{error:?}"))
            .unwrap();
        (results, lua)
    };
    let (expected, _) = run(ExecutionMode::Interpreter);
    let (actual, lua) = run(ExecutionMode::Tracing);
    assert_eq!(actual, expected);
    let stats = lua.vm().jit_statistics().trace;
    assert!(stats.side_compiles >= 1, "{stats:?}");
    assert!(stats.link_recompiles >= 1, "{stats:?}");
    // Without the relink every hit of the guard leaves for the interpreter.
    assert!(stats.exits_of(TraceExitKind::Poll) < 100_000, "{stats:?}");
}
