//! A frame that a tail call replaced reports `istailcall` (what a host sees
//! through `lua_getinfo` with the `t` option), and the count of tail calls
//! the frame took stays right.

use oximond::compat::CompatibilityProfile;
use oximond::runtime::Value;
use oximond::vm::{Vm, VmResult};
use std::sync::Mutex;

static SEEN: Mutex<Vec<bool>> = Mutex::new(Vec::new());
/// The probe records into one list, so the programs take turns.
static TURN: Mutex<()> = Mutex::new(());

/// Records whether the Lua function that called it was entered by a tail
/// call.
#[allow(clippy::unnecessary_wraps)]
fn probe(vm: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
    let info = vm.host_stack_info(1).expect("the calling Lua frame");
    SEEN.lock().unwrap().push(info.is_tail_call);
    Ok(Vec::new())
}

fn flags(profile: CompatibilityProfile, source: &str) -> Vec<bool> {
    let _turn = TURN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut vm = Vm::with_profile(profile);
    let function = vm.register_host_function(probe, 0, Vec::new()).unwrap();
    vm.runtime_mut().set_global("probe", function);
    SEEN.lock().unwrap().clear();
    let chunk = oximond::compiler::compile(source).expect("the program compiles");
    vm.execute(&chunk).unwrap();
    SEEN.lock().unwrap().clone()
}

const PROGRAM: &str = "
    local function g(...) probe() end
    local function f(...) return g(...) end
    local function h(...) g(...) end
    local function k(...) return f(...) end
    f(1, 2)
    h(1, 2)
    k(3)
    g()
";

#[test]
fn tail_called_frames_report_istailcall() {
    // `g` is entered by a tail call from `f`, by a plain call from `h`, by a
    // chain of two tail calls from `k`, and directly from the main chunk.
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        assert_eq!(
            flags(profile, PROGRAM),
            vec![true, false, true, false],
            "{profile:?}"
        );
    }
}

#[test]
fn a_million_tail_calls_leave_one_frame_marked() {
    let source = "
        local function spin(n, ...)
            if n == 0 then probe() return end
            return spin(n - 1, ...)
        end
        spin(1000000, 1, 2)
        local function plain(n) if n == 0 then probe() return end plain(n - 1) end
        plain(3)
    ";
    assert_eq!(
        flags(CompatibilityProfile::Lua54, source),
        vec![true, false]
    );
}
