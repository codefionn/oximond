//! Reads that an `__index` chain answers are cached by metatable and key.
//! Every change to the receiver, a metatable, or a table of the chain must
//! show in the next read, under every profile in both execution modes.

use oximond::bytecode::Chunk;
use oximond::compat::{CompatibilityProfile, ExecutionMode};
use oximond::compiler::{CompilerConfig, compile_with_config};
use oximond::stdlib::install_base_with_profile;
use oximond::vm::Vm;

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

const MODES: [ExecutionMode; 2] = [ExecutionMode::Interpreter, ExecutionMode::Tracing];

fn new_vm(profile: CompatibilityProfile, mode: ExecutionMode) -> Vm {
    let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
    vm.use_synchronous_jit_compilation();
    install_base_with_profile(vm.runtime_mut(), profile).expect("install standard library");
    vm
}

fn compile(source: &str, profile: CompatibilityProfile) -> Chunk {
    compile_with_config(source, CompilerConfig::new(profile)).expect("compile")
}

fn run_everywhere(name: &str, source: &str) {
    for profile in PROFILES {
        let chunk = compile(source, profile);
        for mode in MODES {
            let mut vm = new_vm(profile, mode);
            if let Err(error) = vm.execute(&chunk) {
                panic!("{name} under {profile:?} in {mode:?}: {error:?}");
            }
        }
    }
}

macro_rules! fixture_tests {
    ($($test:ident => $file:literal,)*) => {
        $(
            #[test]
            fn $test() {
                run_everywhere(
                    $file,
                    include_str!(concat!("fixtures/inherited_index/", $file, ".lua")),
                );
            }
        )*
    };
}

fixture_tests! {
    chains_of_depth_one_two_and_three => "chains",
    every_change_to_the_chain_shows_in_the_next_read => "invalidation",
    index_functions_are_never_cached => "functions",
    collections_do_not_leave_stale_answers => "collection",
    coroutines_and_hooks_see_the_same_answers => "coroutines",
}
