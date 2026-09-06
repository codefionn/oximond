//! Per-site inline caches of `GetGlobal`, `GetField` and `SetField` in the
//! interpreter: they must follow every change of the tables they were filled
//! from, and a hot site must stop probing tables.

use oximond::bytecode::{Chunk, OpCode};
use oximond::compat::{CompatibilityProfile, ExecutionMode};
use oximond::compiler::{CompilerConfig, compile_with_config};
use oximond::runtime::Value;
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

/// Runs `source` under every profile in both execution modes.
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
                    include_str!(concat!("fixtures/field_caches/", $file, ".lua")),
                );
            }
        )*
    };
}

fixture_tests! {
    sites_follow_layout_changes_between_hits => "shape_changes",
    sites_follow_deleted_keys => "deleted_keys",
    sites_leave_index_metamethods_alone => "index_metamethods",
    sites_leave_newindex_metamethods_alone => "newindex_metamethods",
    sites_follow_replaced_environments => "environments",
    sites_survive_a_collected_table_and_its_reused_slot => "slot_reuse",
    sites_are_shared_by_coroutines => "coroutines",
    sites_store_every_kind_of_value => "values",
    sites_never_hide_a_finalizer_change => "finalizers",
    sites_follow_constructors_that_share_a_layout => "shared_shapes",
    sites_follow_changes_inside_one_activation => "one_activation",
}

/// Every filled site of `chunk` and its prototypes whose instruction is one
/// of `opcodes`, and how many such instructions there are.
fn filled_sites(chunk: &Chunk, opcodes: &[OpCode]) -> (usize, usize) {
    let mut filled = 0;
    let mut total = 0;
    for (pc, instruction) in chunk.code.iter().enumerate() {
        if instruction
            .opcode()
            .is_ok_and(|opcode| opcodes.contains(&opcode))
        {
            total += 1;
            let site = if matches!(
                instruction.opcode(),
                Ok(OpCode::SetGlobal | OpCode::SetField)
            ) {
                chunk.store_site_at(chunk.code.as_ptr().wrapping_add(pc))
            } else {
                chunk.field_site(pc)
            };
            if site.is_some_and(|site| site.load().0 != 0) {
                filled += 1;
            }
        }
    }
    for prototype in &chunk.prototypes {
        let (inner_filled, inner_total) = filled_sites(prototype, opcodes);
        filled += inner_filled;
        total += inner_total;
    }
    (filled, total)
}

// The loop lives in a function: the VM runs a copy of the main chunk, but
// shares the prototypes of its functions, so their sites can be inspected.
const HOT_PROGRAM: &str = r#"
    counter = 0
    local function run()
        local ordered = { a = 1, b = 2, c = 3, d = 4 }
        local hashed = {}
        for i = 1, 40 do hashed["f" .. i] = i end
        hashed.x = 1
        local function shaped(i) return { x = i, y = i } end
        local pair = shaped(1)
        local total = 0
        for i = 1, 5000 do
            total = total + counter + ordered.d + hashed.x + pair.y + math.pi
            hashed.x = i
            pair.x = i
            ordered.c = i
            counter = counter
        end
        return total
    end
    return run()
"#;

#[test]
fn hot_sites_fill_their_caches_and_skip_the_string_cache() {
    for profile in PROFILES {
        let chunk = compile(HOT_PROGRAM, profile);
        let mut vm = new_vm(profile, ExecutionMode::Interpreter);
        vm.execute(&chunk).expect("run");
        let (filled, total) = filled_sites(&chunk, &[OpCode::GetGlobal, OpCode::GetField]);
        assert!(total >= 4, "{profile:?}: {total} read sites");
        assert!(
            filled >= total - 1,
            "{profile:?}: {filled} of {total} read sites filled"
        );
        // Three stores run in the loop; the constructor's and the setup's
        // stores run once.
        let (filled, total) = filled_sites(&chunk, &[OpCode::SetField]);
        assert!(
            filled >= 3,
            "{profile:?}: {filled} of {total} store sites filled"
        );
        // 5000 iterations read two globals and four fields. Filled sites
        // answer them without the hashed-read cache or a probe. Lua 5.2 and
        // later read a global with `GetTable` on `_ENV`, which has no site.
        let cache = vm.string_read_cache();
        let global_sites = matches!(
            profile,
            CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21
        );
        let allowed = if global_sites { 200 } else { 16_000 };
        assert!(
            cache.hits() < allowed,
            "{profile:?}: {} cache hits",
            cache.hits()
        );
        assert!(
            cache.installs() < 20,
            "{profile:?}: {} installs",
            cache.installs()
        );
    }
}

#[test]
fn a_site_that_sees_many_tables_stops_refilling() {
    let profile = CompatibilityProfile::Lua54;
    let source = r#"
        local function get(t) return t.x end
        local tables = {}
        for i = 1, 64 do
            local t = {}
            for k = 1, 20 do t["k" .. k] = k end
            t.x = i
            tables[i] = t
        end
        local total = 0
        for round = 1, 20 do
            for i = 1, 64 do total = total + get(tables[i]) end
        end
        return total
    "#;
    let chunk = compile(source, profile);
    let mut vm = new_vm(profile, ExecutionMode::Interpreter);
    let results = vm.execute(&chunk).expect("run");
    assert_eq!(results, vec![Value::Integer(20 * (64 * 65 / 2))]);
}

#[test]
fn readonly_tables_still_refuse_stores_through_a_filled_site() {
    for (profile, mode) in PROFILES.into_iter().flat_map(|p| MODES.map(|m| (p, m))) {
        let setup = compile(
            r"
            ro = { x = 1, y = 2 }
            function put(t, v) t.x = v end
            for i = 1, 50 do put(ro, i) end
            assert(ro.x == 50)
            ",
            profile,
        );
        let mut vm = new_vm(profile, mode);
        vm.execute(&setup).expect("setup");
        let Value::Table(table) = vm.runtime().get_global("ro") else {
            panic!("ro is a table");
        };
        let check = compile(
            r"
            local ok = pcall(put, ro, 99)
            return ok, ro.x
            ",
            profile,
        );
        vm.runtime_mut().table_set_readonly(table, true).unwrap();
        let number = |value: i64| {
            if profile.is_lua53_or_later() {
                Value::Integer(value)
            } else {
                Value::Number(value as f64)
            }
        };
        let results = vm.execute(&check).expect("readonly run");
        assert_eq!(
            results,
            vec![Value::Boolean(false), number(50)],
            "{profile:?} in {mode:?}"
        );
        vm.runtime_mut().table_set_readonly(table, false).unwrap();
        let results = vm.execute(&check).expect("writable run");
        assert_eq!(
            results,
            vec![Value::Boolean(true), number(99)],
            "{profile:?} in {mode:?}"
        );
    }
}
