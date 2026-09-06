use oximond::runtime::{Runtime, RuntimeResult, Value};
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

fn lua(profile: CompatibilityProfile) -> Lua {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Tracing)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua
}

fn record(source: &str) -> (Vec<u8>, Vec<Value>) {
    record_with(CompatibilityProfile::Lua54, source)
}

fn record_with(profile: CompatibilityProfile, source: &str) -> (Vec<u8>, Vec<Value>) {
    let mut lua = lua(profile);
    lua.vm_mut().start_trace_profile_recording();
    let results = lua.exec(source).unwrap();
    assert!(lua.vm().jit_statistics().trace.traces_recorded > 0);
    (lua.vm().trace_profile().unwrap(), results)
}

#[test]
fn a_fresh_vm_runs_the_first_loop_iteration_natively() {
    let source = "local i,s=0,0; while i<300 do s=s+i; i=i+1 end; return s";
    let (profile, expected) = record(source);
    assert_eq!(expected, vec![Value::Integer(44_850)]);
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
    assert_eq!(stats.iterations, 300, "{stats:?}");
}

/// An unrolled inner loop calls `math.random` at one pc several times, with
/// only the `forloop` check between the calls. Loading the profile recompiles
/// the trace, and invariant code motion or value numbering takes that check's
/// constants out, which left the calls touching. The debug validator then read
/// a call's guard as a replay of the previous call's effect, and the compile
/// panicked.
#[test]
fn a_loaded_profile_compiles_an_unrolled_loop_around_an_effect() {
    let source = "math.randomseed(42); local k; \
                  for i = 1, 52 do for j = 1, 2 do k = math.random(7) end end; return k";
    let (profile, expected) = record(source);
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
}

/// Value numbering takes the constants out of the gap between five calls of
/// `math.random` (found by the difftest campaign, lua54 seed 44).
#[test]
fn a_loaded_profile_compiles_a_longer_unrolled_loop_around_an_effect() {
    let source = r"local function nz(x) if x == 0 then return 1 end return x end
math.randomseed(42)
local g1 = (2070 * (-1))
local numeric_strings = numeric_strings
local k47 = ((-8) // nz(g1))
local k48 = math.abs(k47)
local i53 = 0
while i53 < 45 do
  i53 = i53 + 1
  for j54 = 1, 5 do
    k48 = math.random()
  end
  k48 = k47
end
";
    let (profile, expected) = record(source);
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    assert_eq!(replay.vm().jit_statistics().trace.recording_starts, 0);
}

extern "C" fn twice(value: f64) -> f64 {
    value * 2.0
}

fn twice_callback(_: &mut Runtime, args: &[Value]) -> RuntimeResult<Value> {
    let Value::Number(value) = args[0] else {
        panic!("test callback needs a number");
    };
    Ok(Value::Number(twice(value)))
}

fn install_twice(lua: &mut Lua) {
    let handle = lua
        .runtime_mut()
        .register_native_numeric_unary(twice_callback, twice)
        .unwrap();
    lua.runtime_mut()
        .set_global("twice", Value::NativeFunction(handle));
}

#[test]
fn native_callbacks_relocate_when_registration_indices_change() {
    let source = "local i,s=0.0,0.0; while i<300 do s=s+twice(i); i=i+1 end; return s";
    let mut recording = lua(CompatibilityProfile::Lua54);
    install_twice(&mut recording);
    recording.vm_mut().start_trace_profile_recording();
    let expected = recording.exec(source).unwrap();
    assert_eq!(expected, vec![Value::Number(89_700.0)]);
    let profile = recording.vm().trace_profile().unwrap();
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay
        .runtime_mut()
        .register_native_scalar(|_, _| Ok(Value::Nil))
        .unwrap();
    install_twice(&mut replay);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert_eq!(stats.iterations, 300, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
}

#[test]
fn inlined_closures_relocate_with_upvalues_and_table_writes() {
    let source = "local gain=3; local function add(x) return x+gain end; local t={value=0}; local i=0; while i<300 do t.value=t.value+add(i); i=i+1 end; return t.value";
    let (profile, expected) = record(source);
    assert_eq!(expected, vec![Value::Integer(45_750)]);
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.exec("function unrelated() return 1 end").unwrap();
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert_eq!(stats.iterations, 300, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
}

#[test]
fn branch_side_traces_are_replayed_without_recording() {
    let source =
        // The `%` keeps the arms impure, so the branch keeps its guard.
        "local i,s=0,0; while i<3000 do if i%7==0 then s=(s+2)%100000 else s=(s+1)%100000 end; i=i+1 end; return s";
    let mut recording = lua(CompatibilityProfile::Lua54);
    recording.vm_mut().start_trace_profile_recording();
    let expected = recording.exec(source).unwrap();
    let recorded_stats = recording.vm().jit_statistics().trace;
    assert!(recorded_stats.side_installs > 0, "{recorded_stats:?}");
    let profile = recording.vm().trace_profile().unwrap();
    assert_eq!(expected, vec![Value::Integer(3_429)]);
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert_eq!(stats.side_recordings, 0, "{stats:?}");
    assert_eq!(stats.iterations, 3000, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
}

#[test]
fn incompatible_and_damaged_profiles_are_rejected() {
    let (profile, _) = record("local i=0; while i<300 do i=i+1 end; return i");
    let mut incompatible = lua(CompatibilityProfile::Lua53);
    assert!(incompatible.vm_mut().load_trace_profile(&profile).is_err());
    let mut replay = lua(CompatibilityProfile::Lua54);
    for length in [0, 1, profile.len() / 2, profile.len() - 1] {
        assert!(
            replay
                .vm_mut()
                .load_trace_profile(&profile[..length])
                .is_err()
        );
    }
    let mut trailing = profile.clone();
    trailing.push(0);
    assert!(replay.vm_mut().load_trace_profile(&trailing).is_err());
    let mut interpreted = Lua::new().unwrap();
    assert!(interpreted.vm_mut().load_trace_profile(&profile).is_err());
}

#[test]
fn changed_bytecode_does_not_use_an_old_profile() {
    let (profile, _) = record("local i,s=0,0; while i<300 do s=s+i; i=i+1 end; return s");
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(
        replay
            .exec("local i,s=0,0; while i<8 do s=s+2*i; i=i+1 end; return s")
            .unwrap(),
        vec![Value::Integer(56)]
    );
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.native_entries, 0, "{stats:?}");
    assert_eq!(stats.iterations, 0, "{stats:?}");
}

#[test]
fn replayed_guards_fall_back_when_table_element_types_change() {
    let source =
        "local values=input; local i,s=1,0.0; while i<=300 do s=s+values[i]; i=i+1 end; return s";
    let mut recording = lua(CompatibilityProfile::Lua54);
    recording
        .exec("input={}; for i=1,300 do input[i]=1 end")
        .unwrap();
    recording.vm_mut().start_trace_profile_recording();
    assert_eq!(recording.exec(source).unwrap(), vec![Value::Number(300.0)]);
    let profile = recording.vm().trace_profile().unwrap();
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay
        .exec("input={}; for i=1,300 do if i<=150 then input[i]=1 else input[i]=2.0 end end")
        .unwrap();
    let before = replay.vm().jit_statistics().trace;
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), vec![Value::Number(450.0)]);
    let after = replay.vm().jit_statistics().trace;
    assert!(after.native_entries > before.native_entries, "{after:?}");
    assert!(
        after.side_exits.iter().sum::<u64>() > before.side_exits.iter().sum::<u64>(),
        "{after:?}"
    );
}

#[test]
fn binary_profiles_preserve_non_finite_float_constants_and_signed_zero() {
    let source = "local i,a,b,z=0,0.0,0.0,0.0; while i<300 do a=a+(1.0/0.0); b=b+(0.0/0.0); z=-0.0; i=i+1 end; return a,b,z";
    let (profile, recorded) = record(source);
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    let results = replay.exec(source).unwrap();
    for values in [recorded, results] {
        let [
            Value::Number(infinity),
            Value::Number(nan),
            Value::Number(zero),
        ] = values.as_slice()
        else {
            panic!("expected three floats, got {values:?}");
        };
        assert_eq!(*infinity, f64::INFINITY);
        assert!(nan.is_nan());
        assert_eq!(zero.to_bits(), (-0.0_f64).to_bits());
    }
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert_eq!(stats.iterations, 300, "{stats:?}");
}

#[test]
fn an_installed_profile_runs_again_on_later_function_calls() {
    let source = "local function sum(start) local i,s=0,start; while i<300 do s=s+i; i=i+1 end; return s end; local first=sum(0); return first,sum(0)";
    let (profile, expected) = record(source);
    assert_eq!(
        expected,
        vec![Value::Integer(44_850), Value::Integer(44_850)]
    );
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert!(stats.native_entries >= 2, "{stats:?}");
    assert_eq!(stats.iterations, 600, "{stats:?}");
}

#[test]
fn multiple_entry_type_versions_replay_from_their_first_iteration() {
    let source = "local function sum(start) local i,s=0,start; while i<300 do s=s+i; i=i+1 end; return s end; local integer=sum(0); return integer,sum(0.0)";
    let mut recording = lua(CompatibilityProfile::Lua54);
    recording.vm_mut().start_trace_profile_recording();
    let expected = recording.exec(source).unwrap();
    assert_eq!(
        expected,
        vec![Value::Integer(44_850), Value::Number(44_850.0)]
    );
    let recorded_stats = recording.vm().jit_statistics().trace;
    assert!(recorded_stats.version_installs > 0, "{recorded_stats:?}");
    let profile = recording.vm().trace_profile().unwrap();
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert!(stats.native_entries >= 2, "{stats:?}");
    assert_eq!(stats.iterations, 600, "{stats:?}");
}

#[test]
fn specialized_native_calls_reject_a_different_builtin_at_the_saved_path() {
    let source = "local band=bit.band; local i,s=0LL,0LL; while i<300LL do s=band(s,i); i=i+1LL end; return tonumber(s)";
    let mut recording = lua(CompatibilityProfile::LuaJit21);
    recording.vm_mut().start_trace_profile_recording();
    assert_eq!(recording.exec(source).unwrap(), vec![Value::Number(0.0)]);
    let recorded_stats = recording.vm().jit_statistics().trace;
    assert!(recorded_stats.traces_recorded > 0, "{recorded_stats:?}");
    let profile = recording.vm().trace_profile().unwrap();

    let mut interpreted = Lua::builder()
        .profile(CompatibilityProfile::LuaJit21)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    interpreted.exec("bit.band=bit.bor").unwrap();
    let expected = interpreted.exec(source).unwrap();
    assert_eq!(expected, vec![Value::Number(511.0)]);

    let mut replay = lua(CompatibilityProfile::LuaJit21);
    replay.exec("bit.band=bit.bor").unwrap();
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert!(stats.recording_starts > 0, "{stats:?}");
    assert!(stats.traces_recorded > 0, "{stats:?}");
}

#[test]
fn helper_calls_are_replayed_from_a_profile_by_id() {
    // `t[i % 4 + 1]` is a string borrowed from a table slot, so the length
    // is a helper call (`string.len_ref`) in the recorded trace.
    let source = "local t={'a','bb','ccc','dddd'}; local s,i=0,0; \
                  while i<300 do s=s+string.len(t[i%4+1]); i=i+1 end; return s";
    let (profile, expected) = record(source);
    assert_eq!(expected, vec![Value::Integer(750)]);
    let mut replay = lua(CompatibilityProfile::Lua54);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert_eq!(stats.iterations, 300, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
    assert_eq!(replay.vm().trace_helper_calls(), ["string.len_ref"]);
}

#[test]
fn metatable_helpers_and_the_bit_operations_are_replayed_from_a_profile() {
    // `setmetatable` and `getmetatable` are helpers, a virtual table gets a
    // checked metatable, and `bit.bxor` of numbers is an integer operation.
    let source = "local t, mt = {}, {} local virt = {__index = {}} local s, i = 0, 0 \
         while i < 300 do \
           setmetatable(t, mt) \
           if getmetatable(t) == mt then s = s + 1 end \
           local c = setmetatable({v = i}, virt) c.w = i s = s + c.v + c.w \
           s = s + bit.bxor(i, 5) + bit.rol(i, 3) \
           i = i + 1 \
         end return s";
    let record = |profile: CompatibilityProfile| {
        let mut lua = lua(profile);
        lua.vm_mut().start_trace_profile_recording();
        let results = lua.exec(source).unwrap();
        (lua.vm().trace_profile().unwrap(), results)
    };
    let mut interpreted = Lua::builder()
        .profile(CompatibilityProfile::LuaJit21)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    let expected = interpreted.exec(source).unwrap();
    let (profile, recorded) = record(CompatibilityProfile::LuaJit21);
    assert_eq!(recorded, expected);
    let mut replay = lua(CompatibilityProfile::LuaJit21);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
    let mut helpers = replay.vm().trace_helper_calls();
    helpers.sort_unstable();
    helpers.dedup();
    assert_eq!(
        helpers,
        ["getmetatable", "plain_metatable", "setmetatable"],
        "{stats:?}"
    );
}

#[test]
fn a_profile_with_the_old_format_version_is_rejected() {
    let source = "local i,s=0,0; while i<300 do s=s+i; i=i+1 end; return s";
    let (mut profile, _) = record(source);
    assert!(profile.starts_with(b"OXITRACE\x09"));
    // The same bytes under the version before the host functions were
    // renumbered.
    profile[8] = 1;
    let mut replay = lua(CompatibilityProfile::Lua54);
    let error = replay.vm_mut().load_trace_profile(&profile).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    let message = error.to_string();
    assert!(message.contains("version 1"), "{message}");
    assert!(message.contains("regenerate"), "{message}");
    // Nothing from it was loaded.
    assert_eq!(replay.exec(source).unwrap(), vec![Value::Integer(44_850)]);
    assert_eq!(replay.vm().jit_statistics().trace.recording_starts, 1);
}

#[test]
fn constant_find_and_match_replay_through_the_prepared_helpers_at_level_one() {
    use oximond::compat::TracingOptLevel;
    let source = "local s, n = 'key=value; other=thing', 0 \
                  for i = 1, 3000 do local a, b = string.find(s, '%w+=%w+') \
                  local k = string.match(s, '(%w+)=') n = n + a + b + #k end return n";
    let level_one = || {
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua54)
            .execution_mode(ExecutionMode::Tracing)
            .tracing_opt_flags(TracingOptLevel::O1.flags())
            .build()
            .unwrap();
        lua.vm_mut().use_synchronous_jit_compilation();
        lua
    };
    let mut recording = level_one();
    recording.vm_mut().start_trace_profile_recording();
    let expected = recording.exec(source).unwrap();
    let profile = recording.vm().trace_profile().unwrap();
    let mut replay = level_one();
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert_eq!(stats.recording_starts, 0, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
    assert!(replay.vm().trace_prepared_helper_calls() > 0);
}

/// A side of the recorded tree inlines a `__tostring` closure the root does
/// not, so the side pins one closure more than the root. The run must pin the
/// side's closures too, or an exit inside the inlined call finds no closure
/// for its frame and panics (found by the difftest campaign, lua51 seed
/// 8653177503505483396).
#[test]
fn a_loaded_tree_pins_the_closures_its_sides_inline() {
    let source = r#"local function fold(v) end
local A = {__tostring = function(o) return "H" .. o.hits end}
local B = {__tostring = function(o) return "B" end}
local objs = {}
for k = 1, 8 do objs[k] = setmetatable({hits = 0}, k % 2 == 0 and A or B) end
local last
for i = 1, 2277 do
  last = tostring(objs[(i * 3 + 1) % 8 + 1])
  fold(last)
end
return last
"#;
    let (profile, expected) = record_with(CompatibilityProfile::Lua51, source);
    let mut replay = lua(CompatibilityProfile::Lua51);
    replay.vm_mut().load_trace_profile(&profile).unwrap();
    assert_eq!(replay.exec(source).unwrap(), expected);
    let stats = replay.vm().jit_statistics().trace;
    assert!(stats.native_entries > 0, "{stats:?}");
}
