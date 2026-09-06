//! Loops that call one prototype through a different closure per object,
//! whose closures call functions they hold in cells (Mineclonia's xoroshiro
//! self-test, random.lua:1159: `fork:next_integer`, a new closure for each
//! fork, calls `fn`, the fork's own `getlong`, which calls the ULL helpers
//! the module's main chunk holds). The call of a closure read through the
//! live closure's cell is guarded on its prototype when each closure of the
//! reader captured the cell anew, and on its identity when the cell is the
//! enclosing closure's own. The loops run natively, and every program gives
//! the interpreter's result under each profile, also when a cell or a field
//! takes another function, a callable table or a native mid-loop.

use oximond::runtime::Value;
use oximond::vm::{TraceAbortReason, TraceGuardKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua};
use std::fmt::Write as _;

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua51,
    CompatibilityProfile::LuaJit21,
];

fn outcome(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` in both modes of every profile; the results agree.
fn run_each(source: &str) -> Vec<TraceStatistics> {
    let mut all = Vec::new();
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?}"
        );
        all.push(lua.vm().jit_statistics().trace);
    }
    all
}

/// The generator of the xoroshiro self-test, cut down: `make_fn` makes the
/// step closure over a state table, `make` the object whose method calls
/// it, and the step calls `addk`, a local of the main chunk that every step
/// closure shares, and `floor`, a native.
const FORKS: &str = r"
local floor = math.floor
local function addk(a, k)
  a[1] = (a[1] * 75 + k) % 65537
end
local function make_fn(seed)
  local lo = { seed }
  local function getlong()
    addk(lo, 74)
    return lo
  end
  return getlong
end
local function make(seed)
  local fn = make_fn(seed)
  return {
    next_integer = function(self)
      local ull = fn()
      return floor(ull[1] / 3) % 3000
    end,
  }
end
";

/// The loop of the self-test: a new fork for each run of the inner loop.
const FORK_LOOP: &str = r"
local other = make(7)
local total = 0
for i = 1, 60 do
  local fork = make(i)
  for j = 1, 500 do
    local a, b = fork:next_integer(), other:next_integer()
    total = (total + a * 3 + b) % 1000003
  end
end
";

#[test]
fn a_method_of_a_new_closure_per_object_runs_natively_in_the_main_chunk() {
    // The loop runs in the main chunk, as the self-test does: `addk` is a
    // captured local of the loop's frame, which each fork's step reads
    // through its closure.
    let source = format!("{FORKS}{FORK_LOOP}return total\n");
    for stats in run_each(&source) {
        assert!(stats.iterations > 20_000, "{stats:?}");
        assert!(stats.closure_helper_calls > 0, "{stats:?}");
        assert!(
            stats.side_exits_of(TraceGuardKind::Callee) < 200,
            "{stats:?}"
        );
    }
}

#[test]
fn a_method_of_a_new_closure_per_object_runs_natively_in_a_function() {
    let source = format!(
        "{FORKS}
local function run()
{FORK_LOOP}
  return total
end
return run()
"
    );
    for stats in run_each(&source) {
        assert!(stats.iterations > 20_000, "{stats:?}");
        assert!(
            stats.side_exits_of(TraceGuardKind::Callee) < 200,
            "{stats:?}"
        );
    }
}

/// A fork whose step cell takes a closure of another prototype half way:
/// the prototype guard of the call through the cell leaves, and the
/// interpreter runs the new step.
#[test]
fn a_cell_that_takes_a_closure_of_another_prototype_leaves_to_the_interpreter() {
    let source = format!(
        "{FORKS}
local function make_swappable(seed)
  local fn = make_fn(seed)
  local object = {{}}
  object.next_integer = function(self)
    local ull = fn()
    return floor(ull[1] / 3) % 3000
  end
  object.swap = function()
    local n = 0
    fn = function() n = n + 1 return {{ n * 7 }} end
  end
  return object
end
local other = make(7)
local total = 0
for i = 1, 40 do
  local fork = make_swappable(i)
  for j = 1, 400 do
    if j == 200 and i % 3 == 0 then fork.swap() end
    local a, b = fork:next_integer(), other:next_integer()
    total = (total + a * 3 + b) % 1000003
  end
end
return total
"
    );
    for stats in run_each(&source) {
        assert!(stats.iterations > 1_000, "{stats:?}");
    }
}

/// A step cell that takes a callable table (set with `debug.setupvalue`):
/// no closure of the prototype, so the read of the cell leaves before the
/// call.
#[test]
fn a_cell_that_takes_a_callable_table_leaves_to_the_interpreter() {
    let source = format!(
        "{FORKS}
local function set_fn(object, value)
  local i = 1
  while true do
    local name = debug.getupvalue(object.next_integer, i)
    if name == nil then return end
    if name == 'fn' then
      debug.setupvalue(object.next_integer, i, value)
      return
    end
    i = i + 1
  end
end
local callable = setmetatable({{}}, {{ __call = function() return {{ 11 }} end }})
local other = make(7)
local total = 0
for i = 1, 40 do
  local fork = make(i)
  for j = 1, 400 do
    if j == 150 and i % 4 == 3 then set_fn(fork, callable) end
    local a, b = fork:next_integer(), other:next_integer()
    total = (total + a * 3 + b) % 1000003
  end
end
return total
"
    );
    for stats in run_each(&source) {
        assert!(stats.iterations > 1_000, "{stats:?}");
    }
}

/// The method field takes a native mid-loop, and an object gets its method
/// from a metatable instead of its own field.
#[test]
fn a_method_field_that_turns_native_or_moves_to_a_metatable_matches_the_interpreter() {
    let source = format!(
        "{FORKS}
local other = make(7)
local total = 0
for i = 1, 40 do
  local fork = make(i)
  if i % 5 == 0 then
    fork = setmetatable({{}}, {{ __index = fork }})
  end
  for j = 1, 400 do
    if j == 250 and i % 7 == 0 then
      fork.next_integer = type
    end
    local a = fork:next_integer()
    if type(a) ~= 'number' then a = 1 end
    local b = other:next_integer()
    total = (total + a * 3 + b) % 1000003
  end
end
return total
"
    );
    for stats in run_each(&source) {
        assert!(stats.iterations > 1_000, "{stats:?}");
    }
}

/// The step writes a counter that the loop's frame holds as a captured
/// local, and the loop reads it: a write through a closure would leave the
/// trace's register stale, so such a trace keeps the identity guard or
/// leaves; the counts match the interpreter's.
#[test]
fn a_step_that_writes_a_captured_local_of_the_loop_matches_the_interpreter() {
    let source = r"
local floor = math.floor
local steps = 0
local function make_fn(seed)
  local lo = { seed }
  return function()
    steps = steps + 1
    lo[1] = (lo[1] * 75 + 74) % 65537
    return lo
  end
end
local function make(seed)
  local fn = make_fn(seed)
  return {
    next_integer = function(self)
      local ull = fn()
      return floor(ull[1] / 3) % 3000
    end,
  }
end
local other = make(7)
local total = 0
for i = 1, 40 do
  local fork = make(i)
  for j = 1, 400 do
    local a, b = fork:next_integer(), other:next_integer()
    total = (total + a * 3 + b + steps) % 1000003
  end
end
return total, steps
";
    run_each(source);
}

/// Forty closures of one constructor, each in a field of its own object,
/// called in turn through `p.test`: more callees than a chain of sides holds
/// (`MAX_SIDE_DEPTH`). The root's identity guard of the call keeps leaving
/// past the chain; the root retires, its loop remembers the call site, and
/// the next recording guards the call on the prototype, which runs every
/// closure in one trace.
const MANY_CLOSURES: &str = r"
local floor = math.floor
local function mix(a, b) return (a * 31 + b) % 65521 end
local function build(spacing, salt)
  return function(x)
    return mix(floor(x / spacing), salt)
  end
end
local objects = {}
for i = 1, 40 do objects[i] = { test = build(i + 2, i * 7) } end
local function starts(p, x)
  local v = p.test(x)
  return v
end
";

#[test]
fn a_call_site_that_outgrows_a_chain_of_sides_is_guarded_on_the_prototype() {
    let source = format!(
        "{MANY_CLOSURES}
local function run()
  local s = 0
  for i = 1, 150000 do
    local p = objects[i % 40 + 1]
    s = (s + starts(p, i)) % 1000003
  end
  return s
end
return run()
"
    );
    for stats in run_each(&source) {
        // The tree of identity guards retires within the first 60000
        // iterations; the prototype-guarded root runs the rest.
        assert!(stats.iterations > 80_000, "{stats:?}");
        assert!(stats.closure_helper_calls > 0, "{stats:?}");
    }
}

/// The same call site, where every tenth object holds a closure of another
/// prototype and two objects swap theirs mid-loop: the prototype guard
/// leaves for those, and the results match the interpreter's.
#[test]
fn a_prototype_guarded_call_site_that_meets_another_prototype_matches_the_interpreter() {
    let source = format!(
        "{MANY_CLOSURES}
local function other(salt)
  return function(x) return (x * 3 + salt) % 977 end
end
for i = 10, 40, 10 do objects[i] = {{ test = other(i) }} end
local function run()
  local s = 0
  for i = 1, 150000 do
    if i == 90000 then
      objects[3].test = other(5)
      objects[10].test = build(4, 9)
    end
    local p = objects[i % 40 + 1]
    s = (s + starts(p, i)) % 1000003
  end
  return s
end
return run()
"
    );
    for stats in run_each(&source) {
        assert!(stats.iterations > 10_000, "{stats:?}");
    }
}

/// Six closures of one constructor called in turn: fewer than a chain of
/// sides holds, so a root of identity guards would keep a side per closure
/// and never retire (structures.lua loop@744's tree grew five to seven such
/// sides and was evicted for room). The second side asked for at the call
/// names the site instead, and the loop records again with the call guarded
/// on the prototype: one root and at most one side carry every closure.
#[test]
fn a_call_site_that_takes_turns_between_a_few_closures_names_itself_before_its_tree_grows() {
    let source = format!(
        "{MANY_CLOSURES}
local function run()
  local s = 0
  for i = 1, 60000 do
    local p = objects[i % 6 + 1]
    s = (s + starts(p, i)) % 1000003
  end
  return s
end
return run()
"
    );
    for stats in run_each(&source) {
        assert!(stats.iterations > 55_000, "{stats:?}");
        assert!(stats.closure_helper_calls > 0, "{stats:?}");
        assert!(stats.side_installs <= 2, "{stats:?}");
        assert!(stats.traces_compiled <= 3, "{stats:?}");
    }
}

/// Two closures taking turns at a call: one side beside the root carries
/// the second, and the call keeps its identity guard (no read through the
/// closure).
#[test]
fn a_call_site_that_takes_turns_between_two_closures_keeps_identity_guards() {
    let source = format!(
        "{MANY_CLOSURES}
local function run()
  local s = 0
  for i = 1, 60000 do
    local p = objects[i % 2 + 1]
    s = (s + starts(p, i)) % 1000003
  end
  return s
end
return run()
"
    );
    for stats in run_each(&source) {
        assert!(stats.iterations > 55_000, "{stats:?}");
        assert_eq!(stats.closure_helper_calls, 0, "{stats:?}");
        assert!(stats.side_installs >= 1, "{stats:?}");
    }
}

/// Closures of one constructor with a branch whose other arm turns hot after
/// the prototype-guarded root installed: the side for that arm starts
/// inside the inlined call, in a frame that runs whichever closure of the
/// prototype the call had, and reads the closure's cells through the
/// closure the exit pushed the frame with.
#[test]
fn a_side_starts_inside_a_call_guarded_on_the_prototype() {
    let source = r"
local floor = math.floor
local function mix(a, b) return (a * 31 + b) % 65521 end
local function build(spacing, salt)
  return function(x, wide)
    if wide then
      return mix(floor(x / spacing), salt) + spacing
    end
    return mix(x % spacing, salt)
  end
end
local objects = {}
for i = 1, 40 do objects[i] = { test = build(i + 2, i * 7) } end
local function starts(p, x, wide)
  local v = p.test(x, wide)
  return v
end
local function run()
  local s = 0
  for i = 1, 200000 do
    local p = objects[i % 40 + 1]
    s = (s + starts(p, i, i > 100000 and i % 2 == 0)) % 1000003
  end
  return s
end
return run()
";
    for stats in run_each(source) {
        assert!(stats.iterations > 100_000, "{stats:?}");
        assert!(stats.side_installs > 0, "{stats:?}");
        assert!(stats.closure_helper_calls > 0, "{stats:?}");
    }
}

/// The shape of `get_structure_starts` (structures.lua:744): a loop over
/// the structure sets whose branches test `cx` and `cz`, arguments that hold
/// one value for a whole run of the loop and another in the next. The
/// optimizer hoists those branch guards in front of the loop, where no side
/// can start, so the first root leaves at loop entry whenever a run takes
/// the other way. Its retirement keeps the guards in the loop for the next
/// recording, whose sides then carry the other ways.
const HOISTED: &str = r"
local floor = math.floor
local sets = {}
for i = 1, 24 do sets[i] = { spacing = i + 3, salt = i * 11 } end
local function shift(v, set) return floor(v / set.spacing) + set.salt end
local function starts(cx, cz)
  local n = 0
  for _, set in ipairs(sets) do
    local v = (cx * set.salt + cz) % set.spacing
    if cx < 0 then v = shift(v - cx, set) end
    if cz < 0 then v = shift(v * 3 - cz, set) end
    n = n + v
  end
  return n
end
";

/// Runs `starts` 4000 times with the signs of `cx` and `cz` taking turns
/// between runs; `edit` runs before run `r`.
fn hoisted_source(edit: &str) -> String {
    format!(
        "{HOISTED}
local total = 0
for r = 1, 4000 do
  {edit}
  local cx = r % 2 == 0 and -r or r
  local cz = floor(r / 2) % 2 == 0 and -r or r
  total = (total + starts(cx, cz)) % 1000003
end
return total
"
    )
}

#[test]
fn a_loop_that_leaves_at_hoisted_guards_compiles_again_with_them_in_the_loop() {
    // 4000 runs of 24 iterations: 96000. The first root runs a few dozen
    // iterations before it leaves at its hoisted guards in a quarter of its
    // entries; it then compiles again in place with the guards in the loop
    // (`InstalledTrace::keep_hoisted_guards`), or, on the retirement path,
    // a second root does. That code and its sides run nearly all the rest.
    for stats in run_each(&hoisted_source("")) {
        assert!(
            stats.traces_compiled + stats.link_recompiles >= 2,
            "{stats:?}"
        );
        assert!(stats.side_installs >= 2, "{stats:?}");
        assert!(stats.iterations > 80_000, "{stats:?}");
        // The branch guards no longer leave at each run's entry: the sides
        // continue them, and the few exits are the ones before the sides
        // landed.
        assert!(
            stats.side_exits_of(TraceGuardKind::Branch) < 400,
            "{stats:?}"
        );
    }
}

/// The loop with kept guards meets a set that turns into a table with a
/// metatable, a set whose field goes missing, and a set whose `salt` turns
/// into a string: each leaves to the interpreter, which gives the same
/// result (or error) as it would alone.
#[test]
fn a_loop_with_kept_guards_matches_the_interpreter_when_its_data_changes() {
    for edit in [
        "if r == 3000 then sets[5] = setmetatable({}, { __index = { spacing = 9, salt = 4 } }) end",
        "if r == 3000 then sets[7].spacing = 2.5 end",
        "if r == 3000 then sets[9].salt = '13' end",
        "if r == 3000 then sets[30] = sets[2]; sets[25] = sets[3] end",
        "if r == 3000 then sets = { sets[1], sets[2] } end",
    ] {
        for stats in run_each(&hoisted_source(edit)) {
            assert!(stats.iterations > 40_000, "{edit}: {stats:?}");
        }
    }
}

/// A call site that takes turns between closures of one prototype whose
/// cells hold a string, a different one in each closure (the `id` of
/// Mineclonia's concentric ring `chunk_test`, structures.lua:404, which
/// indexes a table with it). The recording that guards the call on the
/// prototype reads the string through the live closure into a lane of the
/// run, and the loop runs natively. `edit` runs before iteration `i`.
fn string_cell_source(edit: &str) -> String {
    format!(
        r#"
local floor, assert = math.floor, assert
local hashes = {{}}
local function build(spacing, salt, id)
  return function(x)
    local h = hashes[id]
    assert(h, "no hash")
    return (floor(x / spacing) * salt + h) % 977
  end
end
local objects = {{}}
for i = 1, 40 do
  local id = "set" .. i
  hashes[id] = i * 3
  objects[i] = {{ test = build(i + 2, i * 7, id) }}
end
local function starts(p, x)
  local v = p.test(x)
  return v
end
local function run()
  local s = 0
  for i = 1, 150000 do
    {edit}
    local p = objects[i % 40 + 1]
    s = (s + starts(p, i)) % 1000003
  end
  return s
end
return run()
"#
    )
}

#[test]
fn a_prototype_guarded_call_that_reads_a_string_cell_runs_natively() {
    for stats in run_each(&string_cell_source("")) {
        assert!(stats.iterations > 80_000, "{stats:?}");
        assert!(stats.closure_helper_calls > 0, "{stats:?}");
        assert!(
            stats.side_exits_of(TraceGuardKind::Callee) < 400,
            "{stats:?}"
        );
    }
}

/// The string cell's closure meets a table with a metatable, a missing key
/// (the closure's `assert` raises the interpreter's error) and a string
/// cell that turns into a number: each leaves to the interpreter, which
/// gives the same result, or error, as it would alone.
#[test]
fn a_string_cell_read_through_the_closure_matches_the_interpreter_when_it_changes() {
    for edit in [
        "if i == 90000 then setmetatable(hashes, { __index = function() return 5 end }); hashes.set3 = nil end",
        "if i == 90000 then objects[7] = { test = build(4, 9, 'missing') } end",
        "if i == 90000 then objects[9] = { test = build(4, 9, 12) }; hashes[12] = 8 end",
    ] {
        let source = string_cell_source(edit);
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(
                format!("{actual:?}"),
                format!("{expected:?}"),
                "{edit}: {profile:?}"
            );
            let stats = lua.vm().jit_statistics().trace;
            assert!(stats.iterations > 40_000, "{edit}: {stats:?}");
        }
    }
}

/// A call site that takes turns between closures of one prototype whose
/// cells hold a value a frame that reads cells through the live closure
/// cannot read (a coroutine): the recording that guards the call on the
/// prototype aborts. The loop goes back to identity guards instead of
/// aborting at the same cell until the header is rejected, and the results
/// match the interpreter's.
#[test]
fn a_prototype_guarded_call_that_reads_a_coroutine_cell_goes_back_to_identity_guards() {
    let source = r"
local floor = math.floor
local function build(spacing, salt)
  local co = coroutine.create(function() end)
  return function(x)
    local h = co and 1 or 0
    return (floor(x / spacing) * salt + h) % 977
  end
end
local objects = {}
for i = 1, 40 do
  objects[i] = { test = build(i + 2, i * 7) }
end
local function starts(p, x)
  local v = p.test(x)
  return v
end
local function run()
  local s = 0
  for i = 1, 150000 do
    local p = objects[i % 40 + 1]
    s = (s + starts(p, i)) % 1000003
  end
  return s
end
return run()
";
    for stats in run_each(source) {
        // One abort at the coroutine cell, not the three in a row that
        // reject the header; the identity-guarded roots compile again
        // after it.
        assert!(
            stats.aborts(TraceAbortReason::UnsupportedOperand) <= 1,
            "{stats:?}"
        );
        assert!(stats.traces_compiled >= 2, "{stats:?}");
    }
}

/// Closures of one constructor that read globals (`assert` and `bonus`), as
/// Mineclonia's concentric ring `chunk_test` reads `assert`
/// (structures.lua:405), called in turn at one site. `edit` runs before
/// iteration `i`.
fn global_reader_source(edit: &str) -> String {
    format!(
        r#"
local floor = math.floor
local function build(spacing, salt)
  return function(x)
    local v = floor(x / spacing) * salt
    assert(v >= 0, "negative")
    return v % 977 + bonus
  end
end
bonus = 3
local objects = {{}}
for i = 1, 40 do objects[i] = {{ test = build(i + 2, i * 7) }} end
local function starts(p, x)
  local v = p.test(x)
  return v
end
local function run()
  local s = 0
  for i = 1, 150000 do
    {edit}
    local p = objects[i % 40 + 1]
    s = (s + starts(p, i)) % 1000003
  end
  return s
end
return run()
"#
    )
}

/// In Lua 5.1 and `LuaJIT` a global read is `GetGlobal` through the
/// closure's environment: the frame guarded on the prototype checks that
/// the live closure has the pinned closure's environment and reads the
/// pinned closure's global, and the loop runs natively. Lua 5.4 reads it
/// with `GetTabUp` of `_ENV`, which such a frame does not record; the loop
/// keeps identity guards there, and the results match in every profile.
#[test]
fn a_prototype_guarded_call_that_reads_globals_runs_natively() {
    let source = global_reader_source("");
    let all = run_each(&source);
    for (profile, stats) in PROFILES.iter().zip(&all) {
        if *profile == CompatibilityProfile::Lua54 {
            continue;
        }
        assert!(stats.iterations > 80_000, "{profile:?}: {stats:?}");
        assert!(stats.closure_helper_calls > 0, "{profile:?}: {stats:?}");
        assert!(
            stats.side_exits_of(TraceGuardKind::Callee) < 400,
            "{profile:?}: {stats:?}"
        );
    }
}

/// A closure that `setfenv` gives another environment mid-loop leaves at
/// the environment guard, and an `assert` that fails raises the
/// interpreter's error. A loop that stores a global its callee reads does
/// not compile (a trace that holds a global in a slot does not store it),
/// and gives the interpreter's result.
#[test]
fn globals_read_through_a_prototype_guarded_frame_match_the_interpreter_when_they_change() {
    for (edit, fenv, native) in [
        (
            "if i == 90000 then setfenv(objects[5].test, { assert = assert, bonus = 100 }) end",
            true,
            true,
        ),
        ("if i == 90000 then bonus = 11 end", false, false),
        (
            "if i == 90000 then objects[6] = { test = build(-3, 5) } end",
            false,
            true,
        ),
    ] {
        let source = global_reader_source(edit);
        for profile in PROFILES {
            if fenv && profile == CompatibilityProfile::Lua54 {
                continue;
            }
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(
                format!("{actual:?}"),
                format!("{expected:?}"),
                "{edit}: {profile:?}"
            );
            // Lua 5.4 keeps identity guards at this site (see above).
            if native && profile != CompatibilityProfile::Lua54 {
                let stats = lua.vm().jit_statistics().trace;
                assert!(stats.iterations > 40_000, "{edit}: {profile:?}: {stats:?}");
            }
        }
    }
}

/// A loop of three trips whose body makes 32 calls, each reading fields of
/// the element: the shape of Mineclonia's loop over a chunk's structure
/// starts (`structures.lua:744`), which makes a dozen calls in about two
/// trips per entry. `edit` runs in the loop's caller on call `n`.
fn heavy_short_loop_source(edit: &str) -> String {
    let mut source = String::from("local calls = {}\n");
    for k in 1..=32 {
        let _ = writeln!(
            source,
            "calls[{k}] = function(s) return (s.x * {k} + s.y - s.z) % 1009 end"
        );
    }
    source.push_str("local ");
    source.push_str(
        &(1..=32)
            .map(|k| format!("f{k}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    source.push_str(" = ");
    source.push_str(
        &(1..=32)
            .map(|k| format!("calls[{k}]"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    source.push_str("\nlocal function visit(starts)\n  local sum = 0\n  for _, s in ipairs(starts) do\n    sum = sum");
    for k in 1..=32 {
        let _ = write!(source, " + f{k}(s)");
    }
    let _ = write!(
        source,
        "
  end
  return sum
end
local starts = {{ {{ x = 1, y = 2, z = 3 }}, {{ x = 4, y = 5, z = 6 }}, {{ x = 7, y = 8, z = 9 }} }}
local total = 0
for n = 1, 3000 do
  {edit}
  total = (total + visit(starts)) % 1000003
end
return total
"
    );
    source
}

/// Its trace has more work per iteration than an entry and exit cost, so it
/// stays installed at two loop-backs per entry, where a bar of three or eight
/// would retire it.
#[test]
fn a_loop_of_a_few_trips_whose_body_makes_many_calls_stays_native() {
    for stats in run_each(&heavy_short_loop_source("")) {
        assert!(stats.iterations >= 5_000, "{stats:?}");
    }
}

/// The same loop leaves to the interpreter and gives its result when the
/// trips change, an element loses a field, or a field turns float.
#[test]
fn a_heavy_short_loop_matches_the_interpreter_when_its_trips_or_calls_change() {
    for edit in [
        "if n == 2000 then starts[4] = { x = 1, y = 1, z = 1 } end",
        "if n == 2000 then starts[3] = nil end",
        "if n == 2000 then starts = {} end",
        "if n == 2000 then starts[2].y = 2.5 end",
        "if n == 2900 then starts[2].y = nil end",
    ] {
        let source = heavy_short_loop_source(edit);
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, _) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(
                format!("{actual:?}"),
                format!("{expected:?}"),
                "{edit}: {profile:?}"
            );
        }
    }
}
