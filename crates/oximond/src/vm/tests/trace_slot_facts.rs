//! Slot facts (`recorder::slot_facts`): a read of an array slot the
//! recording already read or stored is a move of the value it knows, with no
//! guard. The shape is Mineclonia's 64-bit helpers on `{ lo, hi }` pairs: a
//! call that reads both halves of two pairs and stores both halves of the
//! first, made twice per iteration.

use super::*;
use crate::jit_ir::trace::{TraceIr, TraceOp};
use crate::vm::jit::trace::TraceStatistics;

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

/// The root trace the recorder queued for `source`, which it does not compile.
fn recorded(profile: CompatibilityProfile, source: &str) -> TraceIr {
    let mut lua = crate::Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Tracing)
        .build()
        .unwrap();
    lua.vm_mut().jit.trace.set_native_compilation(false);
    lua.exec(source).unwrap();
    let queued = lua.vm().jit.trace.queued();
    queued.front().expect("a recording").ir.clone()
}

/// `source`'s results under `mode`, with synchronous compilation, and the
/// trace statistics.
fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Vec<Value>, TraceStatistics) {
    let mut lua = crate::Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}"));
    (results, lua.vm().jit_statistics().trace)
}

/// Runs `source` in the interpreter and traced: the results must agree, and
/// the loop must have run in native code without an abort.
fn agrees_natively(profile: CompatibilityProfile, source: &str) -> TraceStatistics {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(actual, expected, "{profile:?}");
    assert!(stats.traces_compiled >= 1, "{profile:?}: {stats:?}");
    assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
    assert!(stats.iterations > 2_000, "{profile:?}: {stats:?}");
    assert_eq!(stats.total_aborts(), 0, "{profile:?}: {stats:?}");
    stats
}

fn array_loads(ir: &TraceIr) -> usize {
    ir.body
        .iter()
        .chain(&ir.preheader)
        .filter(|instruction| {
            matches!(
                instruction.op,
                TraceOp::ArrayLoad { .. } | TraceOp::LaneArrayLoad { .. }
            )
        })
        .count()
}

/// `add (x, y)` twice, with `x` and `y` in the window slots `with` names.
fn pairs(with: &str) -> String {
    format!(
        "
local function add(a, b)
  local lo = a[1] + b[1]
  local hi = a[2] + b[2]
  a[1] = lo % 65536
  a[2] = hi % 65536
end
local x, y = {{1, 2}}, {{3, 4}}
{with}
local function run(n)
  for i = 1, n do
    add(x, y)
    add(x, y)
  end
  return x[1], x[2], y[1], y[2]
end
return run(4000)
"
    )
}

#[test]
fn a_slot_the_iteration_stored_is_not_read_again() {
    for profile in PROFILES {
        let ir = recorded(profile, &pairs(""));
        ir.verify().unwrap();
        // The first call reads x[1], y[1], x[2], y[2]. Its stores into x[1]
        // and x[2] keep x's facts and kill y's (any table may be x), so the
        // second call reads only y[1] and y[2] again.
        assert_eq!(array_loads(&ir), 6, "{profile:?}");
        let stats = agrees_natively(profile, &pairs(""));
        // The second call's reads of x[1] and x[2].
        assert_eq!(stats.slot_reads_forwarded, 2, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_second_name_for_the_stored_table_reads_the_stored_value() {
    for profile in PROFILES {
        // `y` is `x`: the store through `a` must reach the read through `b`.
        let source = pairs("y = x");
        let ir = recorded(profile, &source);
        ir.verify().unwrap();
        agrees_natively(profile, &source);
        // Two names in two window slots are two operands: the reads through
        // `b` are loads, of what the stores through `a` left.
        assert_eq!(array_loads(&ir), 6, "{profile:?}");
    }
}

#[test]
fn a_trace_recorded_on_two_tables_runs_right_on_one() {
    for profile in PROFILES {
        // The trace is recorded with `a` and `b` two tables, then entered
        // with both one table: the store through `a` must reach the second
        // call's read through `b`, which therefore cannot be forwarded from
        // the first call's read.
        let source = "
local function add(a, b)
  local lo = a[1] + b[1]
  local hi = a[2] + b[2]
  a[1] = lo % 65536
  a[2] = hi % 65536
end
local function run(n, a, b)
  for i = 1, n do
    add(a, b)
    add(a, b)
  end
  return a[1], a[2]
end
local x, y = {1, 2}, {3, 4}
local p, q = run(4000, x, y)
local z = {5, 6}
local r, s = run(4000, z, z)
return p, q, r, s
";
        let stats = agrees_natively(profile, source);
        assert!(stats.native_entries >= 2, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_call_with_one_table_twice_forwards_its_own_reads() {
    for profile in PROFILES {
        let source = "
local function add(a, b)
  local lo = a[1] + b[1]
  local hi = a[2] + b[2]
  a[1] = lo % 65536
  a[2] = hi % 65536
end
local x = {1, 2}
local function run(n)
  for i = 1, n do add(x, x) end
  return x[1], x[2]
end
return run(4000)
";
        let ir = recorded(profile, source);
        ir.verify().unwrap();
        // b[1] is a[1], b[2] is a[2].
        assert_eq!(array_loads(&ir), 2, "{profile:?}");
        agrees_natively(profile, source);
    }
}

#[test]
fn a_store_under_an_unknown_key_kills_the_facts() {
    for profile in PROFILES {
        // `t[k]` may be `x[1]`: the second call reads both pairs again.
        let source = "
local function add(a, b)
  local lo = a[1] + b[1]
  local hi = a[2] + b[2]
  a[1] = lo % 65536
  a[2] = hi % 65536
end
local x, y = {1, 2}, {3, 4}
local t, k = x, 1
local function run(n)
  for i = 1, n do
    add(x, y)
    t[k] = 7
    add(x, y)
  end
  return x[1], x[2], y[1], y[2]
end
return run(4000)
";
        let ir = recorded(profile, source);
        ir.verify().unwrap();
        assert_eq!(array_loads(&ir), 8, "{profile:?}");
        agrees_natively(profile, source);
    }
}

#[test]
fn a_library_call_that_writes_tables_kills_the_facts() {
    for profile in PROFILES {
        // `rawset` stores through the VM, behind the trace's back.
        let source = "
local function add(a, b)
  local lo = a[1] + b[1]
  local hi = a[2] + b[2]
  a[1] = lo % 65536
  a[2] = hi % 65536
end
local x, y = {1, 2}, {3, 4}
local rawset = rawset
local function run(n)
  for i = 1, n do
    add(x, y)
    rawset(x, 1, i)
    add(x, y)
  end
  return x[1], x[2], y[1], y[2]
end
return run(4000)
";
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
    }
}

#[test]
fn a_slot_whose_type_changes_leaves_through_the_first_read() {
    for profile in PROFILES {
        // From iteration 3000 on, x[1] holds a string: the first read's
        // guard leaves, and the forwarded read never runs on it.
        let source = "
local function add(a, b)
  local lo = a[1] + b[1]
  local hi = a[2] + b[2]
  a[1] = lo % 65536
  a[2] = hi % 65536
end
local x, y = {1, 2}, {3, 4}
local function run(n)
  local s = 0
  for i = 1, n do
    if i == 3000 then x[1] = '5' end
    add(x, y)
    add(x, y)
    s = s + x[1]
  end
  return s, x[1], x[2]
end
return run(4000)
";
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(stats.traces_compiled >= 1, "{profile:?}: {stats:?}");
    }
}

fn ops_matching(ir: &TraceIr, pick: impl Fn(&TraceOp) -> bool) -> usize {
    ir.body
        .iter()
        .chain(&ir.preheader)
        .filter(|instruction| pick(&instruction.op))
        .count()
}

#[test]
fn a_store_into_a_slot_the_iteration_reached_takes_no_guard() {
    for profile in PROFILES {
        let ir = recorded(profile, &pairs(""));
        ir.verify().unwrap();
        // The first call read x[1] and x[2] before storing them, and the
        // second call stores what the first stored: all four stores are
        // proven, and none takes a guard or a snapshot.
        assert_eq!(
            ops_matching(&ir, |op| matches!(op, TraceOp::ArrayOverwrite { .. })),
            4,
            "{profile:?}"
        );
        assert_eq!(
            ops_matching(&ir, |op| matches!(op, TraceOp::ArrayStore { .. })),
            0,
            "{profile:?}"
        );
        let stats = agrees_natively(profile, &pairs(""));
        assert_eq!(stats.array_overwrites, 4, "{profile:?}: {stats:?}");
    }
}

#[test]
fn the_verifier_refuses_an_overwrite_with_no_earlier_access() {
    for profile in PROFILES {
        let mut ir = recorded(profile, &pairs(""));
        // Point the first overwrite at a key no access reached: the value
        // it stores, a number the iteration computed.
        let at = ir
            .body
            .iter()
            .position(|instruction| matches!(instruction.op, TraceOp::ArrayOverwrite { .. }))
            .expect("an overwrite");
        if let TraceOp::ArrayOverwrite { key, value, .. } = &mut ir.body[at].op {
            *key = *value;
        }
        let error = ir.verify().unwrap_err().to_string();
        assert!(error.contains("no earlier access"), "{profile:?}: {error}");
    }
}

#[test]
fn a_metatable_that_appears_mid_loop_keeps_the_overwrites_right() {
    for profile in PROFILES {
        // `__newindex` runs only for a key whose raw value is nil, so the
        // overwrites of x's present slots do not call it; the `rawset` in
        // it would add one to the value if they did. `setmetatable` is a
        // library call between the two calls: it kills the facts, and the
        // second call's stores are guarded again in that recording.
        let source = "
local function add(a, b)
  local lo = a[1] + b[1]
  local hi = a[2] + b[2]
  a[1] = lo % 65536
  a[2] = hi % 65536
end
local x, y = {1, 2}, {3, 4}
local hits = 0
local function run(n)
  for i = 1, n do
    add(x, y)
    if i == 3000 then
      setmetatable(x, {__newindex = function(t, k, v) hits = hits + 1; rawset(t, k, v + 1) end})
    end
    add(x, y)
  end
  return x[1], x[2], y[1], y[2], hits, x[3]
end
return run(4000)
";
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(stats.traces_compiled >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
        assert!(stats.array_overwrites >= 4, "{profile:?}: {stats:?}");
        assert!(
            stats.exits.iter().sum::<u64>() >= 1,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_hole_made_through_another_name_leaves_before_the_overwrite() {
    for profile in PROFILES {
        // `y` is `x`: at iteration 3000 a store through `y` makes x[1] nil
        // and fills it again, between the two calls. The store under `y`
        // kills x's facts of key 1, so the second call reads x[1] again
        // with a guard; a later iteration makes the hole permanent with a
        // float in x[2], and the guards leave.
        let source = "
local function add(a, b)
  local lo = a[1] + b[1]
  local hi = a[2] + b[2]
  a[1] = lo % 65536
  a[2] = hi % 65536
end
local x = {1, 2}
local y = x
local z = {3, 4}
local function run(n)
  for i = 1, n do
    add(x, z)
    if i == 3000 then y[1] = nil; y[1] = 5 end
    if i == 3500 then y[2] = 0.5 end
    add(x, z)
  end
  return x[1], x[2], z[1], z[2]
end
return run(4000)
";
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(stats.traces_compiled >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
        assert!(stats.array_overwrites >= 2, "{profile:?}: {stats:?}");
        assert!(
            stats.exits.iter().sum::<u64>() >= 1,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_set_between_the_read_and_the_store_keeps_the_store_guarded() {
    for profile in PROFILES {
        // An append to `b` may move any array, so the stores after it are
        // not proven by the reads before it. With `b` the table `a` the
        // append grows the very array the stores write.
        let source = "
local function add(a, b)
  local lo = a[1] + b[1]
  local hi = a[2] + b[2]
  b[#b + 1] = 1
  a[1] = lo % 65536
  a[2] = hi % 65536
end
local function run(n, a, b)
  for i = 1, n do add(a, b) end
  return a[1], a[2], #a, #b
end
local p = {run(3000, {1, 2}, {3, 4})}
local x = {5, 6}
local q = {run(3000, x, x)}
return p[1], p[2], p[3], p[4], q[1], q[2], q[3], q[4]
";
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let ir = recorded(profile, source);
        ir.verify().unwrap();
        assert_eq!(
            ops_matching(&ir, |op| matches!(op, TraceOp::ArrayOverwrite { .. })),
            0,
            "{profile:?}"
        );
    }
}

/// Bumps slot `idx` of the array a region's `solids` field holds, then reads
/// it back. The regions are met by lane (`regions[k % 4 + 1]`).
const LANE_STORES: &str = "
local regions = {}
for i = 1, 4 do
  local solids = {}
  for j = 1, 8 do solids[j] = 0 end
  regions[i] = { solids = solids, size = 8 }
end
local function run(n)
  local seen = 0
  for k = 1, n do
    local region = regions[k % 4 + 1]
    local s = region.solids
    local idx = k % region.size + 1
    s[idx] = s[idx] + 1
    seen = seen + s[idx]
  end
  return seen
end
return run(6000)
";

#[test]
fn a_lane_store_overwrites_in_place_and_answers_the_read_after_it() {
    for profile in PROFILES {
        let ir = recorded(profile, LANE_STORES);
        ir.verify().unwrap();
        let ops = || ir.body.iter().chain(&ir.preheader).map(|i| &i.op);
        // The store is inline, not the raw-set call, so a row read and the
        // store sit in one recording.
        assert_eq!(
            ops()
                .filter(|op| matches!(op, TraceOp::LaneArrayStore { .. }))
                .count(),
            1,
            "{profile:?}"
        );
        assert!(
            !ops().any(|op| matches!(
                op,
                TraceOp::CallHelper { helper, .. }
                    if *helper == crate::jit_ir::trace::HelperId::TABLE_SET_INDEX
            )),
            "{profile:?}"
        );
        // `s[idx]` is read once, before the store; the read after it is the
        // stored value.
        assert_eq!(
            ops()
                .filter(|op| matches!(op, TraceOp::LaneArrayLoad { .. }))
                .count(),
            1,
            "{profile:?}"
        );
        let stats = agrees_natively(profile, LANE_STORES);
        assert!(stats.slot_reads_forwarded >= 1, "{profile:?}: {stats:?}");
        assert!(
            stats.lane_array_stores_compiled >= 1,
            "{profile:?}: {stats:?}"
        );
    }
}
