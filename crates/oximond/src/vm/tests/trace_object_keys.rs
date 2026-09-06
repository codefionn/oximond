//! Reads keyed by a table or a function (`TableKey::Object` in the
//! recorder): the cache `ordinal_sturdy[shape]` of Mineclonia's
//! `get_sturdy_faces_1` (nodeprops.lua:705), read in the floor patch of a
//! mineshaft corridor (mineshaft.lua:1045). The recorder emits one lookup
//! call through the table's lane, with the key as an operand. A store under
//! such a key still aborts the recording. The same call reads a table under
//! a sparse number key in the hash part of a bound table (structures.lua:856),
//! and the file also holds the recording of `math.max` of three numbers
//! (structures.lua:1271), the other refusal of Mineclonia's structure loops.

use super::*;
use crate::jit_ir::trace::{HelperId, TraceIr, TraceOp};
use crate::vm::jit::trace::{TraceAbortReason, TraceStatistics};

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

/// The root trace the recorder queued for `source`, which it does not
/// compile, and the statistics of the run.
fn recorded(profile: CompatibilityProfile, source: &str) -> (Option<TraceIr>, TraceStatistics) {
    let mut lua = crate::Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Tracing)
        .build()
        .unwrap();
    lua.vm_mut().jit.trace.set_native_compilation(false);
    lua.exec(source).unwrap();
    let queued = lua.vm().jit.trace.queued();
    let ir = queued.front().map(|request| request.ir.clone());
    (ir, lua.vm().jit_statistics().trace)
}

fn lookups(ir: &TraceIr, wanted: HelperId) -> usize {
    ir.body
        .iter()
        .chain(&ir.preheader)
        .filter(|i| matches!(&i.op, TraceOp::CallHelper { helper, .. } if *helper == wanted))
        .count()
}

#[test]
fn a_read_under_a_table_key_records_one_lookup_call() {
    let source = r"
        local shapes = {}
        for i = 1, 4 do shapes[i] = { id = i } end
        local cache = {}
        for i = 1, 4 do cache[shapes[i]] = i * 16 end
        local s = 0
        for i = 1, 400 do s = s + cache[shapes[i % 4 + 1]] end
        return s
    ";
    for profile in PROFILES {
        let (ir, stats) = recorded(profile, source);
        let ir = ir.unwrap_or_else(|| panic!("{profile:?}: no recording {stats:?}"));
        assert_eq!(
            stats.recording_aborts[TraceAbortReason::UnsupportedOperand.index()],
            0,
            "{profile:?}"
        );
        // Lua 5.1 holds the values as floats, the others as integers.
        assert_eq!(
            lookups(&ir, HelperId::TABLE_GET_INTEGER) + lookups(&ir, HelperId::TABLE_GET_FLOAT),
            1,
            "{profile:?}: {ir:?}"
        );
    }
}

#[test]
fn a_read_under_a_function_key_records_the_lookup_for_the_value_type() {
    // The values are strings: the call is the one that returns a string,
    // and leaves for any other type.
    let source = r#"
        local fns = {}
        for i = 1, 3 do fns[i] = function () return i end end
        local names = {}
        for i = 1, 3 do names[fns[i]] = "f" .. i end
        local n = 0
        for i = 1, 400 do n = n + #names[fns[i % 3 + 1]] end
        return n
    "#;
    for profile in PROFILES {
        let (ir, stats) = recorded(profile, source);
        let ir = ir.unwrap_or_else(|| panic!("{profile:?}: no recording {stats:?}"));
        assert_eq!(
            lookups(&ir, HelperId::TABLE_GET_STRING),
            1,
            "{profile:?}: {ir:?}"
        );
    }
}

#[test]
fn a_store_under_a_table_key_aborts_the_recording() {
    let source = r"
        local shapes = {}
        for i = 1, 4 do shapes[i] = {} end
        local cache = {}
        for i = 1, 400 do cache[shapes[i % 4 + 1]] = i end
        return cache[shapes[1]]
    ";
    for profile in PROFILES {
        let (ir, stats) = recorded(profile, source);
        assert!(ir.is_none(), "{profile:?}");
        assert!(
            stats.recording_aborts[TraceAbortReason::UnsupportedOperand.index()] >= 1,
            "{profile:?}: {stats:?}"
        );
    }
}

/// `starts[hash].starts` (Mineclonia's structures.lua:856): a table under a
/// sparse number key sits in the hash part of a bound table. The recorder
/// reads it through the same lookup call, as `rawget` does, where it used to
/// refuse any value but a number or a boolean there.
#[test]
fn a_table_under_a_sparse_number_key_records_one_lookup_call() {
    let source = r"
        local starts = {}
        for x = 0, 7 do starts[(x + 4096) * 8192 + 3] = { n = x } end
        local s = 0
        for i = 1, 400 do s = s + starts[(i % 8 + 4096) * 8192 + 3].n end
        return s
    ";
    for profile in PROFILES {
        let (ir, stats) = recorded(profile, source);
        let ir = ir.unwrap_or_else(|| panic!("{profile:?}: no recording {stats:?}"));
        assert_eq!(
            stats.recording_aborts[TraceAbortReason::UnsupportedOperand.index()],
            0,
            "{profile:?}"
        );
        assert_eq!(
            lookups(&ir, HelperId::TABLE_GET_TABLE),
            1,
            "{profile:?}: {ir:?}"
        );
        assert!(
            ir.body
                .iter()
                .any(|i| matches!(i.op, TraceOp::TableFieldLoad { .. })),
            "{profile:?}: {ir:?}"
        );
    }
}

/// A number or boolean under such a key keeps its inline hash load; only a
/// value with no word native code can carry takes the call.
#[test]
fn a_number_under_a_sparse_number_key_keeps_the_hash_load() {
    let source = r"
        local starts = {}
        for x = 0, 7 do starts[(x + 4096) * 8192 + 3] = x * 2 end
        local s = 0
        for i = 1, 400 do s = s + starts[(i % 8 + 4096) * 8192 + 3] end
        return s
    ";
    for profile in PROFILES {
        let (ir, stats) = recorded(profile, source);
        let ir = ir.unwrap_or_else(|| panic!("{profile:?}: no recording {stats:?}"));
        assert_eq!(
            lookups(&ir, HelperId::TABLE_GET_TABLE),
            0,
            "{profile:?}: {ir:?}"
        );
        assert!(
            ir.body
                .iter()
                .any(|i| matches!(i.op, TraceOp::HashLoad { .. })),
            "{profile:?}: {ir:?}"
        );
    }
}

/// `mathmax (0, piece_bbox[1] - x, x - piece_bbox[4])` (Mineclonia's
/// structures.lua:1271): `math.max` and `math.min` of three or four numbers
/// fold the two-argument form from the left, one step per extra argument:
/// floats through the native's own binary function, integers (Lua 5.4)
/// through the host compare.
#[test]
fn math_max_of_three_numbers_records_two_steps() {
    let source = r"
        local mathmax, mathmin = math.max, math.min
        local s = 0
        for i = 1, 400 do
            s = s + mathmax (0, 3 - i % 7, i % 5 - 2) + mathmin (i % 3, 4, i % 4, 9)
        end
        return s
    ";
    for profile in PROFILES {
        let (ir, stats) = recorded(profile, source);
        let ir = ir.unwrap_or_else(|| panic!("{profile:?}: no recording {stats:?}"));
        assert_eq!(
            stats.recording_aborts[TraceAbortReason::UnsupportedOperand.index()],
            0,
            "{profile:?}"
        );
        let steps = ir
            .body
            .iter()
            .chain(&ir.preheader)
            .filter(|i| {
                matches!(
                    i.op,
                    TraceOp::NativeBinary { .. }
                        | TraceOp::HostCall {
                            function: crate::jit_ir::trace::HostFunction::MathMax
                                | crate::jit_ir::trace::HostFunction::MathMin,
                            ..
                        }
                )
            })
            .count();
        assert_eq!(steps, 2 + 3, "{profile:?}: {ir:?}");
    }
}

/// An inner `ipairs` loop whose first step ends it (an empty list,
/// structures.lua:857) is followed by the recording of the loop around it:
/// one `TABLE_ABSENT_INDEX` check stands for the step, and the recording
/// closes instead of missing.
#[test]
fn an_empty_inner_ipairs_records_the_absent_check() {
    let source = r"
        local lists = {}
        for i = 1, 4 do lists[i] = {} end
        local s = 0
        for i = 1, 400 do
            for _, v in ipairs (lists[i % 4 + 1]) do s = s + v end
            s = s + 1
        end
        return s
    ";
    for profile in PROFILES {
        let (ir, stats) = recorded(profile, source);
        let ir = ir.unwrap_or_else(|| panic!("{profile:?}: no recording {stats:?}"));
        assert_eq!(
            stats.recording_aborts[TraceAbortReason::TableMiss.index()],
            0,
            "{profile:?}"
        );
        assert_eq!(
            lookups(&ir, HelperId::TABLE_ABSENT_INDEX),
            1,
            "{profile:?}: {ir:?}"
        );
    }
}
