//! Array sets through a header and nil reads past the end of an array, MIR
//! against Cranelift.
//!
//! The sets run over tables of a real runtime, since an append grows the
//! array through the runtime's own function and a pop writes the table's
//! border and its record of deleted keys. Each case builds the table twice,
//! runs each backend over its own copy and compares the outcome, the exit
//! area, the appended count and the table afterwards.

use super::*;
use crate::jit::trace::exit::{HostEntry, TraceBinding, TraceTableSpace, TraceTables};
use crate::jit_ir::trace::{BindingKind, TableBinding};
use crate::runtime::{Runtime, TableHandle, Value};

/// A VM that stores nothing: a set that is not done in place is refused, and
/// the trace leaves through the set's guard.
extern "C" fn refuse_set(
    _context: usize,
    _words: *mut TraceInputSlot,
    _binding: i64,
    _key_tag: i64,
    _key_bits: i64,
    _value_tag: i64,
    _value_bits: i64,
) -> i64 {
    -1
}

extern "C" fn refuse_store(
    _context: usize,
    _binding: i64,
    _tag: i64,
    _bits: i64,
    _slot: i64,
    _replace: i64,
) -> i64 {
    -1
}

thread_local! {
    /// The calls of the store entry a run made: binding, tag, bits and
    /// replace.
    static STORES: std::cell::RefCell<Vec<(i64, i64, i64, i64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// A store entry that stores an integer in the slot and remembers the call.
extern "C" fn accept_store(
    _context: usize,
    binding: i64,
    tag: i64,
    bits: i64,
    slot: i64,
    replace: i64,
) -> i64 {
    STORES.with(|stores| stores.borrow_mut().push((binding, tag, bits, replace)));
    // SAFETY: `slot` is the address of a slot of the table's array part, or of
    // its spare capacity, which the trace made room for.
    #[allow(unsafe_code)]
    unsafe {
        (slot as *mut Value).write(Value::Integer(99));
    }
    0
}

struct Refusing;

/// A host with a store entry that stores.
struct Storing;

impl TraceTables for Storing {
    fn lane(&mut self, _value: &Value) -> Option<usize> {
        None
    }

    fn row(&mut self, _lane: usize, _row: &mut [usize]) {}

    fn host_entry(&mut self) -> Option<HostEntry> {
        Some(HostEntry {
            function: 0,
            context: 0,
            set: refuse_set as *const () as usize,
            store: accept_store as *const () as usize,
        })
    }
}

impl TraceTables for Refusing {
    fn lane(&mut self, _value: &Value) -> Option<usize> {
        None
    }

    fn row(&mut self, _lane: usize, _row: &mut [usize]) {}

    fn host_entry(&mut self) -> Option<HostEntry> {
        Some(HostEntry {
            function: 0,
            context: 0,
            set: refuse_set as *const () as usize,
            store: refuse_store as *const () as usize,
        })
    }
}

fn snap(phase: SnapshotPhase, effects: u32, registers: Vec<RegisterEntry>) -> Snapshot {
    Snapshot::new(
        phase,
        effects,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(0),
            registers: registers.into(),
        }],
    )
}

/// `t[k] = v`, then `r3 = #t` and `r4 = t[k]` (a read of the type of the
/// value, for a value that is not nil), over an array, a length and a header
/// binding (when `header` says whether it is strict) of the table in register
/// 0. A nil value is a constant, and a string is not read back.
fn set_trace(key_ty: TraceType, value_ty: TraceType, header: Option<bool>) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let mut b = TraceBuilder::new(TraceId(0), frame(5));
    b.mark_written(FrameId(0), 3);
    b.mark_written(FrameId(0), 4);
    let table = b.input(0, TraceType::Table);
    let key = b.input(1, key_ty);
    let value = if value_ty == TraceType::Nil {
        b.emit(
            site(0),
            TraceOp::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        )
        .unwrap()
    } else {
        b.input(2, value_ty)
    };
    let mut bindings = vec![
        TableBinding {
            register: 0,
            kind: BindingKind::Array { first: None },
        },
        TableBinding {
            register: 0,
            kind: BindingKind::Length {
                ty: TraceType::Integer,
                raw: false,
            },
        },
    ];
    bindings.extend(header.map(|strict| TableBinding {
        register: 0,
        kind: BindingKind::Header { strict },
    }));
    b.set_bindings(bindings);
    let nil = Const(TraceScalar::Nil);
    let held = vec![Unchanged, Unchanged, Unchanged];
    let regs = |first: RegisterEntry, second: RegisterEntry| {
        let mut regs = held.clone();
        regs.extend([first, second]);
        regs
    };
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        b.effect_position(),
        regs(nil, nil),
    ));
    let g0 = b.guard(site(0), s0);
    let effect = b.next_effect();
    b.emit(
        site(0),
        TraceOp::ArraySet {
            binding: 0,
            table,
            key,
            value,
            effect,
            guard: g0,
        },
        None,
    );
    let length = b
        .emit(
            site(1),
            TraceOp::TableLength { binding: 1, table },
            Some(TraceType::Integer),
        )
        .unwrap();
    let read = if matches!(
        value_ty,
        TraceType::Nil | TraceType::String | TraceType::Table
    ) {
        None
    } else {
        let s1 = b.snapshot(snap(
            SnapshotPhase::Before { pc: 2 },
            b.effect_position(),
            regs(Val(length), nil),
        ));
        let g1 = b.guard(site(2), s1);
        Some(
            b.emit(
                site(2),
                TraceOp::ArrayLoad {
                    binding: 0,
                    table,
                    key,
                    expected: value_ty,
                    past_end: false,
                    guard: g1,
                },
                Some(value_ty),
            )
            .unwrap(),
        )
    };
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        b.effect_position(),
        regs(Val(length), read.map_or(nil, Val)),
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// How a test table is made: the integer keys set in order, then a string key
/// when `named`.
struct Spec {
    label: &'static str,
    sets: &'static [(i64, SpecValue)],
    named: bool,
}

#[derive(Clone, Copy)]
enum SpecValue {
    Integer,
    Boolean,
    Text,
}

impl SpecValue {
    fn value(self, key: i64) -> Value {
        match self {
            SpecValue::Integer => Value::Integer(key * 10),
            SpecValue::Boolean => Value::Boolean(key % 2 == 0),
            SpecValue::Text => Value::from("text"),
        }
    }
}

use SpecValue::{Boolean, Integer, Text};

/// Tables of every kind a header tells apart: dense lists of several lengths
/// (so appends fit the capacity and grow it), a holey one (a boolean two past
/// the end), a holey one of 40 slots with room for more (so a boolean at the
/// end of the reach fits and the one after it is refused for the reach, not
/// for the capacity), one that is not solid (a string key, or a deleted
/// key), and lists that hold a string.
const SPECS: &[Spec] = &[
    Spec {
        label: "empty",
        sets: &[],
        named: false,
    },
    Spec {
        label: "one",
        sets: &[(1, Integer)],
        named: false,
    },
    Spec {
        label: "three",
        sets: &[(1, Integer), (2, Integer), (3, Integer)],
        named: false,
    },
    Spec {
        label: "four",
        sets: &[(1, Integer), (2, Integer), (3, Integer), (4, Integer)],
        named: false,
    },
    Spec {
        label: "five",
        sets: &[
            (1, Integer),
            (2, Integer),
            (3, Integer),
            (4, Integer),
            (5, Integer),
        ],
        named: false,
    },
    Spec {
        label: "booleans",
        sets: &[(1, Boolean), (2, Boolean), (3, Boolean)],
        named: false,
    },
    Spec {
        label: "holey",
        sets: &[(1, Boolean), (2, Boolean), (3, Boolean), (5, Boolean)],
        named: false,
    },
    Spec {
        label: "holey early",
        sets: &[(1, Boolean), (3, Boolean)],
        named: false,
    },
    Spec {
        label: "holey long",
        sets: &[
            (1, Boolean),
            (2, Boolean),
            (3, Boolean),
            (4, Boolean),
            (5, Boolean),
            (6, Boolean),
            (7, Boolean),
            (8, Boolean),
            (9, Boolean),
            (10, Boolean),
            (11, Boolean),
            (12, Boolean),
            (13, Boolean),
            (14, Boolean),
            (15, Boolean),
            (16, Boolean),
            (17, Boolean),
            (18, Boolean),
            (19, Boolean),
            (20, Boolean),
            (21, Boolean),
            (22, Boolean),
            (23, Boolean),
            (24, Boolean),
            (25, Boolean),
            (26, Boolean),
            (27, Boolean),
            (28, Boolean),
            (29, Boolean),
            (30, Boolean),
            (31, Boolean),
            (32, Boolean),
            (33, Boolean),
            (34, Boolean),
            (35, Boolean),
            (36, Boolean),
            (37, Boolean),
            (38, Boolean),
            (40, Boolean),
        ],
        named: false,
    },
    Spec {
        label: "named",
        sets: &[(1, Integer), (2, Integer)],
        named: true,
    },
    Spec {
        label: "text",
        sets: &[(1, Text), (2, Integer), (3, Text)],
        named: false,
    },
];

fn build(runtime: &mut Runtime, spec: &Spec) -> TableHandle {
    let table = runtime.create_table().unwrap();
    for (key, kind) in spec.sets {
        runtime
            .table_set(table, &Value::Integer(*key), kind.value(*key))
            .unwrap();
    }
    if spec.named {
        runtime
            .table_set(table, &Value::from("name"), Value::Integer(1))
            .unwrap();
    }
    table
}

/// What a run leaves: the outcome, the exit area, the appended slots and the
/// fuel, and the table (its border and the slots to past its array).
type Seen = (TraceOutcome, Vec<i64>, (i64, i64), String);

fn observe_set(
    trace: &CompiledTrace,
    spec: &Spec,
    (key, value): (TraceScalar, TraceScalar),
) -> Seen {
    observe_set_by(trace, spec, (key, value), &mut Refusing).0
}

/// [`observe_set`] with the host `tables`, and the calls its store entry got.
fn observe_set_by(
    trace: &CompiledTrace,
    spec: &Spec,
    (key, value): (TraceScalar, TraceScalar),
    tables: &mut dyn TraceTables,
) -> (Seen, Vec<(i64, i64, i64, i64)>) {
    STORES.with(|stores| stores.borrow_mut().clear());
    let mut runtime = Runtime::new();
    let table = build(&mut runtime, spec);
    let header = runtime.table_native_header(table).unwrap();
    let array = runtime.table_trace_array(table).unwrap();
    let border = runtime.table_length(table).unwrap();
    let registers = [TraceScalar::Table(0), key, value];
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    let mut bindings = vec![
        TraceBinding::array(array),
        TraceBinding::integer_length(border as i64),
    ];
    if trace.bindings().len() > 2 {
        bindings.push(TraceBinding::header(header.word(), header.address));
    }
    let mut area = vec![0x5a5a_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(100);
    let mut space = TraceTableSpace::new();
    let outcome = trace.run_with_tables(
        &inputs,
        &bindings,
        &mut control,
        &mut area,
        &mut space,
        tables,
    );
    let length = runtime.table_trace_array(table).unwrap().len();
    let slots: Vec<Value> = (1..=length as i64 + 2)
        .map(|key| runtime.table_get(table, &Value::Integer(key)).unwrap())
        .collect();
    let after = runtime.table_native_header(table).unwrap();
    let stores = STORES.with(|stores| stores.borrow().clone());
    (
        (
            outcome,
            area,
            (control.appended, control.fuel),
            format!(
                "border {} flags {} pending {} {slots:?}",
                runtime.table_length(table).unwrap(),
                after.word(),
                pending_deletions(header.address)
            ),
        ),
        stores,
    )
}

/// The word of the table at `address` that records the deleted array keys
/// native pops left unrecorded, which a pop raises.
#[allow(unsafe_code)]
fn pending_deletions(address: usize) -> usize {
    let layout = crate::runtime::NativeTableLayout::get().unwrap();
    // SAFETY: `address` is the address of a live table of the runtime, and the
    // layout names a word in it.
    unsafe { *((address + layout.pending_deletions as usize) as *const usize) }
}

fn compile_both(ir: &TraceIr, tier: OptimizationTier) -> (CompiledTrace, CompiledTrace) {
    ir.verify().unwrap();
    let mir = compile(ir, tier, true);
    let cranelift = compile(ir, tier, false);
    assert!(mir.mir_address().is_some(), "compiled by MIR");
    assert!(cranelift.mir_address().is_none());
    assert_eq!(mir.exits(), cranelift.exits());
    (mir, cranelift)
}

/// Whether the table of `spec` holds a value of `value`'s type under `key`.
/// MIR overwrites such a slot in place whatever the table is, as an
/// `ArrayStore` does. Under a strict header Cranelift leaves through the
/// guard instead when the header's flags do not let it fill the slot (a table
/// that is not solid, the slot after the border), where under a lax header
/// its general path makes the same overwrite.
fn holds_type_of(spec: &Spec, key: TraceScalar, value: TraceScalar) -> bool {
    let index = match key {
        TraceScalar::Integer(index) => index,
        TraceScalar::Float(index) if index.fract() == 0.0 && index.abs() < 1e15 => index as i64,
        _ => return false,
    };
    let mut runtime = Runtime::new();
    let table = build(&mut runtime, spec);
    let held = runtime.table_get(table, &Value::Integer(index)).unwrap();
    matches!(
        (held, value),
        (Value::Integer(_), TraceScalar::Integer(_)) | (Value::Boolean(_), TraceScalar::Boolean(_))
    )
}

/// Every key, value and table: MIR stores in place, appends, fills holes,
/// pops and leaves through the guard as Cranelift does, and the table, the
/// appended count and the exit agree afterwards. The one difference is the
/// overwrite of a slot of the value's type under a strict header (see
/// [`holds_type_of`]), which MIR makes as Cranelift does under a lax one.
#[test]
fn array_sets_through_a_header_match_cranelift() {
    let int_keys: Vec<TraceScalar> = [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        20,
        21,
        39,
        41,
        42,
        56,
        57,
        i64::MAX,
        i64::MIN,
    ]
    .iter()
    .map(|key| TraceScalar::Integer(*key))
    .collect();
    let float_keys: Vec<TraceScalar> = [
        0.0,
        -0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        21.0,
        39.0,
        42.0,
        56.0,
        57.0,
        2.5,
        f64::NAN,
        1e300,
        -3.0,
        9.3e18,
        // Keys that are not whole by the least a float can show: below 1
        // (a denormal, whose bits are a small integer), just above 1, and
        // just below 4.
        5e-324,
        0.99,
        -0.5,
        1.000_000_000_000_000_2,
        3.999_999_999_999_999_6,
        f64::INFINITY,
        -9.3e18,
    ]
    .iter()
    .map(|key| TraceScalar::Float(*key))
    .collect();
    let mut appended = 0;
    let mut left = 0;
    let mut stored = 0;
    let mut overwrote = 0;
    for tier in TIERS {
        for strict in [false, true] {
            for (value_ty, value) in [
                (TraceType::Integer, TraceScalar::Integer(-7)),
                (TraceType::Float, TraceScalar::Float(1.5)),
                (TraceType::Boolean, TraceScalar::Boolean(true)),
                (TraceType::Boolean, TraceScalar::Boolean(false)),
                (TraceType::Nil, TraceScalar::Nil),
            ] {
                for (key_ty, keys) in [
                    (TraceType::Integer, &int_keys),
                    (TraceType::Float, &float_keys),
                ] {
                    let ir = set_trace(key_ty, value_ty, Some(strict));
                    let (mir, cranelift) = compile_both(&ir, tier);
                    let lax = compile(&set_trace(key_ty, value_ty, Some(false)), tier, false);
                    for spec in SPECS {
                        for &key in keys {
                            let a = observe_set(&mir, spec, (key, value));
                            let mut b = observe_set(&cranelift, spec, (key, value));
                            if strict && a != b && holds_type_of(spec, key, value) {
                                assert_eq!(b.0, TraceOutcome::Exit { index: 0 });
                                b = observe_set(&lax, spec, (key, value));
                                overwrote += 1;
                            }
                            assert_eq!(
                                a, b,
                                "{tier:?} strict {strict} {value:?} under {key:?} in {}",
                                spec.label
                            );
                            appended += usize::from(a.2.0 > 0);
                            match a.0 {
                                TraceOutcome::Exit { index: 0 } => left += 1,
                                TraceOutcome::Exit { .. } => stored += 1,
                                other => panic!("{other:?}"),
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        appended > 100 && left > 100 && stored > 100 && overwrote > 0,
        "{appended} appends, {left} exits at the set, {stored} sets that went on, \
         {overwrote} overwrites Cranelift left for"
    );
}

/// A table with no header takes the general path only: a scalar overwrites a
/// slot of its own type in place, and the VM's store entry (this host refuses)
/// takes everything else, which leaves through the guard.
#[test]
fn array_sets_with_no_header_match_cranelift() {
    let keys: Vec<TraceScalar> = [0, 1, 2, 3, 4, 5, 6, i64::MAX]
        .iter()
        .map(|key| TraceScalar::Integer(*key))
        .chain([1.0, 2.0, 3.0, 2.5, f64::NAN, 1e300].map(float))
        .collect();
    let mut stored = 0;
    let mut left = 0;
    for tier in TIERS {
        for (value_ty, value) in [
            (TraceType::Integer, TraceScalar::Integer(-7)),
            (TraceType::Float, TraceScalar::Float(1.5)),
            (TraceType::Boolean, TraceScalar::Boolean(true)),
            (TraceType::Nil, TraceScalar::Nil),
        ] {
            for key_ty in [TraceType::Integer, TraceType::Float] {
                let ir = set_trace(key_ty, value_ty, None);
                let (mir, cranelift) = compile_both(&ir, tier);
                for spec in SPECS {
                    for &key in keys.iter().filter(|key| {
                        matches!(key, TraceScalar::Float(_)) == (key_ty == TraceType::Float)
                    }) {
                        let a = observe_set(&mir, spec, (key, value));
                        let b = observe_set(&cranelift, spec, (key, value));
                        assert_eq!(a, b, "{tier:?} {value:?} under {key:?} in {}", spec.label);
                        match a.0 {
                            TraceOutcome::Exit { index: 0 } => left += 1,
                            TraceOutcome::Exit { .. } => stored += 1,
                            other => panic!("{other:?}"),
                        }
                    }
                }
            }
        }
    }
    assert!(stored > 50 && left > 50, "{stored} stored, {left} left");
}

/// A string or a table goes through the VM's store entry when a dense list
/// takes it in place, and leaves through the guard under a strict header
/// otherwise. This host refuses every store, so a string never lands, and the
/// run must leave at the set the same way under both backends.
#[test]
fn a_string_set_through_a_header_leaves_like_cranelift() {
    for tier in TIERS {
        for strict in [false, true] {
            let ir = set_trace(TraceType::Integer, TraceType::String, Some(strict));
            let (mir, cranelift) = compile_both(&ir, tier);
            for spec in SPECS {
                for key in [0, 1, 2, 3, 4, 6, 7] {
                    let case = (TraceScalar::Integer(key), TraceScalar::String(0));
                    assert_eq!(
                        observe_set(&mir, spec, case),
                        observe_set(&cranelift, spec, case),
                        "{tier:?} strict {strict} key {key} in {}",
                        spec.label
                    );
                }
            }
        }
    }
}

/// A host whose store entry stores: a dense list takes a string in a new slot
/// or over a slot, and a nil pops a slot that holds one (the entry releases
/// it), with the same calls to the entry and the same table afterwards.
#[test]
fn the_store_entry_gets_the_same_calls_under_a_header_as_from_cranelift() {
    let mut calls = 0;
    for tier in TIERS {
        for strict in [false, true] {
            for (value_ty, value) in [
                (TraceType::String, TraceScalar::String(0)),
                (TraceType::Nil, TraceScalar::Nil),
            ] {
                let ir = set_trace(TraceType::Integer, value_ty, Some(strict));
                let (mir, cranelift) = compile_both(&ir, tier);
                for spec in SPECS {
                    for key in [0, 1, 2, 3, 4, 5, 6, 7] {
                        let case = (TraceScalar::Integer(key), value);
                        let a = observe_set_by(&mir, spec, case, &mut Storing);
                        let b = observe_set_by(&cranelift, spec, case, &mut Storing);
                        assert_eq!(
                            a, b,
                            "{tier:?} strict {strict} {value_ty:?} key {key} in {}",
                            spec.label
                        );
                        calls += a.1.len();
                    }
                }
            }
        }
    }
    assert!(calls > 50, "{calls} calls of the store entry");
}

/// A loop that sets constant key 1 to a constant (`set_case` of the store
/// tests) under a header, run for a few iterations over every table: the key
/// and the value are constants, so the compares of the in-place sets have an
/// immediate on both sides.
#[test]
fn a_loop_that_sets_a_constant_key_through_a_header_matches_cranelift() {
    use crate::jit::trace::store_tests::set_case;
    let observe = |trace: &CompiledTrace, spec: &Spec, fuel: i64| {
        let mut runtime = Runtime::new();
        let table = build(&mut runtime, spec);
        let header = runtime.table_native_header(table).unwrap();
        let array = runtime.table_trace_array(table).unwrap();
        let inputs: Vec<TraceInputSlot> = trace
            .entry_contract()
            .iter()
            .map(|_| TraceInputSlot::from_scalar(TraceScalar::Table(0)))
            .collect();
        let bindings = [
            TraceBinding::array(array),
            TraceBinding::header(header.word(), header.address),
        ];
        let mut area = vec![0x5a5a_i64; trace.exit_area_len()];
        let mut control = TraceControl::new(fuel);
        let mut space = TraceTableSpace::new();
        let outcome = trace.run_with_tables(
            &inputs,
            &bindings,
            &mut control,
            &mut area,
            &mut space,
            &mut Storing,
        );
        let length = runtime.table_trace_array(table).unwrap().len();
        let slots: Vec<Value> = (1..=length as i64 + 1)
            .map(|key| runtime.table_get(table, &Value::Integer(key)).unwrap())
            .collect();
        (
            outcome,
            area,
            (control.appended, control.fuel),
            format!("{slots:?} {}", runtime.table_length(table).unwrap()),
        )
    };
    let mut ran = 0;
    for tier in TIERS {
        for strict in [false, true] {
            for value in [
                TraceScalar::Integer(3),
                TraceScalar::Float(0.5),
                TraceScalar::Boolean(true),
                TraceScalar::Nil,
            ] {
                let ir = set_case(Some(BindingKind::Header { strict }), value);
                let (mir, cranelift) = compile_both(&ir, tier);
                let lax = set_case(Some(BindingKind::Header { strict: false }), value);
                let lax = compile(&lax, tier, false);
                for spec in SPECS {
                    for fuel in [0, 1, 4] {
                        let a = observe(&mir, spec, fuel);
                        let mut b = observe(&cranelift, spec, fuel);
                        // The overwrite of a slot of the value's type, which
                        // MIR makes under a strict header too.
                        if strict && a != b && holds_type_of(spec, TraceScalar::Integer(1), value) {
                            assert_eq!(b.0, TraceOutcome::Exit { index: 0 });
                            b = observe(&lax, spec, fuel);
                        }
                        assert_eq!(
                            a, b,
                            "{tier:?} strict {strict} {value:?} fuel {fuel} in {}",
                            spec.label
                        );
                        ran += 1;
                    }
                }
            }
        }
    }
    assert!(ran > 400);
}

/// `r3 = t[k]` as a nil read past the end of the array of a table with a
/// header and no metatable.
fn nil_read_trace(key_ty: TraceType) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    b.mark_written(FrameId(0), 3);
    let table = b.input(0, TraceType::Table);
    let key = b.input(1, key_ty);
    b.set_bindings(vec![
        TableBinding {
            register: 0,
            kind: BindingKind::Array { first: None },
        },
        TableBinding {
            register: 0,
            kind: BindingKind::NoMetatable,
        },
        TableBinding {
            register: 0,
            kind: BindingKind::Header { strict: false },
        },
    ]);
    let nil = Const(TraceScalar::Nil);
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![Unchanged, Unchanged, Unchanged, nil],
    ));
    let g0 = b.guard(site(0), s0);
    let read = b
        .emit(
            site(0),
            TraceOp::ArrayLoad {
                binding: 0,
                table,
                key,
                expected: TraceType::Nil,
                past_end: true,
                guard: g0,
            },
            Some(TraceType::Nil),
        )
        .unwrap();
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        vec![Unchanged, Unchanged, Unchanged, Val(read)],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// A key outside the array (past its end, below 1, fractional or NaN) reads
/// nil only when the header says the table is solid, and a key inside reads
/// nil only from a nil slot, as Cranelift does.
#[test]
fn a_nil_read_past_the_end_matches_cranelift() {
    let slot_sets: [Vec<Value>; 4] = [
        vec![Value::Number(1.0), Value::Nil, Value::Integer(3)],
        vec![Value::Nil],
        Vec::new(),
        vec![Value::Boolean(true), Value::Nil, Value::Nil, Value::Nil],
    ];
    let int_keys = [-1, 0, 1, 2, 3, 4, 5, 100, i64::MIN, i64::MAX];
    let float_keys = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        1.5,
        f64::NAN,
        1e300,
        -2.0,
        0.5,
    ];
    let mut nil_reads = 0;
    let mut exits = 0;
    for tier in TIERS {
        for key_ty in [TraceType::Integer, TraceType::Float] {
            let ir = nil_read_trace(key_ty);
            let (mir, cranelift) = compile_both(&ir, tier);
            let keys: Vec<TraceScalar> = if key_ty == TraceType::Float {
                float_keys.iter().map(|key| float(*key)).collect()
            } else {
                int_keys
                    .iter()
                    .map(|key| TraceScalar::Integer(*key))
                    .collect()
            };
            for slots in &slot_sets {
                for flags in [0, 1, 2] {
                    for key in &keys {
                        let observe = |trace: &CompiledTrace| {
                            let mut slots = slots.clone();
                            let inputs =
                                [TraceInputSlot::table(0), TraceInputSlot::from_scalar(*key)];
                            let bindings = [
                                TraceBinding::raw_array(slots.len(), slots.as_mut_ptr() as usize),
                                TraceBinding::callee(),
                                TraceBinding::header(flags, 0),
                            ];
                            let mut area = vec![0x5a5a_i64; trace.exit_area_len()];
                            let mut control = TraceControl::new(50);
                            let mut space = TraceTableSpace::new();
                            let outcome = trace.run_with_tables(
                                &inputs,
                                &bindings,
                                &mut control,
                                &mut area,
                                &mut space,
                                &mut Refusing,
                            );
                            (outcome, area, control.fuel)
                        };
                        let (a, b) = (observe(&mir), observe(&cranelift));
                        assert_eq!(a, b, "{tier:?} {key:?} flags {flags} in {slots:?}");
                        match a.0 {
                            TraceOutcome::Exit { index: 0 } => exits += 1,
                            TraceOutcome::Exit { .. } => nil_reads += 1,
                            other => panic!("{other:?}"),
                        }
                    }
                }
            }
        }
    }
    assert!(
        nil_reads > 100 && exits > 100,
        "{nil_reads} nil reads, {exits} exits"
    );
}
