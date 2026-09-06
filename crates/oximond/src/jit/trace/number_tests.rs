//! Numeric operations that convert, and registers that are a float or an
//! integer at run time, MIR against Cranelift.
//!
//! The module is declared inside the MIR tests (`mir/tests.rs`), which own the
//! harness it uses: `same_everywhere` compiles a case with each backend,
//! requires the MIR backend to have made its code, and compares the outcome,
//! the control block and every word of the exit area over many states.

use super::*;
use crate::jit_ir::trace::NumberRegister;

/// Integral floats in the range of an integer, the values the recorder lets a
/// conversion see (see [`NumericOp::FloatToInteger`]).
const INTEGRAL: [f64; 14] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -7.0,
    255.0,
    65_536.0,
    4_294_967_296.0,
    -4_294_967_297.0,
    9_007_199_254_740_992.0,
    -9_007_199_254_740_991.0,
    4_611_686_018_427_387_904.0,
    -9_223_372_036_854_775_808.0,
];

fn states_of(values: &[f64]) -> Vec<Vec<TraceScalar>> {
    values
        .iter()
        .map(|x| vec![float(*x), TraceScalar::Nil])
        .collect()
}

#[test]
fn a_float_converted_to_an_integer_matches_cranelift() {
    let ir = op_trace(&[TraceType::Float], TraceType::Integer, |v| {
        TraceOp::Numeric {
            op: NumericOp::FloatToInteger,
            source: v[0],
        }
    });
    // Saturating: NaN is zero and what is out of range gives an extreme.
    let mut values = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e19,
        -1e19,
        1e308,
        9_223_372_036_854_775_807.0,
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_777_856.0,
        0.5,
        -0.5,
        2.75,
        -2.75,
    ];
    values.extend(INTEGRAL);
    same_everywhere("float to integer", &ir, &states_of(&values), 100);
}

/// The rounding trick of `bit.*`: defined for every float, NaN and the
/// infinities included.
#[test]
fn the_low_word_of_a_rounded_float_matches_cranelift() {
    let ir = op_trace(&[TraceType::Float], TraceType::Integer, |v| {
        TraceOp::Numeric {
            op: NumericOp::TrickBits,
            source: v[0],
        }
    });
    let mut values = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.5,
        1.5,
        2.5,
        -0.5,
        -1.5,
        3.001,
        -583.5,
        2_251_799_813_685_248.0,
        -2_251_799_813_685_249.0,
        4_294_967_295.5,
        4_294_967_296.5,
        1e300,
        5e-324,
        f64::MAX,
        f64::MIN_POSITIVE,
    ];
    values.extend(INTEGRAL);
    same_everywhere("rounding trick", &ir, &states_of(&values), 100);
}

#[test]
fn the_absolute_value_of_an_integer_matches_cranelift() {
    let ir = op_trace(&[TraceType::Integer], TraceType::Integer, |v| {
        TraceOp::Numeric {
            op: NumericOp::Abs,
            source: v[0],
        }
    });
    let states: Vec<Vec<TraceScalar>> = INTEGERS
        .iter()
        .chain(&[i64::MIN, i64::MIN + 1, 1 << 53, -(1 << 53)])
        .map(|x| vec![TraceScalar::Integer(*x), TraceScalar::Nil])
        .collect();
    same_everywhere("integer abs", &ir, &states, 100);
}

/// A conversion and a use of its result in the same trace: the integer feeds
/// integer arithmetic, then goes back to a float.
#[test]
fn a_converted_integer_feeds_integer_arithmetic_like_cranelift() {
    let mut b = TraceBuilder::new(TraceId(0), frame(2));
    b.mark_written(FrameId(0), 1);
    let x = b.input(0, TraceType::Float);
    let whole = b
        .emit(
            site(0),
            TraceOp::Numeric {
                op: NumericOp::FloatToInteger,
                source: x,
            },
            Some(TraceType::Integer),
        )
        .unwrap();
    let magnitude = b
        .emit(
            site(0),
            TraceOp::Numeric {
                op: NumericOp::Abs,
                source: whole,
            },
            Some(TraceType::Integer),
        )
        .unwrap();
    let back = b
        .emit(
            site(0),
            TraceOp::IntToFloat { source: magnitude },
            Some(TraceType::Float),
        )
        .unwrap();
    let exit = b.snapshot(snapshot(
        after(0),
        0,
        vec![RegisterEntry::Value(x), RegisterEntry::Value(back)],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    same_everywhere(
        "convert, abs, convert back",
        &ir,
        &states_of(&INTEGRAL),
        100,
    );
}

// ---- a register of either subtype, entering a side ---------------------------------------

/// The loop of `tree_with_a_leaving_side` with the numbers `i` (register 2)
/// and `no` (register 1) restored as an integer or a float by the booleans in
/// registers 4 and 5, and a side that takes `no` and `i` as `no_ty` and
/// `i_ty`: `yes = yes + i`, then leave.
fn tree(no_ty: TraceType, i_ty: TraceType) -> TraceIr {
    tree_ending(no_ty, i_ty, false)
}

/// [`tree`], with a side that, when `rejoin`, does a step of the loop
/// itself (`no = no + i`, `i = i + 1`, as floats) and links back to it
/// instead of leaving.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "the names of the loop tree in `mir/tests.rs`"
)]
fn tree_ending(no_ty: TraceType, i_ty: TraceType, rejoin: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(6));
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no_is_integer = b.input(4, TraceType::Boolean);
    let i_is_integer = b.input(5, TraceType::Boolean);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let f = |b: &mut TraceBuilder, pc, op, ty| b.emit(site(pc), op, Some(ty)).unwrap();
    let seven = f(&mut b, 0, TraceOp::Constant(float(7.0)), TraceType::Float);
    let rem = f(
        &mut b,
        0,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Modulo,
            left: i,
            right: seven,
        },
        TraceType::Float,
    );
    let three = f(&mut b, 1, TraceOp::Constant(float(3.0)), TraceType::Float);
    let c = f(
        &mut b,
        1,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: rem,
            right: three,
        },
        TraceType::Boolean,
    );
    let regs = |no: ValueId, i: ValueId, c: ValueId| {
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Value(c),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ]
    };
    let with_numbers = snapshot(after(2), 2, regs(no, i, c)).with_numbers(vec![
        NumberRegister {
            register: 1,
            is_integer: RegisterEntry::Value(no_is_integer),
        },
        NumberRegister {
            register: 2,
            is_integer: RegisterEntry::Value(i_is_integer),
        },
    ]);
    let snap = b.snapshot(with_numbers);
    let guard = b.guard(site(2), snap);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: c,
            expected: false,
            guard,
        },
        None,
    );
    let no2 = f(
        &mut b,
        5,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: no,
            right: i,
        },
        TraceType::Float,
    );
    let one = f(&mut b, 6, TraceOp::Constant(float(1.0)), TraceType::Float);
    let i2 = f(
        &mut b,
        6,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: one,
        },
        TraceType::Float,
    );
    let top = f(&mut b, 6, TraceOp::Constant(float(100.0)), TraceType::Float);
    let more = f(
        &mut b,
        6,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i2,
            right: top,
        },
        TraceType::Boolean,
    );
    let snap = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        6,
        regs(no2, i2, c),
    ));
    let guard = b.guard(site(6), snap);
    b.emit(
        site(6),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        regs(no2, i2, c),
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    ir.header_live = vec![0, 1, 2];
    // The side: `yes = yes + i`, then leave.
    let mut s = TraceBuilder::new(TraceId(1), frame(6));
    s.mark_written(FrameId(0), 0);
    let yes = s.input(0, TraceType::Float);
    let no = s.input(1, no_ty);
    let i = s.input(2, i_ty);
    let as_float = |s: &mut TraceBuilder, value: ValueId, ty: TraceType| {
        if ty == TraceType::Integer {
            s.emit(
                site(3),
                TraceOp::IntToFloat { source: value },
                Some(TraceType::Float),
            )
            .unwrap()
        } else {
            value
        }
    };
    let i_float = as_float(&mut s, i, i_ty);
    let yes2 = s
        .emit(
            site(3),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: yes,
                right: i_float,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    if rejoin {
        let no_float = as_float(&mut s, no, no_ty);
        let no2 = s
            .emit(
                site(4),
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left: no_float,
                    right: i_float,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let one = s
            .emit(
                site(4),
                TraceOp::Constant(float(1.0)),
                Some(TraceType::Float),
            )
            .unwrap();
        let i2 = s
            .emit(
                site(4),
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left: i_float,
                    right: one,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let link = s.snapshot(snapshot(
            SnapshotPhase::After {
                pc: 4,
                successor: 0,
            },
            4,
            vec![
                RegisterEntry::Value(yes2),
                RegisterEntry::Value(no2),
                RegisterEntry::Value(i2),
                RegisterEntry::Const(TraceScalar::Nil),
                RegisterEntry::Unchanged,
                RegisterEntry::Unchanged,
            ],
        ));
        ir.sides.push(TraceSide {
            parent: None,
            guard: 0,
            ir: Box::new(s.finish(TraceTerminator::Link { snapshot: link })),
        });
        return ir;
    }
    let out = s.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 4,
            successor: 12,
        },
        4,
        vec![
            RegisterEntry::Value(yes2),
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Const(TraceScalar::Nil),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ],
    ));
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(s.finish(TraceTerminator::Exit { snapshot: out })),
    });
    ir
}

fn number_states() -> Vec<Vec<TraceScalar>> {
    let mut states = Vec::new();
    for start in [1.0, 2.0, 3.0, 4.0, 8.0, 50.5, 99.0, 100.0] {
        for yes in [0.0, 5.0] {
            for (no_integer, i_integer) in
                [(false, false), (false, true), (true, false), (true, true)]
            {
                states.push(vec![
                    float(yes),
                    float(6.0),
                    float(start),
                    TraceScalar::Nil,
                    TraceScalar::Boolean(no_integer),
                    TraceScalar::Boolean(i_integer),
                ]);
            }
        }
    }
    states
}

/// The side runs when each number has the subtype it was recorded for, and
/// the guard leaves through its own exit otherwise, as Cranelift's `leave`
/// does.
#[test]
fn a_side_that_takes_numbers_of_a_run_time_subtype_matches_cranelift() {
    use TraceType::{Float, Integer};
    for (no_ty, i_ty) in [
        (Float, Float),
        (Float, Integer),
        (Integer, Float),
        (Integer, Integer),
    ] {
        let ir = tree(no_ty, i_ty);
        for fuel in [2, 3, 10_000] {
            same_everywhere(
                &format!("numbers {no_ty:?} and {i_ty:?} to a side, fuel {fuel}"),
                &ir,
                &number_states(),
                fuel,
            );
        }
        let mir = compile(&ir, OPTIMIZED, true);
        assert_eq!(mir.member_count(), 2, "the side is part of the tree");
    }
}

/// The same side, rejoining the loop instead of leaving: MIR takes the tree
/// (it handed it to Cranelift until the number load and the rejoin got
/// cheaper, see `guard_target`), converts the numbers in the guard's skip
/// and links back like Cranelift.
#[test]
fn a_rejoining_side_that_takes_numbers_of_a_run_time_subtype_matches_cranelift() {
    use TraceType::{Float, Integer};
    for (no_ty, i_ty) in [
        (Float, Float),
        (Float, Integer),
        (Integer, Float),
        (Integer, Integer),
    ] {
        let ir = tree_ending(no_ty, i_ty, true);
        for fuel in [2, 3, 7, 10_000] {
            same_everywhere(
                &format!("numbers {no_ty:?} and {i_ty:?} to a rejoining side, fuel {fuel}"),
                &ir,
                &number_states(),
                fuel,
            );
        }
        let mir = compile(&ir, OPTIMIZED, true);
        assert_eq!(mir.member_count(), 2, "the side is part of the tree");
    }
}

/// A loop side that continues a terminal exit takes `i` as a float, and the
/// exit records the register as a number that is an integer or not. A
/// constant says which, so one tree has the loop run and the other leave.
#[test]
fn a_loop_side_entered_with_a_number_of_a_recorded_subtype_matches_cranelift() {
    use super::super::super::tree_tests::loop_side;
    for is_integer in [false, true] {
        let mut ir = loop_side::tree(100.0, 40.0);
        let TraceTerminator::Exit { snapshot } = ir.sides[0].ir.terminator else {
            panic!("the side ends in an exit");
        };
        let snapshot = snapshot.index();
        ir.sides[0].ir.snapshots[snapshot].numbers = vec![NumberRegister {
            register: 2,
            is_integer: RegisterEntry::Const(TraceScalar::Boolean(is_integer)),
        }];
        let states: Vec<Vec<TraceScalar>> = [1.0, 2.0, 3.0, 4.0, 7.0, 15.0, 39.0, 41.0, -3.0]
            .iter()
            .map(|start| vec![float(10.0), float(20.0), float(*start), TraceScalar::Nil])
            .collect();
        for fuel in [1_000_000, 40, 3] {
            same_everywhere(
                &format!("loop side, integer {is_integer}, fuel {fuel}"),
                &ir,
                &states,
                fuel,
            );
        }
    }
}

// ---- the fields of a table the trace makes, filled in place ---------------------------------

/// A loop that makes a table and fills it each iteration runs the same
/// through MIR and Cranelift: the helper calls, what they were given, the
/// fields stored in place and counted, and the slots' contents. A fill that
/// may not trust the slots (a second store under the key, a store by another
/// helper that may move them, a nil that deletes the entry, an integer key
/// without a prefilled array) is the helper's in both.
#[test]
fn the_fields_of_a_made_table_are_filled_in_place_like_cranelift() {
    use super::super::super::new_table_tests::{Fill, run_fill};
    for fill in [
        Fill::Name,
        Fill::NameTwice,
        Fill::HelperThenName,
        Fill::NilThenName,
        Fill::Integer,
        Fill::NameInteger,
        Fill::NameBoolean,
        Fill::Array,
    ] {
        for give_slots in [true, false] {
            for (iterations, miss_on) in [(5, 0), (10, 3), (1, 1)] {
                for tier in TIERS {
                    let mut seen = Vec::new();
                    for mir in [true, false] {
                        FORCE_BACKEND.with(|force| force.set(Some(mir)));
                        LAST_FALLBACK.with(|last| last.set(None));
                        let (calls, class, initialized) =
                            run_fill(tier, fill, iterations, (miss_on, give_slots));
                        let refused = LAST_FALLBACK.with(Cell::get);
                        FORCE_BACKEND.with(|force| force.set(None));
                        assert_eq!(refused, None, "mir {mir}, give_slots {give_slots}");
                        let stored: Vec<_> =
                            calls.slots.iter().map(|slots| slots[0].clone()).collect();
                        seen.push((
                            class,
                            calls.made,
                            calls.filled,
                            calls.set,
                            calls.last_new,
                            calls.last_init,
                            calls.lanes,
                            initialized,
                            format!("{stored:?}"),
                        ));
                    }
                    assert_eq!(
                        seen[0], seen[1],
                        "fill {}, slots {give_slots}, {iterations} iterations, miss {miss_on}, {tier:?}",
                        fill as u8
                    );
                }
            }
        }
    }
}

// ---- shifts by a count known at run time --------------------------------------------------

/// Lua's shifts with the distance in a register: every sign, 63 and 64 places,
/// the most negative distance, and values with the top bit set.
#[test]
fn shifts_by_a_run_time_distance_match_cranelift() {
    let distances = [
        0,
        1,
        -1,
        5,
        -5,
        31,
        32,
        -32,
        62,
        63,
        64,
        -63,
        -64,
        65,
        -65,
        1 << 40,
        -(1 << 40),
        i64::MAX,
        i64::MIN,
        i64::MIN + 1,
    ];
    let values = [0, 1, -1, 2, 0x1234_5678_9abc_def0, i64::MIN, i64::MAX, 255];
    let states: Vec<Vec<TraceScalar>> = values
        .iter()
        .flat_map(|value| {
            distances.iter().map(move |distance| {
                vec![
                    TraceScalar::Integer(*value),
                    TraceScalar::Integer(*distance),
                    TraceScalar::Nil,
                ]
            })
        })
        .collect();
    for (name, op) in [
        ("left", ArithmeticOp::ShiftLeft),
        ("right", ArithmeticOp::ShiftRight),
    ] {
        let ir = op_trace(&[TraceType::Integer; 2], TraceType::Integer, |v| {
            TraceOp::Arithmetic {
                op,
                left: v[0],
                right: v[1],
            }
        });
        same_everywhere(&format!("shift {name} by a value"), &ir, &states, 100);
    }
}

/// Without BMI2 the machine has no shift by a register: the tree goes to
/// Cranelift with the reason, and still runs the same.
#[test]
fn a_shift_by_a_value_stays_with_cranelift_on_a_target_without_bmi2() {
    use oximond_mir::x64::X64;
    let ir = op_trace(&[TraceType::Integer; 2], TraceType::Integer, |v| {
        TraceOp::Arithmetic {
            op: ArithmeticOp::ShiftLeft,
            left: v[0],
            right: v[1],
        }
    });
    super::super::support::with_target(X64::sse2(), || {
        FORCE_BACKEND.with(|force| force.set(Some(true)));
        LAST_FALLBACK.with(|last| last.set(None));
        let trace = compile_trace(&ir, OPTIMIZED).unwrap();
        let reason = LAST_FALLBACK.with(Cell::get);
        FORCE_BACKEND.with(|force| force.set(None));
        assert!(trace.mir_address().is_none());
        assert_eq!(reason, Some("shift by a value on a target without BMI2"));
    });
}

// ---- the native step of a boxed `next` loop ----------------------------------------------

/// Loops over tables that have a hash part keep the control value in a lane of
/// the bank, and the step is made in line before the helper is called
/// (`steps.rs`). MIR and Cranelift run the same loops to the same results and
/// the same exits, in every profile (keys of the array part are floats in
/// Lua 5.1 and 5.2 and `LuaJIT`, integers after), over arrays with holes and
/// values of every kind a step takes, and over several tables of different
/// sizes in turn, which reuse the lanes and the snapshot of the run before.
#[test]
fn a_boxed_next_step_matches_cranelift() {
    use crate::{CompatibilityProfile, ExecutionMode, Lua};
    const SOURCES: [&str; 9] = [
        "local n, c = 0, 0 local t = {} for i = 1, 60 do t[i] = i end for i = 1, 60 do t['k' .. i] = i + 100 end for r = 1, 4000 do for k, v in pairs(t) do n = n + v c = c + 1 end end print(n, c)",
        "local n,c=0,0 local t={} for i=1,90 do t[i]=i*2 end for i=1,90,4 do t[i]=nil end t.x=7 t.y=9 for r=1,4000 do for k,v in pairs(t) do n=n+v c=c+1 end end print(n,c)",
        "local n,c=0,0 local t={} for i=1,50 do t[i]=i+0.25 end t.a=1.5 t.b=2.5 for r=1,4000 do for k,v in pairs(t) do n=n+v c=c+1 end end print(n,c)",
        "local n,c=0,0 local t={} for i=1,50 do t[i]=(i%3==0) end t.a=true t.b=false for r=1,4000 do for k,v in pairs(t) do if v then n=n+1 end c=c+1 end end print(n,c)",
        "local n,c=0,0 local t={} for i=1,40 do t[i]=i end for i=1,40 do t[\"s\"..i]=i end for r=1,4000 do for k in pairs(t) do if type(k)==\"number\" then n=n+k end c=c+1 end end print(n,c)",
        "local n,c=0,0 local t={} for i=1,50 do t[i]=i end t.a=1 t.b=2 for r=1,4000 do for k,v in pairs(t) do c=c+1 end end print(n,c)",
        "local n,c=0,0 local t={} for i=1,50 do t[i]=i end t.a=1 t.b=2 for r=1,4000 do for k,v in next,t do n=n+v c=c+1 end end print(n,c)",
        "local ts={} for j=1,4 do local t={} for i=1,j*13 do t[i]=(i%2==0) and i+0.5 or i end for i=1,j*5 do t[\"z\"..i]=j+i end ts[j]=t end local n,c=0,0 for r=1,6000 do local t=ts[r%4+1] for k,v in pairs(t) do n=n+v c=c+1 end end print(n,c)",
        "local n,c=0,0 local t={} for i=1,60 do t[i]=i end for i=1,10 do t[\"k\"..i]=i end for r=1,3000 do for k,v in pairs(t) do if k==5 then t[6]=nil end if k==\"k3\" then t.k4=nil end n=n+v c=c+1 end for i=1,60 do t[i]=i end for i=1,10 do t[\"k\"..i]=i end end print(n,c)",
    ];
    const PROFILES: [CompatibilityProfile; 6] = [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
        CompatibilityProfile::LuaJit21,
    ];
    let run = |profile, mode, mir: Option<bool>, source: &str| {
        FORCE_BACKEND.with(|force| force.set(mir));
        LAST_FALLBACK.with(|last| last.set(None));
        let mut lua = Lua::builder()
            .profile(profile)
            .execution_mode(mode)
            .build()
            .unwrap();
        lua.vm_mut().use_synchronous_jit_compilation();
        let results = lua.exec(source).map(|values| format!("{values:?}"));
        let reason = LAST_FALLBACK.with(Cell::get);
        FORCE_BACKEND.with(|force| force.set(None));
        let stats = lua.vm().jit_statistics().trace;
        (
            results.map_err(|error| format!("{error:?}")),
            (stats.iterations, stats.exits, stats.side_exits),
            reason,
        )
    };
    for source in SOURCES {
        for profile in PROFILES {
            let interpreted = run(profile, ExecutionMode::Interpreter, None, source).0;
            let cranelift = run(profile, ExecutionMode::Tracing, Some(false), source);
            let mir = run(profile, ExecutionMode::Tracing, Some(true), source);
            assert_eq!(cranelift.0, interpreted, "{profile:?} cranelift: {source}");
            assert_eq!(mir.0, interpreted, "{profile:?} mir: {source}");
            assert_eq!(mir.1, cranelift.1, "{profile:?} statistics: {source}");
            assert!(
                mir.2.is_none_or(|reason| !reason.contains("next step")),
                "{profile:?}: {:?}: {source}",
                mir.2
            );
        }
    }
}
