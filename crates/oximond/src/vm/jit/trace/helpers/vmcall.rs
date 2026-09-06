//! The helper that makes a host C function call from a trace
//! ([`HelperId::vm_call`]).
//!
//! The call is the interpreter's own `Call`: the helper writes the state of
//! the call's `Before` snapshot back to the frame (the restore of the exit
//! native code spilled just before calling here), makes the call through the
//! host boundary with the arguments in their registers
//! ([`Vm::host_call_for_trace`]), and hands the results to the trace when
//! they are of the kinds the recording saw. It runs the VM, and so follows
//! the rules of `resume` ([`super::resume`]): the run keeps no reference into
//! the VM, the stack, the runtime or the bank across the call, and reads
//! every array, length and header binding again afterwards
//! ([`RunTables::refresh_bindings`]).
//!
//! It misses before the call, having changed nothing but the frame (which
//! the exit writes again), when the interpreter would not make the call
//! itself: a hook, a coroutine, another callee. It misses after the call
//! when the trace cannot take the answer. An error or other results wait in
//! the VM for the interpreter's replay of the call; a call that completed
//! leaves the frame after it (`TracePolicy::vmcall_landed`).

use super::super::host::publish_string;
use super::super::native::RunTables;
use crate::jit::{TraceInputSlot, install_helper};
use crate::jit_ir::trace::{HelperId, MAX_RESUME_VALUES};
use crate::runtime::Value;
use crate::vm::Vm;
use crate::vm::fast_path::TraceHostCall;
use crate::vm::stack::CallStack;

/// The call of a host function that gave `VALUES` values of the kinds
/// `KINDS` says (see [`HelperId::vm_call`]). `area` is the exit area native
/// code spilled the `Before` snapshot's values into, and `exit` that
/// snapshot's exit. One instance per shape only names the shape: the work is
/// [`call`].
extern "C" fn helper_vm_call<const VALUES: usize, const KINDS: usize>(
    context: usize,
    out: *mut i64,
    words: *mut TraceInputSlot,
    area: *const i64,
    exit: i64,
) -> i64 {
    call(
        context,
        (out, words, area, exit),
        (VALUES, KINDS),
        gives_objects(KINDS),
    )
}

/// Whether a shape of results holds a table or a string, which need a lane.
const fn gives_objects(mut kinds: usize) -> bool {
    while kinds > 0 {
        if kinds % 6 >= 4 {
            return true;
        }
        kinds /= 6;
    }
    false
}

/// The objects the results of a call hold, by place.
type Objects = [Option<Value>; MAX_RESUME_VALUES];

/// The words of `values` if they are of the kinds `shape` says (see
/// [`HelperId::vm_call`]). A table or string result has no word yet: the
/// value waits in `objects` for a lane of the run's bank. A missing value is
/// nil. A call that returns all its results (`tail`) must give exactly as
/// many as the shape has.
fn fits(
    values: &[Value],
    tail: bool,
    (count, kinds): (usize, usize),
    objects: &mut Objects,
) -> Option<[i64; MAX_RESUME_VALUES]> {
    if tail && values.len() != count {
        return None;
    }
    let mut words = [0; MAX_RESUME_VALUES];
    let mut digits = kinds;
    for (position, word) in words.iter_mut().enumerate().take(count) {
        let value = values.get(position).unwrap_or(&Value::Nil);
        *word = match (digits % 6, value) {
            (0, Value::Integer(integer)) => *integer,
            (1, Value::Number(number)) => i64::from_ne_bytes(number.to_bits().to_ne_bytes()),
            (2, Value::Boolean(flag)) => i64::from(*flag),
            (3, Value::Nil) => 0,
            (4, Value::Table(_)) | (5, Value::String(_)) => {
                objects[position] = Some(value.clone());
                0
            }
            _ => return None,
        };
        digits /= 6;
    }
    Some(words)
}

/// The call [`helper_vm_call`] makes, for results of `shape`; `objects_given`
/// says whether some are tables or strings.
#[allow(unsafe_code)]
#[inline(never)]
fn call(
    context: usize,
    (out, words, area, exit): (*mut i64, *mut TraceInputSlot, *const i64, i64),
    shape: (usize, usize),
    objects_given: bool,
) -> i64 {
    // SAFETY: `context` is the address of the `RunTables` that handed out the
    // helper entry, which outlives the run and is used by nothing else while
    // native code calls here.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let Some((vm_address, frames_address)) = tables
        .resume
        .as_ref()
        .and_then(|state| state.host.as_ref())
        .map(|host| (host.vm, host.frames))
    else {
        return 1;
    };
    let Ok(exit) = usize::try_from(exit) else {
        return 1;
    };
    // SAFETY: the run lent the VM and the stack for exactly this call and
    // keeps no reference to either; this one ends before `tables` is used.
    if !unsafe { lent_vm(vm_address) }.vmcall_ready() {
        return 1;
    }
    // SAFETY: as above.
    let Some((pc, pushed)) =
        tables.restore_exit(exit, area, unsafe { lent_frames(frames_address) })
    else {
        return 1;
    };
    // What the trace writes to cells and globals is in memory now (the
    // restore stored it). The call may change it, and the trace holds it in
    // native values.
    // SAFETY: as above.
    let written = tables.written_externals(unsafe { lent_frames(frames_address) });
    // SAFETY: the uses of `tables` are over, and nothing else holds the VM.
    let (vm, frames) = unsafe { (lent_vm(vm_address), lent_frames(frames_address)) };
    let frame_index = frames.len() - 1;
    let mut objects: Objects = [None, None, None];
    // What the trace reads from cells closed at entry holds after a call that
    // wrote no cell.
    let cell_writes = vm.runtime.cell_writes();
    let mut words_out = match vm.host_call_for_trace(frames, frame_index, pc, |values, tail| {
        fits(values, tail, shape, &mut objects)
    }) {
        TraceHostCall::Answered(words) => words,
        // The call did not run: the exit writes the state again, so the
        // frames it pushed come off.
        TraceHostCall::Declined => {
            take_off(frames, pushed);
            return 1;
        }
        // The call ran and the frames stand as it left them.
        TraceHostCall::Missed => return 1,
    };
    let cells_written = vm.runtime.cell_writes() != cell_writes;
    // The call may have run any Lua code. The tables the run's bindings name
    // must still be the same, and so must the cells it read. The tables and
    // strings it gave get lanes of the run's bank, which roots them until
    // the run ends: nothing else names them but the frame's registers, which
    // the trace overwrites.
    let holds = tables.refresh_bindings(words)
        // SAFETY: `vm` is not used again before it is made again.
        && tables.externals_hold(
            unsafe { lent_frames(frames_address) },
            &written,
            cells_written,
        )
        && (!objects_given || give_lanes(tables, &objects, &mut words_out));
    if !holds {
        // The call is complete and the frames stand after it, the inlined
        // ones too.
        // SAFETY: as above.
        unsafe { lent_vm(vm_address) }
            .jit
            .trace
            .set_vmcall_landed(pc + 1, pc);
        return 1;
    }
    // The rows of the tables it met by lane name slots the call may have
    // moved, and the tables it gave have none yet.
    tables.refill_rows_in_use(area);
    if objects_given {
        for (word, object) in words_out.iter().zip(&objects) {
            if matches!(object, Some(Value::Table(_))) {
                tables.fill_row(usize::try_from(*word).unwrap_or(usize::MAX));
            }
        }
    }
    // The trace carries on in its own frames: the ones the call saw come off.
    // SAFETY: as above.
    take_off(unsafe { lent_frames(frames_address) }, pushed);
    // SAFETY: `out` is the control block's `helper_out`, which holds a word
    // per result and which nothing else touches during the call.
    unsafe {
        for (position, word) in words_out.iter().enumerate().take(shape.0) {
            *out.add(position) = *word;
        }
    }
    0
}

/// Gives each of `objects` a lane of the run's bank and puts it in `words`.
/// Whether the run had room.
fn give_lanes(
    tables: &mut RunTables<'static>,
    objects: &Objects,
    words: &mut [i64; MAX_RESUME_VALUES],
) -> bool {
    for (position, object) in objects.iter().enumerate() {
        let lane = match object {
            None => continue,
            Some(Value::Table(table)) => tables.result_table_lane(*table),
            Some(Value::String(text)) => {
                let (_, bank, _, state, lengths) = tables.library();
                let mut slot = TraceInputSlot::nil();
                publish_string(bank, state, lengths, text.clone(), &mut slot)
                    .then(|| usize::try_from(slot.bits()).unwrap_or(usize::MAX))
            }
            Some(_) => None,
        };
        let Some(lane) = lane else {
            return false;
        };
        words[position] = i64::try_from(lane).unwrap_or(i64::MAX);
    }
    true
}

/// Takes the `count` frames of inlined calls that the restore pushed off the
/// stack, once the host function has returned or been declined. Their
/// records stay for the restore of the next call to reuse.
fn take_off(frames: &mut CallStack, count: usize) {
    for _ in 0..count {
        frames.take_off_top();
    }
}

/// The VM the run lent.
///
/// # Safety
///
/// As `resume`'s `lent_vm`: `address` is the `vm` of a `ResumeHost` of a run
/// in flight, and no reference to the VM is live.
#[allow(unsafe_code)]
unsafe fn lent_vm<'a>(address: usize) -> &'a mut Vm {
    // SAFETY: as above.
    unsafe { &mut *(address as *mut Vm) }
}

/// The call stack the run lent, as [`lent_vm`].
///
/// # Safety
///
/// As [`lent_vm`], for the `frames` of the host.
#[allow(unsafe_code)]
unsafe fn lent_frames<'a>(address: usize) -> &'a mut CallStack {
    // SAFETY: as above.
    unsafe { &mut *(address as *mut CallStack) }
}

/// Installs the addresses of the helpers of [`HelperId::vm_call`].
pub(super) fn install() {
    macro_rules! shapes {
        ($(($values:literal, $kinds:literal)),* $(,)?) => {
            [$((($values, $kinds), helper_vm_call::<$values, $kinds> as *const () as usize)),*]
        };
    }
    #[rustfmt::skip]
    let table: [((usize, usize), usize); 259] = shapes![
        (0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16),
        (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24),
        (2, 25), (2, 26), (2, 27), (2, 28), (2, 29), (2, 30), (2, 31), (2, 32),
        (2, 33), (2, 34), (2, 35), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
        (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12),
        (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20),
        (3, 21), (3, 22), (3, 23), (3, 24), (3, 25), (3, 26), (3, 27), (3, 28),
        (3, 29), (3, 30), (3, 31), (3, 32), (3, 33), (3, 34), (3, 35), (3, 36),
        (3, 37), (3, 38), (3, 39), (3, 40), (3, 41), (3, 42), (3, 43), (3, 44),
        (3, 45), (3, 46), (3, 47), (3, 48), (3, 49), (3, 50), (3, 51), (3, 52),
        (3, 53), (3, 54), (3, 55), (3, 56), (3, 57), (3, 58), (3, 59), (3, 60),
        (3, 61), (3, 62), (3, 63), (3, 64), (3, 65), (3, 66), (3, 67), (3, 68),
        (3, 69), (3, 70), (3, 71), (3, 72), (3, 73), (3, 74), (3, 75), (3, 76),
        (3, 77), (3, 78), (3, 79), (3, 80), (3, 81), (3, 82), (3, 83), (3, 84),
        (3, 85), (3, 86), (3, 87), (3, 88), (3, 89), (3, 90), (3, 91), (3, 92),
        (3, 93), (3, 94), (3, 95), (3, 96), (3, 97), (3, 98), (3, 99), (3, 100),
        (3, 101), (3, 102), (3, 103), (3, 104), (3, 105), (3, 106), (3, 107), (3, 108),
        (3, 109), (3, 110), (3, 111), (3, 112), (3, 113), (3, 114), (3, 115), (3, 116),
        (3, 117), (3, 118), (3, 119), (3, 120), (3, 121), (3, 122), (3, 123), (3, 124),
        (3, 125), (3, 126), (3, 127), (3, 128), (3, 129), (3, 130), (3, 131), (3, 132),
        (3, 133), (3, 134), (3, 135), (3, 136), (3, 137), (3, 138), (3, 139), (3, 140),
        (3, 141), (3, 142), (3, 143), (3, 144), (3, 145), (3, 146), (3, 147), (3, 148),
        (3, 149), (3, 150), (3, 151), (3, 152), (3, 153), (3, 154), (3, 155), (3, 156),
        (3, 157), (3, 158), (3, 159), (3, 160), (3, 161), (3, 162), (3, 163), (3, 164),
        (3, 165), (3, 166), (3, 167), (3, 168), (3, 169), (3, 170), (3, 171), (3, 172),
        (3, 173), (3, 174), (3, 175), (3, 176), (3, 177), (3, 178), (3, 179), (3, 180),
        (3, 181), (3, 182), (3, 183), (3, 184), (3, 185), (3, 186), (3, 187), (3, 188),
        (3, 189), (3, 190), (3, 191), (3, 192), (3, 193), (3, 194), (3, 195), (3, 196),
        (3, 197), (3, 198), (3, 199), (3, 200), (3, 201), (3, 202), (3, 203), (3, 204),
        (3, 205), (3, 206), (3, 207), (3, 208), (3, 209), (3, 210), (3, 211), (3, 212),
        (3, 213), (3, 214), (3, 215),
    ];
    for ((values, kinds), function) in table {
        install_helper(HelperId::vm_call(values, kinds), function);
    }
}
