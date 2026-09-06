//! The helper that resumes a coroutine from a trace ([`HelperId::coroutine_resume`]).
//!
//! The call is the fast loop's own `coroutine.resume`: the thread runs to its
//! next switch on the VM's heap stacks, with the trace's frames as its
//! resumer (see [`Vm::resume_for_trace`]). The helper is the one place a
//! trace runs Lua code in the middle of its run, and it takes the VM and the
//! call stack from the run ([`ResumeHost`](super::super::native::ResumeHost)),
//! which lent their addresses for exactly this. The run keeps no reference
//! to either, nor to the runtime or the root bank (`RunTables` holds
//! pointers), so the helper makes the `&mut Vm` and `&mut CallStack` the
//! resumed code needs, and makes them again after anything of the run's has
//! been used. What it may not do is trust anything the run holds about
//! tables afterwards: it reads every array, length and header binding word
//! again from the tables ([`RunTables::refresh_bindings`]).
//!
//! It answers only the call the recording saw: the thread yielded or returned
//! values of exactly the recorded types after the leading `true`. Any other
//! answer is a miss *after* the thread ran. The VM keeps what the call gave
//! ([`Vm::remember_resume`]), the trace leaves before the call as for any
//! missed library call, and the interpreter's execution of the call takes the
//! answer from the VM instead of running the thread again.

use super::super::host;
use super::super::native::RunTables;
use crate::jit::{TRACE_TAG_THREAD, TraceInputSlot, install_helper};
use crate::jit_ir::trace::{HelperId, MAX_RESUME_VALUES};
use crate::runtime::{ThreadHandle, Value};
use crate::vm::Vm;
use crate::vm::coroutines::{Direct, Resumed};
use crate::vm::stack::CallStack;

/// The coroutine an argument slot names: a lane of the run's bank.
fn thread_of(bank: &[Value], slot: TraceInputSlot) -> Option<ThreadHandle> {
    if slot.tag() != TRACE_TAG_THREAD {
        return None;
    }
    match bank.get(usize::try_from(slot.bits()).ok()?)? {
        Value::Thread(thread) => Some(*thread),
        _ => None,
    }
}

/// The value an argument slot stands for.
#[inline(always)]
fn argument_of(bank: &[Value], roots: &[Value], slot: TraceInputSlot) -> Option<Value> {
    if slot.tag() == TRACE_TAG_THREAD {
        return match bank.get(usize::try_from(slot.bits()).ok()?)? {
            held @ Value::Thread(_) => Some(held.clone()),
            _ => None,
        };
    }
    slot.scalar_value()
        .or_else(|| host::lane_value(slot, bank, roots).cloned())
}

/// The words of `values`, the ones after the `true` of a resume that
/// succeeded, if they are of the kinds `KINDS` says (the base-3 digits: 0 an
/// integer, 1 a float, 2 a boolean). A missing value is nil, which no kind
/// takes.
pub(super) fn fits<const VALUES: usize, const KINDS: usize>(
    values: &[Value],
) -> Option<[i64; MAX_RESUME_VALUES]> {
    let mut words = [0; MAX_RESUME_VALUES];
    let mut digits = KINDS;
    for (position, word) in words.iter_mut().enumerate().take(VALUES) {
        *word = match (digits % 3, values.get(position)?) {
            (0, Value::Integer(integer)) => *integer,
            (1, Value::Number(number)) => i64::from_ne_bytes(number.to_bits().to_ne_bytes()),
            (2, Value::Boolean(flag)) => i64::from(*flag),
            _ => return None,
        };
        digits /= 3;
    }
    Some(words)
}

/// The VM the run lent.
///
/// # Safety
///
/// `address` is the `vm` of a [`ResumeHost`](super::super::native::ResumeHost)
/// of a run in flight, and no reference to the VM, or to anything in it, is
/// live: the run keeps none across its native call, and the caller makes no
/// second. The reference ends before the caller next uses the run's runtime
/// or bank, which are parts of the VM.
#[allow(unsafe_code)]
unsafe fn lent_vm<'a>(address: usize) -> &'a mut Vm {
    // SAFETY: as above.
    unsafe { &mut *(address as *mut Vm) }
}

/// The call stack the run lent.
///
/// # Safety
///
/// As [`lent_vm`], for the `frames` of the host: the run keeps no reference
/// into the stack across its native call.
#[allow(unsafe_code)]
unsafe fn lent_frames<'a>(address: usize) -> &'a mut CallStack {
    // SAFETY: as above.
    unsafe { &mut *(address as *mut CallStack) }
}

/// The resume both helpers make: `thread_of` finds the thread in the run's
/// bank and `passed` are the arguments the call gives it. `wrapped` is a call
/// of a function `coroutine.wrap` made.
///
/// The run is in the middle of a native call, with no reference into the VM,
/// the stack, the runtime or the bank (see `ResumeHost`). The references this
/// makes follow one rule: the `&mut Vm` and `&mut CallStack` that the resume
/// uses are made after the last use of `tables` before it, and every use of
/// `tables` after it (which reaches the runtime and the bank, parts of the
/// VM) comes before any further use of them: they are made again.
#[allow(unsafe_code)]
fn resume_call<const VALUES: usize, const KINDS: usize>(
    tables: &mut RunTables<'static>,
    (out, words): (*mut i64, *mut TraceInputSlot),
    thread_of: impl FnOnce(&[Value]) -> Option<ThreadHandle>,
    passed: usize,
    slot_of: impl Fn(usize) -> TraceInputSlot,
    wrapped: bool,
) -> i64 {
    let Some((vm_address, frames_address)) = tables
        .resume
        .as_ref()
        .and_then(|state| state.host.as_ref())
        .map(|host| (host.vm, host.frames))
    else {
        return 1;
    };
    let (bank, roots) = tables.lanes();
    let Some(thread) = thread_of(bank) else {
        return 1;
    };
    // The arguments wait in a small array: the call gives at most four. A
    // resume that ran moved them out, so the array is dropped only when it
    // did not (below), which saves a drop of nils on the usual path.
    let mut arguments = std::mem::ManuallyDrop::new([const { Value::Nil }; 4]);
    let passed = passed.min(arguments.len());
    for (index, slot) in arguments.iter_mut().enumerate().take(passed) {
        if let Some(value) = argument_of(bank, roots, slot_of(index)) {
            *slot = value;
        } else {
            drop(std::mem::ManuallyDrop::into_inner(arguments));
            return 1;
        }
    }
    let given = passed;
    // SAFETY: the run is in a native call and keeps no reference into the VM
    // (`ResumeHost`); `bank` and `roots` point at the run's heap storage, not
    // into the VM's own bytes, and these are the only references to the VM
    // and the stack, as the run lent them, from here until the first use of
    // `tables` below.
    let (vm, frames) = unsafe { (lent_vm(vm_address), lent_frames(frames_address)) };
    // A thread that ran only its few instructions to a yield hands over the
    // values from its registers: the words of the recorded kinds, or, for
    // any other answer, the values the interpreter's replay will take.
    let direct = vm.resume_for_trace(
        frames,
        thread,
        &mut arguments[..given],
        wrapped,
        |yielded| {
            fits::<VALUES, KINDS>(yielded).ok_or_else(|| {
                let mut kept = Vec::with_capacity(yielded.len() + 1);
                kept.push(Value::Boolean(true));
                kept.extend(yielded.iter().cloned());
                kept
            })
        },
    );
    let result = match direct {
        None | Some(Direct::Declined) => {
            drop(std::mem::ManuallyDrop::into_inner(arguments));
            return 1;
        }
        // Such a thread wrote nothing the run reads, so there is nothing to
        // read again.
        Some(Direct::Yielded(Ok(answer))) => {
            // SAFETY: `out` is the control block's `helper_out`, which holds
            // a word per result and which nothing else touches during the
            // call.
            unsafe {
                for (position, word) in answer.iter().enumerate().take(VALUES) {
                    *out.add(position) = *word;
                }
            }
            vm.jit.statistics.coroutine_trace_unchecked += 1;
            return 0;
        }
        Some(Direct::Yielded(Err(kept))) => {
            vm.remember_resume(thread, Ok(kept));
            return 1;
        }
        Some(Direct::Ran(result)) => result,
    };
    // The answer must be the recorded one, no hook may have appeared, and the
    // tables the thread ran over must still be the ones the run's bindings
    // name. Any other answer waits in the VM for the interpreter's replay.
    let answer = if vm.hooks_installed() {
        None
    } else {
        match &result {
            Resumed::Yielded => fits::<VALUES, KINDS>(vm.quick.values()),
            Resumed::Done(done) => match &**done {
                Ok(values) if matches!(values.first(), Some(Value::Boolean(true))) => {
                    fits::<VALUES, KINDS>(&values[1..])
                }
                _ => None,
            },
        }
    };
    // A thread that ran only the instructions of `Vm::run_micro` wrote
    // nothing the run reads, so there is nothing to read again.
    let unchanged = matches!(result, Resumed::Yielded) && vm.quick.pure();
    let answer = answer.filter(|_| {
        // SAFETY: `vm` and `frames` are not used again before they are made
        // again below: `tables` reaches the runtime and the bank, which are
        // inside the VM, and it makes its own references to them.
        unchanged
            || (tables.refresh_bindings(words)
                && tables.externals_hold(unsafe { lent_frames(frames_address) }, &[], true))
    });
    // SAFETY: the uses of `tables` are over, and nothing else holds the VM.
    let vm = unsafe { lent_vm(vm_address) };
    let Some(answer) = answer else {
        vm.remember_resumed(thread, result);
        return 1;
    };
    // SAFETY: `out` is the control block's `helper_out`, which holds a word
    // per result and which nothing else touches during the call.
    unsafe {
        for (position, word) in answer.iter().enumerate().take(VALUES) {
            *out.add(position) = *word;
        }
    }
    if unchanged {
        vm.jit.statistics.coroutine_trace_unchecked += 1;
    }
    match result {
        Resumed::Yielded => vm.quick.clear(),
        Resumed::Done(done) => {
            if let Ok(values) = *done {
                vm.recycle_values(values);
            }
        }
    }
    0
}

/// `coroutine.resume (thread, arguments...)` that gave `VALUES` values of the
/// kinds `KINDS` says (see [`HelperId::coroutine_resume`]). `count` is the
/// number of arguments the call gave, the thread included.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_resume<const VALUES: usize, const KINDS: usize>(
    context: usize,
    out: *mut i64,
    words: *mut TraceInputSlot,
    count: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
    t3: i64,
    b3: i64,
    t4: i64,
    b4: i64,
) -> i64 {
    // The slots are built one at a time, from the words as they arrive: an
    // array of them is read back in 16-byte loads that cannot forward from
    // the eight-byte stores of the native caller's stack arguments.
    let given = usize::try_from(count).unwrap_or(0).min(5);
    if given == 0 {
        return 1;
    }
    // SAFETY: `context` is the address of the `RunTables` that handed out the
    // helper entry, which outlives the run and is used by nothing else while
    // native code calls here.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    resume_call::<VALUES, KINDS>(
        tables,
        (out, words),
        |bank| thread_of(bank, TraceInputSlot::from_words(t0, b0)),
        given - 1,
        |index| match index {
            0 => TraceInputSlot::from_words(t1, b1),
            1 => TraceInputSlot::from_words(t2, b2),
            2 => TraceInputSlot::from_words(t3, b3),
            _ => TraceInputSlot::from_words(t4, b4),
        },
        false,
    )
}

/// A call of a function `coroutine.wrap` made, that gave `VALUES` values of
/// the kinds `KINDS` says (see [`HelperId::coroutine_wrap`]). `callee` is the
/// lane of the function, which must be a wrapper; `count` is the number of
/// arguments the call gave, the function included.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_wrap<const VALUES: usize, const KINDS: usize>(
    context: usize,
    out: *mut i64,
    words: *mut TraceInputSlot,
    count: i64,
    callee: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
    t3: i64,
    b3: i64,
    t4: i64,
    b4: i64,
) -> i64 {
    let given = usize::try_from(count).unwrap_or(0).min(5);
    if given == 0 {
        return 1;
    }
    // SAFETY: as in `helper_resume`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    resume_call::<VALUES, KINDS>(
        tables,
        (out, words),
        |bank| match bank.get(usize::try_from(callee).ok()?)? {
            Value::CoroutineWrapper(thread) => Some(*thread),
            _ => None,
        },
        given - 1,
        |index| match index {
            0 => TraceInputSlot::from_words(t1, b1),
            1 => TraceInputSlot::from_words(t2, b2),
            2 => TraceInputSlot::from_words(t3, b3),
            _ => TraceInputSlot::from_words(t4, b4),
        },
        true,
    )
}

/// Installs the addresses of the helpers of [`HelperId::coroutine_resume`].
pub(super) fn install() {
    let table: [((usize, usize), usize); 40] = [
        ((0, 0), helper_resume::<0, 0> as *const () as usize),
        ((1, 0), helper_resume::<1, 0> as *const () as usize),
        ((1, 1), helper_resume::<1, 1> as *const () as usize),
        ((1, 2), helper_resume::<1, 2> as *const () as usize),
        ((2, 0), helper_resume::<2, 0> as *const () as usize),
        ((2, 1), helper_resume::<2, 1> as *const () as usize),
        ((2, 2), helper_resume::<2, 2> as *const () as usize),
        ((2, 3), helper_resume::<2, 3> as *const () as usize),
        ((2, 4), helper_resume::<2, 4> as *const () as usize),
        ((2, 5), helper_resume::<2, 5> as *const () as usize),
        ((2, 6), helper_resume::<2, 6> as *const () as usize),
        ((2, 7), helper_resume::<2, 7> as *const () as usize),
        ((2, 8), helper_resume::<2, 8> as *const () as usize),
        ((3, 0), helper_resume::<3, 0> as *const () as usize),
        ((3, 1), helper_resume::<3, 1> as *const () as usize),
        ((3, 2), helper_resume::<3, 2> as *const () as usize),
        ((3, 3), helper_resume::<3, 3> as *const () as usize),
        ((3, 4), helper_resume::<3, 4> as *const () as usize),
        ((3, 5), helper_resume::<3, 5> as *const () as usize),
        ((3, 6), helper_resume::<3, 6> as *const () as usize),
        ((3, 7), helper_resume::<3, 7> as *const () as usize),
        ((3, 8), helper_resume::<3, 8> as *const () as usize),
        ((3, 9), helper_resume::<3, 9> as *const () as usize),
        ((3, 10), helper_resume::<3, 10> as *const () as usize),
        ((3, 11), helper_resume::<3, 11> as *const () as usize),
        ((3, 12), helper_resume::<3, 12> as *const () as usize),
        ((3, 13), helper_resume::<3, 13> as *const () as usize),
        ((3, 14), helper_resume::<3, 14> as *const () as usize),
        ((3, 15), helper_resume::<3, 15> as *const () as usize),
        ((3, 16), helper_resume::<3, 16> as *const () as usize),
        ((3, 17), helper_resume::<3, 17> as *const () as usize),
        ((3, 18), helper_resume::<3, 18> as *const () as usize),
        ((3, 19), helper_resume::<3, 19> as *const () as usize),
        ((3, 20), helper_resume::<3, 20> as *const () as usize),
        ((3, 21), helper_resume::<3, 21> as *const () as usize),
        ((3, 22), helper_resume::<3, 22> as *const () as usize),
        ((3, 23), helper_resume::<3, 23> as *const () as usize),
        ((3, 24), helper_resume::<3, 24> as *const () as usize),
        ((3, 25), helper_resume::<3, 25> as *const () as usize),
        ((3, 26), helper_resume::<3, 26> as *const () as usize),
    ];
    for ((values, kinds), function) in table {
        install_helper(HelperId::coroutine_resume(values, kinds), function);
    }
    let wrapped: [((usize, usize), usize); 40] = [
        ((0, 0), helper_wrap::<0, 0> as *const () as usize),
        ((1, 0), helper_wrap::<1, 0> as *const () as usize),
        ((1, 1), helper_wrap::<1, 1> as *const () as usize),
        ((1, 2), helper_wrap::<1, 2> as *const () as usize),
        ((2, 0), helper_wrap::<2, 0> as *const () as usize),
        ((2, 1), helper_wrap::<2, 1> as *const () as usize),
        ((2, 2), helper_wrap::<2, 2> as *const () as usize),
        ((2, 3), helper_wrap::<2, 3> as *const () as usize),
        ((2, 4), helper_wrap::<2, 4> as *const () as usize),
        ((2, 5), helper_wrap::<2, 5> as *const () as usize),
        ((2, 6), helper_wrap::<2, 6> as *const () as usize),
        ((2, 7), helper_wrap::<2, 7> as *const () as usize),
        ((2, 8), helper_wrap::<2, 8> as *const () as usize),
        ((3, 0), helper_wrap::<3, 0> as *const () as usize),
        ((3, 1), helper_wrap::<3, 1> as *const () as usize),
        ((3, 2), helper_wrap::<3, 2> as *const () as usize),
        ((3, 3), helper_wrap::<3, 3> as *const () as usize),
        ((3, 4), helper_wrap::<3, 4> as *const () as usize),
        ((3, 5), helper_wrap::<3, 5> as *const () as usize),
        ((3, 6), helper_wrap::<3, 6> as *const () as usize),
        ((3, 7), helper_wrap::<3, 7> as *const () as usize),
        ((3, 8), helper_wrap::<3, 8> as *const () as usize),
        ((3, 9), helper_wrap::<3, 9> as *const () as usize),
        ((3, 10), helper_wrap::<3, 10> as *const () as usize),
        ((3, 11), helper_wrap::<3, 11> as *const () as usize),
        ((3, 12), helper_wrap::<3, 12> as *const () as usize),
        ((3, 13), helper_wrap::<3, 13> as *const () as usize),
        ((3, 14), helper_wrap::<3, 14> as *const () as usize),
        ((3, 15), helper_wrap::<3, 15> as *const () as usize),
        ((3, 16), helper_wrap::<3, 16> as *const () as usize),
        ((3, 17), helper_wrap::<3, 17> as *const () as usize),
        ((3, 18), helper_wrap::<3, 18> as *const () as usize),
        ((3, 19), helper_wrap::<3, 19> as *const () as usize),
        ((3, 20), helper_wrap::<3, 20> as *const () as usize),
        ((3, 21), helper_wrap::<3, 21> as *const () as usize),
        ((3, 22), helper_wrap::<3, 22> as *const () as usize),
        ((3, 23), helper_wrap::<3, 23> as *const () as usize),
        ((3, 24), helper_wrap::<3, 24> as *const () as usize),
        ((3, 25), helper_wrap::<3, 25> as *const () as usize),
        ((3, 26), helper_wrap::<3, 26> as *const () as usize),
    ];
    for ((values, kinds), function) in wrapped {
        install_helper(HelperId::coroutine_wrap(values, kinds), function);
    }
}
