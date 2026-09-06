//! The addresses of the helpers a trace calls
//! ([`oximond::jit_ir::trace::TraceOp::CallHelper`]).
//!
//! The IR names a helper by [`HelperId`] and the table in `oximond::jit_ir`
//! describes its signature. The address of the `extern "C"` function that
//! implements it lives here, in a process-wide table that the crate that owns
//! the function fills before it compiles a trace. The VM installs the ones
//! that read its heap. This crate implements the helpers only its tests call,
//! and only in test builds.
//! Nothing here is persisted: a profile stores ids and resolves them again.

use crate::jit_ir::trace::{HELPER_COUNT, HelperId};
#[cfg(test)]
use std::sync::Once;
#[cfg(test)]
use std::sync::atomic::AtomicI64;
use std::sync::atomic::{AtomicUsize, Ordering};

static ADDRESSES: [AtomicUsize; HELPER_COUNT] = [const { AtomicUsize::new(0) }; HELPER_COUNT];

/// The addresses of the specialized twins of helpers: they take the address of
/// a [`super::Prepared`] object right after the argument count.
static PREPARED_ADDRESSES: [AtomicUsize; HELPER_COUNT] =
    [const { AtomicUsize::new(0) }; HELPER_COUNT];

/// The addresses of the test helpers, which have ids of their own.
#[cfg(test)]
static TEST_ADDRESSES: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];

/// Where the address of `id` is kept.
fn slot(id: HelperId) -> Option<&'static AtomicUsize> {
    if let Some(slot) = ADDRESSES.get(usize::from(id.0)) {
        return Some(slot);
    }
    #[cfg(test)]
    if let Some(index) = id.0.checked_sub(HelperId::TEST_BASE) {
        return TEST_ADDRESSES.get(usize::from(index));
    }
    None
}

/// Installs the function that implements `id`. `address` is the address of an
/// `extern "C"` function with the calling convention in
/// [`oximond::jit_ir::trace`]'s helper module documentation: `([context: usize,]
/// out: *mut i64, args...) -> i64`, where the context is present exactly when
/// the helper's table entry says so. A later install replaces the earlier one.
///
/// # Panics
///
/// Panics if `id` is not in the helper table.
pub fn install_helper(id: HelperId, address: usize) {
    slot(id)
        .expect("an installed helper is in the table")
        .store(address, Ordering::Release);
}

/// The address installed for `id`, or `None` when no crate has installed one
/// (the backend then refuses to compile a trace that calls it).
#[must_use]
pub fn helper_address(id: HelperId) -> Option<usize> {
    #[cfg(test)]
    install_test_helpers();
    match slot(id)?.load(Ordering::Acquire) {
        0 => None,
        address => Some(address),
    }
}

/// Installs the specialized twin of the helper `id`. Its signature is the
/// helper's with one more `usize`, the prepared object's address, after the
/// argument count. A later install replaces the earlier one.
///
/// # Panics
///
/// Panics if `id` is not in the helper table.
pub fn install_prepared_helper(id: HelperId, address: usize) {
    PREPARED_ADDRESSES
        .get(usize::from(id.0))
        .expect("an installed helper is in the table")
        .store(address, Ordering::Release);
}

/// The address of the specialized twin of `id`, if one is installed.
#[must_use]
pub fn prepared_helper_address(id: HelperId) -> Option<usize> {
    match PREPARED_ADDRESSES
        .get(usize::from(id.0))?
        .load(Ordering::Acquire)
    {
        0 => None,
        address => Some(address),
    }
}

/// Installs the test helpers, once.
#[cfg(test)]
fn install_test_helpers() {
    static TESTS: Once = Once::new();
    TESTS.call_once(|| {
        let divmod: extern "C" fn(*mut i64, i64, i64) -> i64 = test_divmod;
        let write: extern "C" fn(*mut i64, i64) -> i64 = test_write;
        let read: extern "C" fn(*mut i64, i64) -> i64 = test_read;
        let draw: extern "C" fn(*mut i64, i64) -> i64 = test_draw;
        let resize: extern "C" fn(*mut i64, usize, i64) -> i64 = test_resize;
        for (id, address) in [
            (HelperId::TEST_DIVMOD, divmod as *const () as usize),
            (HelperId::TEST_WRITE, write as *const () as usize),
            (HelperId::TEST_READ, read as *const () as usize),
            (HelperId::TEST_DRAW, draw as *const () as usize),
            (HelperId::TEST_RESIZE, resize as *const () as usize),
        ] {
            install_helper(id, address);
        }
    });
}

/// The word [`test_write`] stores to, which tests read back.
#[cfg(test)]
pub(crate) static TEST_WRITTEN: AtomicI64 = AtomicI64::new(0);

/// `test.divmod`: the floored quotient and remainder of two integers, as two
/// results. Misses on a zero divisor and on the one overflowing quotient.
#[allow(unsafe_code)]
#[cfg(test)]
extern "C" fn test_divmod(out: *mut i64, left: i64, right: i64) -> i64 {
    if right == 0 || (left == i64::MIN && right == -1) {
        return 1;
    }
    let quotient = left / right;
    let remainder = left % right;
    let (quotient, remainder) = if remainder != 0 && ((remainder < 0) != (right < 0)) {
        (quotient - 1, remainder + right)
    } else {
        (quotient, remainder)
    };
    // SAFETY: `out` is the control block's `helper_out`, which holds
    // `MAX_HELPER_RESULTS` words and which nothing else touches during the
    // call; the table entry says this helper gives two results.
    unsafe {
        *out = quotient;
        *out.add(1) = remainder;
    }
    0
}

/// `test.write`: stores its argument in [`TEST_WRITTEN`] and gives it back.
#[allow(unsafe_code)]
#[cfg(test)]
extern "C" fn test_write(out: *mut i64, value: i64) -> i64 {
    TEST_WRITTEN.store(value, Ordering::SeqCst);
    // SAFETY: as for `test_divmod`.
    unsafe { *out = value };
    0
}

/// `test.read`: gives its argument back.
#[allow(unsafe_code)]
#[cfg(test)]
extern "C" fn test_read(out: *mut i64, value: i64) -> i64 {
    // SAFETY: as for `test_divmod`.
    unsafe { *out = value };
    0
}

/// `test.draw`: gives its argument plus one.
#[allow(unsafe_code)]
#[cfg(test)]
extern "C" fn test_draw(out: *mut i64, value: i64) -> i64 {
    // SAFETY: as for `test_divmod`.
    unsafe { *out = value.wrapping_add(1) };
    0
}

/// The words of a test table [`test_resize`] works on: the array words sit
/// at the offsets `NativeTableLayout` names, and the last two words hold a
/// spare array pointer and length.
#[cfg(test)]
pub(crate) const TEST_TABLE_WORDS: usize = 64;

/// `test.resize`: swaps the array pointer and length of the test table at
/// `address` with the spare pair in its last two words, and gives 0. Like
/// every helper that rewrites arrays it takes the run's binding slots after
/// `out`; the test table has no binding, so it leaves them alone.
#[allow(unsafe_code)]
#[cfg(test)]
extern "C" fn test_resize(out: *mut i64, _bindings: usize, address: i64) -> i64 {
    let layout = crate::runtime::NativeTableLayout::get().expect("a known vector layout");
    let words = address as usize as *mut usize;
    let pointer = usize::try_from(layout.array_pointer).unwrap() / size_of::<usize>();
    let length = usize::try_from(layout.array_length).unwrap() / size_of::<usize>();
    // SAFETY: the test hands the address of a buffer of `TEST_TABLE_WORDS`
    // words that it owns and that nothing else touches during the call, and
    // `out` is as for `test_divmod`.
    unsafe {
        std::ptr::swap(words.add(pointer), words.add(TEST_TABLE_WORDS - 2));
        std::ptr::swap(words.add(length), words.add(TEST_TABLE_WORDS - 1));
        *out = 0;
    }
    0
}
