//! The string-index probe: a `HashLoad` of a string key and the in-place
//! stores `SET_PRESENT`, `SET_REVIVE` and `DELETE_STRING`, answered from the
//! index the run holds as Cranelift answers them (`hash_probe`, `hash_store`
//! and `probe_bucket` in `emit.rs`), with the same conditions and the same
//! exits.
//!
//! The probe sequence is walked in line with nested skips (`Op::If`, see "The
//! forward skip" in `docs/impl/native-backend/mir/backend.md`). The first
//! bucket is read in a straight line. When it holds the very string the key
//! is (the same allocation, which is how a key and the table's own key meet
//! when they come from one place), that is the entry and the skip over the
//! rest is the only branch taken. The straight line is short: the start of
//! the sequence is the high half of the string's cached hash, which is mixed
//! when it is computed (`finish_hash` in `string/mod.rs`), so the code loads
//! those 32 bits and masks them with the index's mask shifted by the size of
//! a bucket, which gives the bucket's offset with no multiply. It compares
//! the bucket's owner word (what a value that holds the key string has in
//! its payload) with the key's payload as loaded, and loads the address of
//! the bucket's value before the compare, so the skipped path has it at
//! once. That is seven instructions from the key's value to the branch.
//!
//! Otherwise the block of the skip runs. Its walk (`step`) looks at up to
//! [`WALK_STEPS`] buckets and ends at the first whose hash word is the key's
//! hash or zero, by skipping the rest. A step that ends nothing costs two
//! compares and the address of the next bucket. `settle` then reads the key
//! of the bucket the walk ended at. No key: an empty bucket, so the key is
//! not in the index. The key's own string: the entry. Any other string is a
//! candidate, or belongs to a static bucket that stands for "cannot tell" (a
//! binding with no index, a key without a hash, a walk that ran out of
//! steps): the words of its key are compared with the key's, once. Equal
//! inline strings in two allocations are the same key. Anything else (a long
//! key, which needs a byte comparison, and two keys with one hash) goes to a
//! conditional call that walks the whole sequence in native code and asks
//! the VM when even that cannot tell, which is the only time the VM is
//! called, as in Cranelift's code.
//!
//! The probe ends with the address of the entry's value. For a missing key
//! that is a static nil (lookup) or a tag no store takes (store). A binding
//! with no index is given the address of two buckets that hold no hash, so
//! nothing here reads address zero.
//!
//! A key that is an inline string equal to the table's in another allocation
//! is found by that comparison of words at every lookup, about thirty
//! instructions where the table's own string costs seven. The first time, a
//! conditional call gives the array slot the key was read from the table's
//! string ([`trace_adopt_key`]), so the next lookup of that slot ends at the
//! first compare.
//!
//! A side recorded from the type guard of a lookup starts by making the same
//! lookup. When nothing can have changed in between, the guard passes the
//! address its probe found and the side does not probe (`takes_probe`).
//!
//! A store is a skip that runs when the table takes it in line, and the VM's
//! helper a skip that runs when it does not, so a store that is not made
//! writes nothing. The guard on the helper's status is in that skip.
//!
//! It is on by default and `OXIMOND_JIT_MIR_PROBE=0` turns it off
//! (`mir_probes`); without it MIR declines a tree that probes and Cranelift
//! compiles it. The in-line walk made the probe match Cranelift on most of
//! the hash benchmarks, and with a side that rejoins going straight to the
//! loop header `table/hash_read_indexed` is below it too (see "String-index
//! probe" in `docs/impl/native-backend/mir/backend.md`).

#[allow(
    clippy::wildcard_imports,
    reason = "a child module of the lowering, which shares its names"
)]
use super::*;
use crate::jit::trace::helpers::helper_address;
use crate::runtime::{
    NATIVE_PROBE_SLOT_HASH, NATIVE_PROBE_SLOT_KEY, NATIVE_PROBE_SLOT_OWNER, NATIVE_PROBE_SLOT_SIZE,
    NATIVE_PROBE_SLOT_VALUE, NATIVE_STRING_HASH_OFFSET, NATIVE_STRING_INLINE_TAG,
    NATIVE_STRING_INLINE_WORDS, NATIVE_STRING_PROBE_START_OFFSET, NATIVE_STRING_STORAGE_OFFSET,
    native_probe_start,
};

use super::super::super::emit::trace_hash_get;
use super::super::super::exit::{PROBE_NO_TOMBSTONES, TRACE_TAG_BORROWED_STRING, TRACE_TAG_NIL};
use crate::jit_ir::trace::{HelperEffects, HelperId};
use crate::runtime::Value;
use crate::string::LuaString;

// The VM answers a lookup as an entry tag and its bits, which the scalar types
// the probe reads share with a value's tag byte and payload word. Any other
// answer has a tag that is none of the four.
const _: () = {
    assert!(TRACE_TAG_NIL == NATIVE_VALUE_TAG_NIL as i64);
    assert!(crate::jit::trace::exit::TRACE_TAG_BOOLEAN == NATIVE_VALUE_TAG_BOOLEAN as i64);
    assert!(crate::jit::trace::exit::TRACE_TAG_INTEGER == NATIVE_VALUE_TAG_INTEGER as i64);
    assert!(crate::jit::trace::exit::TRACE_TAG_FLOAT == NATIVE_VALUE_TAG_NUMBER as i64);
    assert!(crate::jit::trace::exit::TRACE_TAG_OTHER > NATIVE_VALUE_TAG_NUMBER as i64);
    assert!(mem::offset_of!(TraceInputSlot, tag) == 0);
    assert!(mem::offset_of!(TraceInputSlot, bits) == NATIVE_VALUE_PAYLOAD_OFFSET as usize);
    // The statics stand for buckets of the index, so they must have its
    // layout.
    // The tests of the mask's flags shift by these.
    assert!(PROBE_NO_TOMBSTONES == 1 << 62);
    assert!(mem::size_of::<Bucket>() == NATIVE_PROBE_SLOT_SIZE);
    assert!(mem::offset_of!(Bucket, hash) == NATIVE_PROBE_SLOT_HASH);
    assert!(mem::offset_of!(Bucket, key) == NATIVE_PROBE_SLOT_KEY);
    assert!(mem::offset_of!(Bucket, value) == NATIVE_PROBE_SLOT_VALUE);
    assert!(mem::offset_of!(Bucket, owner) == NATIVE_PROBE_SLOT_OWNER);
};

/// A bucket of the index, as the statics below hold them.
#[repr(C)]
struct Bucket {
    hash: u64,
    key: *const u64,
    value: *const u64,
    /// What a string value that holds the key has in its payload: the
    /// address the key's storage follows. Null with no key.
    owner: *const u8,
}

/// The owner word of a static bucket whose key is [`NO_STRING`]: no string
/// value's payload, so the first compare of a probe fails, and the words that
/// follow it as a storage would are those no string has.
const NO_OWNER: *const u8 = (&raw const NO_STRING)
    .cast::<u8>()
    .wrapping_sub(NATIVE_STRING_STORAGE_OFFSET);

// SAFETY: read-only statics that nothing writes.
#[allow(unsafe_code)]
unsafe impl Sync for Bucket {}

/// A value that is nil (a zero tag byte).
static NIL_VALUE: [u64; 2] = [0, 0];

/// The bucket the call of a store answers with when the key is not in the
/// index: no key, and a value no store takes.
static MISS_BUCKET: Bucket = Bucket {
    hash: 0,
    key: std::ptr::null(),
    value: (&raw const MISS_VALUE).cast::<u64>(),
    owner: std::ptr::null(),
};

/// What a store finds for a key the index lacks: a tag that is no value's,
/// so no store takes it for an entry (a nil one is a deleted key, which a
/// revive takes).
static MISS_VALUE: [u64; 2] = [0xfe, 0];

/// The bucket a walk ends in when it cannot tell (a key without a hash, a
/// binding without an index, a sequence longer than the walk): its key is
/// the words no string storage starts with, so the comparison that settles a
/// candidate fails and the native walk answers.
static GIVE_UP_BUCKET: Bucket = Bucket {
    hash: 0,
    key: (&raw const NO_STRING).cast::<u64>(),
    value: std::ptr::null(),
    owner: NO_OWNER,
};

/// Three words that are not the start of a string's storage: its first byte
/// says whether the string is inline, and is never this.
static NO_STRING: [u64; NATIVE_STRING_INLINE_WORDS] = [u64::MAX; NATIVE_STRING_INLINE_WORDS];

/// What a binding with no index holds in place of its slots: two buckets that
/// hold no string (the second stands for the header slot after the index) and
/// a mask of zero. The probe reads a bucket in line. It has no hash, so the
/// walk ends there, and its key is the words no string has, so the comparison
/// fails and the conditional call answers, which knows the table by its
/// address.
static NO_INDEX: [Bucket; 2] = [
    Bucket {
        hash: 0,
        key: (&raw const NO_STRING).cast::<u64>(),
        value: std::ptr::null(),
        owner: NO_OWNER,
    },
    Bucket {
        hash: 0,
        key: (&raw const NO_STRING).cast::<u64>(),
        value: std::ptr::null(),
        owner: NO_OWNER,
    },
];

/// The address of the slots a binding without an index is given.
pub(super) fn no_index() -> i64 {
    NO_INDEX.as_ptr() as usize as i64
}

/// The rest of a probe sequence, for a first bucket that did not hold the
/// key's own string (the loop `probe_bucket` of `emit.rs` emits). `key` is the
/// address of the array slot that holds the key. Returns the address of the
/// bucket that holds the key, or of the empty bucket that ends the sequence
/// (its key word is zero), and zero when native code cannot tell: the binding
/// has no index, the key's hash is not computed, a candidate with the key's
/// hash is another allocation and the key is a long string that needs a byte
/// comparison.
#[allow(unsafe_code)]
fn locate(slots: u64, mask: u64, key: u64) -> u64 {
    let read = |address: u64| -> u64 {
        // SAFETY: the addresses are inside the index the run holds (a bucket
        // of `slots`, which has `mask + 1` of them), or in the storage of the
        // key and of a stored key, which the index keeps alive. The key is a
        // string in an array slot. The run moves no hash storage (see
        // `native_probe`).
        unsafe { std::ptr::read(address as *const u64) }
    };
    let data = read(key + NATIVE_VALUE_PAYLOAD_OFFSET as u64) + NATIVE_STRING_STORAGE_OFFSET as u64;
    let hash = read(data + NATIVE_STRING_HASH_OFFSET as u64);
    if slots == no_index().cast_unsigned() || hash == 0 {
        return 0;
    }
    // The flag bits above the index size are not part of the mask.
    let mask = mask & (PROBE_NO_TOMBSTONES.cast_unsigned() - 1);
    // SAFETY: as above.
    let inline = unsafe { std::ptr::read(data as *const u8) == NATIVE_STRING_INLINE_TAG };
    let bucket_size = NATIVE_PROBE_SLOT_SIZE as u64;
    let mut at = native_probe_start(hash) & mask;
    for _ in 0..=mask {
        let bucket = slots + at * bucket_size;
        let stored = read(bucket + NATIVE_PROBE_SLOT_KEY as u64);
        if stored == 0 {
            return bucket;
        }
        if read(bucket + NATIVE_PROBE_SLOT_HASH as u64) == hash {
            if stored == data {
                return bucket;
            }
            if !inline {
                return 0;
            }
            let words = NATIVE_STRING_INLINE_WORDS as u64;
            if (0..words).all(|word| read(data + word * 8) == read(stored + word * 8)) {
                return bucket;
            }
        }
        at = (at + 1) & mask;
    }
    0
}

/// The conditional call of a store: the bucket of the key, or the miss bucket
/// when the key is not in the index, or the index has none or cannot tell (the
/// VM's helper stores).
#[allow(unsafe_code)]
extern "C" fn trace_probe_bucket(slots: u64, mask: u64, key: u64) -> u64 {
    let miss = &raw const MISS_BUCKET as usize as u64;
    let bucket = locate(slots, mask, key);
    if bucket == 0 {
        return miss;
    }
    // SAFETY: a bucket `locate` found is one of the index's. The empty one
    // that ends a sequence has no value to read: the miss bucket stands in.
    let stored = unsafe { std::ptr::read((bucket + NATIVE_PROBE_SLOT_KEY as u64) as *const u64) };
    if stored == 0 { miss } else { bucket }
}

/// The conditional call of a lookup: the address of the value the key has, a
/// nil for a key the index lacks, and the VM's answer (written where a value
/// is, in the control block's `hash_result`) when native code cannot tell.
#[allow(unsafe_code)]
extern "C" fn trace_probe_value(
    control: *mut TraceControl,
    slots: u64,
    mask: u64,
    binding: u64,
    key: u64,
) -> u64 {
    let bucket = locate(slots, mask, key);
    if bucket != 0 {
        // SAFETY: a bucket `locate` found is one of the index's.
        let (stored, value) = unsafe {
            (
                std::ptr::read((bucket + NATIVE_PROBE_SLOT_KEY as u64) as *const u64),
                std::ptr::read((bucket + NATIVE_PROBE_SLOT_VALUE as u64) as *const u64),
            )
        };
        return if stored == 0 {
            &raw const NIL_VALUE as usize as u64
        } else {
            value
        };
    }
    // SAFETY: `control` is the run's control block, which nothing else
    // touches during the call; the dispatcher it names is live for the run
    // (see `trace_hash_get`).
    unsafe {
        let control = &mut *control;
        trace_hash_get(
            control.tables as *mut _,
            &raw mut control.hash_result,
            i64::try_from(binding).unwrap_or(-1),
            TRACE_TAG_BORROWED_STRING,
            key.cast_signed(),
        );
        &raw const control.hash_result as usize as u64
    }
}

/// How many array slots this thread gave the key string of a table (see
/// [`trace_adopt_key`]), for tests.
#[cfg(test)]
pub(in super::super::super) fn adopted_keys() -> u64 {
    ADOPTION.with(|state| state.adopted.get())
}

/// What [`trace_adopt_key`] remembers between calls on one thread.
struct Adoption {
    /// The strings the last adoptions put into slots, by owner word, each at
    /// the place its address hashes to.
    recent: [std::cell::Cell<usize>; ADOPT_RECENT],
    /// Adoptions that replaced a string an adoption had put there.
    strikes: std::cell::Cell<u32>,
    /// Adoptions that did not, since the last strike was forgiven.
    calm: std::cell::Cell<u32>,
    /// Runs that still adopt nothing.
    rest: std::cell::Cell<u32>,
    /// The strings the last [`ADOPT_RETIRED`] adoptions took out of slots,
    /// and where the next one goes. Code that loaded the word of such a
    /// string before the swap may still read the string (see `key_hash`).
    retired: std::cell::RefCell<[Option<LuaString>; ADOPT_RETIRED]>,
    next: std::cell::Cell<usize>,
    #[cfg(test)]
    adopted: std::cell::Cell<u64>,
}

/// How many adopted strings a thread remembers.
const ADOPT_RECENT: usize = 64;
/// The strikes after which a thread stops adopting.
const ADOPT_STRIKES: u32 = 16;
/// The adoptions without a strike that forgive one.
const ADOPT_CALM: u32 = 8;
/// For how many runs that ask it stops.
const ADOPT_REST: u32 = 256;
/// How many strings taken out of slots stay alive.
const ADOPT_RETIRED: usize = 8;

thread_local! {
    static ADOPTION: Adoption = const {
        Adoption {
            recent: [const { std::cell::Cell::new(0) }; ADOPT_RECENT],
            strikes: std::cell::Cell::new(0),
            calm: std::cell::Cell::new(0),
            rest: std::cell::Cell::new(0),
            retired: std::cell::RefCell::new([const { None }; ADOPT_RETIRED]),
            next: std::cell::Cell::new(0),
            #[cfg(test)]
            adopted: std::cell::Cell::new(0),
        }
    };
}

// A string handle is one word, the one a string value has in its payload
// (`NATIVE_STRING_STORAGE_OFFSET` before the storage `identity` names): the
// adoption below makes a handle from a bucket's owner word.
const _: () = assert!(mem::size_of::<LuaString>() == mem::size_of::<usize>());

/// Gives the array slot at `key` the key string of the table whose bucket
/// `bucket` is, when the slot holds an equal string in another allocation.
///
/// A probe compiled in line knows a key by its address: the bucket's owner
/// word against the word in the key's slot, seven instructions. A slot that
/// holds the same bytes in another allocation pays the walk and a comparison
/// of the strings' words at every lookup instead, about thirty. The two
/// strings are equal and immutable, and nothing in Lua tells one allocation
/// from the other, so the first such lookup of a slot swaps the slot's string
/// for the table's, and every later lookup of that slot in that table takes
/// the short path. The probe calls this once it has compared the words and
/// found them equal, which is only true of two inline strings.
///
/// The slot is one of an array part a table bound for the run holds, which
/// the code borrowed the key from (`TraceType::StringRef`); the borrow names
/// the slot, not the string, so it reads the new string from then on. Code
/// that loaded the slot's word before the call (the probe that called, and
/// the probes of the same key that follow it closely, see `key_hash`) may
/// read the old string through it, so the old string is not dropped here: it
/// stays alive until [`ADOPT_RETIRED`] more strings were taken out of slots.
///
/// One slot looked up in two tables that hold equal keys in two allocations
/// would be swapped at every lookup. An adoption that replaces a string an
/// adoption put there is a strike; after [`ADOPT_STRIKES`] of them the run
/// stops asking (`TraceControl::adopt_off`), and so do the next
/// [`ADOPT_REST`] runs of the thread at their first call. Such a loop then
/// costs what it did before, plus the test of that word.
#[allow(unsafe_code)]
extern "C" fn trace_adopt_key(control: *mut TraceControl, key: u64, bucket: u64) {
    // A thread that is ending has no state left: nothing is adopted.
    let _ = ADOPTION.try_with(|state| {
        // SAFETY: `control` is the run's control block, which nothing else
        // touches during the call.
        let control = unsafe { &mut *control };
        if state.rest.get() != 0 {
            state.rest.set(state.rest.get() - 1);
            control.adopt_off = 1;
            return;
        }
        // SAFETY: `bucket` is a bucket of the index the run holds, whose key
        // the probe just read; `key` is the address of the array slot the
        // code borrowed the key from, which no reference names during a run.
        let (owner, slot) = unsafe {
            (
                std::ptr::read((bucket + NATIVE_PROBE_SLOT_OWNER as u64) as *const usize),
                &mut *(key as usize as *mut Value),
            )
        };
        let Value::String(held) = slot else {
            return;
        };
        let old = held.identity() - NATIVE_STRING_STORAGE_OFFSET;
        if owner == 0 || owner == old {
            return;
        }
        let place = |owner: usize| &state.recent[(owner >> 4) % ADOPT_RECENT];
        if place(old).get() == old {
            state.strikes.set(state.strikes.get() + 1);
            if state.strikes.get() >= ADOPT_STRIKES {
                state.strikes.set(0);
                state.rest.set(ADOPT_REST);
                control.adopt_off = 1;
                return;
            }
        } else {
            state.calm.set(state.calm.get() + 1);
            if state.calm.get() >= ADOPT_CALM {
                state.calm.set(0);
                state.strikes.set(state.strikes.get().saturating_sub(1));
            }
        }
        // SAFETY: the owner word of a full bucket is the word a string value
        // that holds the table's key has in its payload, which is what a
        // handle is (the assertion above, and the layout tests of the
        // runtime). The table holds the key for the run, so the handle read
        // here is live; it is not dropped, only cloned.
        let theirs = unsafe {
            mem::ManuallyDrop::new(std::ptr::read((&raw const owner).cast::<LuaString>()))
        };
        if *held != *theirs {
            return;
        }
        let old = mem::replace(held, (*theirs).clone());
        let at = state.next.get();
        state.next.set((at + 1) % ADOPT_RETIRED);
        // The string that leaves the ring is dropped after the borrow ends.
        let _left = state.retired.borrow_mut()[at].replace(old);
        place(owner).set(owner);
        #[cfg(test)]
        state.adopted.set(state.adopted.get() + 1);
    });
}

/// Where a probe ended: `entry` is the address of the value the key has (for
/// a lookup that finds none, of a nil; for a store, of the miss bucket's
/// value). `payload` is the word of the key's value that names its string
/// when the probe is over (it may have swapped the string in the key's slot
/// for the table's own, see [`trace_adopt_key`]), `slots` and `mask` the
/// index's.
struct Probe {
    entry: Val,
    payload: Val,
    slots: Val,
    mask: Val,
}

/// What a probe is for.
#[derive(Clone, Copy, PartialEq)]
enum Answer {
    Value,
    Bucket,
}

/// How many buckets of the probe sequence the code walks in line. The native
/// walk (`locate`) takes over after them, so the sequence is still walked to
/// its end.
const WALK_STEPS: u32 = 8;

/// What every step of the walk reads.
#[derive(Clone, Copy)]
struct Walk {
    slots: Val,
    mask: Val,
    /// The mask as it masks the offset of a bucket.
    scaled: Val,
    /// The word of the key's value that names its string. Its storage
    /// follows at [`NATIVE_STRING_STORAGE_OFFSET`].
    payload: Val,
    hash: Val,
    zero: Val,
    /// The bucket that stands for "the walk cannot tell" (see
    /// [`GIVE_UP_BUCKET`]).
    give_up: Val,
}

impl<'a> Lowerer<'a, '_> {
    /// Whether this backend probes the string index of the tree as Cranelift
    /// does: Cranelift probes it (`native_probe`) and `OXIMOND_JIT_MIR_PROBE`
    /// is not `0` or empty. With the probe off such a tree's lookups and
    /// in-place stores are declined, and the tree goes to Cranelift.
    pub(super) fn mir_probes(&self) -> bool {
        self.native_probe() && crate::jit::debug_env::mir_probe()
    }

    /// Whether the lookup of `key` expecting `expected` is one Cranelift
    /// probes in line (`hash_load` in `emit.rs`).
    pub(super) fn cranelift_probes_load(&self, key: ValueId, expected: TraceType) -> bool {
        self.ty(key) == TraceType::StringRef
            && matches!(
                expected,
                TraceType::Float | TraceType::Integer | TraceType::Boolean | TraceType::Nil
            )
            && self.native_probe()
    }

    /// Whether this backend probes the lookup of `key` in line (`probed_load`).
    pub(super) fn probes_load(&self, key: ValueId, expected: TraceType) -> bool {
        self.cranelift_probes_load(key, expected) && crate::jit::debug_env::mir_probe()
    }

    /// The binding whose string index a call of one of the in-place store
    /// helpers stores through in Cranelift's code (`native_store_binding` in
    /// `emit.rs`), if the call has one. A forced guard never reaches this
    /// backend.
    pub(super) fn native_store_binding(&self, helper: HelperId, args: &[ValueId]) -> Option<u32> {
        let stores_value = match helper {
            HelperId::SET_PRESENT | HelperId::SET_REVIVE => true,
            HelperId::DELETE_STRING => false,
            _ => return None,
        };
        if !self.mir_probes()
            || self.member != 0
            || args.len() != 2 + usize::from(stores_value)
            || self.ir.value_type(args[1]) != Some(TraceType::StringRef)
            || (stores_value
                && !matches!(
                    self.ir.value_type(args[2]),
                    Some(TraceType::Float | TraceType::Integer | TraceType::Boolean)
                ))
        {
            return None;
        }
        let register = self
            .ir
            .entry
            .inputs
            .iter()
            .find(|input| input.value == args[0])?
            .register;
        let found = self.members[0].ir.bindings.iter().position(|binding| {
            binding.register == register
                && match binding.kind {
                    BindingKind::NoMetatable => true,
                    BindingKind::Hash => helper != HelperId::SET_REVIVE,
                    _ => false,
                }
        })?;
        u32::try_from(found).ok()
    }

    /// Whether the side `side` starts by repeating the lookups its parent
    /// made before the guard that enters it, so that the guard passes the
    /// address of the value its probe found and the side reads that instead
    /// of probing again. Returns the place in the side's body of the lookup
    /// that takes the address.
    ///
    /// A side recorded from the type guard of a `HashLoad` begins by running
    /// the guest instruction of that read again, from the state the guard's
    /// snapshot restores. Its first operations are therefore the lookups the
    /// parent made for the instruction: the one whose guard failed, which
    /// now expects the type the read had (a nil for a key the table lacks),
    /// and before it those that passed (the miss in a proxy table, when the
    /// read went on to an `__index` function that looks the key up in
    /// another). Nothing runs between the parent's probes and the side's
    /// first operations, and a binding names one table for the whole run,
    /// so each of the side's lookups would end where the parent's did:
    ///
    /// - a lookup that passed in the parent and expected a nil is a nil
    ///   again, and the side makes no code for it;
    /// - the lookup whose guard failed has the parent's answer: the value in
    ///   the bucket, a static nil, or the VM's answer in the control block,
    ///   which the next lookup of the VM is the first to overwrite. The side
    ///   still checks the type it expects.
    ///
    /// It holds for a side that runs once per entry (no loop of its own, so
    /// no rejoin runs its first operations again), attached to the guard of
    /// a probed lookup. The lookups pair up in order, each with the table
    /// register of its parent's and with the entry input that the guard's
    /// snapshot gives the parent's key. Metatable guards may stand between
    /// them on either side: they read no entry.
    pub(super) fn takes_probe(&self, side: usize) -> Option<usize> {
        let scalar = |ty: TraceType| {
            matches!(
                ty,
                TraceType::Float | TraceType::Integer | TraceType::Boolean | TraceType::Nil
            )
        };
        if self.function.is_some() || !self.mir_probes() {
            return None;
        }
        let parent = self.members.get(*self.s.parent.get(side)?)?;
        let &(guard, _) = parent.attached.iter().find(|(_, held)| *held == side)?;
        let parent_ir = parent.ir;
        let side_ir = self.members[side].ir;
        if !side_ir.preheader.is_empty()
            || !matches!(
                side_ir.terminator,
                TraceTerminator::Link { .. } | TraceTerminator::Exit { .. }
            )
        {
            return None;
        }
        // The table a lookup reads: the register of its binding.
        let bindings = &self.members[0].ir.bindings;
        let table = |binding: u32| {
            let binding = bindings.get(binding as usize)?;
            matches!(binding.kind, BindingKind::Hash | BindingKind::NoMetatable)
                .then_some(binding.register)
        };
        // The parent's lookups for the instruction, the failed one last.
        let entered = parent_ir.guards.get(guard)?.snapshot;
        let snapshot = parent_ir.snapshots.get(entered.index())?;
        let mut made: [Option<(u16, ValueId, TraceType)>; 4] = [None; 4];
        let mut count = 0;
        let ops = [&parent_ir.preheader, &parent_ir.body];
        let (list, failed) = ops.iter().find_map(|list| {
            let at = list.iter().position(|instruction| {
                matches!(&instruction.op, TraceOp::HashLoad { guard: held, .. } if held.index() == guard)
            })?;
            Some((*list, at))
        })?;
        for instruction in list[..=failed].iter().rev() {
            match &instruction.op {
                TraceOp::HashLoad {
                    binding,
                    key,
                    expected,
                    guard: held,
                    ..
                } => {
                    // The same state to resume from: the snapshot of the
                    // failed guard, or one that restores the same.
                    let same_state = parent_ir
                        .guards
                        .get(held.index())
                        .and_then(|held| parent_ir.snapshots.get(held.snapshot.index()))
                        .is_some_and(|held| {
                            held.phase == snapshot.phase
                                && held.effect_position == snapshot.effect_position
                                && held.counted == snapshot.counted
                                && held.frames == snapshot.frames
                                && held.virtuals == snapshot.virtuals
                                && held.slices == snapshot.slices
                                && held.numbers == snapshot.numbers
                        });
                    if !same_state
                        || count == made.len()
                        || parent_ir.value_type(*key) != Some(TraceType::StringRef)
                        || !scalar(*expected)
                    {
                        break;
                    }
                    made[count] = Some((table(*binding)?, *key, *expected));
                    count += 1;
                }
                TraceOp::MetaGuard { .. } => {}
                _ => break,
            }
        }
        // The side's, in the order it makes them (`made` is the other way
        // round).
        let mut left = count;
        for (at, instruction) in side_ir.body.iter().enumerate() {
            match &instruction.op {
                TraceOp::HashLoad {
                    binding,
                    key,
                    expected,
                    ..
                } => {
                    left = left.checked_sub(1)?;
                    let (register, parent_key, parent_expected) = made[left]?;
                    if table(*binding)? != register
                        || side_ir.value_type(*key) != Some(TraceType::StringRef)
                        || !scalar(*expected)
                    {
                        return None;
                    }
                    // A lookup that passed is not made again: it was a nil.
                    if left != 0
                        && (*expected != TraceType::Nil || parent_expected != TraceType::Nil)
                    {
                        return None;
                    }
                    let input =
                        side_ir.entry.inputs.iter().find(|input| {
                            input.value == *key && input.ty == TraceType::StringRef
                        })?;
                    let register = usize::from(input.register);
                    let same_key = match snapshot.frames.first()?.registers.get(register)? {
                        RegisterEntry::Value(value) => *value == parent_key,
                        RegisterEntry::Unchanged => parent_ir.entry.inputs.iter().any(|held| {
                            held.register == input.register && held.value == parent_key
                        }),
                        _ => false,
                    };
                    if !same_key {
                        return None;
                    }
                    if left == 0 {
                        return Some(at);
                    }
                }
                TraceOp::MetaGuard { .. } => {}
                _ => return None,
            }
        }
        None
    }

    /// The offset of the bucket after the one at offset `at`, and its
    /// address.
    fn next_bucket(&mut self, walk: &Walk, at: Val) -> (Val, Val) {
        let size = self.iconst(i64::try_from(NATIVE_PROBE_SLOT_SIZE).expect("small"));
        let next = self.f.bin(Op::Add, at, size);
        let wrapped = self.f.bin(Op::And, next, walk.scaled);
        (wrapped, self.f.bin(Op::Add, walk.slots, wrapped))
    }

    fn bucket_word(&mut self, bucket: Val, at: usize) -> Val {
        self.load_at(
            Ty::I64,
            bucket,
            Off::K(0),
            i64::try_from(at).expect("small"),
        )
    }

    /// The word that names the string in the array slot `key_slot`, the
    /// value of the borrowed key `key` (the address its storage follows),
    /// and the high half of its hash, which the scaled mask of an index
    /// turns into the offset of the key's first bucket (see
    /// `NATIVE_PROBE_SHIFT`): one 32-bit load. A second probe with the same
    /// key in the same straight-line code (a proxy's miss and then the table
    /// behind it) reads them from the first: the slot holds the same string
    /// for as long as the borrow lives.
    ///
    /// The third answer says the words came from an earlier probe. That
    /// probe may have read the hash before the VM computed it (a string has
    /// none until its first lookup), so a later one must not walk from it.
    ///
    /// A probe may swap the string in the slot for the table's own
    /// ([`trace_adopt_key`]), and the word read before then names a string
    /// the slot no longer holds. That string is equal to the slot's, so a
    /// probe that walks with it finds the same entry, and the adoption keeps
    /// the strings it took out of slots alive until [`ADOPT_RETIRED`] more
    /// were taken. Each probe takes at most one, so words are reused only by
    /// the probes that follow closely enough, and read again after that.
    fn key_hash(&mut self, key: ValueId, key_slot: Val) -> (Val, Val, bool) {
        let index = key.index() as u32;
        self.s.probes += 1;
        let now = self.s.probes;
        // From the probe that read the words to this one, both counted.
        let near = |made: u32| now - made < ADOPT_RETIRED as u32;
        if let Some(&(_, payload, high, _)) = self
            .s
            .keys
            .iter()
            .find(|(held, .., made)| *held == index && near(*made))
        {
            return (payload, high, true);
        }
        self.s.keys.retain(|(held, ..)| *held != index);
        let payload = self.load_at(
            Ty::I64,
            key_slot,
            Off::K(0),
            i64::from(NATIVE_VALUE_PAYLOAD_OFFSET),
        );
        let storage = i64::try_from(NATIVE_STRING_STORAGE_OFFSET).expect("small");
        let high = self.load32(
            payload,
            storage + i64::try_from(NATIVE_STRING_PROBE_START_OFFSET).expect("small"),
        );
        self.s.keys.push((index, payload, high, now));
        (payload, high, false)
    }

    /// The storage of the string a value names with the word `payload`.
    fn storage_of(&mut self, payload: Val) -> Val {
        let storage = self.iconst(i64::try_from(NATIVE_STRING_STORAGE_OFFSET).expect("small"));
        self.f.bin(Op::Add, payload, storage)
    }

    /// The probe of the string index bound as `binding` for the key `key`
    /// (see the module comment). `binding` for a lookup is the index of the
    /// binding the VM's lookup takes.
    ///
    /// The value of the first bucket is read before the key is compared, so
    /// the bucket that holds the key's own string costs the compare and one
    /// branch that jumps over the rest with nothing to move: the join takes
    /// that value as it is. Everything the rest reads again (the bucket's
    /// key, the key's hash) it loads itself, so the straight line keeps no
    /// register for it.
    fn probe_index(&mut self, binding: u32, key: ValueId, answer: Answer) -> Res<Probe> {
        let words = self.words_of(binding)?;
        if words.base == NONE || words.len == NONE {
            return self.reject("probe of a binding without index words");
        }
        let (slots, mask, scaled) = (words.base, words.len, words.lenf);
        let key_slot = self.get(key)?;
        let (payload, high, reused) = self.key_hash(key, key_slot);
        // The first bucket, by its offset.
        let start = self.f.bin(Op::And, high, scaled);
        let owner = self.load_at(
            Ty::I64,
            slots,
            Off::V(start),
            i64::try_from(NATIVE_PROBE_SLOT_OWNER).expect("small"),
        );
        // An empty bucket and a binding with no index have no value here:
        // the word is read and, as the key is not theirs, not used.
        let first = self.load_at(
            Ty::I64,
            slots,
            Off::V(start),
            i64::try_from(NATIVE_PROBE_SLOT_VALUE).expect("small"),
        );
        let miss = match answer {
            Answer::Value => &raw const NIL_VALUE as usize,
            Answer::Bucket => &raw const MISS_VALUE as usize,
        };
        let zero = self.iconst(0);
        let miss = self.iconst(i64::try_from(miss).expect("an address"));
        let give_up =
            self.iconst(i64::try_from(&raw const GIVE_UP_BUCKET as usize).expect("an address"));
        // The key's own string in the first bucket is the answer with no
        // other work. Everything else is in the skipped block.
        let open = self.f.begin_if(Cc::Ne, owner, payload);
        // The walk starts where the hash it compares says, so a start word
        // an earlier probe read is read again here, next to that hash.
        let start = if reused {
            let storage = i64::try_from(NATIVE_STRING_STORAGE_OFFSET).expect("small");
            let high = self.load32(
                payload,
                storage + i64::try_from(NATIVE_STRING_PROBE_START_OFFSET).expect("small"),
            );
            self.f.bin(Op::And, high, scaled)
        } else {
            start
        };
        let bucket = self.f.bin(Op::Add, slots, start);
        let hash = self.load_at(
            Ty::I64,
            payload,
            Off::K(0),
            i64::try_from(NATIVE_STRING_STORAGE_OFFSET + NATIVE_STRING_HASH_OFFSET).expect("small"),
        );
        let walk = Walk {
            slots,
            mask,
            scaled,
            payload,
            hash,
            zero,
            give_up,
        };
        // A key with no hash cannot say where it would have been.
        let hashed = self.f.begin_if(Cc::Ne, hash, zero);
        let seen = self.bucket_word(bucket, NATIVE_PROBE_SLOT_HASH);
        let found = self.step(&walk, 0, (start, bucket, seen));
        let found = self.f.end_if(hashed, &[(found, give_up)]);
        let later = self.settle(&walk, answer, binding, key_slot, found, miss);
        let entry = self.f.end_if(open, &[(later, first)]);
        Ok(Probe {
            entry,
            payload,
            slots,
            mask,
        })
    }

    /// The address of the value in the bucket `found`, where the walk
    /// ([`Self::step`]) ended, as the block of the skip over a first bucket
    /// that does not hold the key's own string.
    ///
    /// The bucket's key says what it is. None: an empty bucket, which ends
    /// the probe sequence, so the key is not in the index and the answer is
    /// `miss`. The key's own string: the entry (the walk met it past the
    /// first bucket). Any other string has the key's hash, or belongs to the
    /// bucket that stands for "cannot tell": its words are compared with the
    /// key's, and when they differ the rest is the native walk's and then
    /// the VM's, in a conditional call.
    fn settle(
        &mut self,
        walk: &Walk,
        answer: Answer,
        binding: u32,
        key_slot: Val,
        found: Val,
        miss: Val,
    ) -> Val {
        // The keys by their owner words, as the key's value names its own.
        let stored = self.bucket_word(found, NATIVE_PROBE_SLOT_OWNER);
        let full = self.f.begin_if(Cc::Ne, stored, walk.zero);
        let other = self.f.begin_if(Cc::Ne, stored, walk.payload);
        let different = self.words_differ(walk.payload, stored);
        // Equal words: the same key in another allocation. The slot takes
        // the table's string, so its next lookup here ends at the first
        // compare (`trace_adopt_key`), unless the run has stopped doing
        // that. A conditional call and no skip of its own: a skip here cost
        // the paths that never reach it (the register allocator settles the
        // values a skip joins on every path into it).
        let off = self.control_word(mem::offset_of!(TraceControl, adopt_off));
        let kept = self.f.bin(Op::Or, different, off);
        self.f.call_if(
            Cc::Eq,
            (kept, walk.zero),
            trace_adopt_key as *const () as usize,
            false,
            &[self.control, key_slot, found],
        );
        let call = self.f.begin_if(Cc::Ne, different, walk.zero);
        let answer_at = mem::offset_of!(TraceControl, admit_lane);
        let result_at = Some(u32::try_from(answer_at).expect("small"));
        let late = match answer {
            Answer::Bucket => {
                self.f.call_if_storing(
                    Cc::Ne,
                    (different, walk.zero),
                    trace_probe_bucket as *const () as usize,
                    false,
                    &[walk.slots, walk.mask, key_slot],
                    result_at,
                );
                self.control_word(answer_at)
            }
            Answer::Value => {
                let binding = self.iconst(i64::from(binding));
                self.f.call_if_storing(
                    Cc::Ne,
                    (different, walk.zero),
                    trace_probe_value as *const () as usize,
                    false,
                    &[self.control, walk.slots, walk.mask, binding, key_slot],
                    result_at,
                );
                // The call answers with the address of a value, where the
                // entry's value would be if the control block's word were a
                // bucket's last.
                let before = self.iconst(
                    offset(answer_at) - i64::try_from(NATIVE_PROBE_SLOT_VALUE).expect("small"),
                );
                self.f.bin(Op::Add, self.control, before)
            }
        };
        let settled = self.f.end_if(call, &[(late, found)]);
        let settled = self.f.end_if(other, &[(settled, found)]);
        let entry = self.bucket_word(settled, NATIVE_PROBE_SLOT_VALUE);
        self.f.end_if(full, &[(entry, miss)])
    }

    /// One bucket of the walk and the buckets after it. Returns the bucket
    /// the walk ends at, for [`Self::settle`] to look at: one with the key's
    /// hash (the key's own string or a candidate), one with no hash (an empty
    /// bucket, or a bucket of a binding with no index, whose key is no
    /// string's), or the one that stands for "cannot tell" after
    /// [`WALK_STEPS`] buckets. Each of the first two ends the walk by
    /// skipping the rest of it, so a step costs two compares and the address
    /// of the next bucket. `step` counts the buckets walked before this one,
    /// `(at, bucket, seen)` is its offset in the index, its address and its
    /// hash word.
    fn step(&mut self, walk: &Walk, step: u32, (at, bucket, seen): (Val, Val, Val)) -> Val {
        let other = self.f.begin_if(Cc::Ne, seen, walk.hash);
        let full = self.f.begin_if(Cc::Ne, seen, walk.zero);
        let found = if step + 1 < WALK_STEPS {
            let (at, next) = self.next_bucket(walk, at);
            let seen = self.bucket_word(next, NATIVE_PROBE_SLOT_HASH);
            self.step(walk, step + 1, (at, next, seen))
        } else {
            walk.give_up
        };
        let found = self.f.end_if(full, &[(found, bucket)]);
        self.f.end_if(other, &[(found, bucket)])
    }

    /// Whether the first words of the storages of the strings that the
    /// payload words `ours` and `theirs` name differ (zero when they are the
    /// same). Two inline strings are equal
    /// exactly when these words are: the tag, the length and the zero-padded
    /// bytes. A long string's words are its tag and where its bytes are, so
    /// they differ from an inline string's and from those of an equal long
    /// string in another allocation. A difference therefore does not say the
    /// strings differ, and the caller asks the native walk, which compares
    /// inline strings again and leaves long ones to the VM.
    fn words_differ(&mut self, ours: Val, theirs: Val) -> Val {
        let mut different = None;
        for word in 0..NATIVE_STRING_INLINE_WORDS {
            let at = i64::try_from(NATIVE_STRING_STORAGE_OFFSET + word * 8).expect("small");
            let a = self.load_at(Ty::I64, ours, Off::K(0), at);
            let b = self.load_at(Ty::I64, theirs, Off::K(0), at);
            let x = self.f.bin(Op::Xor, a, b);
            different = Some(match different {
                None => x,
                Some(before) => self.f.bin(Op::Or, before, x),
            });
        }
        different.expect("an inline string has words")
    }

    /// `HashLoad` of a string key through the index (`hash_probe` in
    /// `emit.rs`). A value of another type, and a missing key when the trace
    /// expects a value, leave through the guard. The VM answers (inside the
    /// conditional call) when the probe cannot tell, and the guard checks the
    /// tag of its answer.
    pub(super) fn probed_load(
        &mut self,
        binding: u32,
        key: ValueId,
        expected: TraceType,
        guard: GuardId,
    ) -> Res<Val> {
        let wanted_native = match expected {
            TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
            TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
            TraceType::Boolean => NATIVE_VALUE_TAG_BOOLEAN,
            _ => NATIVE_VALUE_TAG_NIL,
        };
        // A side that starts with the lookups its parent made has the
        // parent's answers (`takes_probe`): a nil for the ones that passed,
        // and the address the guard passed for the last.
        let entry = match self.s.handed {
            Some((_, last)) if self.at < last => return Ok(self.iconst(0)),
            Some((handed, last)) if self.at == last => {
                self.s.handed = None;
                handed
            }
            _ => self.probe_index(binding, key, Answer::Value)?.entry,
        };
        let tag = self.load8(entry, Off::K(0), 0);
        let wanted = self.iconst(i64::from(wanted_native));
        self.s.hand = Some(entry);
        let guarded = self.guard_cc(Cc::Eq, tag, wanted, guard);
        self.s.hand = None;
        guarded?;
        let payload = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
        Ok(match expected {
            TraceType::Nil => self.iconst(0),
            TraceType::Float => self.load_at(Ty::F64, entry, Off::K(0), payload),
            TraceType::Boolean => self.load8(entry, Off::K(0), payload),
            _ => self.load_at(Ty::I64, entry, Off::K(0), payload),
        })
    }

    /// A call of `SET_PRESENT`, `SET_REVIVE` or `DELETE_STRING` with the
    /// store in line (`hash_store` in `emit.rs`), after the probe found the
    /// key's bucket. The entry changes where it is when the table allows it
    /// and the entry has the type the store needs; the VM's helper answers
    /// every other case, then stores or leaves through the guard.
    ///
    /// `bad` is zero when the table takes the store in line. The store is a
    /// skip that runs on that, and the helper's call is a skip that runs on
    /// the opposite.
    pub(super) fn probed_store(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        binding: u32,
        guard: GuardId,
    ) -> Res<()> {
        let probe = self.probe_index(binding, args[1], Answer::Bucket)?;
        let zero = self.iconst(0);
        let stored = (helper != HelperId::DELETE_STRING).then(|| self.ty(args[2]));
        let value = match stored {
            Some(_) => Some(self.get(args[2])?),
            None => None,
        };
        let entry = probe.entry;
        let tag = self.load8(entry, Off::K(0), 0);
        // The index says whether this table takes the store, and the entry has
        // the type the store needs (the miss bucket's value has a tag that
        // fits none of them).
        let bad = if let (HelperId::SET_PRESENT, Some(ty)) = (helper, stored) {
            let wanted = self.iconst(i64::from(native_tag_of(ty)));
            let wrong = self.f.bin(Op::Xor, tag, wanted);
            let sixty_three = self.iconst(63);
            let read_only = self.f.bin(Op::Shr, probe.mask, sixty_three);
            self.f.bin(Op::Or, wrong, read_only)
        } else {
            let wrong = if helper == HelperId::SET_REVIVE {
                tag
            } else {
                // A number, an integer or a boolean: the tags 1 to 3.
                let one = self.iconst(1);
                let shifted = self.f.bin(Op::Sub, tag, one);
                let three = self.iconst(3);
                let yes = self.iconst(0);
                let no = self.iconst(1);
                self.f.select(Cc::Ult, shifted, three, yes, no)
            };
            self.not_stored_in_line(wrong, &probe)
        };
        self.prepare_helpers(&[(helper, args, guard)])?;
        let payload = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
        let open = self.f.begin_if(Cc::Eq, bad, zero);
        if helper == HelperId::SET_PRESENT {
            let (Some(value), Some(ty)) = (value, stored) else {
                unreachable!("a store has a value");
            };
            self.store_scalar(entry, payload, ty, value);
        } else {
            let counter = self.live_counter(&probe);
            let live = self.load_at(Ty::I64, counter, Off::K(0), 0);
            let one = self.iconst(1);
            if let (Some(value), Some(ty)) = (value, stored) {
                self.store_scalar(entry, payload, ty, value);
                let new_tag = self.iconst(i64::from(native_tag_of(ty)));
                self.store8(entry, Off::K(0), 0, new_tag);
                let more = self.f.bin(Op::Add, live, one);
                self.store_at(counter, Off::K(0), 0, more);
            } else {
                let nil = self.iconst(i64::from(NATIVE_VALUE_TAG_NIL));
                self.store8(entry, Off::K(0), 0, nil);
                let fewer = self.f.bin(Op::Sub, live, one);
                self.store_at(counter, Off::K(0), 0, fewer);
            }
        }
        self.f.end_if(open, &[]);
        self.store_by_helpers(&[(helper, args, guard)], bad)
    }

    /// The address of the table's live-entry counter, which sits in the
    /// header slot after the index, whose size the mask's low bits give.
    fn live_counter(&mut self, probe: &Probe) -> Val {
        let low = self.iconst(PROBE_NO_TOMBSTONES - 1);
        let last = self.f.bin(Op::And, probe.mask, low);
        let one = self.iconst(1);
        let count = self.f.bin(Op::Add, last, one);
        let size = self.iconst(i64::try_from(NATIVE_PROBE_SLOT_SIZE).expect("small"));
        let skip = self.f.bin(Op::Mul, count, size);
        let header = self.f.bin(Op::Add, probe.slots, skip);
        self.load_at(
            Ty::I64,
            header,
            Off::K(0),
            i64::try_from(NATIVE_PROBE_SLOT_HASH).expect("small"),
        )
    }

    /// What the helpers' call needs before the skip that makes it opens: the
    /// values of the arguments (a pending comparison is made where it is first
    /// used, which must not be inside a skip) and the exits of the guards on
    /// their status, whose snapshots are made once and kept.
    pub(super) fn prepare_helpers(&mut self, calls: &[(HelperId, &[ValueId], GuardId)]) -> Res<()> {
        for &(helper, args, guard) in calls {
            for &arg in args {
                self.get(arg)?;
            }
            if helper
                .info()
                .is_some_and(|info| info.effects.contains(HelperEffects::MAY_EXIT))
            {
                self.guard_target(guard)?;
            }
        }
        Ok(())
    }

    /// The VM's stores, in order, when `bad` is not zero (the table did not
    /// take the store in line): one skip that runs on it, with a conditional
    /// call of each helper and the helper's guard on its status, which is
    /// zeroed first. [`Self::prepare_helpers`] ran before.
    pub(super) fn store_by_helpers(
        &mut self,
        calls: &[(HelperId, &[ValueId], GuardId)],
        bad: Val,
    ) -> Res<()> {
        let zero = self.iconst(0);
        let open = self.f.begin_if(Cc::Ne, bad, zero);
        for &(helper, args, guard) in calls {
            let Some(info) = helper.info() else {
                return self.reject("helper without a description");
            };
            let Some(address) = helper_address(helper) else {
                return self.reject("helper nobody installed");
            };
            let mut list = mem::take(&mut self.s.list);
            list.clear();
            let built = self.helper_arguments(helper, info, args, &mut list);
            if let Err(error) = built {
                self.s.list = list;
                return Err(error);
            }
            let set_at = mem::offset_of!(TraceControl, set_status);
            self.store_at(self.control, Off::K(0), offset(set_at), zero);
            self.f.call_if_storing(
                Cc::Ne,
                (bad, zero),
                address,
                false,
                &list,
                Some(u32::try_from(set_at).expect("small")),
            );
            self.s.list = list;
            if info.effects.contains(HelperEffects::MAY_EXIT) {
                let status = self.control_word(set_at);
                self.guard_cc(Cc::Eq, status, zero, guard)?;
            }
        }
        self.f.end_if(open, &[]);
        Ok(())
    }

    /// The key a `t[k] = nil` followed by `t[k] = v` of a number or boolean
    /// fuses with (`fused_delete_stores` in `emit.rs`), seen from the delete
    /// at `at`: the arguments and the guard of the store after it, with only
    /// pure operations between.
    pub(super) fn store_after_delete(&self, at: usize) -> Option<(&'a [ValueId], GuardId)> {
        let TraceOp::CallHelper {
            helper: HelperId::DELETE_STRING,
            args: delete,
            ..
        } = self.op_at(at)?
        else {
            return None;
        };
        for later in at + 1.. {
            match self.op_at(later)? {
                TraceOp::Constant(_)
                | TraceOp::Move { .. }
                | TraceOp::Arithmetic { .. }
                | TraceOp::Compare { .. }
                | TraceOp::Select { .. } => {}
                TraceOp::CallHelper {
                    helper: HelperId::SET_REVIVE,
                    args,
                    guard,
                    ..
                } if args.len() == 3
                    && args[..2] == delete[..]
                    && matches!(
                        self.ir.value_type(args[2]),
                        Some(TraceType::Float | TraceType::Integer | TraceType::Boolean)
                    ) =>
                {
                    return Some((args.as_slice(), *guard));
                }
                _ => return None,
            }
        }
        None
    }

    /// The delete a `SET_REVIVE` at `at` fuses with, seen from the store: the
    /// same pair [`Self::store_after_delete`] finds from the other end.
    pub(super) fn delete_before_store(
        &self,
        at: usize,
        guard: GuardId,
    ) -> Option<(&'a [ValueId], GuardId)> {
        for earlier in (0..at).rev() {
            match self.op_at(earlier)? {
                TraceOp::Constant(_)
                | TraceOp::Move { .. }
                | TraceOp::Arithmetic { .. }
                | TraceOp::Compare { .. }
                | TraceOp::Select { .. } => {}
                TraceOp::CallHelper {
                    helper: HelperId::DELETE_STRING,
                    args,
                    guard: delete_guard,
                    ..
                } => {
                    let (_, store) = self.store_after_delete(earlier)?;
                    return (store == guard).then_some((args.as_slice(), *delete_guard));
                }
                _ => return None,
            }
        }
        None
    }

    /// The operation `at` places into the member, counting the preheader and
    /// then the body.
    fn op_at(&self, at: usize) -> Option<&'a TraceOp> {
        let ir: &'a TraceIr = self.ir;
        let preheader = ir.preheader.len();
        let instruction = if at < preheader {
            ir.preheader.get(at)
        } else {
            ir.body.get(at - preheader)
        };
        instruction.map(|instruction| &instruction.op)
    }

    /// `t[k] = nil` then `t[k] = v` of one key, probed once (`hash_overwrite`
    /// in `emit.rs`). A live number, integer or boolean takes the value where
    /// it stands (the delete and the store would drop the live-entry counter
    /// and raise it again). A deleted key (nil) takes it and the counter
    /// rises. Anything else goes to both helpers in order, each with its own
    /// guard and snapshot.
    pub(super) fn probed_overwrite(
        &mut self,
        (delete, delete_guard): (&[ValueId], GuardId),
        (store, store_guard): (&[ValueId], GuardId),
        binding: u32,
    ) -> Res<()> {
        let probe = self.probe_index(binding, store[1], Answer::Bucket)?;
        let zero = self.iconst(0);
        let ty = self.ty(store[2]);
        let value = self.get(store[2])?;
        let entry = probe.entry;
        let tag = self.load8(entry, Off::K(0), 0);
        // A number, an integer, a boolean or a nil (the tags 0 to 3); the
        // miss bucket's value has another.
        let three = self.iconst(3);
        let over = self.f.bin(Op::Sub, three, tag);
        let sixty_three = self.iconst(63);
        let wrong = self.f.bin(Op::Shr, over, sixty_three);
        let bad = self.not_stored_in_line(wrong, &probe);
        let calls = [
            (HelperId::DELETE_STRING, delete, delete_guard),
            (HelperId::SET_REVIVE, store, store_guard),
        ];
        self.prepare_helpers(&calls)?;
        let payload = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
        let open = self.f.begin_if(Cc::Eq, bad, zero);
        self.store_scalar(entry, payload, ty, value);
        let new_tag = self.iconst(i64::from(native_tag_of(ty)));
        self.store8(entry, Off::K(0), 0, new_tag);
        // Only a nil that comes back to life is a new live entry.
        let revived = self.f.begin_if(Cc::Eq, tag, zero);
        let counter = self.live_counter(&probe);
        let live = self.load_at(Ty::I64, counter, Off::K(0), 0);
        let one = self.iconst(1);
        let more = self.f.bin(Op::Add, live, one);
        self.store_at(counter, Off::K(0), 0, more);
        self.f.end_if(revived, &[]);
        self.f.end_if(open, &[]);
        self.store_by_helpers(&calls, bad)
    }

    /// Adds to `wrong` (zero when the entry has a type the store takes) the
    /// reasons the table does not take a delete or a revive in line: its index
    /// says no, or the key may name a metamethod. A key that may name a
    /// metamethod changes what a metatable's lookups answer: the VM's. An
    /// inline string has its bytes after the tag and the length, zero-padded;
    /// a long one is the VM's too. The result is zero when the store is made
    /// in line.
    fn not_stored_in_line(&mut self, wrong: Val, probe: &Probe) -> Val {
        // The flag of a table that takes no delete is bit 62 of the mask, and
        // a table that is read-only (bit 63) is one of those.
        let flag_bits = self.iconst(62);
        let flags = self.f.bin(Op::Shr, probe.mask, flag_bits);
        let bad = self.f.bin(Op::Or, wrong, flags);
        let data = self.storage_of(probe.payload);
        let form = self.load8(data, Off::K(0), 0);
        let head = self.load32(data, 0);
        let sixteen = self.iconst(16);
        let bytes = self.f.bin(Op::Shr, head, sixteen);
        // One for the bytes `__`, which are the only two that make the
        // difference zero.
        let dunder = self.iconst(0x5f5f);
        let apart = self.f.bin(Op::Xor, bytes, dunder);
        let one = self.iconst(1);
        let below = self.f.bin(Op::Sub, apart, one);
        let sixty_three = self.iconst(63);
        let named = self.f.bin(Op::Shr, below, sixty_three);
        let keyed = self.f.bin(Op::Or, form, named);
        self.f.bin(Op::Or, bad, keyed)
    }

    /// Stores the scalar `value` of type `ty` at `[base + disp]`.
    fn store_scalar(&mut self, base: Val, disp: i64, ty: TraceType, value: Val) {
        match ty {
            TraceType::Boolean => self.store8(base, Off::K(0), disp, value),
            _ => self.store_at(base, Off::K(0), disp, value),
        }
    }
}

/// The tag of a value in an entry, for a scalar type the stores handle.
fn native_tag_of(ty: TraceType) -> u8 {
    match ty {
        TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
        TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
        _ => NATIVE_VALUE_TAG_BOOLEAN,
    }
}
