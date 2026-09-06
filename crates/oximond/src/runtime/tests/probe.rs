//! The string index native trace code probes ([`Runtime::table_native_probe`])
//! and the string layout that probe reads.

use super::*;
use crate::runtime::native_views::{
    NATIVE_PROBE_SLOT_HASH, NATIVE_PROBE_SLOT_KEY, NATIVE_PROBE_SLOT_SIZE, NATIVE_PROBE_SLOT_VALUE,
    NATIVE_STRING_STORAGE_OFFSET, NATIVE_VALUE_PAYLOAD_OFFSET, native_probe_start,
};
use crate::runtime::{
    NATIVE_STRING_HASH_OFFSET, NATIVE_STRING_INLINE_TAG, NATIVE_STRING_INLINE_WORDS, TableShape,
};
use crate::string::LuaString;

/// Reads a word `offset` bytes past `address`.
///
/// # Safety
///
/// `address + offset` is readable memory of a live object.
#[allow(unsafe_code)]
unsafe fn word(address: usize, offset: usize) -> u64 {
    // SAFETY: the caller's promise.
    unsafe { ((address + offset) as *const u64).read_unaligned() }
}

/// The storage address of the string a `Value::String` holds, the way native
/// code finds it: the payload word plus [`NATIVE_STRING_STORAGE_OFFSET`].
#[allow(unsafe_code)]
fn storage_of(value: &Value) -> usize {
    assert!(matches!(value, Value::String(_)));
    let slot = std::ptr::from_ref(value) as usize;
    // SAFETY: `value` is a live 16-byte `Value`, whose payload word is at 8.
    let payload = unsafe { word(slot, NATIVE_VALUE_PAYLOAD_OFFSET as usize) };
    payload as usize + NATIVE_STRING_STORAGE_OFFSET
}

/// The words native code compares a borrowed string with a short constant
/// by ([`crate::runtime::native_inline_words`]) are the first words of the
/// storage of every string of those bytes, however it was made, for every
/// inline length; bytes too long to be inline have none. Strings of other
/// bytes differ from them in some word, and so does a long string.
#[test]
fn the_inline_words_of_a_constant_are_those_of_its_strings() {
    use crate::runtime::native_inline_words;
    let alphabet = b"abcdefghijklmnopqrstuvwxyz";
    let words_of = |value: &Value| -> [u64; NATIVE_STRING_INLINE_WORDS] {
        let storage = storage_of(value);
        // SAFETY: `storage` is the live storage of the string, at least
        // `NATIVE_STRING_INLINE_WORDS` words long.
        #[allow(unsafe_code)]
        std::array::from_fn(|index| unsafe { word(storage, 8 * index) })
    };
    for length in 0..=24 {
        let bytes = &alphabet[..length];
        let words = native_inline_words(bytes);
        assert_eq!(words.is_some(), length <= 22, "length {length}");
        let made = [
            Value::from(LuaString::from(bytes)),
            Value::from(LuaString::from(bytes.to_vec())),
            Value::from(LuaString::from(String::from_utf8(bytes.to_vec()).unwrap())),
        ];
        for value in &made {
            match words {
                Some(words) => assert_eq!(words_of(value), words, "length {length}"),
                // A long string's first word has another tag.
                None => assert_ne!(
                    words_of(value)[0].to_ne_bytes()[0],
                    NATIVE_STRING_INLINE_TAG,
                    "length {length}"
                ),
            }
        }
    }
    // Zero bytes inside the string are not its padding.
    assert_ne!(native_inline_words(b"a"), native_inline_words(b"a\0"));
    assert_ne!(native_inline_words(b""), native_inline_words(b"\0"));
    let long = Value::from(LuaString::from(&alphabet[..23]));
    let short = native_inline_words(&alphabet[..22]).unwrap();
    assert_ne!(words_of(&long), short);
}

#[test]
fn the_string_layout_native_code_reads_holds() {
    let inline = Value::from("field_12");
    let Value::String(text) = &inline else {
        unreachable!()
    };
    // The payload plus two reference counts is the storage `identity` names.
    assert_eq!(storage_of(&inline), text.identity());
    let storage = text.identity();
    let hash = text.cached_hash();
    #[allow(unsafe_code)]
    // SAFETY: `storage` is the live storage of `text`, 32 bytes.
    unsafe {
        assert_eq!(word(storage, NATIVE_STRING_HASH_OFFSET), hash);
        // Tag, length, bytes, zero padding.
        let first = word(storage, 0).to_le_bytes();
        assert_eq!(first[0], NATIVE_STRING_INLINE_TAG);
        assert_eq!(first[1], 8);
        assert_eq!(&first[2..], b"field_");
        let second = word(storage, 8).to_le_bytes();
        assert_eq!(&second[..2], b"12");
        assert!(second[2..].iter().all(|byte| *byte == 0));
        assert_eq!(word(storage, 16), 0);
    }
    assert_eq!(NATIVE_STRING_INLINE_WORDS, 3);
    // Two allocations with the same bytes agree on every inline word.
    let other = Value::from(String::from("field_12"));
    assert_ne!(storage_of(&other), storage);
    for index in 0..NATIVE_STRING_INLINE_WORDS {
        #[allow(unsafe_code)]
        // SAFETY: both are live storages of at least 24 bytes.
        unsafe {
            assert_eq!(
                word(storage_of(&other), index * 8),
                word(storage, index * 8)
            );
        }
    }
    // A string that does not fit inline is tagged otherwise, so native code
    // sends it to the runtime.
    let long = Value::from("a string that is longer than the inline limit");
    #[allow(unsafe_code)]
    // SAFETY: the storage is live.
    let tag = unsafe { word(storage_of(&long), 0) }.to_le_bytes()[0];
    assert_ne!(tag, NATIVE_STRING_INLINE_TAG);
    // The longest inline string uses every byte.
    let full = Value::from("0123456789abcdefghijkl");
    #[allow(unsafe_code)]
    // SAFETY: the storage is live.
    unsafe {
        let tag = word(storage_of(&full), 0).to_le_bytes()[0];
        assert_eq!(tag, NATIVE_STRING_INLINE_TAG);
        assert_eq!(word(storage_of(&full), 0).to_le_bytes()[1], 22);
    }
}

/// What the emitted probe does with the key in `key`, in Rust: `Err` when it
/// calls the runtime instead, else the address of the value it found (`None`
/// for a miss).
#[allow(unsafe_code)]
fn emulate(probe: NativeStringProbe, key: &Value) -> Result<Option<*const Value>, ()> {
    let data = storage_of(key);
    // SAFETY: `data` is the live storage of the key, whose layout the first
    // test pins; the slots are the index the caller took, still valid.
    unsafe {
        let hash = word(data, NATIVE_STRING_HASH_OFFSET);
        if hash == 0 {
            return Err(());
        }
        let mask = probe.mask as usize;
        let mut at = native_probe_start(hash) as usize & mask;
        loop {
            let slot = probe.slots + at * NATIVE_PROBE_SLOT_SIZE;
            let stored = word(slot, NATIVE_PROBE_SLOT_KEY) as usize;
            if stored == 0 {
                return Ok(None);
            }
            if word(slot, NATIVE_PROBE_SLOT_HASH) == hash {
                let value = word(slot, NATIVE_PROBE_SLOT_VALUE) as usize as *const Value;
                if stored == data {
                    return Ok(Some(value));
                }
                if word(data, 0).to_le_bytes()[0] != NATIVE_STRING_INLINE_TAG {
                    return Err(());
                }
                if word(stored, 0).to_le_bytes()[0] == NATIVE_STRING_INLINE_TAG
                    && (0..NATIVE_STRING_INLINE_WORDS)
                        .all(|index| word(stored, index * 8) == word(data, index * 8))
                {
                    return Ok(Some(value));
                }
            }
            at = (at + 1) & mask;
        }
    }
}

/// A string value whose hash is computed, as the key of a lookup is once any
/// lookup has used it (native code sends a key without a hash to the runtime).
fn string(text: impl Into<String>) -> Value {
    let value = Value::from(text.into());
    if let Value::String(text) = &value {
        let _ = text.cached_hash();
    }
    value
}

fn hashed(table: &crate::runtime::table_storage::Table) -> bool {
    matches!(&table.entries, TableEntries::Other(other) if matches!(other.view(), OtherRef::Indexed(_)))
}

/// Builds the table a case probes: `count` string keys `k0..` holding the
/// integers `i * 3`, and an integer key besides.
fn filled(runtime: &mut Runtime, count: usize) -> TableHandle {
    let table = runtime.create_table().unwrap();
    for index in 0..count {
        runtime
            .table_set(
                table,
                &string(format!("k{index}")),
                Value::Integer(index as i64 * 3),
            )
            .unwrap();
    }
    runtime
        .table_set(table, &Value::Integer(1000), Value::Integer(7))
        .unwrap();
    table
}

fn read(value: Result<Option<*const Value>, ()>) -> Option<Value> {
    #[allow(unsafe_code)]
    // SAFETY: a found address is a value of the table the probe indexed.
    value
        .expect("native")
        .map(|value| unsafe { (*value).clone() })
}

fn check_every_key(runtime: &mut Runtime, table: TableHandle, count: usize) {
    let mut scratch = NativeStringProbeScratch::default();
    let probe = runtime
        .table_native_probe(table, &mut scratch)
        .expect("an index");
    assert!(probe.mask.is_power_of_two() || (probe.mask + 1).is_power_of_two());
    for index in 0..count {
        // A key allocated elsewhere, as most lookups have.
        let key = string(format!("k{index}"));
        assert_eq!(
            read(emulate(probe, &key)),
            Some(Value::Integer(index as i64 * 3)),
            "k{index}"
        );
    }
    for missing in ["absent", "k", "k999999", "1000"] {
        assert_eq!(read(emulate(probe, &string(missing))), None, "{missing}");
    }
}

#[test]
fn an_ordered_table_is_indexed_in_scratch() {
    let mut runtime = Runtime::new();
    let table = filled(&mut runtime, 10);
    assert!(!hashed(runtime.table(table).unwrap()));
    check_every_key(&mut runtime, table, 10);
}

#[test]
fn a_shaped_table_is_indexed_in_scratch() {
    let mut runtime = Runtime::new();
    let keys = ["alpha", "beta", "gamma", "delta"];
    let shape = TableShape::new(keys.iter().map(|key| LuaString::from(*key))).unwrap();
    let table = runtime.create_table_with_shape(&shape).unwrap();
    runtime
        .table_set(table, &string("beta"), Value::Integer(2))
        .unwrap();
    runtime
        .table_set(table, &string("delta"), Value::Number(4.5))
        .unwrap();
    assert!(runtime.table(table).unwrap().entries.is_shaped());
    let mut scratch = NativeStringProbeScratch::default();
    let probe = runtime.table_native_probe(table, &mut scratch).unwrap();
    assert_eq!(
        read(emulate(probe, &string("beta"))),
        Some(Value::Integer(2))
    );
    assert_eq!(
        read(emulate(probe, &string("delta"))),
        Some(Value::Number(4.5))
    );
    // A key of the shape with no value reads nil, as an absent one does.
    assert_eq!(read(emulate(probe, &string("alpha"))), Some(Value::Nil));
    assert_eq!(read(emulate(probe, &string("other"))), None);
}

#[test]
fn a_hashed_table_keeps_one_index_per_layout() {
    let mut runtime = Runtime::new();
    for count in [20, 63, 64, 300] {
        let table = filled(&mut runtime, count);
        assert!(hashed(runtime.table(table).unwrap()));
        let mut scratch = NativeStringProbeScratch::default();
        // A big table earns its index on the second request for a layout.
        let first = runtime.table_native_probe(table, &mut scratch);
        // The integer key counts too.
        assert_eq!(first.is_some(), count + 1 < 64, "{count}");
        let probe = runtime
            .table_native_probe(table, &mut scratch)
            .expect("an index");
        let again = runtime.table_native_probe(table, &mut scratch).unwrap();
        assert_eq!(probe, again, "the layout did not change");
        check_every_key(&mut runtime, table, count);
    }
}

#[test]
fn overwrites_and_deletes_in_place_keep_the_index_current() {
    let mut runtime = Runtime::new();
    let table = filled(&mut runtime, 100);
    let mut scratch = NativeStringProbeScratch::default();
    let _ = runtime.table_native_probe(table, &mut scratch);
    let probe = runtime.table_native_probe(table, &mut scratch).unwrap();
    let key = string("k5");
    let before = emulate(probe, &key).unwrap().unwrap();
    // A new value, of another type, is read through the same address.
    runtime.table_set(table, &key, Value::Number(0.5)).unwrap();
    assert_eq!(
        runtime.table_native_probe(table, &mut scratch),
        Some(probe),
        "an overwrite moves nothing"
    );
    assert_eq!(emulate(probe, &key).unwrap().unwrap(), before);
    assert_eq!(read(emulate(probe, &key)), Some(Value::Number(0.5)));
    // A delete leaves a tombstone, which reads nil at the same address, and
    // reviving it puts the value back there.
    assert!(
        runtime
            .table_delete_string_in_place(table, &LuaString::from("k5"))
            .unwrap()
    );
    assert_eq!(runtime.table_native_probe(table, &mut scratch), Some(probe));
    assert_eq!(read(emulate(probe, &key)), Some(Value::Nil));
    runtime.table_set(table, &key, Value::Integer(9)).unwrap();
    assert_eq!(runtime.table_native_probe(table, &mut scratch), Some(probe));
    assert_eq!(emulate(probe, &key).unwrap().unwrap(), before);
    assert_eq!(read(emulate(probe, &key)), Some(Value::Integer(9)));
}

#[test]
fn every_change_of_layout_retires_the_index() {
    let mut runtime = Runtime::new();
    let mut scratch = NativeStringProbeScratch::default();
    type Change = fn(&mut Runtime, TableHandle);
    let changes: [(&str, Change); 8] = [
        ("a new string key", |runtime, table| {
            runtime
                .table_set(table, &Value::from("fresh"), Value::Integer(1))
                .unwrap();
        }),
        ("a new integer key", |runtime, table| {
            runtime
                .table_set(table, &Value::Integer(5000), Value::Integer(1))
                .unwrap();
        }),
        ("a removed integer key", |runtime, table| {
            runtime
                .table_set(table, &Value::Integer(1000), Value::Nil)
                .unwrap();
        }),
        ("a clear", |runtime, table| {
            runtime.table_clear(table).unwrap();
        }),
        ("a reserve", |runtime, table| {
            runtime.table_reserve(table, 0, 4096).unwrap();
        }),
        ("a pop of the first entry", |runtime, table| {
            assert!(
                runtime
                    .table_mut(table)
                    .unwrap()
                    .entries
                    .pop_first()
                    .is_some()
            );
        }),
        ("a compaction", |runtime, table| {
            // Deleting most keys in place leaves tombstones that the next
            // insertion of a new key drops.
            for index in 0..290 {
                let key = LuaString::from(format!("k{index}"));
                assert!(runtime.table_delete_string_in_place(table, &key).unwrap());
            }
            runtime
                .table_set(table, &Value::from("after"), Value::Integer(1))
                .unwrap();
        }),
        (
            "a table that gets sparse integer keys absorbed",
            |runtime, table| {
                for index in 1..=40 {
                    runtime
                        .table_set(table, &Value::Integer(index), Value::Integer(index))
                        .unwrap();
                }
            },
        ),
    ];
    for (name, change) in changes {
        let table = filled(&mut runtime, 300);
        let _ = runtime.table_native_probe(table, &mut scratch);
        assert!(
            runtime.table_native_probe(table, &mut scratch).is_some(),
            "{name}: indexed"
        );
        change(&mut runtime, table);
        assert!(
            hashed(runtime.table(table).unwrap()),
            "{name}: still hashed"
        );
        // The old index is gone. A table that is still big starts over (the
        // next request is a first one); a small one gets a new index. Either
        // way no stale index is offered.
        let keys: Vec<Value> = (0..300)
            .map(|index| string(format!("k{index}")))
            .chain([string("fresh"), string("after")])
            .collect();
        match runtime.table_native_probe(table, &mut scratch) {
            None => {}
            Some(probe) => {
                for key in &keys {
                    let expected = runtime.table_get(table, key).unwrap();
                    let found = read(emulate(probe, key)).unwrap_or(Value::Nil);
                    assert_eq!(found, expected, "{name}: {key:?}");
                }
            }
        }
    }
}

/// The live-entry counter the header slot after the index names, as native
/// code finds it: the slot `mask + 1` places past the first.
#[allow(unsafe_code)]
fn counter_of(probe: NativeStringProbe) -> (usize, usize) {
    let header = probe.slots + (probe.mask as usize + 1) * NATIVE_PROBE_SLOT_SIZE;
    // SAFETY: the header slot belongs to the index, and holds the address of
    // the table's counter.
    let counter = unsafe { word(header, NATIVE_PROBE_SLOT_HASH) } as usize;
    // SAFETY: the counter is a field of the table the probe indexes.
    (counter, unsafe { (counter as *const usize).read() })
}

#[test]
fn the_index_says_what_native_code_may_store_through_it() {
    let mut runtime = Runtime::new();
    let mut scratch = NativeStringProbeScratch::default();
    // A hashed table that is writable takes overwrites, deletes and revives.
    let table = filled(&mut runtime, 100);
    let _ = runtime.table_native_probe(table, &mut scratch);
    let probe = runtime.table_native_probe(table, &mut scratch).unwrap();
    assert!(probe.writable() && probe.tombstones());
    // The header slot names the table's counter of live entries, which a
    // delete and a revive through the runtime move as native code would.
    let (counter, live) = counter_of(probe);
    assert_eq!(live, 101);
    assert!(
        runtime
            .table_delete_string_in_place(table, &LuaString::from("k5"))
            .unwrap()
    );
    assert_eq!(counter_of(probe), (counter, 100));
    assert!(
        runtime
            .table_revive_string(table, &LuaString::from("k5"), &Value::Integer(1))
            .unwrap()
    );
    assert_eq!(counter_of(probe), (counter, 101));
    // The counter's address is written at every request.
    let again = runtime.table_native_probe(table, &mut scratch).unwrap();
    assert_eq!(counter_of(again).0, counter);

    // A deleted key of the other kind (an integer key leaves the map) stops
    // the deletes and revives, not the overwrites.
    runtime
        .table_set(table, &Value::Integer(1000), Value::Nil)
        .unwrap();
    let _ = runtime.table_native_probe(table, &mut scratch);
    let probe = runtime.table_native_probe(table, &mut scratch).unwrap();
    assert!(probe.writable() && !probe.tombstones());

    // A read-only table, a table other tables' cached reads depend on, and the
    // table a collection traces in chunks take no store at all.
    let table = filled(&mut runtime, 100);
    runtime.table_set_readonly(table, true).unwrap();
    let _ = runtime.table_native_probe(table, &mut scratch);
    let probe = runtime.table_native_probe(table, &mut scratch).unwrap();
    assert!(!probe.writable() && !probe.tombstones());
    runtime.table_set_readonly(table, false).unwrap();
    runtime.table_mut(table).unwrap().watched = true;
    let probe = runtime.table_native_probe(table, &mut scratch).unwrap();
    assert!(!probe.writable() && !probe.tombstones());
    runtime.table_mut(table).unwrap().watched = false;
    runtime.gc_partial_table = Some(table);
    let probe = runtime.table_native_probe(table, &mut scratch).unwrap();
    assert!(!probe.writable() && !probe.tombstones());
    runtime.gc_partial_table = None;
    let probe = runtime.table_native_probe(table, &mut scratch).unwrap();
    assert!(probe.writable() && probe.tombstones());

    // A small table is indexed in scratch: it keeps no tombstones, and its
    // values may still be overwritten in place.
    let small = filled(&mut runtime, 10);
    assert!(!hashed(runtime.table(small).unwrap()));
    let probe = runtime.table_native_probe(small, &mut scratch).unwrap();
    assert!(probe.writable() && !probe.tombstones());
}

#[test]
fn the_first_request_of_a_run_indexes_a_big_table_at_once() {
    let mut runtime = Runtime::new();
    let mut scratch = NativeStringProbeScratch::default();
    let table = filled(&mut runtime, 300);
    assert!(hashed(runtime.table(table).unwrap()));
    assert!(
        runtime
            .table_native_probe_now(table, &mut scratch)
            .is_some(),
        "a trace's first run does not wait for a second request"
    );
    // Later requests find the index built for the layout, as before.
    let probe = runtime.table_native_probe(table, &mut scratch).unwrap();
    assert_eq!(
        runtime.table_native_probe_now(table, &mut scratch),
        Some(probe)
    );
    // The same table is gated for the requests that are not a first run.
    let other = filled(&mut runtime, 300);
    assert!(runtime.table_native_probe(other, &mut scratch).is_none());
    assert!(runtime.table_native_probe(other, &mut scratch).is_some());
}

/// Keys that differ only in a number at their end start their probes apart.
/// The start is the high half of a hash whose halves are folded and
/// multiplied when it is cached (`finish_hash`, `native_probe_start`).
/// Without that mix the average key of 20000 keys `player_name_N` sat
/// 15 buckets past its first, and a probe compiled in line gives up after
/// eight.
#[test]
fn numbered_keys_do_not_pile_up_in_the_index() {
    let mut runtime = Runtime::new();
    let mut scratch = NativeStringProbeScratch::default();
    for prefix in [
        "k",
        "field_",
        "player_name_",
        "some.long.module.path.entry_",
    ] {
        for count in [12, 300, 2000, 20000] {
            let table = runtime.create_table().unwrap();
            let keys: Vec<Value> = (1..=count)
                .map(|index| string(format!("{prefix}{index}")))
                .collect();
            for key in &keys {
                runtime.table_set(table, key, Value::Integer(1)).unwrap();
            }
            let probe = runtime
                .table_native_probe_now(table, &mut scratch)
                .expect("an index");
            let mask = probe.mask as usize;
            let mut walked = 0_usize;
            for key in &keys {
                // The table holds its own copy of the key, so the walk ends
                // at the bucket with the key's hash and bytes.
                let found = emulate_steps(probe.slots, mask, key);
                walked += found;
            }
            assert!(
                walked <= count,
                "{count} keys {prefix}N walk {walked} buckets past their first"
            );
        }
    }
}

/// How many buckets past its first a probe for `key`, which the index holds,
/// reads before it meets a bucket with the key's hash.
#[allow(unsafe_code)]
fn emulate_steps(slots: usize, mask: usize, key: &Value) -> usize {
    let data = storage_of(key);
    // SAFETY: `data` is the live storage of the key and the slots are an
    // index the caller just took, which holds the key.
    unsafe {
        let hash = word(data, NATIVE_STRING_HASH_OFFSET);
        let mut at = native_probe_start(hash) as usize & mask;
        let mut steps = 0;
        loop {
            let slot = slots + at * NATIVE_PROBE_SLOT_SIZE;
            assert_ne!(
                word(slot, NATIVE_PROBE_SLOT_KEY),
                0,
                "the index holds the key"
            );
            if word(slot, NATIVE_PROBE_SLOT_HASH) == hash {
                return steps;
            }
            at = (at + 1) & mask;
            steps += 1;
        }
    }
}

/// What a probe compiled in line reads of a slot and of a key: the slot
/// names its key by the word a value that holds the string has in its
/// payload, and the high half of the key's hash under the mask shifted by
/// the size of a slot is the offset of the slot the probe starts at.
#[test]
fn a_slot_names_its_owner_and_starts_at_the_masked_hash() {
    use crate::runtime::native_views::{
        NATIVE_PROBE_SLOT_OWNER, NATIVE_PROBE_SLOT_SHIFT, NATIVE_STRING_PROBE_START_OFFSET,
    };
    let mut runtime = Runtime::new();
    let mut scratch = NativeStringProbeScratch::default();
    for count in [1, 12, 300, 5000] {
        let table = runtime.create_table().unwrap();
        let keys: Vec<Value> = (0..count)
            .map(|index| string(format!("owner_{index}")))
            .collect();
        for key in &keys {
            runtime.table_set(table, key, Value::Integer(1)).unwrap();
        }
        let probe = runtime
            .table_native_probe_now(table, &mut scratch)
            .expect("an index");
        let mask = probe.mask as usize;
        let scaled = (mask << NATIVE_PROBE_SLOT_SHIFT) as u64;
        assert!(scaled <= u64::from(u32::MAX));
        let mut owned = 0;
        // SAFETY: the slots are the index just taken, `mask + 1` of them,
        // and the keys are live strings whose layout the tests above pin.
        #[allow(unsafe_code)]
        unsafe {
            for at in 0..=mask {
                let slot = probe.slots + at * NATIVE_PROBE_SLOT_SIZE;
                let stored = word(slot, NATIVE_PROBE_SLOT_KEY) as usize;
                let named = word(slot, NATIVE_PROBE_SLOT_OWNER) as usize;
                if stored == 0 {
                    assert_eq!(named, 0);
                } else {
                    assert_eq!(named + NATIVE_STRING_STORAGE_OFFSET, stored);
                    owned += 1;
                }
            }
            for key in &keys {
                let data = storage_of(key);
                let hash = word(data, NATIVE_STRING_HASH_OFFSET);
                let high = word(data, NATIVE_STRING_PROBE_START_OFFSET) & 0xffff_ffff;
                assert_eq!(
                    high & scaled,
                    (native_probe_start(hash) & mask as u64) << NATIVE_PROBE_SLOT_SHIFT
                );
            }
        }
        assert_eq!(owned, count);
    }
}
