use super::*;
use crate::string::LuaString;

#[test]
fn large_sparse_numeric_maps_keep_ordered_lookup_and_traversal() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let start = 1_000_u32;
    for index in 0..=HASHED_TABLE_ENTRY_THRESHOLD {
        runtime
            .table_set(
                table,
                &Value::Number(f64::from(start) + f64::from(u32::try_from(index).unwrap())),
                Value::Number(f64::from(u32::try_from(index).unwrap())),
            )
            .unwrap();
    }

    assert!(runtime.table(table).unwrap().entries.is_indexed());
    assert_eq!(
        runtime
            .table_get(
                table,
                &Value::Number(f64::from(
                    start + u32::try_from(HASHED_TABLE_ENTRY_THRESHOLD - 1).unwrap()
                ))
            )
            .unwrap(),
        Value::Number(f64::from(
            u32::try_from(HASHED_TABLE_ENTRY_THRESHOLD - 1).unwrap()
        ))
    );
    assert_eq!(
        runtime.table_next(table, &Value::Nil).unwrap(),
        (Value::Number(f64::from(start)), Value::Number(0.0))
    );
}

#[test]
fn arena_reuses_retired_slots_without_reviving_stale_handles() {
    let mut arena = Arena::default();
    let first = arena.insert("first").unwrap();
    assert_eq!(arena.remove(first), Some("first"));
    let second = arena.insert("second").unwrap();
    assert_eq!(first.slot, second.slot);
    assert_ne!(first.generation, second.generation);
    assert_eq!(arena.get(first), None);
    assert_eq!(arena.get(second), Some(&"second"));
}

#[test]
fn string_keys_are_found_before_anyone_hashed_them() {
    // Strings hash on first use, and ordered tables compare the stored key's
    // hash without computing it, so storing a key has to hash it.
    for length in [5_u8, 22, 120] {
        let bytes: Vec<u8> = (0..length).map(|index| b'a' + index % 26).collect();
        let mut runtime = Runtime::new();
        for entries in [4, u32::try_from(HASHED_TABLE_ENTRY_THRESHOLD).unwrap() + 4] {
            let table = runtime.create_table().unwrap();
            for index in 0..entries {
                runtime
                    .table_set(
                        table,
                        &Value::from(format!("filler{index}").as_str()),
                        Value::Number(f64::from(index)),
                    )
                    .unwrap();
            }
            let stored = LuaString::from(bytes.clone());
            assert_eq!(stored.known_hash(), 0);
            runtime
                .table_set(table, &Value::String(stored), Value::Number(-1.0))
                .unwrap();
            let probe = LuaString::from(bytes.clone());
            assert_eq!(probe.known_hash(), 0);
            assert_eq!(
                runtime.table_get_string(table, &probe).unwrap(),
                Value::Number(-1.0)
            );
            let mut other = bytes.clone();
            other[0] ^= 1;
            assert_eq!(
                runtime
                    .table_get_string(table, &LuaString::from(other))
                    .unwrap(),
                Value::Nil
            );
        }
    }
}

/// `direct_native_kind` reads a reverse index of `direct_natives`. It must
/// answer as a scan of every kind in declaration order would: nothing for
/// a plain native or one past the index, the first kind of a handle that
/// two kinds share, and nothing for a handle a kind moved away from.
#[test]
fn direct_native_kinds_follow_the_recorded_handles() {
    let mut runtime = Runtime::new();
    let plain = runtime.register_native(|_, _| Ok(Vec::new())).unwrap();
    let find = runtime.register_native(|_, _| Ok(Vec::new())).unwrap();
    let shared = runtime.register_native(|_, _| Ok(Vec::new())).unwrap();
    let later = runtime.register_native(|_, _| Ok(Vec::new())).unwrap();
    assert_eq!(runtime.direct_native_kind(plain), None);
    assert_eq!(runtime.direct_native_kind(later), None);
    runtime.set_direct_native_kind(find, DirectNativeKind::Find);
    runtime.set_direct_native_kind(shared, DirectNativeKind::Print);
    runtime.set_direct_native_kind(shared, DirectNativeKind::TableInsert);
    assert_eq!(runtime.direct_native_kind(plain), None);
    assert_eq!(
        runtime.direct_native_kind(find),
        Some(DirectNativeKind::Find)
    );
    // TableInsert is declared before Print.
    assert_eq!(
        runtime.direct_native_kind(shared),
        Some(DirectNativeKind::TableInsert)
    );
    assert_eq!(runtime.direct_native_kind(later), None);
    // `string.find` replaced: the old handle is plain again.
    runtime.set_direct_native_kind(later, DirectNativeKind::Find);
    assert_eq!(runtime.direct_native_kind(find), None);
    assert_eq!(
        runtime.direct_native_kind(later),
        Some(DirectNativeKind::Find)
    );
    runtime.set_direct_native_kind(later, DirectNativeKind::TableInsert);
    assert_eq!(
        runtime.direct_native_kind(shared),
        Some(DirectNativeKind::Print)
    );
    assert_eq!(
        runtime.direct_native_kind(later),
        Some(DirectNativeKind::Find)
    );
}
