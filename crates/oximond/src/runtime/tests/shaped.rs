//! Shaped tables against ordered tables built with the same writes.

use super::*;
use crate::runtime::TableShape;
use crate::runtime::gc::heap::table_accounted_bytes;
use crate::runtime::table_storage::key::TableKey;
use crate::string::LuaString;

const SHAPE_KEYS: [&str; 4] = ["d", "b", "a", "c"];
const OTHER_KEYS: [&str; 3] = ["aa", "e", "zz"];

fn shape() -> Arc<TableShape> {
    TableShape::new(SHAPE_KEYS.iter().map(|key| LuaString::from(*key))).unwrap()
}

struct Pair {
    runtime: Runtime,
    ordered: TableHandle,
    shaped: TableHandle,
}

impl Pair {
    fn new() -> Self {
        let mut runtime = Runtime::new();
        let ordered = runtime.create_table().unwrap();
        let shaped = runtime.create_table_with_shape(&shape()).unwrap();
        Self {
            runtime,
            ordered,
            shaped,
        }
    }

    /// Applies one write to both tables and checks it charged the same.
    fn set(&mut self, key: &Value, value: &Value) {
        let mut debts = Vec::new();
        for table in [self.ordered, self.shaped] {
            let before = self.runtime.gc_debt;
            self.runtime.table_set(table, key, value.clone()).unwrap();
            debts.push(self.runtime.gc_debt - before);
        }
        assert_eq!(debts[0], debts[1], "charge for {key:?} = {value:?}");
    }

    fn traverse(&self, table: TableHandle) -> Vec<(Value, Value)> {
        let mut out = Vec::new();
        let mut key = Value::Nil;
        loop {
            let (next, value) = self.runtime.table_next(table, &key).unwrap();
            if matches!(next, Value::Nil) {
                return out;
            }
            out.push((next.clone(), value));
            key = next;
        }
    }

    fn check(&self) {
        assert_eq!(self.traverse(self.ordered), self.traverse(self.shaped));
        let ordered = self.runtime.table(self.ordered).unwrap();
        let shaped = self.runtime.table(self.shaped).unwrap();
        assert_eq!(ordered.entries.len(), shaped.entries.len());
        assert_eq!(ordered.entries.is_empty(), shaped.entries.is_empty());
        assert_eq!(
            ordered.entries.allocated_bytes(),
            shaped.entries.allocated_bytes()
        );
        assert_eq!(
            table_accounted_bytes(ordered),
            table_accounted_bytes(shaped)
        );
        assert_eq!(ordered.border, shaped.border);
        let mut keys: Vec<Value> = SHAPE_KEYS
            .iter()
            .chain(&OTHER_KEYS)
            .map(|key| Value::from(*key))
            .collect();
        keys.extend((1..=5).map(Value::Integer));
        keys.push(Value::Number(3.5));
        keys.push(Value::Boolean(true));
        for key in &keys {
            assert_eq!(
                self.runtime.table_get(self.ordered, key),
                self.runtime.table_get(self.shaped, key),
                "get {key:?}"
            );
            assert_eq!(
                ordered.has_raw_value(key).unwrap(),
                shaped.has_raw_value(key).unwrap(),
                "contains {key:?}"
            );
            if let Value::String(string) = key {
                assert_eq!(
                    self.runtime.table_get_string_ref(self.ordered, string),
                    self.runtime.table_get_string_ref(self.shaped, string),
                );
                assert_eq!(
                    self.runtime.table_get_string_hit(self.ordered, string),
                    self.runtime.table_get_string_hit(self.shaped, string),
                );
            }
        }
        assert_eq!(
            self.runtime.table_length(self.ordered),
            self.runtime.table_length(self.shaped)
        );
    }

    fn shaped_is_shaped(&self) -> bool {
        self.runtime.table(self.shaped).unwrap().entries.is_shaped()
    }
}

#[test]
fn shape_sorts_dedups_and_limits_keys() {
    let shape = shape();
    assert_eq!(shape.len(), 4);
    let names: Vec<_> = shape
        .keys()
        .iter()
        .map(super::super::table_storage::key::TableKey::value)
        .collect();
    assert_eq!(
        names,
        ["a", "b", "c", "d"].map(Value::from).to_vec(),
        "byte order, like TableKey"
    );
    let twice = TableShape::new(["x", "x", "y"].map(LuaString::from)).unwrap();
    assert_eq!(twice.len(), 2);
    assert!(TableShape::new([]).is_none());
    let nine = (0..9).map(|index| LuaString::from(format!("k{index}").as_str()));
    assert!(TableShape::new(nine).is_none());
    let eight = (0..8).map(|index| LuaString::from(format!("k{index}").as_str()));
    assert!(TableShape::new(eight).is_some());
}

#[test]
fn creation_charges_like_create_table_and_starts_empty() {
    let mut runtime = Runtime::new();
    let before = runtime.gc_debt;
    runtime.create_table().unwrap();
    let plain = runtime.gc_debt - before;
    let before = runtime.gc_debt;
    let table = runtime.create_table_with_shape(&shape()).unwrap();
    assert_eq!(runtime.gc_debt - before, plain);
    let entries = &runtime.table(table).unwrap().entries;
    assert!(entries.is_shaped());
    assert!(entries.is_empty());
    assert_eq!(entries.len(), 0);
    assert_eq!(
        runtime.table_next(table, &Value::Nil),
        Ok((Value::Nil, Value::Nil))
    );
    for key in SHAPE_KEYS {
        assert_eq!(runtime.table_get(table, &Value::from(key)), Ok(Value::Nil));
        assert!(
            !runtime
                .table(table)
                .unwrap()
                .has_raw_value(&Value::from(key))
                .unwrap()
        );
    }
}

#[test]
fn filling_an_absent_slot_charges_one_new_key() {
    let mut pair = Pair::new();
    let key = Value::from("b");
    let before = pair.runtime.gc_debt;
    pair.runtime
        .table_set(pair.shaped, &key, Value::Integer(1))
        .unwrap();
    assert_eq!(
        pair.runtime.gc_debt - before,
        std::mem::size_of::<TableKey>() + std::mem::size_of::<Value>()
    );
    let before = pair.runtime.gc_debt;
    pair.runtime
        .table_set(pair.shaped, &key, Value::Integer(2))
        .unwrap();
    assert_eq!(pair.runtime.gc_debt, before, "overwrites are free");
    assert!(pair.shaped_is_shaped());
}

#[test]
fn string_writes_in_shape_stay_shaped_and_match_ordered() {
    let mut pair = Pair::new();
    for (index, key) in ["c", "a", "d", "b"].into_iter().enumerate() {
        pair.set(&Value::from(key), &Value::Integer(index as i64));
        pair.check();
        assert!(pair.shaped_is_shaped());
    }
    pair.set(&Value::from("a"), &Value::from("text"));
    pair.check();
    assert!(pair.shaped_is_shaped());
}

#[test]
fn out_of_shape_nil_and_non_string_writes_materialize() {
    let writes: [(Value, Value); 4] = [
        (Value::from("zz"), Value::Integer(1)),
        (Value::Integer(2), Value::Integer(1)),
        (Value::Boolean(true), Value::Integer(1)),
        (Value::from("a"), Value::Nil),
    ];
    for (key, value) in writes {
        let mut pair = Pair::new();
        pair.set(&Value::from("b"), &Value::Integer(7));
        pair.set(&Value::from("c"), &Value::Integer(8));
        pair.set(&key, &value);
        pair.check();
        pair.set(&Value::from("d"), &Value::Integer(9));
        pair.check();
        if matches!(value, Value::Nil) {
            // Storing nil to an absent slot of an empty-handed key.
            assert!(
                !pair.shaped_is_shaped()
                    || pair.runtime.table(pair.shaped).unwrap().entries.len() == 2
            );
        } else {
            assert!(!pair.shaped_is_shaped());
        }
    }
}

#[test]
fn materialization_keeps_values_and_skips_absent_slots() {
    let mut pair = Pair::new();
    pair.set(&Value::from("d"), &Value::Integer(4));
    pair.set(&Value::from("b"), &Value::from("two"));
    let table = pair.runtime.table_mut(pair.shaped).unwrap();
    table.entries.materialize();
    assert!(matches!(table.entries, TableEntries::Ordered(_)));
    assert_eq!(table.entries.len(), 2);
    pair.check();
}

#[test]
fn deleting_the_current_key_during_traversal_keeps_next_working() {
    for deleted in 0..4 {
        let mut pair = Pair::new();
        for (index, key) in SHAPE_KEYS.iter().enumerate() {
            pair.set(&Value::from(*key), &Value::Integer(index as i64 + 1));
        }
        let mut seen = [Vec::new(), Vec::new()];
        for (slot, table) in [pair.ordered, pair.shaped].into_iter().enumerate() {
            let mut key = Value::Nil;
            let mut step = 0;
            loop {
                let (next, value) = pair.runtime.table_next(table, &key).unwrap();
                if matches!(next, Value::Nil) {
                    break;
                }
                seen[slot].push((next.clone(), value));
                if step == deleted {
                    pair.runtime.table_set(table, &next, Value::Nil).unwrap();
                }
                key = next;
                step += 1;
            }
        }
        assert_eq!(seen[0], seen[1]);
        assert_eq!(seen[0].len(), 4);
        pair.check();
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % bound as u64) as usize
    }
}

#[test]
fn random_operation_sequences_match_the_ordered_representation() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..400 {
        let mut pair = Pair::new();
        for _ in 0..=rng.next(14) {
            let key = match rng.next(10) {
                0..=5 => Value::from(SHAPE_KEYS[rng.next(4)]),
                6 | 7 => Value::from(OTHER_KEYS[rng.next(3)]),
                8 => Value::Integer(rng.next(5) as i64 + 1),
                _ => [Value::Number(3.5), Value::Boolean(true)][rng.next(2)].clone(),
            };
            let value = match rng.next(6) {
                0 => Value::Nil,
                1 => Value::from("s"),
                2 => Value::Boolean(false),
                _ => Value::Integer(rng.next(100) as i64),
            };
            pair.set(&key, &value);
            pair.check();
        }
        // Whatever the sequence did, the tables still iterate the same
        // when deleting every entry during one traversal.
        for table in [pair.ordered, pair.shaped] {
            let mut key = Value::Nil;
            loop {
                let (next, _) = pair.runtime.table_next(table, &key).unwrap();
                if matches!(next, Value::Nil) {
                    break;
                }
                pair.runtime.table_set(table, &next, Value::Nil).unwrap();
                key = next;
            }
        }
        pair.check();
    }
}

#[test]
fn gc_traversal_sees_only_present_slots() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table_with_shape(&shape()).unwrap();
    let child = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::from("c"), Value::Table(child))
        .unwrap();
    let mut seen = Vec::new();
    runtime
        .table(table)
        .unwrap()
        .entries
        .for_each(|key, value| seen.push((key.value(), value.clone())));
    assert_eq!(seen, [(Value::from("c"), Value::Table(child))]);
    assert_eq!(
        runtime
            .collect_full(incremental_roots(&Value::Table(table)))
            .reclaimed(),
        0
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("c")),
        Ok(Value::Table(child))
    );
}

#[test]
fn shaped_values_lists_every_slot_in_key_order_and_only_for_shaped_tables() {
    let mut runtime = Runtime::new();
    let shaped = runtime.create_table_with_shape(&shape()).unwrap();
    let ordered = runtime.create_table().unwrap();
    runtime
        .table_set(shaped, &Value::from("c"), Value::Integer(3))
        .unwrap();
    runtime
        .table_set(shaped, &Value::from("a"), Value::Integer(1))
        .unwrap();
    runtime
        .table_set(ordered, &Value::from("a"), Value::Integer(1))
        .unwrap();
    // The shape sorts its keys: a, b, c, d. "b" and "d" hold nil.
    let values = runtime.table(shaped).unwrap().entries.shaped_values();
    assert_eq!(
        values,
        Some(&[Value::Integer(1), Value::Nil, Value::Integer(3), Value::Nil][..])
    );
    let ordered_entries = &runtime.table(ordered).unwrap().entries;
    assert!(ordered_entries.shaped_values().is_none());
}

#[test]
fn collection_keeps_every_object_a_shaped_table_holds_and_frees_the_rest() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table_with_shape(&shape()).unwrap();
    let children = (0..3)
        .map(|_| runtime.create_table().unwrap())
        .collect::<Vec<_>>();
    let orphan = runtime.create_table().unwrap();
    // Slots a, c and d hold objects; b stays nil, so the values sit apart.
    for (key, child) in ["a", "c", "d"].into_iter().zip(&children) {
        runtime
            .table_set(table, &Value::from(key), Value::Table(*child))
            .unwrap();
    }
    for _ in 0..2 {
        runtime.collect_full(incremental_roots(&Value::Table(table)));
        for (key, child) in ["a", "c", "d"].into_iter().zip(&children) {
            assert_eq!(
                runtime.table_get(table, &Value::from(key)),
                Ok(Value::Table(*child))
            );
        }
        assert!(runtime.table(orphan).is_err(), "an unreferenced table dies");
    }
}

/// The shaped variant must not grow the entry storage: every table pays for
/// it in memory and in the bytes charged at creation.
#[test]
fn shaped_variant_does_not_grow_tables() {
    assert_eq!(
        std::mem::size_of::<TableEntries>(),
        std::mem::size_of::<Vec<u8>>()
    );
}

/// A young value stored into an old record is remembered, so the next minor
/// collection keeps it.
#[test]
fn slot_stores_run_the_write_barrier() {
    let mut runtime = Runtime::new();
    let record = runtime.create_table_with_shape(&shape()).unwrap();
    let roots = incremental_roots(&Value::Table(record));
    runtime.collect_full(roots.clone());
    runtime.collect_full(roots);
    assert!(runtime.tables.is_old(record.0));
    let child = runtime.create_table().unwrap();
    assert!(!runtime.tables.is_old(child.0));
    assert_eq!(
        runtime.table_set_string_if_no_newindex(record, &LuaString::from("b"), Value::Table(child)),
        Ok(true)
    );
    runtime.collect_full(incremental_roots(&Value::Table(record)));
    assert_eq!(
        runtime.table_get(record, &Value::from("b")),
        Ok(Value::Table(child))
    );
    assert!(runtime.table(child).is_ok());
}

/// Blocks of dead record tables are reused: a new record over a recycled
/// block holds nothing of the old values, and the old values are released.
#[test]
fn recycled_blocks_start_empty_and_release_their_values() {
    for convert in [false, true] {
        let mut runtime = Runtime::new();
        let held = LuaString::from("held by a dead record");
        let weak = held.downgrade();
        let record = runtime.create_table_with_shape(&shape()).unwrap();
        for key in SHAPE_KEYS {
            assert_eq!(
                runtime.table_set_string_if_no_newindex(
                    record,
                    &LuaString::from(key),
                    Value::String(held.clone())
                ),
                Ok(true)
            );
        }
        drop(held);
        if convert {
            runtime
                .table_set(record, &Value::from("not in shape"), Value::Integer(1))
                .unwrap();
            assert!(!runtime.table(record).unwrap().entries.is_shaped());
        }
        assert!(weak.is_alive());
        runtime.collect_full(incremental_roots(&Value::Nil));
        assert!(runtime.table(record).is_err());
        assert!(!weak.is_alive());
        for _ in 0..3 {
            let fresh = runtime.create_table_with_shape(&shape()).unwrap();
            for key in SHAPE_KEYS {
                assert_eq!(runtime.table_get(fresh, &Value::from(key)), Ok(Value::Nil));
            }
            assert!(runtime.table(fresh).unwrap().entries.is_empty());
        }
    }
}

/// The block pool keeps a bounded number of dead record blocks and frees the
/// rest, so memory records released does not stay with the pool.
#[test]
fn dead_record_blocks_are_pooled_up_to_a_cap() {
    use crate::runtime::table_storage::entries::block_pool_listed;
    let mut runtime = Runtime::new();
    let keys = SHAPE_KEYS.len();
    let count = 4 * crate::runtime::table_storage::entries::BLOCK_POOL_CAP;
    for _ in 0..count {
        runtime.create_table_with_shape(&shape()).unwrap();
    }
    runtime.collect_full(incremental_roots(&Value::Nil));
    let listed = block_pool_listed(keys);
    assert!(
        listed
            <= crate::runtime::table_storage::entries::BLOCK_POOL_CAP
                + crate::runtime::table_storage::entries::BLOCK_REGION_BLOCKS
    );
    assert!(listed > 0);
    // Blocks taken from the list are listed again once their tables die.
    for _ in 0..8 {
        runtime.create_table_with_shape(&shape()).unwrap();
    }
    runtime.collect_full(incremental_roots(&Value::Nil));
    assert!(
        block_pool_listed(keys)
            <= crate::runtime::table_storage::entries::BLOCK_POOL_CAP
                + crate::runtime::table_storage::entries::BLOCK_REGION_BLOCKS
    );
}

/// A shaped metatable whose `__newindex` slot is empty, with a table that
/// uses it.
fn shaped_metatable_pair(
    runtime: &mut Runtime,
    shape: &Arc<TableShape>,
) -> (TableHandle, TableHandle) {
    let metatable = runtime.create_table_with_shape(shape).unwrap();
    let table = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    (table, metatable)
}

#[test]
fn shaped_slot_fill_of_newindex_ends_raw_stores() {
    let mut runtime = Runtime::new();
    let meta_shape = TableShape::new(["__newindex", "x"].map(LuaString::from)).unwrap();
    let (table, metatable) = shaped_metatable_pair(&mut runtime, &meta_shape);
    let name = LuaString::from("a");
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &name, Value::Integer(1)),
        Ok(true)
    );
    // Fills the empty slot of the shape, as a constructor's store would.
    let key = LuaString::from("__newindex");
    assert_eq!(
        runtime.table_set_string_if_no_newindex(metatable, &key, Value::Boolean(true)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &LuaString::from("b"), Value::Integer(1)),
        Ok(false)
    );
    assert_eq!(
        runtime.table_set_if_no_newindex(table, &Value::Integer(3), Value::Integer(1)),
        Ok(false)
    );
}

/// Fills the write site of pc 0 of `chunk` for `key` from a donor table of
/// `shape` that has the key set. The site then serves every table of the
/// shape, including the ones that lack the field.
fn filled_site(
    runtime: &mut Runtime,
    chunk: &crate::bytecode::Chunk,
    shape: &Arc<TableShape>,
    key: &str,
) {
    let donor = runtime.create_table_with_shape(shape).unwrap();
    runtime
        .table_set(donor, &Value::from(key), Value::Integer(0))
        .unwrap();
    runtime.table_site_fill_store(donor, chunk, 0, &LuaString::from(key));
    assert!(chunk.store_site(0).unwrap().load_pair().is_some());
}

fn site_chunk() -> crate::bytecode::Chunk {
    crate::bytecode::Chunk::new(1, Vec::new(), vec![Instruction::return_(0)])
}

#[test]
fn site_store_adds_to_a_shaped_table_and_clears_the_cache() {
    // The site store is a writer that fills a slot without `table_mut`: it
    // must clear the absence bits itself.
    let mut runtime = Runtime::new();
    let meta_shape = TableShape::new(["__newindex", "x"].map(LuaString::from)).unwrap();
    let (table, metatable) = shaped_metatable_pair(&mut runtime, &meta_shape);
    // Cache the absence of __newindex in the shaped metatable.
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &LuaString::from("a"), Value::Integer(1)),
        Ok(true)
    );
    let chunk = site_chunk();
    filled_site(&mut runtime, &chunk, &meta_shape, "__newindex");
    let site = chunk.store_site(0).unwrap();
    assert_eq!(
        runtime.table_site_store(metatable, site, &Value::Boolean(true)),
        SiteStore::Added
    );
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &LuaString::from("b"), Value::Integer(1)),
        Ok(false)
    );
    assert_eq!(
        runtime.table_set_if_no_newindex(table, &Value::from("c"), Value::Integer(1)),
        Ok(false)
    );
}

#[test]
fn site_store_into_a_table_with_a_metatable_follows_newindex() {
    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    let chunk = site_chunk();
    let shape = shape();
    let mut objects = Vec::new();
    for _ in 0..3 {
        let object = runtime.create_table_with_shape(&shape).unwrap();
        runtime
            .raw_set_metatable(&Value::Table(object), Some(metatable))
            .unwrap();
        objects.push(object);
    }
    filled_site(&mut runtime, &chunk, &shape, "a");
    let site = chunk.store_site(0).unwrap();
    // No __newindex: the add is raw.
    assert_eq!(
        runtime.table_site_store(objects[0], site, &Value::Integer(1)),
        SiteStore::Added
    );
    assert_eq!(
        runtime.table_site_store(objects[0], site, &Value::Integer(2)),
        SiteStore::Overwrote
    );
    assert_eq!(
        runtime.table_site_store(objects[1], site, &Value::Integer(1)),
        SiteStore::Added
    );
    // After __newindex appears the site must decline an absent field.
    runtime
        .table_set(metatable, &Value::from("__newindex"), Value::Boolean(true))
        .unwrap();
    assert_eq!(
        runtime.table_site_store(objects[2], site, &Value::Integer(1)),
        SiteStore::Missed
    );
    // A present field is still overwritten raw.
    assert_eq!(
        runtime.table_site_store(objects[1], site, &Value::Integer(5)),
        SiteStore::Overwrote
    );
    // And it is raw again once __newindex is removed.
    runtime
        .table_set(metatable, &Value::from("__newindex"), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_site_store(objects[2], site, &Value::Integer(1)),
        SiteStore::Added
    );
}

#[test]
fn site_store_into_a_self_metatable_shaped_table_declines_after_newindex() {
    let mut runtime = Runtime::new();
    let shape = TableShape::new(["__newindex", "x"].map(LuaString::from)).unwrap();
    let table = runtime.create_table_with_shape(&shape).unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(table))
        .unwrap();
    let chunk = site_chunk();
    filled_site(&mut runtime, &chunk, &shape, "x");
    let site = chunk.store_site(0).unwrap();
    // Fills "x" through the metatable check, which caches the absence of
    // __newindex in this very table.
    assert_eq!(
        runtime.table_site_store(table, site, &Value::Integer(1)),
        SiteStore::Added
    );
    runtime
        .table_set(table, &Value::from("__newindex"), Value::Boolean(true))
        .unwrap();
    runtime
        .table_set(table, &Value::from("x"), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &LuaString::from("y"), Value::Integer(1)),
        Ok(false)
    );
}

#[test]
fn site_store_add_to_a_watched_table_bumps_the_inherit_epoch() {
    let mut runtime = Runtime::new();
    let shape = shape();
    let table = runtime.create_table_with_shape(&shape).unwrap();
    let chunk = site_chunk();
    filled_site(&mut runtime, &chunk, &shape, "a");
    let site = chunk.store_site(0).unwrap();
    runtime.tables.get_mut(table.0).unwrap().watched = true;
    let before = runtime.inherit_epoch;
    assert_eq!(
        runtime.table_site_store(table, site, &Value::Integer(1)),
        SiteStore::Added
    );
    assert!(runtime.inherit_epoch > before);
}

#[test]
fn site_store_declines_nil_and_readonly_but_overwrites_a_present_field() {
    let mut runtime = Runtime::new();
    let shape = shape();
    let table = runtime.create_table_with_shape(&shape).unwrap();
    let chunk = site_chunk();
    filled_site(&mut runtime, &chunk, &shape, "a");
    let site = chunk.store_site(0).unwrap();
    assert_eq!(
        runtime.table_site_store(table, site, &Value::Nil),
        SiteStore::Missed
    );
    assert_eq!(
        runtime.table_site_store(table, site, &Value::Integer(1)),
        SiteStore::Added
    );
    assert_eq!(
        runtime.table_site_store(table, site, &Value::Nil),
        SiteStore::Missed
    );
    runtime.table_set_readonly(table, true).unwrap();
    assert_eq!(
        runtime.table_site_store(table, site, &Value::Integer(2)),
        SiteStore::Missed
    );
    runtime.table_set_readonly(table, false).unwrap();
    assert_eq!(
        runtime.table_site_store(table, site, &Value::Integer(2)),
        SiteStore::Overwrote
    );
}

#[test]
fn site_store_does_not_add_a_deleted_key_of_an_unshaped_table() {
    // A nil slot of an ordered table is a deleted key, which the general
    // store re-adds; only a shaped table's nil slot is an add.
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::from("a"), Value::Integer(1))
        .unwrap();
    let chunk = site_chunk();
    runtime.table_site_fill_store(table, &chunk, 0, &LuaString::from("a"));
    let site = chunk.store_site(0).unwrap();
    assert!(site.load_pair().is_some());
    assert_eq!(
        runtime.table_site_store(table, site, &Value::Integer(2)),
        SiteStore::Overwrote
    );
    runtime
        .table_set(table, &Value::from("a"), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_site_store(table, site, &Value::Integer(3)),
        SiteStore::Missed
    );
}

#[test]
fn tables_over_a_pinned_shape_borrow_its_count_and_leave_it_balanced() {
    let shape = shape();
    assert!(shape.pin());
    assert!(shape.is_pinned());
    // Pinning is idempotent and leaves one count behind for good.
    assert!(!shape.pin());
    let pinned = Arc::strong_count(&shape);
    assert_eq!(pinned, 2);

    let mut runtime = Runtime::new();
    let tables: Vec<_> = (0..64)
        .map(|_| runtime.create_table_with_shape(&shape).unwrap())
        .collect();
    assert_eq!(Arc::strong_count(&shape), pinned);
    for (index, table) in tables.iter().enumerate() {
        runtime
            .table_set(*table, &Value::from("a"), Value::Integer(index as i64))
            .unwrap();
    }
    // Materializing a table clones and drops a real count.
    runtime
        .table_set(tables[0], &Value::from("extra"), Value::Integer(1))
        .unwrap();
    assert_eq!(Arc::strong_count(&shape), pinned);
    for (index, table) in tables.iter().enumerate() {
        assert_eq!(
            runtime.table_get(*table, &Value::from("a")).unwrap(),
            Value::Integer(index as i64)
        );
    }
    // Unreachable, all of them are freed together with their borrowed counts.
    runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(Arc::strong_count(&shape), pinned);
    drop(runtime);
    assert_eq!(Arc::strong_count(&shape), pinned);

    // An unpinned shape is counted per table and released with it.
    let plain = shape_of(&["p", "q"]);
    let mut runtime = Runtime::new();
    let table = runtime.create_table_with_shape(&plain).unwrap();
    assert_eq!(Arc::strong_count(&plain), 2);
    let _ = table;
    runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(Arc::strong_count(&plain), 1);
}

fn shape_of(keys: &[&str]) -> Arc<TableShape> {
    TableShape::new(keys.iter().map(|key| LuaString::from(*key))).unwrap()
}

#[test]
fn site_store_of_a_closure_over_a_closure_drops_the_cached_handler() {
    // A watched metatable's `__index` closure is overwritten through a write
    // site, which must advance the epoch the cached handler depends on.
    let mut runtime = Runtime::new();
    let chunk = Arc::new(site_chunk());
    let first = runtime
        .create_closure(Arc::clone(&chunk), Vec::new())
        .unwrap();
    let second = runtime.create_closure(chunk, Vec::new()).unwrap();
    let meta_shape = TableShape::new(["__index"].map(LuaString::from)).unwrap();
    let (object, metatable) = shaped_metatable_pair(&mut runtime, &meta_shape);
    runtime
        .table_set(metatable, &Value::from("__index"), Value::Closure(first))
        .unwrap();
    let key = LuaString::from("field");
    assert_eq!(runtime.table_inherited_resolve(object, &key), None);
    assert_eq!(runtime.table_inherited_handler(object, &key), Some(first));

    let store_chunk = site_chunk();
    filled_site(&mut runtime, &store_chunk, &meta_shape, "__index");
    let site = store_chunk.store_site(0).unwrap();
    for value in [second, first, second] {
        assert_eq!(
            runtime.table_site_store(metatable, site, &Value::Closure(value)),
            SiteStore::Overwrote
        );
        assert_eq!(runtime.table_inherited_handler(object, &key), None);
        assert_eq!(runtime.table_inherited_resolve(object, &key), None);
        assert_eq!(runtime.table_inherited_handler(object, &key), Some(value));
    }
}

/// The creation native code uses gives the table `create_table_with_shape`
/// gives and the slots `table_shaped_slots` reports for it: the same
/// charge, the same accounted bytes, and every key absent.
#[test]
fn a_table_made_with_its_slots_is_the_table_made_without_them() {
    let mut runtime = Runtime::new();
    let debt = runtime.gc_debt;
    let plain = runtime.create_table_with_shape(&shape()).unwrap();
    let charged = runtime.gc_debt - debt;
    let debt = runtime.gc_debt;
    let (made, slots) = runtime.create_table_with_shape_slots(&shape()).unwrap();
    assert_eq!(runtime.gc_debt - debt, charged);
    assert_eq!(runtime.table_shaped_slots(made), Some(slots));
    assert_ne!(slots, 0);
    assert_eq!(
        table_accounted_bytes(runtime.tables.get(made.0).unwrap()),
        table_accounted_bytes(runtime.tables.get(plain.0).unwrap())
    );
    for key in SHAPE_KEYS {
        assert_eq!(
            runtime.table_get(made, &Value::from(key)).unwrap(),
            Value::Nil
        );
    }
    assert_eq!(
        runtime.table_next(made, &Value::Nil).unwrap(),
        (Value::Nil, Value::Nil)
    );
    // Writes behave as on the other table, in the order stored.
    for (index, key) in SHAPE_KEYS.iter().enumerate() {
        for table in [plain, made] {
            runtime
                .table_set(table, &Value::from(*key), Value::Integer(index as i64))
                .unwrap();
        }
    }
    let mut after = Value::Nil;
    let mut met = 0;
    loop {
        let next = runtime.table_next(plain, &after).unwrap();
        assert_eq!(runtime.table_next(made, &after).unwrap(), next);
        if next.0 == Value::Nil {
            break;
        }
        met += 1;
        after = next.0;
    }
    assert_eq!(met, SHAPE_KEYS.len());
}

/// The sweep skips weighing a shaped table's entries, on the bound that
/// they never count more than `SHAPED_MAX_ALLOCATED_BYTES`: a shape of the
/// most keys with every key stored stays within it.
#[test]
fn a_full_shaped_table_stays_within_the_bound_the_sweep_assumes() {
    use crate::runtime::table_storage::entries::SHAPED_MAX_ALLOCATED_BYTES;
    use crate::runtime::table_storage::shape::MAX_SHAPE_KEYS;
    let keys: Vec<String> = (0..MAX_SHAPE_KEYS)
        .map(|index| format!("k{index}"))
        .collect();
    let shape = TableShape::new(keys.iter().map(|key| LuaString::from(key.as_str()))).unwrap();
    let mut runtime = Runtime::new();
    let table = runtime.create_table_with_shape(&shape).unwrap();
    let weigh = |runtime: &Runtime| {
        let entries = &runtime.tables.get(table.0).unwrap().entries;
        (entries.is_shaped(), entries.allocated_bytes())
    };
    assert!(weigh(&runtime).1 <= SHAPED_MAX_ALLOCATED_BYTES);
    for key in &keys {
        runtime
            .table_set(table, &Value::from(key.as_str()), Value::Integer(1))
            .unwrap();
        let (shaped, bytes) = weigh(&runtime);
        assert!(shaped);
        assert!(bytes <= SHAPED_MAX_ALLOCATED_BYTES, "{bytes}");
    }
    assert!(!crate::runtime::table_reclaim::eligible(
        runtime.tables.get(table.0).unwrap()
    ));
}
