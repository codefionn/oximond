//! Redis's bundled Lua modules, implemented without a native Lua dependency.

mod cjson;
mod cmsgpack;
mod structlib;

#[cfg(test)]
mod tests;

use crate::runtime::{Runtime, RuntimeResult, TableHandle, Value};

/// Registers a loader for each module in `package.preload`. A module is built
/// by its first `require`, which also stores it in `package.loaded`, as for a
/// module outside the interpreter. Building the five up front, with their
/// 50 natives and the tables to hold them, took 61,000 instructions of every
/// start, a fifth of the standard library's install.
pub(super) fn install(runtime: &mut Runtime, preload: TableHandle) -> RuntimeResult<()> {
    for (name, kind) in [
        ("cjson", Module::Cjson(false)),
        ("cjson.safe", Module::Cjson(true)),
        ("struct", Module::Struct),
        ("cmsgpack", Module::Cmsgpack(false)),
        ("cmsgpack_safe", Module::Cmsgpack(true)),
    ] {
        // The loader stays in `package.preload`, so clearing the cache of a
        // module lets its next require build it again.
        let callback = runtime.register_native_closure(move |runtime, _| {
            kind.create(runtime).map(|table| vec![Value::Table(table)])
        })?;
        runtime.table_set(preload, &Value::from(name), Value::NativeFunction(callback))?;
    }
    Ok(())
}

/// The `cjson.null` value, made by the first module that needs it.
fn cjson_null(runtime: &mut Runtime) -> RuntimeResult<Value> {
    // Upstream uses a NULL lightuserdata, which cannot be collected. Keep the
    // equivalent opaque value rooted even when scripts retain only a decoder.
    let registry = runtime.registry();
    let null_key = Value::from("oximond.cjson.null");
    if let null @ Value::Userdata(_) = runtime.table_get(registry, &null_key)? {
        return Ok(null);
    }
    let null = Value::Userdata(runtime.create_userdata()?);
    runtime.table_set(registry, &null_key, null.clone())?;
    Ok(null)
}

#[derive(Clone, Copy)]
enum Module {
    Cjson(bool),
    Cmsgpack(bool),
    Struct,
}

impl Module {
    fn create(self, runtime: &mut Runtime) -> RuntimeResult<TableHandle> {
        match self {
            Self::Cjson(safe) => {
                let null = cjson_null(runtime)?;
                cjson::module(runtime, null, safe)
            }
            Self::Cmsgpack(safe) => cmsgpack::module(runtime, safe),
            Self::Struct => structlib::module(runtime),
        }
    }
}

fn entries(runtime: &Runtime, table: TableHandle) -> RuntimeResult<Vec<(Value, Value)>> {
    let mut entries = Vec::new();
    let mut key = Value::Nil;
    loop {
        let (next, value) = runtime.table_next(table, &key)?;
        if matches!(next, Value::Nil) {
            return Ok(entries);
        }
        key = next.clone();
        entries.push((next, value));
    }
}

fn set(runtime: &mut Runtime, table: TableHandle, name: &str, value: Value) -> RuntimeResult<()> {
    runtime.table_set(table, &Value::from(name), value)
}
