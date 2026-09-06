#![forbid(unsafe_code)]

//! Lua-facing native FFI extension. Native declarations are trusted input.
//!
//! The parser accepts scalar and pointer function declarations. Fixed scalar
//! arrays can be allocated with `ffi.new`. Structs, unions, callbacks, C
//! variables, and `LuaJIT` metatypes are not implemented; unsupported
//! declarations return an error from `ffi.cdef`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::ffi_native::{FfiArray, FfiEngine, FfiScalar, FfiType, FfiValue, Library};
use crate::runtime::{Runtime, RuntimeError, RuntimeResult, TableHandle, UserdataHandle, Value};

#[derive(Default)]
struct Cdata {
    values: HashMap<UserdataHandle, CdataValue>,
    types: HashMap<UserdataHandle, String>,
    metatable: Option<TableHandle>,
}

enum CdataValue {
    CType(String),
    Pointer(usize),
    Scalar(FfiScalar),
    Array(FfiArray),
}

// Call sites pass `format!` strings and literals; taking it by value keeps them simple.
#[allow(clippy::needless_pass_by_value)]
fn error(message: impl ToString) -> RuntimeError {
    RuntimeError::message(message.to_string())
}

fn string_argument(value: Option<&Value>, name: &str) -> RuntimeResult<String> {
    match value {
        Some(Value::String(value)) => String::from_utf8(value.as_bytes().to_vec())
            .map_err(|_| error(format!("{name} requires a UTF-8 string"))),
        Some(value) => Err(error(format!(
            "{name} requires a string, got {}",
            value.type_name()
        ))),
        None => Err(error(format!("{name} requires a string"))),
    }
}

fn type_argument(value: Option<&Value>, data: &Cdata, name: &str) -> RuntimeResult<String> {
    if let Some(Value::Userdata(handle)) = value
        && let Some(CdataValue::CType(ctype)) = data.values.get(handle)
    {
        return Ok(ctype.clone());
    }
    string_argument(value, name)
}

fn cdata_type(value: Option<&Value>, data: &Cdata, name: &str) -> RuntimeResult<String> {
    let Some(Value::Userdata(handle)) = value else {
        return Err(error(format!("{name} requires FFI cdata")));
    };
    if matches!(data.values.get(handle), Some(CdataValue::CType(_))) {
        return Err(error(format!("{name} requires FFI cdata")));
    }
    data.types
        .get(handle)
        .cloned()
        .ok_or_else(|| error(format!("{name} requires FFI cdata")))
}

fn owned_capacity(value: &Value, data: &Cdata, engine: &FfiEngine) -> RuntimeResult<Option<usize>> {
    let Value::Userdata(handle) = value else {
        return Ok(None);
    };
    let Some(CdataValue::Array(_)) = data.values.get(handle) else {
        return Ok(None);
    };
    let name = data
        .types
        .get(handle)
        .ok_or_else(|| error("missing FFI array type"))?;
    Ok(Some(engine.sizeof_type(name).map_err(error)?))
}

fn check_owned_capacity(
    value: &Value,
    length: usize,
    data: &Cdata,
    engine: &FfiEngine,
    name: &str,
) -> RuntimeResult<()> {
    if owned_capacity(value, data, engine)?.is_some_and(|capacity| length > capacity) {
        return Err(error(format!("{name} length exceeds array bounds")));
    }
    Ok(())
}

fn same_ctype(engine: &FfiEngine, left: &str, right: &str) -> bool {
    if left.split_whitespace().collect::<String>() == right.split_whitespace().collect::<String>() {
        return true;
    }
    if left.contains('[') || right.contains('[') {
        let (Some((left_base, left_count)), Some((right_base, right_count))) =
            (left.rsplit_once('['), right.rsplit_once('['))
        else {
            return false;
        };
        return left_count.trim() == right_count.trim()
            && same_ctype(engine, left_base.trim(), right_base.trim());
    }
    if left.contains('*') && right.contains('*') {
        return false;
    }
    match (engine.type_of(left), engine.type_of(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn is_byte_array_type(engine: &FfiEngine, name: &str) -> bool {
    name.split_once('[')
        .and_then(|(base, _)| engine.type_of(base.trim()).ok())
        .is_some_and(|ty| matches!(ty, FfiType::Char | FfiType::UChar))
}

fn integer_argument(value: Option<&Value>, name: &str) -> RuntimeResult<i64> {
    match value {
        Some(Value::Integer(value)) => Ok(*value),
        Some(Value::Number(value))
            if value.is_finite()
                && value.fract() == 0.0
                && *value >= i64::MIN as f64
                && *value < 9_223_372_036_854_775_808.0 =>
        {
            Ok(*value as i64)
        }
        _ => Err(error(format!("{name} requires an integer"))),
    }
}

fn length_argument(value: Option<&Value>, name: &str) -> RuntimeResult<usize> {
    usize::try_from(integer_argument(value, name)?)
        .map_err(|_| error(format!("{name} requires a nonnegative length")))
}

fn cdata_address(value: Option<&Value>, data: &Cdata, name: &str) -> RuntimeResult<usize> {
    match value {
        Some(Value::Userdata(handle)) => match data.values.get(handle) {
            Some(CdataValue::Pointer(pointer)) => Ok(*pointer),
            Some(CdataValue::Scalar(scalar)) => Ok(scalar.pointer()),
            Some(CdataValue::Array(array)) => Ok(array.pointer()),
            Some(CdataValue::CType(_)) | None => Err(error(format!("{name} requires FFI cdata"))),
        },
        _ => Err(error(format!("{name} requires FFI cdata"))),
    }
}

fn to_ffi(value: &Value, data: &Cdata) -> RuntimeResult<FfiValue> {
    Ok(match value {
        Value::Nil => FfiValue::Pointer(0),
        Value::Integer(value) | Value::Int64(value) => FfiValue::Integer(*value),
        Value::Number(value) => FfiValue::Double(*value),
        Value::UInt64(value) => FfiValue::Unsigned(*value),
        Value::String(value) => FfiValue::CString(
            String::from_utf8(value.as_bytes().to_vec())
                .map_err(|_| error("FFI strings must be UTF-8"))?,
        ),
        Value::Userdata(handle) => match data.values.get(handle) {
            Some(CdataValue::Pointer(pointer)) => FfiValue::Pointer(*pointer),
            Some(CdataValue::Scalar(scalar)) => FfiValue::Pointer(scalar.pointer()),
            Some(CdataValue::Array(array)) => FfiValue::Pointer(array.pointer()),
            Some(CdataValue::CType(_)) => return Err(error("a C type is not an FFI value")),
            None => return Err(error("userdata is not FFI cdata")),
        },
        other => {
            return Err(error(format!(
                "unsupported FFI argument: {}",
                other.type_name()
            )));
        }
    })
}

fn to_ffi_typed(value: &Value, ty: Option<FfiType>, data: &Cdata) -> RuntimeResult<FfiValue> {
    if let Value::Number(number) = value
        && matches!(
            ty,
            Some(
                FfiType::Char
                    | FfiType::UChar
                    | FfiType::Short
                    | FfiType::UShort
                    | FfiType::Int
                    | FfiType::UInt
                    | FfiType::Long
                    | FfiType::ULong
                    | FfiType::LongLong
                    | FfiType::ULongLong
                    | FfiType::SizeT
            )
        )
        && number.is_finite()
        && number.fract() == 0.0
        && *number >= i64::MIN as f64
        && *number < 9_223_372_036_854_775_808.0
    {
        return Ok(FfiValue::Integer(*number as i64));
    }
    if let Value::Userdata(handle) = value
        && let Some(CdataValue::Scalar(scalar)) = data.values.get(handle)
        && !matches!(ty, Some(FfiType::Pointer | FfiType::CString))
    {
        // A scalar passes its value, to a variadic parameter too, as in LuaJIT.
        return scalar
            .value()
            .map_err(error)
            .map(|value| to_float(value, ty));
    }
    to_ffi(value, data).map(|value| to_float(value, ty))
}

/// Converts an integer for a float or double parameter, as C does.
#[allow(clippy::cast_precision_loss)]
fn to_float(value: FfiValue, ty: Option<FfiType>) -> FfiValue {
    if !matches!(ty, Some(FfiType::Float | FfiType::Double)) {
        return value;
    }
    match value {
        FfiValue::Integer(n) => FfiValue::Double(n as f64),
        FfiValue::Unsigned(n) => FfiValue::Double(n as f64),
        other => other,
    }
}

fn make_cdata(
    runtime: &mut Runtime,
    data: &Rc<RefCell<Cdata>>,
    value: CdataValue,
    ctype: &str,
) -> RuntimeResult<Value> {
    let handle = runtime.create_userdata()?;
    let metatable = data.borrow().metatable;
    data.borrow_mut().values.insert(handle, value);
    data.borrow_mut().types.insert(handle, ctype.to_string());
    runtime.raw_set_metatable(&Value::Userdata(handle), metatable)?;
    Ok(Value::Userdata(handle))
}

fn set_array_value(
    array: &mut FfiArray,
    index: usize,
    value: &Value,
    data: &Cdata,
) -> RuntimeResult<()> {
    let input = to_ffi(value, data)?;
    array
        .set(index, &input)
        .or_else(|first| {
            if let Value::Number(number) = value
                && number.is_finite()
                && number.fract() == 0.0
                && *number >= i64::MIN as f64
                && *number < 9_223_372_036_854_775_808.0
            {
                array.set(index, &FfiValue::Integer(*number as i64))
            } else {
                Err(first)
            }
        })
        .map_err(error)
}

fn from_ffi(
    runtime: &mut Runtime,
    value: FfiValue,
    data: &Rc<RefCell<Cdata>>,
    ctype: Option<&str>,
) -> RuntimeResult<Value> {
    Ok(match value {
        FfiValue::Void => Value::Nil,
        FfiValue::Integer(value) => Value::Integer(value),
        FfiValue::Unsigned(value) => {
            i64::try_from(value).map_or(Value::Number(value as f64), Value::Integer)
        }
        FfiValue::Float(value) => Value::Number(f64::from(value)),
        FfiValue::Double(value) => Value::Number(value),
        FfiValue::CString(value) => Value::from(value.as_str()),
        FfiValue::Pointer(value) => make_cdata(
            runtime,
            data,
            CdataValue::Pointer(value),
            ctype.unwrap_or("void *"),
        )?,
    })
}

fn namespace(
    runtime: &mut Runtime,
    engine: Rc<RefCell<FfiEngine>>,
    data: Rc<RefCell<Cdata>>,
    library: Option<Rc<Library>>,
) -> RuntimeResult<Value> {
    let table = runtime.create_table()?;
    let metatable = runtime.create_table()?;
    let lookup = runtime.register_native_closure(move |runtime, args| {
        let name = string_argument(args.get(1), "FFI symbol name")?;
        if engine.borrow().declaration(&name).is_none() {
            return Err(error(format!("undeclared C symbol: {name}")));
        }
        let engine = engine.clone();
        let data = data.clone();
        let library = library.clone();
        let symbol = name.clone();
        let function = runtime.register_native_closure(move |runtime, args| {
            let declaration = engine
                .borrow()
                .declaration(&symbol)
                .cloned()
                .ok_or_else(|| error(format!("undeclared C symbol: {symbol}")))?;
            let values = args
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    to_ffi_typed(value, declaration.args.get(index).copied(), &data.borrow())
                })
                .collect::<RuntimeResult<Vec<_>>>()?;
            let result = if let Some(library) = &library {
                engine.borrow().call_in(library, &symbol, &values)
            } else {
                engine.borrow().call(&symbol, &values)
            }
            .map_err(error)?;
            let result_type = match declaration.result {
                FfiType::CString => Some("const char *"),
                FfiType::Pointer => Some("void *"),
                _ => None,
            };
            Ok(vec![from_ffi(runtime, result, &data, result_type)?])
        })?;
        let value = Value::NativeFunction(function);
        if let Some(Value::Table(namespace)) = args.first() {
            runtime.table_set(*namespace, &Value::from(name.as_str()), value.clone())?;
        }
        Ok(vec![value])
    })?;
    runtime.table_set(
        metatable,
        &Value::from("__index"),
        Value::NativeFunction(lookup),
    )?;
    runtime.raw_set_metatable(&Value::Table(table), Some(metatable))?;
    Ok(Value::Table(table))
}

/// Install LuaJIT-style `ffi` backed by libffi and the process dynamic loader.
/// C declarations and native calls are trusted and can crash the host process.
pub(crate) fn install_ffi(runtime: &mut Runtime) -> RuntimeResult<()> {
    let engine = Rc::new(RefCell::new(FfiEngine::new().map_err(error)?));
    let data = Rc::new(RefCell::new(Cdata::default()));
    let cdata_metatable = runtime.create_table()?;
    let gc_data = data.clone();
    let gc = runtime.register_native_closure(move |_, args| {
        if let Some(Value::Userdata(handle)) = args.first() {
            gc_data.borrow_mut().values.remove(handle);
            gc_data.borrow_mut().types.remove(handle);
        }
        Ok(Vec::new())
    })?;
    runtime.table_set(
        cdata_metatable,
        &Value::from("__gc"),
        Value::NativeFunction(gc),
    )?;
    let tostring_data = data.clone();
    let tostring = runtime.register_native_closure(move |_, args| {
        let Value::Userdata(handle) = args.first().ok_or_else(|| error("FFI cdata expected"))?
        else {
            return Err(error("FFI cdata expected"));
        };
        let data = tostring_data.borrow();
        let value = data
            .values
            .get(handle)
            .ok_or_else(|| error("FFI cdata expected"))?;
        let ty = data.types.get(handle).map_or("void *", String::as_str);
        let description = match value {
            CdataValue::CType(_) => format!("ctype<{ty}>"),
            CdataValue::Pointer(pointer) => format!("cdata<{ty}>: 0x{pointer:x}"),
            CdataValue::Scalar(scalar) => {
                format!("cdata<{ty}>: {:?}", scalar.value().map_err(error)?)
            }
            CdataValue::Array(array) => format!("cdata<{ty}>: 0x{:x}", array.pointer()),
        };
        Ok(vec![Value::from(description.as_str())])
    })?;
    runtime.table_set(
        cdata_metatable,
        &Value::from("__tostring"),
        Value::NativeFunction(tostring),
    )?;
    let eq_data = data.clone();
    let eq = runtime.register_native_closure(move |_, args| {
        let (Some(Value::Userdata(left)), Some(Value::Userdata(right))) =
            (args.first(), args.get(1))
        else {
            return Ok(vec![Value::Boolean(false)]);
        };
        let data = eq_data.borrow();
        let equal = match (data.values.get(left), data.values.get(right)) {
            (Some(CdataValue::Pointer(a)), Some(CdataValue::Pointer(b))) => a == b,
            (Some(CdataValue::CType(a)), Some(CdataValue::CType(b))) => a == b,
            (Some(CdataValue::Scalar(a)), Some(CdataValue::Scalar(b))) => {
                data.types.get(left) == data.types.get(right) && a.value().ok() == b.value().ok()
            }
            (Some(CdataValue::Array(a)), Some(CdataValue::Array(b))) => a.pointer() == b.pointer(),
            _ => false,
        };
        Ok(vec![Value::Boolean(equal)])
    })?;
    runtime.table_set(
        cdata_metatable,
        &Value::from("__eq"),
        Value::NativeFunction(eq),
    )?;
    let index_data = data.clone();
    let index = runtime.register_native_closure(move |runtime, args| {
        let Value::Userdata(handle) = args.first().ok_or_else(|| error("FFI array expected"))?
        else {
            return Err(error("FFI array expected"));
        };
        let index = length_argument(args.get(1), "FFI array index")?;
        let value = {
            let data = index_data.borrow();
            let Some(CdataValue::Array(array)) = data.values.get(handle) else {
                return Err(error("FFI cdata is not an array"));
            };
            array.get(index).map_err(error)?
        };
        Ok(vec![from_ffi(runtime, value, &index_data, None)?])
    })?;
    runtime.table_set(
        cdata_metatable,
        &Value::from("__index"),
        Value::NativeFunction(index),
    )?;
    let newindex_data = data.clone();
    let newindex = runtime.register_native_closure(move |_, args| {
        let Value::Userdata(handle) = args.first().ok_or_else(|| error("FFI array expected"))?
        else {
            return Err(error("FFI array expected"));
        };
        let index = length_argument(args.get(1), "FFI array index")?;
        let input = args
            .get(2)
            .ok_or_else(|| error("FFI array assignment requires a value"))?;
        let mut data = newindex_data.borrow_mut();
        let value = to_ffi(input, &data)?;
        let Some(CdataValue::Array(array)) = data.values.get_mut(handle) else {
            return Err(error("FFI cdata is not an array"));
        };
        array
            .set(index, &value)
            .or_else(|first| {
                if let Value::Number(number) = input
                    && number.is_finite()
                    && number.fract() == 0.0
                    && *number >= i64::MIN as f64
                    && *number < 9_223_372_036_854_775_808.0
                {
                    array.set(index, &FfiValue::Integer(*number as i64))
                } else {
                    Err(first)
                }
            })
            .map_err(error)?;
        Ok(Vec::new())
    })?;
    runtime.table_set(
        cdata_metatable,
        &Value::from("__newindex"),
        Value::NativeFunction(newindex),
    )?;
    // Native closures hold only Rust handles. The Lua collector cannot trace
    // through them, so keep the cdata metatable in the runtime registry.
    runtime.table_set(
        runtime.registry(),
        &Value::from("oximond.ffi.cdata_metatable"),
        Value::Table(cdata_metatable),
    )?;
    data.borrow_mut().metatable = Some(cdata_metatable);
    let ffi = runtime.create_table()?;
    let os = if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "macos") {
        "OSX"
    } else if cfg!(target_os = "windows") {
        "Windows"
    } else if cfg!(target_os = "freebsd") {
        "BSD"
    } else {
        "Other"
    };
    let arch = if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "x86") {
        "x86"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "arm") {
        "arm"
    } else {
        "other"
    };
    runtime.table_set(ffi, &Value::from("os"), Value::from(os))?;
    runtime.table_set(ffi, &Value::from("arch"), Value::from(arch))?;
    let c = namespace(runtime, engine.clone(), data.clone(), None)?;
    runtime.table_set(ffi, &Value::from("C"), c)?;

    let cdef_engine = engine.clone();
    let cdef = runtime.register_native_closure(move |_, args| {
        let declaration = string_argument(args.first(), "ffi.cdef")?;
        cdef_engine.borrow_mut().cdef(&declaration).map_err(error)?;
        Ok(Vec::new())
    })?;
    runtime.table_set(ffi, &Value::from("cdef"), Value::NativeFunction(cdef))?;

    let typeof_engine = engine.clone();
    let typeof_data = data.clone();
    let type_of = runtime.register_native_closure(move |runtime, args| {
        let name = if let Some(Value::Userdata(handle)) = args.first() {
            typeof_data
                .borrow()
                .types
                .get(handle)
                .cloned()
                .ok_or_else(|| error("ffi.typeof requires a C type or FFI cdata"))?
        } else {
            string_argument(args.first(), "ffi.typeof")?
        };
        if name.contains('[') {
            typeof_engine.borrow().sizeof_type(&name).map_err(error)?;
        } else {
            typeof_engine.borrow().type_of(&name).map_err(error)?;
        }
        Ok(vec![make_cdata(
            runtime,
            &typeof_data,
            CdataValue::CType(name.clone()),
            &name,
        )?])
    })?;
    runtime.table_set(ffi, &Value::from("typeof"), Value::NativeFunction(type_of))?;

    let istype_engine = engine.clone();
    let istype_data = data.clone();
    let istype = runtime.register_native_closure(move |_, args| {
        let name = type_argument(args.first(), &istype_data.borrow(), "ffi.istype")?;
        if name.contains('[') {
            istype_engine.borrow().sizeof_type(&name).map_err(error)?;
        } else {
            istype_engine.borrow().type_of(&name).map_err(error)?;
        }
        let actual = cdata_type(args.get(1), &istype_data.borrow(), "ffi.istype");
        Ok(vec![Value::Boolean(actual.is_ok_and(|actual| {
            same_ctype(&istype_engine.borrow(), &name, &actual)
        }))])
    })?;
    runtime.table_set(ffi, &Value::from("istype"), Value::NativeFunction(istype))?;

    let load_engine = engine.clone();
    let load_data = data.clone();
    let load = runtime.register_native_closure(move |runtime, args| {
        let name = string_argument(args.first(), "ffi.load")?;
        let library = Rc::new(load_engine.borrow().load(&name).map_err(error)?);
        Ok(vec![namespace(
            runtime,
            load_engine.clone(),
            load_data.clone(),
            Some(library),
        )?])
    })?;
    runtime.table_set(ffi, &Value::from("load"), Value::NativeFunction(load))?;

    let sizeof_engine = engine.clone();
    let sizeof_data = data.clone();
    let sizeof = runtime.register_native_closure(move |_, args| {
        let name = if let Some(Value::Userdata(handle)) = args.first() {
            sizeof_data
                .borrow()
                .types
                .get(handle)
                .cloned()
                .ok_or_else(|| error("ffi.sizeof requires a C type or FFI cdata"))?
        } else {
            string_argument(args.first(), "ffi.sizeof")?
        };
        let size = sizeof_engine.borrow().sizeof_type(&name).map_err(error)?;
        Ok(vec![Value::Integer(size as i64)])
    })?;
    runtime.table_set(ffi, &Value::from("sizeof"), Value::NativeFunction(sizeof))?;

    let alignof_engine = engine.clone();
    let alignof_data = data.clone();
    let alignof = runtime.register_native_closure(move |_, args| {
        let name = type_argument(args.first(), &alignof_data.borrow(), "ffi.alignof")?;
        let alignment = alignof_engine.borrow().alignof_type(&name).map_err(error)?;
        Ok(vec![Value::Integer(alignment as i64)])
    })?;
    runtime.table_set(ffi, &Value::from("alignof"), Value::NativeFunction(alignof))?;

    let abi = runtime.register_native_closure(|_, args| {
        let flag = string_argument(args.first(), "ffi.abi")?;
        Ok(vec![Value::Boolean(FfiEngine::abi(&flag))])
    })?;
    runtime.table_set(ffi, &Value::from("abi"), Value::NativeFunction(abi))?;

    let errno = runtime.register_native_closure(|_, args| {
        let previous = FfiEngine::errno();
        if let Some(value) = args.first() {
            let number = integer_argument(Some(value), "ffi.errno")?;
            let number = i32::try_from(number)
                .map_err(|_| error("ffi.errno value is outside the C int range"))?;
            FfiEngine::set_errno(number);
        }
        Ok(vec![Value::Integer(i64::from(previous))])
    })?;
    runtime.table_set(ffi, &Value::from("errno"), Value::NativeFunction(errno))?;

    let copy_data = data.clone();
    let copy_engine = engine.clone();
    let copy = runtime.register_native_closure(move |_, args| {
        let destination_value = args
            .first()
            .ok_or_else(|| error("ffi.copy requires a destination"))?;
        let destination = cdata_address(
            Some(destination_value),
            &copy_data.borrow(),
            "ffi.copy destination",
        )?;
        let source = args
            .get(1)
            .ok_or_else(|| error("ffi.copy requires a source"))?;
        if let Value::String(source) = source {
            let bytes = source.as_bytes();
            if let Some(length) = args.get(2) {
                let length = length_argument(Some(length), "ffi.copy length")?;
                if length > bytes.len().saturating_add(1) {
                    return Err(error("ffi.copy length exceeds source string"));
                }
                check_owned_capacity(
                    destination_value,
                    length,
                    &copy_data.borrow(),
                    &copy_engine.borrow(),
                    "ffi.copy",
                )?;
                if length == bytes.len() + 1 {
                    let mut terminated = bytes.to_vec();
                    terminated.push(0);
                    FfiEngine::copy_bytes(destination, &terminated).map_err(error)?;
                } else {
                    FfiEngine::copy_bytes(destination, &bytes[..length]).map_err(error)?;
                }
            } else {
                let mut nul_terminated = bytes.to_vec();
                nul_terminated.push(0);
                check_owned_capacity(
                    destination_value,
                    nul_terminated.len(),
                    &copy_data.borrow(),
                    &copy_engine.borrow(),
                    "ffi.copy",
                )?;
                FfiEngine::copy_bytes(destination, &nul_terminated).map_err(error)?;
            }
        } else {
            let source = cdata_address(Some(source), &copy_data.borrow(), "ffi.copy source")?;
            let length = length_argument(args.get(2), "ffi.copy length")?;
            check_owned_capacity(
                destination_value,
                length,
                &copy_data.borrow(),
                &copy_engine.borrow(),
                "ffi.copy destination",
            )?;
            check_owned_capacity(
                &args[1],
                length,
                &copy_data.borrow(),
                &copy_engine.borrow(),
                "ffi.copy source",
            )?;
            FfiEngine::copy(destination, source, length).map_err(error)?;
        }
        Ok(Vec::new())
    })?;
    runtime.table_set(ffi, &Value::from("copy"), Value::NativeFunction(copy))?;

    let fill_data = data.clone();
    let fill_engine = engine.clone();
    let fill = runtime.register_native_closure(move |_, args| {
        let destination = cdata_address(args.first(), &fill_data.borrow(), "ffi.fill destination")?;
        let length = length_argument(args.get(1), "ffi.fill length")?;
        check_owned_capacity(
            &args[0],
            length,
            &fill_data.borrow(),
            &fill_engine.borrow(),
            "ffi.fill",
        )?;
        let byte = match args.get(2) {
            Some(value) => integer_argument(Some(value), "ffi.fill byte")? as u8,
            None => 0,
        };
        FfiEngine::fill(destination, byte, length).map_err(error)?;
        Ok(Vec::new())
    })?;
    runtime.table_set(ffi, &Value::from("fill"), Value::NativeFunction(fill))?;

    let cast_data = data.clone();
    let cast_engine = engine.clone();
    let cast = runtime.register_native_closure(move |runtime, args| {
        let name = type_argument(args.first(), &cast_data.borrow(), "ffi.cast")?;
        let value = args
            .get(1)
            .ok_or_else(|| error("ffi.cast requires a value"))?;
        let ty = cast_engine.borrow().type_of(&name).map_err(error)?;
        let input = to_ffi_typed(value, Some(ty), &cast_data.borrow())?;
        let result = cast_engine
            .borrow()
            .cast_type(&name, &input)
            .or_else(|first| {
                if let Value::Number(number) = value
                    && number.is_finite()
                    && number.fract() == 0.0
                    && *number >= i64::MIN as f64
                    && *number < 9_223_372_036_854_775_808.0
                {
                    cast_engine
                        .borrow()
                        .cast_type(&name, &FfiValue::Integer(*number as i64))
                } else {
                    Err(first)
                }
            })
            .map_err(error)?;
        let value = match result {
            FfiValue::Pointer(pointer) => CdataValue::Pointer(pointer),
            result => CdataValue::Scalar(
                cast_engine
                    .borrow()
                    .new_scalar_type(&name, &result)
                    .map_err(error)?,
            ),
        };
        Ok(vec![make_cdata(runtime, &cast_data, value, &name)?])
    })?;
    runtime.table_set(ffi, &Value::from("cast"), Value::NativeFunction(cast))?;

    let new_data = data.clone();
    let new_engine = engine.clone();
    let new = runtime.register_native_closure(move |runtime, args| {
        let name = type_argument(args.first(), &new_data.borrow(), "ffi.new")?;
        if name.contains('[') {
            let mut array = new_engine.borrow().new_array_type(&name).map_err(error)?;
            let initializers = &args[1..];
            if initializers.len() == 1 {
                match &initializers[0] {
                    Value::String(source) if is_byte_array_type(&new_engine.borrow(), &name) => {
                        let bytes = source.as_bytes();
                        if bytes.len() > array.len() {
                            return Err(error("FFI character array initializer is too long"));
                        }
                        FfiEngine::copy_bytes(array.pointer(), bytes).map_err(error)?;
                        return Ok(vec![make_cdata(
                            runtime,
                            &new_data,
                            CdataValue::Array(array),
                            &name,
                        )?]);
                    }
                    Value::Table(table) => {
                        let length = runtime.table_length(*table)? as usize;
                        if length > array.len() {
                            return Err(error("too many FFI array initializers"));
                        }
                        for index in 0..length {
                            let value =
                                runtime.table_get(*table, &Value::Integer((index + 1) as i64))?;
                            set_array_value(&mut array, index, &value, &new_data.borrow())?;
                        }
                        return Ok(vec![make_cdata(
                            runtime,
                            &new_data,
                            CdataValue::Array(array),
                            &name,
                        )?]);
                    }
                    _ => {}
                }
            }
            if initializers.len() > array.len() {
                return Err(error("too many FFI array initializers"));
            }
            for (index, value) in initializers.iter().enumerate() {
                set_array_value(&mut array, index, value, &new_data.borrow())?;
            }
            return Ok(vec![make_cdata(
                runtime,
                &new_data,
                CdataValue::Array(array),
                &name,
            )?]);
        }
        let input = match args.get(1) {
            Some(value) => {
                let ty = new_engine.borrow().type_of(&name).map_err(error)?;
                to_ffi_typed(value, Some(ty), &new_data.borrow())?
            }
            None => FfiValue::Integer(0),
        };
        let scalar = new_engine
            .borrow()
            .new_scalar_type(&name, &input)
            .or_else(|first| {
                if let Some(Value::Number(number)) = args.get(1)
                    && number.is_finite()
                    && number.fract() == 0.0
                    && *number >= i64::MIN as f64
                    && *number < 9_223_372_036_854_775_808.0
                {
                    new_engine
                        .borrow()
                        .new_scalar_type(&name, &FfiValue::Integer(*number as i64))
                } else {
                    Err(first)
                }
            })
            .map_err(error)?;
        Ok(vec![make_cdata(
            runtime,
            &new_data,
            CdataValue::Scalar(scalar),
            &name,
        )?])
    })?;
    runtime.table_set(ffi, &Value::from("new"), Value::NativeFunction(new))?;
    runtime.table_set(
        cdata_metatable,
        &Value::from("__call"),
        Value::NativeFunction(new),
    )?;

    let string_data = data.clone();
    let string_engine = engine.clone();
    let string = runtime.register_native_closure(move |_, args| {
        let value = args
            .first()
            .ok_or_else(|| error("ffi.string requires cdata"))?;
        let length = args
            .get(1)
            .map(|value| length_argument(Some(value), "ffi.string length"))
            .transpose()?;
        match value {
            Value::String(value) => {
                let bytes = value.as_bytes();
                let count = length.unwrap_or(bytes.len());
                if count > bytes.len() {
                    return Err(error("ffi.string length exceeds source string"));
                }
                Ok(vec![Value::from(bytes[..count].to_vec())])
            }
            Value::Userdata(handle) => {
                let data = string_data.borrow();
                let pointer = match data.values.get(handle) {
                    Some(CdataValue::Pointer(pointer)) => *pointer,
                    Some(CdataValue::Array(array)) => array.pointer(),
                    Some(CdataValue::Scalar(scalar)) => match scalar.value().map_err(error)? {
                        FfiValue::Pointer(pointer) => pointer,
                        FfiValue::CString(value) => {
                            let bytes = value.as_bytes();
                            let count = length.unwrap_or(bytes.len());
                            if count > bytes.len() {
                                return Err(error("ffi.string length exceeds source string"));
                            }
                            return Ok(vec![Value::from(bytes[..count].to_vec())]);
                        }
                        _ => return Err(error("ffi.string requires a character pointer")),
                    },
                    Some(CdataValue::CType(_)) => {
                        return Err(error("ffi.string requires cdata"));
                    }
                    None => return Err(error("userdata is not FFI cdata")),
                };
                if let Some(count) = length {
                    check_owned_capacity(
                        value,
                        count,
                        &data,
                        &string_engine.borrow(),
                        "ffi.string",
                    )?;
                    if count > isize::MAX as usize || pointer.checked_add(count).is_none() {
                        return Err(error("ffi.string length is too large"));
                    }
                    if count == 0 {
                        return Ok(vec![Value::from(Vec::<u8>::new())]);
                    }
                    if pointer == 0 {
                        return Err(error("null C string pointer"));
                    }
                    let bytes = FfiEngine::string_bytes(pointer, count).map_err(error)?;
                    Ok(vec![Value::from(bytes)])
                } else if let Some(capacity) =
                    owned_capacity(value, &data, &string_engine.borrow())?
                {
                    let bytes = FfiEngine::string_bytes(pointer, capacity).map_err(error)?;
                    let end = bytes
                        .iter()
                        .position(|byte| *byte == 0)
                        .unwrap_or(bytes.len());
                    Ok(vec![Value::from(bytes[..end].to_vec())])
                } else {
                    Ok(vec![Value::from(
                        FfiEngine::string(pointer).map_err(error)?.as_str(),
                    )])
                }
            }
            _ => Err(error("ffi.string requires a string or character pointer")),
        }
    })?;
    runtime.table_set(ffi, &Value::from("string"), Value::NativeFunction(string))?;

    runtime.set_global("ffi", Value::Table(ffi));
    if let Value::Table(package) = runtime.get_global("package")
        && let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded"))?
    {
        runtime.table_set(loaded, &Value::from("ffi"), Value::Table(ffi))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{Lua, LuaJitExtension, StandardLibrary, Value};

    #[test]
    fn ffi_memory_and_platform_helpers() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .luajit_extension(LuaJitExtension::Ffi)
            .build()
            .unwrap();
        let result = lua
            .exec(
                "local p = ffi.new('char[4]'); ffi.fill(p, 4, 0); ffi.copy(p, 'A', 1); local old = ffi.errno(7); return ffi.string(p), p[0], ffi.alignof('int'), ffi.abi('64bit') == (ffi.sizeof('void *') == 8), ffi.errno()",
            )
            .unwrap();
        assert_eq!(
            result,
            vec![
                Value::from("A"),
                Value::Integer(65),
                Value::Integer(4),
                Value::Boolean(true),
                Value::Integer(7),
            ]
        );
    }

    #[test]
    fn ffi_array_receives_native_output() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .luajit_extension(LuaJitExtension::Ffi)
            .build()
            .unwrap();
        let result = lua
            .exec(
                "ffi.cdef('int sscanf(const char *, const char *, ...);'); local out = ffi.new('int[1]'); local count = ffi.C.sscanf('42', '%d', out); local before = out[0]; out[0] = 19; return count, before, out[0]",
            )
            .unwrap();
        assert_eq!(
            result,
            vec![Value::Integer(1), Value::Integer(42), Value::Integer(19)]
        );
        assert_eq!(
            lua.exec("local a = ffi.new('int[2]', 7, 8); local b = ffi.new('int[2]', {9, 10}); local c = ffi.new('char[4]', 'cat'); return a[0], a[1], b[0], b[1], ffi.string(c)")
                .unwrap(),
            vec![
                Value::Integer(7),
                Value::Integer(8),
                Value::Integer(9),
                Value::Integer(10),
                Value::from("cat"),
            ]
        );
    }

    #[test]
    fn ffi_type_objects_and_binary_strings() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .luajit_extension(LuaJitExtension::Ffi)
            .build()
            .unwrap();
        let result = lua.exec(
            "local t = ffi.typeof('int[2]'); local a = t({4, 5}); local p = ffi.new('char[4]'); ffi.copy(p, 'a\\0b', 3); return ffi.istype(t, a), ffi.istype('int[2]', a), ffi.istype('int[3]', a), ffi.sizeof(a), ffi.sizeof(t), ffi.string(p, 3), tostring(t), a[1]",
        ).unwrap();
        assert_eq!(
            result[..6],
            [
                Value::Boolean(true),
                Value::Boolean(true),
                Value::Boolean(false),
                Value::Integer(8),
                Value::Integer(8),
                Value::from(vec![b'a', 0, b'b']),
            ]
        );
        assert!(matches!(&result[6], Value::String(value) if value.as_bytes() == b"ctype<int[2]>"));
        assert_eq!(result[7], Value::Integer(5));
    }

    #[test]
    fn ffi_pointer_return_requires_explicit_string_conversion() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .luajit_extension(LuaJitExtension::Ffi)
            .build()
            .unwrap();
        let result = lua.exec(
            "ffi.cdef('char *strchr(const char *, int); size_t strlen(const char *);'); local b = ffi.new('char[4]'); ffi.copy(b, 'abc', 4); local p = ffi.C.strchr(b, 98); return type(p), ffi.string(p), ffi.string(p, 1), ffi.C.strlen(b), ffi.string(b, 4)",
        ).unwrap();
        assert_eq!(
            result,
            vec![
                Value::from("userdata"),
                Value::from("bc"),
                Value::from("b"),
                Value::Integer(3),
                Value::from(vec![b'a', b'b', b'c', 0]),
            ]
        );
    }

    #[test]
    fn ffi_scalars_pass_their_values() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .luajit_extension(LuaJitExtension::Ffi)
            .build()
            .unwrap();
        let result = lua.exec(
            "ffi.cdef('int sprintf(char *, const char *, ...); double sqrt(double);'); local b = ffi.new('char[64]'); ffi.C.sprintf(b, '%d %lld %llu %g %g', ffi.new('int', 42), ffi.new('int64_t', -7), ffi.new('uint64_t', 9), ffi.new('double', 1.5), ffi.new('float', 2.5)); return ffi.string(b), ffi.C.sqrt(ffi.new('int', 16)), ffi.C.sqrt(ffi.new('uint64_t', 25))",
        ).unwrap();
        assert_eq!(
            result,
            vec![
                Value::from("42 -7 9 1.5 2.5"),
                Value::Number(4.0),
                Value::Number(5.0),
            ]
        );
    }

    #[test]
    fn ffi_aliases_and_owned_array_bounds() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .luajit_extension(LuaJitExtension::Ffi)
            .build()
            .unwrap();
        let result = lua.exec(
            "ffi.cdef('typedef unsigned int counter_t; typedef char byte_t;'); local t = ffi.typeof('counter_t[2]'); local a = t({3, 7}); local b = ffi.new('byte_t[4]', 'cat'); local ok1 = pcall(ffi.string, a, 9); local ok2 = pcall(ffi.fill, a, 9); local ok3 = pcall(ffi.copy, a, '123456789', 9); return ffi.sizeof(t), ffi.istype(t, a), a[0], a[1], ok1, ok2, ok3, ffi.string(b)",
        ).unwrap();
        assert_eq!(
            result,
            vec![
                Value::Integer(8),
                Value::Boolean(true),
                Value::Integer(3),
                Value::Integer(7),
                Value::Boolean(false),
                Value::Boolean(false),
                Value::Boolean(false),
                Value::from("cat"),
            ]
        );
    }

    #[test]
    fn ffi_scalar_cast_and_pointer_alias_type() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .luajit_extension(LuaJitExtension::Ffi)
            .build()
            .unwrap();
        let result = lua.exec(
            "ffi.cdef('typedef char *p_t; int abs(int);'); local n = ffi.cast('int', -6); local p = ffi.new('p_t'); return type(n), ffi.istype('int', n), ffi.istype(ffi.typeof(n), n), ffi.C.abs(n), ffi.istype('char *', p)",
        ).unwrap();
        assert_eq!(
            result,
            vec![
                Value::from("userdata"),
                Value::Boolean(true),
                Value::Boolean(true),
                Value::Integer(6),
                Value::Boolean(true),
            ]
        );
    }
}
