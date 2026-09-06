//! Registers the standard library into a runtime for a compatibility profile.

use crate::compat::CompatibilityProfile;
use crate::runtime::{
    BuiltinFunction, DirectNativeKind, LinesFormats, NativeCallback, NativeFixedResultsProbe,
    NativeFunctionHandle, NativeNumericBinary, NativeNumericUnary, NativeProbe, NativeResultsProbe,
    NativeScalarCallback, NonFiniteNumerals, NumberFormat, Runtime, RuntimeError, RuntimeResult,
    TableHandle, Value,
};
use crate::stdlib::args::table_argument;
use crate::stdlib::base::{
    TONUMBER_RULES_LUA51, TONUMBER_RULES_LUAJIT, TONUMBER_RULES_MODERN, TonumberRules, assert,
    assert_lua52, assert_luajit, assert_string_message, debug_getmetatable, debug_setmetatable,
    error, gcinfo, getmetatable, io_write, newproxy, numeric_tonumber, os_clock, os_date,
    os_difftime, os_execute, os_exit, os_exit_lua51, os_exit_truncating, os_getenv, os_remove,
    os_rename, os_setlocale, os_time, os_tmpname, print, print_lua51, rawequal, rawget, rawlen,
    rawset, select, setmetatable, table_concat, table_create, table_getn, table_insert, table_maxn,
    table_move, table_pack, table_remove, tonumber_probe_lua51, tonumber_probe_luajit,
    tonumber_probe_modern, tonumber_scalar, tonumber_scalar_lua51, tonumber_scalar_luajit, type_of,
    unpack, warn,
};
use crate::stdlib::base::{
    table_concat_meta, table_insert_meta, table_move_meta, table_remove_meta, unpack_meta,
};
use crate::stdlib::bit::{
    arshift, band, bnot, bor, bswap, bxor, lshift, numeric_arshift, numeric_band, numeric_bnot,
    numeric_bor, numeric_bswap, numeric_bxor, numeric_lshift, numeric_rol, numeric_ror,
    numeric_rshift, numeric_tobit, rol, ror, rshift, tobit, tohex, wide_arshift, wide_band,
    wide_bor, wide_bxor, wide_lshift, wide_rol, wide_ror, wide_rshift,
};
use crate::stdlib::jit_ext::{
    flush as jit_flush, off as jit_off, on as jit_on, opt_start as jit_opt_start,
    status as jit_status,
};
use crate::stdlib::math::{
    file_close, file_flush, file_gc, file_lines, file_read, file_seek, file_setvbuf, file_tostring,
    file_write, floor_scalar, io_close, io_flush, io_input, io_lines, io_open, io_output, io_popen,
    io_read, io_tmpfile, io_type, math_abs, math_acos, math_asin, math_atan, math_atan_modern,
    math_atan2, math_ceil, math_cos, math_cosh, math_deg, math_exp, math_fmod, math_frexp,
    math_ldexp, math_log, math_log_luajit, math_log_with_base, math_log_with_base_lua52,
    math_log10, math_max_scalar, math_max_scalar_luajit, math_min_scalar, math_min_scalar_luajit,
    math_mod, math_modf, math_pow, math_rad, math_random, math_random_lua53, math_random_lua54,
    math_random_luajit, math_randomseed, math_randomseed_luajit, math_randomseed_modern, math_sinh,
    math_sqrt, math_tan, math_tanh, math_tointeger, math_type, math_ult, numeric_acos,
    numeric_asin, numeric_cos, numeric_cosh, numeric_deg, numeric_exp, numeric_fmod, numeric_log,
    numeric_log_base, numeric_log_base_lua52, numeric_log_base_luajit, numeric_log10, numeric_max,
    numeric_max_luajit, numeric_min, numeric_min_luajit, numeric_pow, numeric_rad, numeric_sin,
    numeric_sinh, numeric_sqrt, numeric_tan, numeric_tanh, sin_scalar,
};
use crate::stdlib::string_arith::{
    string_add, string_div, string_idiv, string_mod, string_mul, string_pow, string_sub, string_unm,
};
use crate::stdlib::string_pack::{
    string_byte, string_char, string_char_51, string_len_scalar, string_pack,
    string_pack_fixed_probe, string_pack_probe, string_packsize, string_packsize_probe,
    string_unpack, string_unpack_clamped, string_unpack_clamped_fixed_probe,
    string_unpack_clamped_probe, string_unpack_fixed_probe, string_unpack_probe,
};
use crate::stdlib::strings::{
    string_argument, string_find, string_find_pair, string_format_luajit_scalar,
    string_format_modern, string_format_scalar, string_format_with_pointer, string_gmatch,
    string_gmatch_init, string_lower, string_match, string_rep, string_rep_with_separator,
    string_reverse, string_sub_scalar, string_upper, utf8_char, utf8_char_fixed_probe,
    utf8_char_lua53, utf8_char_lua53_fixed_probe, utf8_char_lua53_probe, utf8_char_probe,
    utf8_codepoint, utf8_codepoint_fixed_probe, utf8_codepoint_lua53,
    utf8_codepoint_lua53_fixed_probe, utf8_codepoint_lua53_probe, utf8_codepoint_probe,
    utf8_codes_with_iterators, utf8_len, utf8_len_fixed_probe, utf8_len_lua53,
    utf8_len_lua53_fixed_probe, utf8_len_lua53_probe, utf8_len_probe, utf8_next, utf8_next_lax,
    utf8_next_lax_probe, utf8_next_lua53, utf8_next_lua53_probe, utf8_next_probe, utf8_offset,
    utf8_offset_fixed_probe, utf8_offset_lua53, utf8_offset_lua53_fixed_probe,
    utf8_offset_lua53_probe, utf8_offset_lua55, utf8_offset_lua55_fixed_probe,
    utf8_offset_lua55_probe, utf8_offset_probe,
};
use crate::string::LuaString;

/// Stores `value` under `name` in `table`. A native keeps the name it is
/// stored under, qualified by the library `table` is (`math.floor`).
fn set(runtime: &mut Runtime, table: TableHandle, name: &str, value: Value) -> RuntimeResult<()> {
    if let Value::NativeFunction(function) = &value {
        match runtime.library_name(table) {
            Some(library) => {
                runtime.set_native_name_in_library(*function, library, name);
            }
            None => runtime.set_native_name(*function, name),
        }
    }
    // A shared constant, so code that reads the field by name matches it by identity.
    let key = Value::String(crate::string::shared_key(name.as_bytes()));
    runtime.table_set(table, &key, value)
}

/// Registers each callback and stores it under its name, in order.
fn set_natives(
    runtime: &mut Runtime,
    table: TableHandle,
    entries: &[(&str, NativeCallback)],
) -> RuntimeResult<()> {
    for &(name, callback) in entries {
        let function = runtime.register_native(callback)?;
        set(runtime, table, name, Value::NativeFunction(function))?;
    }
    Ok(())
}

/// A single-result bit operation with its pure numeric fast path.
#[derive(Clone, Copy)]
enum BitNative {
    Unary(NativeScalarCallback, NativeNumericUnary),
    /// A binary operation with a pure form for 64-bit integer operands.
    Wide(
        NativeScalarCallback,
        NativeNumericBinary,
        crate::runtime::NativeWideBinary,
    ),
}

/// Registers each bit operation and stores it under its name, in order.
fn set_bit_natives(
    runtime: &mut Runtime,
    table: TableHandle,
    entries: &[(&str, BitNative)],
) -> RuntimeResult<()> {
    for &(name, native) in entries {
        let function = match native {
            BitNative::Unary(callback, unary) => {
                runtime.register_native_numeric_unary(callback, unary)?
            }
            BitNative::Wide(callback, binary, wide) => {
                runtime.register_native_wide_binary(callback, binary, wide)?
            }
        };
        set(runtime, table, name, Value::NativeFunction(function))?;
    }
    Ok(())
}

/// Stores each VM-implemented builtin under its name, in order.
fn set_builtins(
    runtime: &mut Runtime,
    table: TableHandle,
    entries: &[(&str, BuiltinFunction)],
) -> RuntimeResult<()> {
    for &(name, builtin) in entries {
        set(runtime, table, name, Value::BuiltinFunction(builtin))?;
    }
    Ok(())
}

fn set_global_builtins(runtime: &mut Runtime, entries: &[(&str, BuiltinFunction)]) {
    for &(name, builtin) in entries {
        runtime.set_global(name, Value::BuiltinFunction(builtin));
    }
}

/// Installs base-library functions and returns the handle for `assert`.
///
/// # Errors
///
/// Returns an error if the runtime has exhausted native-function handles.
pub fn install_base(runtime: &mut Runtime) -> RuntimeResult<NativeFunctionHandle> {
    install_base_with_profile(runtime, CompatibilityProfile::default())
}

/// Installs base-library functions for a selected compatibility profile.
///
/// The profile controls exposed runtime identity, including `_VERSION`. APIs
/// outside Oximond's implemented library surface are not installed merely
/// because a newer Lua target names them.
///
/// # Errors
///
/// Returns an error if the runtime has exhausted native-function handles.
pub fn install_base_with_profile(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<NativeFunctionHandle> {
    install_full_with_profile_and_host_libraries(runtime, compatibility, true, true)
}

/// Installs the standard library without registering the default `io` table.
///
/// Hosts can install their own `io` table before calling this function. The
/// package installer then copies that table into `package.loaded.io`.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_base_with_profile_without_io(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<NativeFunctionHandle> {
    install_full_with_profile_and_host_libraries(runtime, compatibility, false, true)
}

/// Installs the full library while selecting the built-in `io` and `os` tables.
///
/// A host can register either table before this call when its corresponding
/// `default_*` flag is false. The installed `package.loaded` table then uses
/// those host tables. The rest of the full library is installed as usual.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_full_with_profile_and_host_libraries(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
    default_io: bool,
    default_os: bool,
) -> RuntimeResult<NativeFunctionHandle> {
    install_with_profile(
        runtime,
        compatibility,
        LibrarySet::Full {
            default_io,
            default_os,
        },
    )
}

/// Installs the pure standard library with the default compatibility profile.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_pure(runtime: &mut Runtime) -> RuntimeResult<NativeFunctionHandle> {
    install_pure_with_profile(runtime, CompatibilityProfile::default())
}

/// Installs Lua functions that do not access the host system.
///
/// File loading, module search, process and file I/O, host output, and debug
/// access are omitted. The in-memory `load` function remains available.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_pure_with_profile(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<NativeFunctionHandle> {
    install_with_profile(runtime, compatibility, LibrarySet::Pure)
}

/// Installs the pure library plus VM-backed source loading and package tables.
///
/// Use this only when the embedding VM supplies restricted implementations for
/// file loading and module search. It does not install `io`, `os`, host output,
/// or debug access.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_pure_with_profile_and_loading(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<NativeFunctionHandle> {
    install_with_profile(runtime, compatibility, LibrarySet::PureWithLoading)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LibrarySet {
    Full { default_io: bool, default_os: bool },
    Pure,
    PureWithLoading,
}

fn install_with_profile(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
    libraries: LibrarySet,
) -> RuntimeResult<NativeFunctionHandle> {
    let lua51 = compatibility.is_lua51_family();
    runtime.set_error_number_positions(lua51 || compatibility == CompatibilityProfile::Lua52);
    runtime.set_table_finalizers_enabled(!lua51);
    runtime.set_ephemerons_enabled(!lua51);
    runtime.set_tagged_integers(compatibility.is_lua53_or_later());
    runtime.set_typed_argument_errors(matches!(
        compatibility,
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    ));
    runtime.set_nonfinite_numerals(match compatibility {
        CompatibilityProfile::Lua51 => NonFiniteNumerals::Strtod,
        CompatibilityProfile::LuaJit21 => NonFiniteNumerals::LuaJit,
        CompatibilityProfile::Lua52
        | CompatibilityProfile::Lua53
        | CompatibilityProfile::Lua54
        | CompatibilityProfile::Lua55 => NonFiniteNumerals::Rejected,
    });
    runtime.set_lines_formats(match compatibility {
        CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21 => LinesFormats::Ignored,
        CompatibilityProfile::Lua52 => LinesFormats::Lua52,
        CompatibilityProfile::Lua53 => LinesFormats::Lua53,
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55 => LinesFormats::Lua54,
    });
    runtime.set_legacy_pattern_errors(compatibility.is_lua51_family());
    runtime.set_string_init_clamps(compatibility.is_lua51_family());
    runtime.set_number_format(match compatibility {
        CompatibilityProfile::Lua51 | CompatibilityProfile::Lua52 => NumberFormat::Lua51,
        CompatibilityProfile::LuaJit21 => NumberFormat::LuaJit,
        CompatibilityProfile::Lua53 | CompatibilityProfile::Lua54 => NumberFormat::Lua53,
        CompatibilityProfile::Lua55 => NumberFormat::Lua55,
    });
    runtime.set_global(
        "_VERSION",
        Value::from(compatibility.identity().version_string()),
    );
    let assert = match compatibility {
        CompatibilityProfile::Lua51 => {
            runtime.register_global_native("assert", assert_string_message)?
        }
        CompatibilityProfile::Lua52 => runtime.register_global_native("assert", assert_lua52)?,
        CompatibilityProfile::LuaJit21 => {
            runtime.register_global_native("assert", assert_luajit)?
        }
        _ => runtime.register_global_native("assert", assert)?,
    };
    runtime.set_direct_native_kind(assert, DirectNativeKind::Assert);
    let error = runtime.register_global_native("error", error)?;
    runtime.set_direct_native_kind(error, DirectNativeKind::Error);
    // One result each, so the interpreter calls them without a result vector.
    for (name, callback) in [
        ("getmetatable", getmetatable as NativeScalarCallback),
        ("setmetatable", setmetatable),
    ] {
        let function = runtime.register_native_scalar(callback)?;
        runtime.set_native_name(function, name);
        match name {
            "getmetatable" => {
                runtime.set_direct_native_kind(function, DirectNativeKind::GetMetatable);
            }
            "setmetatable" => {
                runtime.set_setmetatable_native(function);
                runtime.set_direct_native_kind(function, DirectNativeKind::SetMetatable);
            }
            _ => {}
        }
        runtime.set_global(name, Value::NativeFunction(function));
    }
    let full = matches!(libraries, LibrarySet::Full { .. });
    let loading = full || libraries == LibrarySet::PureWithLoading;
    if full {
        let print: NativeCallback = if compatibility == CompatibilityProfile::Lua51 {
            print_lua51
        } else {
            print
        };
        let function = runtime.register_global_native("print", print)?;
        runtime.set_direct_native_kind(function, DirectNativeKind::Print);
    }
    // One result each, as for `getmetatable` above.
    for (name, callback, kind) in [
        (
            "rawequal",
            rawequal as NativeScalarCallback,
            DirectNativeKind::RawEqual,
        ),
        ("rawget", rawget, DirectNativeKind::RawGet),
        ("rawset", rawset, DirectNativeKind::RawSet),
    ] {
        let function = runtime.register_native_scalar(callback)?;
        runtime.set_native_name(function, name);
        runtime.set_direct_native_kind(function, kind);
        runtime.set_global(name, Value::NativeFunction(function));
    }
    let function = runtime.register_global_native("select", select)?;
    runtime.set_direct_native_kind(function, DirectNativeKind::Select);
    // The callback and the probe that predicts its answer come from one
    // match, so they cannot disagree on a profile.
    let (tonumber, probe, rules): (NativeScalarCallback, NativeProbe, TonumberRules) =
        match compatibility {
            CompatibilityProfile::Lua51 => (
                tonumber_scalar_lua51,
                tonumber_probe_lua51,
                TONUMBER_RULES_LUA51,
            ),
            CompatibilityProfile::LuaJit21 => (
                tonumber_scalar_luajit,
                tonumber_probe_luajit,
                TONUMBER_RULES_LUAJIT,
            ),
            CompatibilityProfile::Lua52
            | CompatibilityProfile::Lua53
            | CompatibilityProfile::Lua54
            | CompatibilityProfile::Lua55 => (
                tonumber_scalar,
                tonumber_probe_modern,
                TONUMBER_RULES_MODERN,
            ),
        };
    runtime.set_tonumber_rules(rules);
    let tonumber = runtime.register_native_numeric_unary(tonumber, numeric_tonumber)?;
    runtime.set_direct_native_kind(tonumber, DirectNativeKind::Tonumber);
    runtime.set_native_probe(tonumber, probe);
    runtime.set_native_name(tonumber, "tonumber");
    runtime.set_global("tonumber", Value::NativeFunction(tonumber));
    let type_function = runtime.register_native_scalar(type_of)?;
    runtime.set_native_name(type_function, "type");
    runtime.set_global("type", Value::NativeFunction(type_function));
    runtime.set_direct_native_kind(type_function, DirectNativeKind::Type);
    if lua51 {
        runtime.register_global_native("gcinfo", gcinfo)?;
        runtime.register_global_native("newproxy", newproxy)?;
        let function = runtime.register_global_native("unpack", unpack)?;
        runtime.set_direct_native_kind(function, DirectNativeKind::Unpack);
    } else {
        let function = runtime.register_global_native("rawlen", rawlen)?;
        runtime.set_direct_native_kind(function, DirectNativeKind::RawLen);
    }
    if full
        && matches!(
            compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        )
    {
        runtime.register_global_native("warn", warn)?;
    }
    set_global_builtins(
        runtime,
        &[
            ("pcall", BuiltinFunction::ProtectedCall),
            ("xpcall", BuiltinFunction::ExtendedProtectedCall),
            ("next", BuiltinFunction::Next),
            ("pairs", BuiltinFunction::Pairs),
            ("ipairs", BuiltinFunction::IPairs),
            ("load", BuiltinFunction::Load),
            ("tostring", BuiltinFunction::ToString),
            ("collectgarbage", BuiltinFunction::CollectGarbage),
        ],
    );
    if loading {
        set_global_builtins(
            runtime,
            &[
                ("loadfile", BuiltinFunction::LoadFile),
                ("dofile", BuiltinFunction::DoFile),
                ("require", BuiltinFunction::Require),
            ],
        );
    }
    if lua51 {
        set_global_builtins(
            runtime,
            &[
                (
                    "loadstring",
                    // Stock 5.1 has `load` take a reader and `loadstring` a
                    // string; LuaJIT makes them one function.
                    if compatibility == CompatibilityProfile::Lua51 {
                        BuiltinFunction::LoadString
                    } else {
                        BuiltinFunction::Load
                    },
                ),
                ("getfenv", BuiltinFunction::GetFenv),
                ("setfenv", BuiltinFunction::SetFenv),
            ],
        );
        if loading {
            set_global_builtins(runtime, &[("module", BuiltinFunction::Module)]);
        }
    }
    install_math_with_profile(runtime, compatibility)?;
    if let LibrarySet::Full {
        default_io,
        default_os,
    } = libraries
    {
        if default_io {
            install_io(runtime)?;
            if runtime.lines_formats() == LinesFormats::Lua54 {
                install_file_close(runtime)?;
            }
        }
        if default_os {
            install_os_with_profile(runtime, compatibility)?;
        }
        install_debug_with_profile(runtime, compatibility)?;
    }
    install_table_with_profile(runtime, compatibility)?;
    install_string_with_profile(runtime, compatibility)?;
    if compatibility == CompatibilityProfile::LuaJit21 {
        install_bit(runtime)?;
    }
    if matches!(
        compatibility,
        CompatibilityProfile::Lua52 | CompatibilityProfile::Lua53
    ) {
        install_bit32(runtime)?;
    }
    install_coroutine_with_profile(runtime, compatibility)?;
    if compatibility.capabilities().libraries.utf8_library {
        install_utf8(runtime, compatibility)?;
    }
    if loading {
        install_package(runtime)?;
    }
    if loading && compatibility.capabilities().libraries.package_searchers {
        let Value::Table(package) = runtime.get_global("package") else {
            unreachable!("package was installed")
        };
        let searchers = runtime.table_get(package, &Value::from("loaders"))?;
        set(runtime, package, "searchers", searchers.clone())?;
        if matches!(
            compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ) {
            let Value::Table(searchers) = searchers else {
                unreachable!("package.loaders was installed")
            };
            let preload = runtime.register_native(package_preload_searcher_modern)?;
            runtime.table_set(
                searchers,
                &Value::Integer(1),
                Value::NativeFunction(preload),
            )?;
        }
    }
    if loading && compatibility != CompatibilityProfile::Lua51 {
        let Value::Table(package) = runtime.get_global("package") else {
            unreachable!("package was installed")
        };
        set_builtins(
            runtime,
            package,
            &[("searchpath", BuiltinFunction::PackageSearchPath)],
        )?;
    }
    if loading && compatibility == CompatibilityProfile::LuaJit21 {
        install_table_extensions(runtime)?;
    }
    Ok(assert)
}

/// Installs the package tables used by the VM-backed `require` builtin.
///
/// The fallback deliberately stays local and predictable for embedded
/// programs. Hosts that accept `LUA_PATH` can call
/// [`install_package_with_path`] at their boundary instead.
///
/// # Errors
///
/// Returns allocation errors from the runtime.
pub fn install_package(runtime: &mut Runtime) -> RuntimeResult<()> {
    install_package_with_path(runtime, LuaString::from("./?.lua;./?/init.lua"))
}

/// Installs package tables with a host-selected byte path.
///
/// # Errors
///
/// Returns allocation errors from the runtime.
///
/// # Panics
///
/// Panics if the built-in searcher list ever grows past 255 entries, where the
/// `u8` index used for `package.loaders` keys would overflow.
pub fn install_package_with_path(
    runtime: &mut Runtime,
    path: impl Into<LuaString>,
) -> RuntimeResult<()> {
    let package = runtime.create_table()?;
    runtime.set_library_name(package, "package");
    let loaded = runtime.create_table()?;
    let preload = runtime.create_table()?;
    let loaders = runtime.create_table()?;
    set(runtime, package, "loaded", Value::Table(loaded))?;
    set(runtime, package, "preload", Value::Table(preload))?;
    set(runtime, package, "loaders", Value::Table(loaders))?;
    set(runtime, package, "path", Value::String(path.into()))?;
    #[cfg(windows)]
    set(runtime, package, "config", Value::from("\\\n;\n?\n!\n-\n"))?;
    #[cfg(not(windows))]
    set(runtime, package, "config", Value::from("/\n;\n?\n!\n-\n"))?;
    #[cfg(windows)]
    set(
        runtime,
        package,
        "cpath",
        Value::from(".\\?.dll;.\\loadall.dll"),
    )?;
    #[cfg(not(windows))]
    set(
        runtime,
        package,
        "cpath",
        Value::from("./?.so;./loadall.so"),
    )?;
    set_natives(
        runtime,
        package,
        &[("loadlib", package_loadlib), ("seeall", package_seeall)],
    )?;
    // Lua 5.1 exposes four searchers. Compilation still belongs to the VM
    // because loading source can suspend. `require` uses the same preload,
    // source, and native order represented by this public table.
    let native_searcher = runtime.register_native(package_native_searcher)?;
    for (index, searcher) in [
        Value::NativeFunction(runtime.register_native(package_preload_searcher)?),
        Value::BuiltinFunction(BuiltinFunction::PackageSourceSearcher),
        Value::NativeFunction(native_searcher),
        Value::NativeFunction(native_searcher),
    ]
    .into_iter()
    .enumerate()
    {
        runtime.table_set(
            loaders,
            &Value::Number(f64::from(u8::try_from(index + 1).expect("four searchers"))),
            searcher,
        )?;
    }
    for name in [
        "string",
        "table",
        "math",
        "io",
        "os",
        "debug",
        "bit",
        "coroutine",
        "utf8",
    ] {
        let value = runtime.get_global(name);
        if !matches!(value, Value::Nil) {
            set(runtime, loaded, name, value)?;
        }
    }
    // As in Lua, the global table and `package` itself are loaded modules.
    let globals = runtime.global_environment();
    set(runtime, loaded, "_G", Value::Table(globals))?;
    set(runtime, loaded, "package", Value::Table(package))?;
    crate::stdlib::bundled::install(runtime, preload)?;
    runtime.set_package_require_table(package)?;
    runtime.set_global("package", Value::Table(package));
    Ok(())
}

/// Replaces the configured path on an already-installed package table.
///
/// This lets a host apply process configuration after [`install_base`] without
/// rebuilding the `loaded` and `preload` tables.
///
/// # Errors
///
/// Returns an error if the package table was replaced or is unavailable.
pub fn set_package_path(runtime: &mut Runtime, path: impl Into<LuaString>) -> RuntimeResult<()> {
    let Value::Table(package) = runtime.get_global("package") else {
        return Err(RuntimeError::message("package table is unavailable"));
    };
    set(runtime, package, "path", Value::String(path.into()))
}

/// Reports that loading a native module is outside Oximond's Rust-only ABI.
///
/// Lua 5.1 specifies a three-result failure tuple. `absent` is permitted when
/// the host cannot provide dynamic loading at all.
pub(crate) fn package_loadlib(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let path = string_argument(arguments.first(), 1)?;
    let symbol = string_argument(arguments.get(1), 2)?;
    Ok(vec![
        Value::Nil,
        Value::from(format!(
            "native library loading is unavailable in this Rust-only build: '{}' ({})",
            String::from_utf8_lossy(&path),
            String::from_utf8_lossy(&symbol)
        )),
        Value::from("absent"),
    ])
}

/// Adds the global table as a module's `__index` fallback.
pub(crate) fn package_seeall(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let table = table_argument(arguments.first(), 1)?;
    let metatable = match runtime.raw_metatable(&Value::Table(table))? {
        Some(metatable) => metatable,
        None => runtime.create_table()?,
    };
    runtime.table_set(
        metatable,
        &Value::from("__index"),
        Value::Table(runtime.global_environment()),
    )?;
    runtime.raw_set_metatable(&Value::Table(table), Some(metatable))?;
    Ok(Vec::new())
}

/// Produces the native-searcher diagnostic used by `package.loaders`.
pub(crate) fn package_native_searcher(
    _: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let name = string_argument(arguments.first(), 1)?;
    Ok(vec![Value::from(format!(
        "\n\tnative module '{}' is unavailable in this Rust-only build",
        String::from_utf8_lossy(&name)
    ))])
}

pub(crate) fn package_preload_searcher(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let name = string_argument(arguments.first(), 1)?;
    let package = match runtime.package_require_table() {
        Some(package) => package,
        None => match runtime.get_global("package") {
            Value::Table(package) => package,
            _ => return Err(RuntimeError::message("package table is unavailable")),
        },
    };
    let Value::Table(preload) = runtime.table_get(package, &Value::from("preload"))? else {
        return Err(RuntimeError::message("package.preload is unavailable"));
    };
    let loader = runtime.table_get(preload, &Value::bytes(name.to_vec()))?;
    if matches!(loader, Value::Nil) {
        Ok(vec![Value::from(format!(
            "\n\tno field package.preload['{}']",
            String::from_utf8_lossy(&name)
        ))])
    } else {
        Ok(vec![loader])
    }
}

fn package_preload_searcher_modern(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let mut results = package_preload_searcher(runtime, arguments)?;
    if matches!(
        results.first(),
        Some(Value::Closure(_) | Value::NativeFunction(_) | Value::BuiltinFunction(_))
    ) {
        results.push(Value::from(":preload:"));
    }
    Ok(results)
}

/// Installs Lua 5.1's VM-backed coroutine library.
///
/// # Errors
///
/// Returns an error if the runtime cannot allocate or initialize the library table.
pub fn install_coroutine(runtime: &mut Runtime) -> RuntimeResult<()> {
    install_coroutine_with_profile(runtime, CompatibilityProfile::Lua51)
}

pub(crate) fn install_coroutine_with_profile(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<()> {
    let coroutine = runtime.create_table()?;
    runtime.set_library_name(coroutine, "coroutine");
    set_builtins(
        runtime,
        coroutine,
        &[
            ("create", BuiltinFunction::CoroutineCreate),
            ("resume", BuiltinFunction::CoroutineResume),
            ("yield", BuiltinFunction::CoroutineYield),
            ("status", BuiltinFunction::CoroutineStatus),
            ("wrap", BuiltinFunction::CoroutineWrap),
            ("running", BuiltinFunction::CoroutineRunning),
        ],
    )?;
    if compatibility.is_lua53_or_later() || compatibility == CompatibilityProfile::LuaJit21 {
        set_builtins(
            runtime,
            coroutine,
            &[("isyieldable", BuiltinFunction::CoroutineIsYieldable)],
        )?;
    }
    if compatibility.capabilities().coroutines.close {
        set_builtins(
            runtime,
            coroutine,
            &[("close", BuiltinFunction::CoroutineClose)],
        )?;
    }
    runtime.set_global("coroutine", Value::Table(coroutine));
    Ok(())
}

/// Installs VM-backed debugging functions.
///
/// # Errors
///
/// Returns allocation errors from the runtime.
pub fn install_debug(runtime: &mut Runtime) -> RuntimeResult<()> {
    install_debug_with_profile(runtime, CompatibilityProfile::default())
}

/// Installs VM-backed debugging functions for a selected compatibility profile.
///
/// # Errors
///
/// Returns allocation errors from the runtime.
pub fn install_debug_with_profile(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<()> {
    let debug = runtime.create_table()?;
    runtime.set_library_name(debug, "debug");
    set_builtins(
        runtime,
        debug,
        &[
            ("getinfo", BuiltinFunction::DebugGetInfo),
            ("traceback", BuiltinFunction::DebugTraceback),
            ("gethook", BuiltinFunction::DebugGetHook),
            ("sethook", BuiltinFunction::DebugSetHook),
            ("getlocal", BuiltinFunction::DebugGetLocal),
            ("setlocal", BuiltinFunction::DebugSetLocal),
            ("getupvalue", BuiltinFunction::DebugGetUpvalue),
            ("setupvalue", BuiltinFunction::DebugSetupValue),
            ("getregistry", BuiltinFunction::DebugGetRegistry),
            ("debug", BuiltinFunction::DebugDebug),
        ],
    )?;
    if compatibility.capabilities().debug.upvalue_id {
        set_builtins(
            runtime,
            debug,
            &[
                ("upvalueid", BuiltinFunction::DebugUpvalueId),
                ("upvaluejoin", BuiltinFunction::DebugUpvalueJoin),
            ],
        )?;
    }
    if compatibility.capabilities().debug.getuservalue {
        set_builtins(
            runtime,
            debug,
            &[
                ("getuservalue", BuiltinFunction::DebugGetUserValue),
                ("setuservalue", BuiltinFunction::DebugSetUserValue),
            ],
        )?;
    }
    set_builtins(
        runtime,
        debug,
        &[
            ("getfenv", BuiltinFunction::GetFenv),
            ("setfenv", BuiltinFunction::SetFenv),
        ],
    )?;
    set_natives(
        runtime,
        debug,
        &[
            ("getmetatable", debug_getmetatable),
            ("setmetatable", debug_setmetatable),
        ],
    )?;
    runtime.set_global("debug", Value::Table(debug));
    Ok(())
}

/// Installs a fixed-C-locale subset of Lua's `os` library for the default
/// profile.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_os(runtime: &mut Runtime) -> RuntimeResult<()> {
    install_os_with_profile(runtime, CompatibilityProfile::default())
}

/// Installs a fixed-C-locale subset of Lua's `os` library.
///
/// The embedding never changes the host process locale. Locale names other
/// than `"C"` report unavailability so Lua code can take its fallback path.
/// `os.exit` does not exit the process: it raises [`RuntimeError::Exit`],
/// which no Lua code can catch, and leaves exiting to the host.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_os_with_profile(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<()> {
    let os = runtime.create_table()?;
    runtime.set_library_name(os, "os");
    let exit: NativeCallback = match compatibility {
        CompatibilityProfile::Lua51 => os_exit_lua51,
        CompatibilityProfile::Lua52 | CompatibilityProfile::LuaJit21 => os_exit_truncating,
        CompatibilityProfile::Lua53 | CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55 => {
            os_exit
        }
    };
    set_natives(runtime, os, &[("exit", exit)])?;
    set_natives(
        runtime,
        os,
        &[
            ("clock", os_clock),
            ("date", os_date),
            ("difftime", os_difftime),
            ("execute", os_execute),
            ("getenv", os_getenv),
            ("tmpname", os_tmpname),
            ("remove", os_remove),
            ("rename", os_rename),
            ("setlocale", os_setlocale),
            ("time", os_time),
        ],
    )?;
    runtime.set_global("os", Value::Table(os));
    Ok(())
}

/// Installs Lua's `io` library and the shared file metatable.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_io(runtime: &mut Runtime) -> RuntimeResult<()> {
    let io = runtime.create_table()?;
    runtime.set_library_name(io, "io");
    let stdin = runtime.create_stdin_file()?;
    let stdout = runtime.create_stdout_file()?;
    let stderr = runtime.create_stderr_file()?;
    set_natives(
        runtime,
        io,
        &[
            ("tmpfile", io_tmpfile),
            ("write", io_write),
            ("read", io_read),
            ("input", io_input),
            ("output", io_output),
            ("close", io_close),
            ("open", io_open),
            ("popen", io_popen),
            ("lines", io_lines),
            ("flush", io_flush),
            ("type", io_type),
        ],
    )?;
    set(runtime, io, "stdin", Value::File(stdin))?;
    set(runtime, io, "stdout", Value::File(stdout))?;
    set(runtime, io, "stderr", Value::File(stderr))?;

    let file_methods = runtime.create_table()?;

    runtime.set_library_name(file_methods, "file");
    set_natives(
        runtime,
        file_methods,
        &[
            ("write", file_write),
            ("seek", file_seek),
            ("read", file_read),
            ("close", file_close),
            ("flush", file_flush),
            ("setvbuf", file_setvbuf),
            ("lines", file_lines),
        ],
    )?;
    let file_metatable = runtime.create_table()?;
    set(
        runtime,
        file_metatable,
        "__index",
        Value::Table(file_methods),
    )?;
    set_natives(
        runtime,
        file_metatable,
        &[("__gc", file_gc), ("__tostring", file_tostring)],
    )?;
    runtime.raw_set_metatable(&Value::File(stdin), Some(file_metatable))?;
    runtime.set_default_input(Some(stdin))?;
    runtime.set_default_output(Some(stdout))?;
    runtime.set_global("io", Value::Table(io));
    Ok(())
}

/// Makes files to-be-closed values, as in Lua 5.4, where `io.lines(name)`
/// returns its file as the loop's closing value. `__close` is `__gc` itself.
fn install_file_close(runtime: &mut Runtime) -> RuntimeResult<()> {
    let stdin = runtime
        .default_input()
        .ok_or_else(|| RuntimeError::message("default input is unavailable"))?;
    if let Some(metatable) = runtime.raw_metatable(&Value::File(stdin))? {
        let gc = runtime.table_get(metatable, &Value::from("__gc"))?;
        set(runtime, metatable, "__close", gc)?;
    }
    Ok(())
}

/// Installs Lua 5.1's `table` library.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_table(runtime: &mut Runtime) -> RuntimeResult<()> {
    install_table_with_profile(runtime, CompatibilityProfile::Lua51)
}

fn install_table_extensions(runtime: &mut Runtime) -> RuntimeResult<()> {
    let Value::Table(package) = runtime.get_global("package") else {
        unreachable!("package was installed")
    };
    let Value::Table(preload) = runtime.table_get(package, &Value::from("preload"))? else {
        unreachable!("package.preload was installed")
    };
    for (name, callback) in [
        ("new", crate::stdlib::table_ext::new as NativeCallback),
        ("clear", crate::stdlib::table_ext::clear),
    ] {
        let function = runtime.register_native(callback)?;
        let module = format!("table.{name}");
        let loader = runtime.register_native_closure(move |runtime, _| {
            if let Value::Table(table) = runtime.get_global("table") {
                runtime.table_set(table, &Value::from(name), Value::NativeFunction(function))?;
            }
            Ok(vec![Value::NativeFunction(function)])
        })?;
        set(runtime, preload, &module, Value::NativeFunction(loader))?;
    }
    Ok(())
}

pub(crate) fn install_table_with_profile(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<()> {
    let lua51 = compatibility.is_lua51_family();
    let table = runtime.create_table()?;
    runtime.set_library_name(table, "table");
    // `table.concat` always returns one string.
    // Lua 5.3 and later read and write the elements through metamethods; their
    // natives decline a table that has any and the VM runs the call itself.
    let follows_metamethods = compatibility.is_lua53_or_later();
    let concat = runtime.register_native_scalar(if follows_metamethods {
        table_concat_meta
    } else {
        table_concat
    })?;
    runtime.set_direct_native_kind(concat, DirectNativeKind::TableConcat);
    set(runtime, table, "concat", Value::NativeFunction(concat))?;
    if lua51 {
        set_natives(
            runtime,
            table,
            &[("getn", table_getn), ("maxn", table_maxn)],
        )?;
    }
    if follows_metamethods {
        for (name, callback, kind) in [
            (
                "insert",
                table_insert_meta as NativeCallback,
                DirectNativeKind::TableInsert,
            ),
            ("remove", table_remove_meta, DirectNativeKind::TableRemove),
        ] {
            let function = runtime.register_native(callback)?;
            runtime.set_direct_native_kind(function, kind);
            set(runtime, table, name, Value::NativeFunction(function))?;
        }
    } else {
        for (name, callback, kind) in [
            (
                "insert",
                table_insert as NativeCallback,
                DirectNativeKind::TableInsertRaw,
            ),
            ("remove", table_remove, DirectNativeKind::TableRemoveRaw),
        ] {
            let function = runtime.register_native(callback)?;
            runtime.set_direct_native_kind(function, kind);
            set(runtime, table, name, Value::NativeFunction(function))?;
        }
    }
    if lua51 {
        set_builtins(
            runtime,
            table,
            &[
                ("foreach", BuiltinFunction::TableForeach),
                ("foreachi", BuiltinFunction::TableForeachi),
            ],
        )?;
    }
    set_builtins(runtime, table, &[("sort", BuiltinFunction::TableSort)])?;
    if !lua51 {
        set_natives(runtime, table, &[("pack", table_pack)])?;
        let unpack_callback: NativeCallback = if follows_metamethods {
            unpack_meta
        } else {
            unpack
        };
        let function = runtime.register_native(unpack_callback)?;
        runtime.set_direct_native_kind(
            function,
            if follows_metamethods {
                DirectNativeKind::TableUnpack
            } else {
                DirectNativeKind::Unpack
            },
        );
        set(runtime, table, "unpack", Value::NativeFunction(function))?;
    }
    if compatibility.capabilities().libraries.table_move {
        if follows_metamethods {
            let function = runtime.register_native(table_move_meta)?;
            runtime.set_direct_native_kind(function, DirectNativeKind::TableMove);
            set(runtime, table, "move", Value::NativeFunction(function))?;
        } else {
            set_natives(runtime, table, &[("move", table_move)])?;
        }
    }
    if compatibility == CompatibilityProfile::Lua55 {
        set_natives(runtime, table, &[("create", table_create)])?;
    }
    runtime.set_global("table", Value::Table(table));
    Ok(())
}

/// Installs Lua 5.1's `math` library.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_math(runtime: &mut Runtime) -> RuntimeResult<()> {
    install_math_with_profile(runtime, CompatibilityProfile::Lua51)
}

#[allow(
    clippy::too_many_lines,
    reason = "keeps Lua callbacks and their numeric fast paths in the same registration table"
)]
pub(crate) fn install_math_with_profile(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<()> {
    let math = runtime.create_table()?;
    runtime.set_library_name(math, "math");
    let modern_random = matches!(
        compatibility,
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    );
    runtime.set_random_modern(modern_random);
    let luajit = compatibility == CompatibilityProfile::LuaJit21;
    runtime.set_random_luajit(luajit);
    let (log, log_base): (NativeScalarCallback, Option<NativeNumericBinary>) = match compatibility {
        CompatibilityProfile::Lua51 => (math_log, None),
        CompatibilityProfile::LuaJit21 => (math_log_luajit, Some(numeric_log_base_luajit)),
        CompatibilityProfile::Lua52 => (math_log_with_base_lua52, Some(numeric_log_base_lua52)),
        _ => (math_log_with_base, Some(numeric_log_base)),
    };
    let random: NativeCallback = match compatibility {
        CompatibilityProfile::Lua51 | CompatibilityProfile::Lua52 => math_random,
        CompatibilityProfile::LuaJit21 => math_random_luajit,
        CompatibilityProfile::Lua53 => math_random_lua53,
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55 => math_random_lua54,
    };
    let floor = runtime.register_native_scalar(floor_scalar)?;
    for (name, function) in [
        ("abs", runtime.register_native_scalar(math_abs)?),
        (
            "acos",
            runtime.register_native_numeric_unary(math_acos, numeric_acos)?,
        ),
        (
            "asin",
            runtime.register_native_numeric_unary(math_asin, numeric_asin)?,
        ),
        (
            "atan",
            runtime.register_native_scalar(if compatibility.is_lua53_or_later() {
                math_atan_modern
            } else {
                math_atan
            })?,
        ),
        ("atan2", runtime.register_native(math_atan2)?),
        ("ceil", runtime.register_native(math_ceil)?),
        (
            "cos",
            runtime.register_native_numeric_unary(math_cos, numeric_cos)?,
        ),
        (
            "cosh",
            runtime.register_native_numeric_unary(math_cosh, numeric_cosh)?,
        ),
        (
            "deg",
            runtime.register_native_numeric_unary(math_deg, numeric_deg)?,
        ),
        (
            "exp",
            runtime.register_native_numeric_unary(math_exp, numeric_exp)?,
        ),
        ("floor", floor),
        (
            "fmod",
            runtime.register_native_numeric_binary(math_fmod, numeric_fmod)?,
        ),
        ("frexp", runtime.register_native(math_frexp)?),
        ("ldexp", runtime.register_native(math_ldexp)?),
        (
            "log",
            runtime.register_native_numeric_unary(log, numeric_log)?,
        ),
        (
            "log10",
            runtime.register_native_numeric_unary(math_log10, numeric_log10)?,
        ),
        (
            "max",
            if luajit {
                runtime
                    .register_native_numeric_binary(math_max_scalar_luajit, numeric_max_luajit)?
            } else {
                runtime.register_native_numeric_binary(math_max_scalar, numeric_max)?
            },
        ),
        (
            "min",
            if luajit {
                runtime
                    .register_native_numeric_binary(math_min_scalar_luajit, numeric_min_luajit)?
            } else {
                runtime.register_native_numeric_binary(math_min_scalar, numeric_min)?
            },
        ),
        ("mod", runtime.register_native(math_mod)?),
        ("modf", runtime.register_native(math_modf)?),
        ("random", runtime.register_native(random)?),
        (
            "randomseed",
            runtime.register_native(if modern_random {
                math_randomseed_modern
            } else if luajit {
                math_randomseed_luajit
            } else {
                math_randomseed
            })?,
        ),
        (
            "rad",
            runtime.register_native_numeric_unary(math_rad, numeric_rad)?,
        ),
        (
            "sin",
            runtime.register_native_numeric_unary(sin_scalar, numeric_sin)?,
        ),
        (
            "sinh",
            runtime.register_native_numeric_unary(math_sinh, numeric_sinh)?,
        ),
        (
            "sqrt",
            runtime.register_native_numeric_unary(math_sqrt, numeric_sqrt)?,
        ),
        (
            "tan",
            runtime.register_native_numeric_unary(math_tan, numeric_tan)?,
        ),
        (
            "tanh",
            runtime.register_native_numeric_unary(math_tanh, numeric_tanh)?,
        ),
    ] {
        match name {
            "max" => runtime.set_direct_native_kind(function, DirectNativeKind::MathMax),
            "min" => runtime.set_direct_native_kind(function, DirectNativeKind::MathMin),
            "floor" => runtime.set_direct_native_kind(function, DirectNativeKind::MathFloor),
            "ceil" => runtime.set_direct_native_kind(function, DirectNativeKind::MathCeil),
            "abs" => runtime.set_direct_native_kind(function, DirectNativeKind::MathAbs),
            "random" => runtime.set_direct_native_kind(function, DirectNativeKind::MathRandom),
            "modf" => runtime.set_direct_native_kind(function, DirectNativeKind::MathModf),
            "log" => match log_base {
                None => runtime.set_direct_native_kind(function, DirectNativeKind::MathLog),
                Some(binary) => {
                    runtime.set_native_log_base(function, binary);
                    runtime.set_direct_native_kind(function, DirectNativeKind::MathLogBase);
                }
            },
            _ => {}
        }
        set(runtime, math, name, Value::NativeFunction(function))?;
    }
    // Lua 5.5 dropped `math.pow`.
    if compatibility != CompatibilityProfile::Lua55 {
        let pow = runtime.register_native_numeric_binary(math_pow, numeric_pow)?;
        runtime.set_direct_native_kind(pow, DirectNativeKind::MathPow);
        set(runtime, math, "pow", Value::NativeFunction(pow))?;
    }
    set(runtime, math, "huge", Value::Number(f64::INFINITY))?;
    set(runtime, math, "pi", Value::Number(std::f64::consts::PI))?;
    if compatibility.is_lua53_or_later() {
        set_natives(runtime, math, &[("ult", math_ult)])?;
        for (name, callback, kind) in [
            (
                "type",
                math_type as NativeCallback,
                DirectNativeKind::MathType,
            ),
            ("tointeger", math_tointeger, DirectNativeKind::MathToInteger),
        ] {
            let function = runtime.register_native(callback)?;
            runtime.set_direct_native_kind(function, kind);
            set(runtime, math, name, Value::NativeFunction(function))?;
        }
        set(runtime, math, "maxinteger", Value::Integer(i64::MAX))?;
        set(runtime, math, "mininteger", Value::Integer(i64::MIN))?;
    }
    runtime.set_global("math", Value::Table(math));
    Ok(())
}

/// Installs Lua 5.1's `string` library.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_string(runtime: &mut Runtime) -> RuntimeResult<()> {
    install_string_with_profile(runtime, CompatibilityProfile::Lua51)
}

pub(crate) fn install_string_with_profile(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<()> {
    let lua51 = compatibility.is_lua51_family();
    let string = runtime.create_table()?;
    runtime.set_library_name(string, "string");
    let byte = runtime.register_native(string_byte)?;
    runtime.set_direct_native_kind(byte, DirectNativeKind::StringByte);
    set(runtime, string, "byte", Value::NativeFunction(byte))?;
    // Lua 5.1 and LuaJIT word the range error differently from 5.2 and later.
    let char: NativeCallback = if matches!(
        compatibility,
        CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21
    ) {
        string_char_51
    } else {
        string_char
    };
    let char = runtime.register_native(char)?;
    runtime.set_direct_native_kind(char, DirectNativeKind::StringChar);
    set(runtime, string, "char", Value::NativeFunction(char))?;
    let find = runtime.register_native_paired(string_find, string_find_pair)?;
    runtime.set_direct_native_kind(find, DirectNativeKind::Find);
    set(runtime, string, "find", Value::NativeFunction(find))?;
    let match_native = runtime.register_native(string_match)?;
    runtime.set_direct_native_kind(match_native, DirectNativeKind::Match);
    set(
        runtime,
        string,
        "match",
        Value::NativeFunction(match_native),
    )?;

    // Lua 5.1 `string.format` stays a plain native. Later profiles route
    // through the VM builtin so `%s` can invoke `__tostring`.
    let raw_format = match compatibility {
        // The Lua 5.1 and LuaJIT natives return exactly one string, so they
        // register in their single-result form.
        CompatibilityProfile::Lua51 => runtime.register_native_scalar(string_format_scalar)?,
        CompatibilityProfile::LuaJit21 => {
            runtime.register_native_scalar(string_format_luajit_scalar)?
        }
        CompatibilityProfile::Lua52 | CompatibilityProfile::Lua53 => {
            runtime.register_native(string_format_modern)?
        }
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55 => {
            runtime.register_native(string_format_with_pointer)?
        }
    };
    let format = if lua51 {
        // LuaJIT's `%s` honours `__tostring`: its native declines such an
        // argument and the VM converts it (`print`'s way).
        if compatibility == CompatibilityProfile::LuaJit21 {
            runtime.set_direct_native_kind(raw_format, DirectNativeKind::StringFormatLuaJit);
            let unchecked = runtime.register_native_scalar(string_format_scalar)?;
            runtime.set_direct_native_kind(unchecked, DirectNativeKind::StringFormatLuaJitRaw);
        } else {
            // Lua 5.1's `string.format` is the formatter itself, which a
            // trace may call.
            runtime.set_direct_native_kind(raw_format, DirectNativeKind::Format);
        }
        Value::NativeFunction(raw_format)
    } else {
        runtime.set_direct_native_kind(raw_format, DirectNativeKind::Format);
        Value::BuiltinFunction(BuiltinFunction::StringFormat)
    };
    set(runtime, string, "format", format)?;
    set_builtins(runtime, string, &[("gsub", BuiltinFunction::StringGsub)])?;
    // Lua 5.4 added the start position.
    let gmatch = if matches!(
        compatibility,
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    ) {
        runtime.register_native(string_gmatch_init)?
    } else {
        runtime.register_native(string_gmatch)?
    };
    runtime.set_direct_native_kind(gmatch, DirectNativeKind::Gmatch);
    set(runtime, string, "gmatch", Value::NativeFunction(gmatch))?;
    if lua51 {
        // Lua 5.1 keeps the obsolete spelling as the same function value.
        set(runtime, string, "gfind", Value::NativeFunction(gmatch))?;
    }
    let len = runtime.register_native_scalar(string_len_scalar)?;
    runtime.set_direct_native_kind(len, DirectNativeKind::StringLen);
    set(runtime, string, "len", Value::NativeFunction(len))?;
    set_builtins(runtime, string, &[("dump", BuiltinFunction::StringDump)])?;
    let lower = runtime.register_native(string_lower)?;
    runtime.set_direct_native_kind(lower, DirectNativeKind::StringLower);
    set(runtime, string, "lower", Value::NativeFunction(lower))?;
    let sub = runtime.register_native_scalar(string_sub_scalar)?;
    runtime.set_direct_native_kind(sub, DirectNativeKind::StringSlice);
    set(runtime, string, "sub", Value::NativeFunction(sub))?;
    // LuaJIT 2.1 added the separator that stock 5.1 lacks.
    let rep: NativeCallback = if compatibility == CompatibilityProfile::Lua51 {
        string_rep
    } else {
        string_rep_with_separator
    };
    for (name, callback, kind) in [
        ("rep", rep, DirectNativeKind::StringRep),
        ("reverse", string_reverse, DirectNativeKind::StringReverse),
        ("upper", string_upper, DirectNativeKind::StringUpper),
    ] {
        let function = runtime.register_native(callback)?;
        runtime.set_direct_native_kind(function, kind);
        set(runtime, string, name, Value::NativeFunction(function))?;
    }
    if compatibility.is_lua53_or_later() {
        // The same test picks the native and its probes below.
        runtime.set_unpack_clamped(matches!(
            compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ));
        for (name, callback, probe, kind) in [
            (
                "pack",
                string_pack as NativeCallback,
                string_pack_probe as NativeResultsProbe,
                DirectNativeKind::StringPack,
            ),
            if matches!(
                compatibility,
                CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
            ) {
                (
                    "unpack",
                    string_unpack_clamped,
                    string_unpack_clamped_probe,
                    DirectNativeKind::StringUnpack,
                )
            } else {
                (
                    "unpack",
                    string_unpack,
                    string_unpack_probe,
                    DirectNativeKind::StringUnpack,
                )
            },
            (
                "packsize",
                string_packsize,
                string_packsize_probe,
                DirectNativeKind::StringPacksize,
            ),
        ] {
            let function = runtime.register_native(callback)?;
            runtime.set_direct_native_kind(function, kind);
            runtime.set_native_results_probe(function, probe);
            if let Some(fixed) = string_pack_fixed_probe_for(kind, compatibility) {
                runtime.set_native_fixed_results_probe(function, fixed);
            }
            set(runtime, string, name, Value::NativeFunction(function))?;
        }
    }
    runtime.set_global("string", Value::Table(string));
    install_string_metatable(runtime, string, compatibility)
}

/// Gives strings their shared metatable, whose `__index` is the `string`
/// table, so `("x"):upper()` finds the library. Lua 5.4 adds arithmetic
/// metamethods there, which convert numeric strings.
fn install_string_metatable(
    runtime: &mut Runtime,
    string: TableHandle,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<()> {
    let metatable = runtime.create_table()?;
    set(runtime, metatable, "__index", Value::Table(string))?;
    if matches!(
        compatibility,
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    ) {
        for (name, callback, kind) in [
            (
                "__add",
                string_add as NativeScalarCallback,
                DirectNativeKind::StringAdd,
            ),
            ("__sub", string_sub, DirectNativeKind::StringSub),
            ("__mul", string_mul, DirectNativeKind::StringMul),
            ("__mod", string_mod, DirectNativeKind::StringMod),
            ("__pow", string_pow, DirectNativeKind::StringPow),
            ("__div", string_div, DirectNativeKind::StringDiv),
            ("__idiv", string_idiv, DirectNativeKind::StringFloorDiv),
            ("__unm", string_unm, DirectNativeKind::StringUnm),
        ] {
            let function = runtime.register_native_dispatched(callback)?;
            runtime.set_direct_native_kind(function, kind);
            set(runtime, metatable, name, Value::NativeFunction(function))?;
        }
    }
    runtime.raw_set_metatable(&Value::from(""), Some(metatable))
}

/// The probe of `string.pack` or `string.unpack` that allocates no list for
/// its results.
fn string_pack_fixed_probe_for(
    kind: DirectNativeKind,
    compatibility: CompatibilityProfile,
) -> Option<NativeFixedResultsProbe> {
    match kind {
        DirectNativeKind::StringPack => Some(string_pack_fixed_probe),
        DirectNativeKind::StringUnpack => Some(
            if matches!(
                compatibility,
                CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
            ) {
                string_unpack_clamped_fixed_probe
            } else {
                string_unpack_fixed_probe
            },
        ),
        _ => None,
    }
}

/// The probe of a `utf8` native that allocates no list for its results.
fn utf8_fixed_probe_for(
    kind: DirectNativeKind,
    compatibility: CompatibilityProfile,
) -> Option<NativeFixedResultsProbe> {
    let lua53 = compatibility == CompatibilityProfile::Lua53;
    Some(match (kind, compatibility) {
        (DirectNativeKind::Utf8Char, _) if lua53 => utf8_char_lua53_fixed_probe,
        (DirectNativeKind::Utf8Char, _) => utf8_char_fixed_probe,
        (DirectNativeKind::Utf8Codepoint, _) if lua53 => utf8_codepoint_lua53_fixed_probe,
        (DirectNativeKind::Utf8Codepoint, _) => utf8_codepoint_fixed_probe,
        (DirectNativeKind::Utf8Len, _) if lua53 => utf8_len_lua53_fixed_probe,
        (DirectNativeKind::Utf8Len, _) => utf8_len_fixed_probe,
        (DirectNativeKind::Utf8Offset, CompatibilityProfile::Lua53) => {
            utf8_offset_lua53_fixed_probe
        }
        (DirectNativeKind::Utf8Offset, CompatibilityProfile::Lua55) => {
            utf8_offset_lua55_fixed_probe
        }
        (DirectNativeKind::Utf8Offset, _) => utf8_offset_fixed_probe,
        _ => return None,
    })
}

/// Installs the byte-oriented UTF-8 helpers introduced by Lua 5.3.
pub(crate) fn install_utf8(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
) -> RuntimeResult<()> {
    let utf8 = runtime.create_table()?;
    runtime.set_library_name(utf8, "utf8");
    let lua53 = compatibility == CompatibilityProfile::Lua53;
    let (offset, offset_probe): (NativeCallback, NativeResultsProbe) = match compatibility {
        CompatibilityProfile::Lua53 => (utf8_offset_lua53, utf8_offset_lua53_probe),
        CompatibilityProfile::Lua55 => (utf8_offset_lua55, utf8_offset_lua55_probe),
        _ => (utf8_offset, utf8_offset_probe),
    };
    let natives: [(&str, NativeCallback, NativeResultsProbe, DirectNativeKind); 4] = [
        (
            "char",
            if lua53 { utf8_char_lua53 } else { utf8_char },
            if lua53 {
                utf8_char_lua53_probe
            } else {
                utf8_char_probe
            },
            DirectNativeKind::Utf8Char,
        ),
        (
            "codepoint",
            if lua53 {
                utf8_codepoint_lua53
            } else {
                utf8_codepoint
            },
            if lua53 {
                utf8_codepoint_lua53_probe
            } else {
                utf8_codepoint_probe
            },
            DirectNativeKind::Utf8Codepoint,
        ),
        (
            "len",
            if lua53 { utf8_len_lua53 } else { utf8_len },
            if lua53 {
                utf8_len_lua53_probe
            } else {
                utf8_len_probe
            },
            DirectNativeKind::Utf8Len,
        ),
        ("offset", offset, offset_probe, DirectNativeKind::Utf8Offset),
    ];
    for (name, callback, probe, kind) in natives {
        let function = runtime.register_native(callback)?;
        runtime.set_direct_native_kind(function, kind);
        runtime.set_native_results_probe(function, probe);
        if let Some(fixed) = utf8_fixed_probe_for(kind, compatibility) {
            runtime.set_native_fixed_results_probe(function, fixed);
        }
        set(runtime, utf8, name, Value::NativeFunction(function))?;
    }
    let legacy = lua53;
    let strict = runtime.register_native(if legacy { utf8_next_lua53 } else { utf8_next })?;
    runtime.set_direct_native_kind(
        strict,
        if legacy {
            DirectNativeKind::Utf8Next53
        } else {
            DirectNativeKind::Utf8Next
        },
    );
    runtime.set_native_results_probe(
        strict,
        if legacy {
            utf8_next_lua53_probe
        } else {
            utf8_next_probe
        },
    );
    let lax = runtime.register_native(utf8_next_lax)?;
    runtime.set_direct_native_kind(lax, DirectNativeKind::Utf8NextLax);
    runtime.set_native_results_probe(lax, utf8_next_lax_probe);
    // No library table holds the steps, so name them for abort counters.
    runtime.set_native_name(strict, "utf8.codes");
    runtime.set_native_name(lax, "utf8.codes");
    let codes = runtime.register_native_closure(move |runtime, arguments| {
        utf8_codes_with_iterators(runtime, arguments, strict, lax, legacy)
    })?;
    runtime.set_direct_native_kind(codes, DirectNativeKind::Utf8Codes);
    set(runtime, utf8, "codes", Value::NativeFunction(codes))?;
    // Lua exposes this pattern as a convenience for byte-pattern consumers.
    set(
        runtime,
        utf8,
        "charpattern",
        Value::bytes(if compatibility == CompatibilityProfile::Lua53 {
            b"[\x00-\x7f\xc2-\xf4][\x80-\xbf]*".to_vec()
        } else {
            b"[\x00-\x7f\xc2-\xfd][\x80-\xbf]*".to_vec()
        }),
    )?;
    runtime.set_global("utf8", Value::Table(utf8));
    Ok(())
}

/// Installs `LuaJIT`'s `bit` library, regardless of the compatibility profile.
///
/// If `package` is already installed, also registers the table in
/// `package.loaded` so `require("bit")` returns it.
///
/// # Errors
///
/// Returns allocation or registration errors from the runtime.
pub fn install_bit(runtime: &mut Runtime) -> RuntimeResult<()> {
    let bit = runtime.create_table()?;
    runtime.set_library_name(bit, "bit");
    set_bit_natives(
        runtime,
        bit,
        &[
            ("tobit", BitNative::Unary(tobit, numeric_tobit)),
            ("bnot", BitNative::Unary(bnot, numeric_bnot)),
            ("band", BitNative::Wide(band, numeric_band, wide_band)),
            ("bor", BitNative::Wide(bor, numeric_bor, wide_bor)),
            ("bxor", BitNative::Wide(bxor, numeric_bxor, wide_bxor)),
            (
                "lshift",
                BitNative::Wide(lshift, numeric_lshift, wide_lshift),
            ),
            (
                "rshift",
                BitNative::Wide(rshift, numeric_rshift, wide_rshift),
            ),
            (
                "arshift",
                BitNative::Wide(arshift, numeric_arshift, wide_arshift),
            ),
            ("rol", BitNative::Wide(rol, numeric_rol, wide_rol)),
            ("ror", BitNative::Wide(ror, numeric_ror, wide_ror)),
            ("bswap", BitNative::Unary(bswap, numeric_bswap)),
        ],
    )?;
    set_natives(runtime, bit, &[("tohex", tohex)])?;
    // The trace JIT recognizes these by handle.
    for (name, kind) in [
        ("tobit", DirectNativeKind::BitTobit),
        ("bnot", DirectNativeKind::BitBnot),
        ("band", DirectNativeKind::BitBand),
        ("bor", DirectNativeKind::BitBor),
        ("bxor", DirectNativeKind::BitBxor),
        ("lshift", DirectNativeKind::BitLshift),
        ("rshift", DirectNativeKind::BitRshift),
        ("arshift", DirectNativeKind::BitArshift),
        ("rol", DirectNativeKind::BitRol),
        ("ror", DirectNativeKind::BitRor),
    ] {
        if let Value::NativeFunction(function) = runtime.table_get(bit, &Value::from(name))? {
            runtime.set_direct_native_kind(function, kind);
        }
    }
    runtime.set_global("bit", Value::Table(bit));
    if let Value::Table(package) = runtime.get_global("package")
        && let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded"))?
    {
        set(runtime, loaded, "bit", Value::Table(bit))?;
    }
    Ok(())
}

/// Installs Oximond's LuaJIT-style `jit` control table for any profile.
///
/// Global `on`, `off`, `status`, and `flush` operations control the VM's
/// selected JIT engine. An interpreter-only VM always reports disabled.
///
/// # Errors
///
/// Returns allocation or native registration errors from the runtime.
pub fn install_jit(runtime: &mut Runtime) -> RuntimeResult<()> {
    let jit = runtime.create_table()?;
    runtime.set_library_name(jit, "jit");
    set_natives(
        runtime,
        jit,
        &[
            ("on", jit_on),
            ("off", jit_off),
            ("status", jit_status),
            ("flush", jit_flush),
        ],
    )?;
    set(
        runtime,
        jit,
        "version",
        Value::from(concat!("Oximond JIT ", env!("CARGO_PKG_VERSION"))),
    )?;
    // Code written for LuaJIT gates on `jit.version_num >= 20000` (2.0) and
    // feature-tests 2.1 extensions. Report the LuaJIT 2.1 API this matches.
    set(runtime, jit, "version_num", Value::Number(20100.0))?;
    set(runtime, jit, "os", Value::from(jit_os()))?;
    set(runtime, jit, "arch", Value::from(jit_arch()))?;
    let opt = runtime.create_table()?;
    runtime.set_library_name(opt, "jit.opt");
    set_natives(runtime, opt, &[("start", jit_opt_start)])?;
    set(runtime, jit, "opt", Value::Table(opt))?;
    runtime.set_global("jit", Value::Table(jit));
    if let Value::Table(package) = runtime.get_global("package")
        && let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded"))?
    {
        set(runtime, loaded, "jit", Value::Table(jit))?;
        set(runtime, loaded, "jit.opt", Value::Table(opt))?;
    }
    Ok(())
}

fn jit_os() -> &'static str {
    match std::env::consts::OS {
        "linux" => "Linux",
        "windows" => "Windows",
        "macos" => "OSX",
        "freebsd" | "openbsd" | "netbsd" | "dragonfly" => "BSD",
        _ => "Other",
    }
}

fn jit_arch() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x64",
        "x86" => "x86",
        "aarch64" => "arm64",
        "arm" => "arm",
        _ => "other",
    }
}

/// Installs Lua 5.2/5.3's unsigned 32-bit compatibility library.
pub(crate) fn install_bit32(runtime: &mut Runtime) -> RuntimeResult<()> {
    let bit32 = runtime.create_table()?;
    runtime.set_library_name(bit32, "bit32");
    set_natives(
        runtime,
        bit32,
        &[
            ("bnot", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "bnot")
            }),
            ("band", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "band")
            }),
            ("bor", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "bor")
            }),
            ("bxor", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "bxor")
            }),
            ("lshift", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "lshift")
            }),
            ("rshift", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "rshift")
            }),
            ("arshift", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "arshift")
            }),
            ("lrotate", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "lrotate")
            }),
            ("rrotate", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "rrotate")
            }),
            ("btest", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "btest")
            }),
            ("extract", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "extract")
            }),
            ("replace", |runtime, arguments| {
                crate::stdlib::bit::bit32_operation(runtime, arguments, "replace")
            }),
        ],
    )?;
    // The trace JIT recognizes these by handle.
    for (name, kind) in [
        ("band", DirectNativeKind::Bit32Band),
        ("bor", DirectNativeKind::Bit32Bor),
        ("bxor", DirectNativeKind::Bit32Bxor),
        ("bnot", DirectNativeKind::Bit32Bnot),
        ("lshift", DirectNativeKind::Bit32Lshift),
        ("rshift", DirectNativeKind::Bit32Rshift),
        ("arshift", DirectNativeKind::Bit32Arshift),
        ("lrotate", DirectNativeKind::Bit32Lrotate),
        ("rrotate", DirectNativeKind::Bit32Rrotate),
        ("btest", DirectNativeKind::Bit32Btest),
        ("extract", DirectNativeKind::Bit32Extract),
        ("replace", DirectNativeKind::Bit32Replace),
    ] {
        if let Value::NativeFunction(function) = runtime.table_get(bit32, &Value::from(name))? {
            runtime.set_direct_native_kind(function, kind);
        }
    }
    runtime.set_global("bit32", Value::Table(bit32));
    Ok(())
}
