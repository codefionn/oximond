//! Lua's standard library natives and the installers that register them.

use std::sync::atomic::AtomicU64;

use chrono::Weekday;

static TEMP_NAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Installs base-library functions and returns the handle for `assert`.
///
/// # Errors
///
/// Returns an error if the runtime has exhausted native-function handles.
mod args;
mod base;
mod bit;
mod bundled;
mod format;
mod installers;
mod jit_ext;
mod math;
mod string_arith;
mod string_pack;
mod strings;
mod table_ext;

pub(crate) use args::substring_range;
pub(crate) use base::{TonumberRules, tonumber_of_text};
pub use installers::{
    install_base, install_base_with_profile, install_base_with_profile_without_io, install_bit,
    install_coroutine, install_debug, install_debug_with_profile,
    install_full_with_profile_and_host_libraries, install_io, install_jit, install_math,
    install_os, install_os_with_profile, install_package, install_package_with_path, install_pure,
    install_pure_with_profile, install_pure_with_profile_and_loading, install_string,
    install_table, set_package_path,
};
pub(crate) use math::numeric_sqrt_address;
pub(crate) use string_pack::{pack_text, unpack_text};
pub use strings::PreparedFormat;
pub(crate) use strings::ResultList;

#[derive(Debug, Clone)]
struct CalendarTime {
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    weekday: Weekday,
    ordinal: u32,
    iso_year: i32,
    iso_week: u32,
    offset_seconds: i32,
    zone: String,
    isdst: bool,
}

#[derive(Clone, Copy)]
enum PackKind {
    Int { signed: bool, size: usize },
    Float(usize),
    Fixed(usize),
    Zero,
    Sized(usize),
    Pad,
    Align,
}

#[derive(Clone, Copy)]
struct PackItem {
    kind: PackKind,
    endian: bool,
    align: usize,
}

#[derive(Clone, Copy, Default)]
struct FormatSpecification {
    flags: u8,
    width: Option<usize>,
    precision: Option<usize>,
    conversion: u8,
}

#[cfg(test)]
mod tests;
