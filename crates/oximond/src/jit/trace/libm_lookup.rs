//! Finds the libm functions Cranelift calls in place of an instruction.
//!
//! Without SSE4.1 (or FMA), `floor`, `ceil`, `trunc`, `nearest` and `fma` lower
//! to calls of the C functions of those names, which the JIT module binds with
//! `dlsym` on the process. The executable no longer links libm (see
//! `crates/oximond-cli/src/libm.rs`), so the process may not hold them until
//! something opens libm. This lookup opens it for exactly those names.
#![allow(unsafe_code)]

use std::ffi::{CString, c_char, c_int, c_void};

unsafe extern "C" {
    fn dlopen(file: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

/// The address of the libm function `name` if it is one Cranelift calls.
pub(super) fn libm_symbol(name: &str) -> Option<*const u8> {
    const LIBCALLS: [&str; 10] = [
        "floor",
        "floorf",
        "ceil",
        "ceilf",
        "trunc",
        "truncf",
        "nearbyint",
        "nearbyintf",
        "fma",
        "fmaf",
    ];
    if !LIBCALLS.contains(&name) {
        return None;
    }
    let name = CString::new(name).ok()?;
    // SAFETY: the argument is a NUL-terminated string. The handle stays open,
    // so the address `dlsym` returns stays valid.
    let library = unsafe { dlopen(c"libm.so.6".as_ptr(), RTLD_NOW) };
    if library.is_null() {
        return None;
    }
    // SAFETY: `library` is a live handle and `name` is NUL-terminated.
    let address = unsafe { dlsym(library, name.as_ptr()) };
    (!address.is_null()).then_some(address.cast::<u8>().cast_const())
}

#[cfg(test)]
mod tests {
    use super::libm_symbol;

    #[test]
    fn finds_only_the_libcalls_in_libm() {
        let address = libm_symbol("floor").expect("libm has floor");
        // SAFETY: `floor` has the C signature `double (double)`.
        let floor: extern "C" fn(f64) -> f64 = unsafe { std::mem::transmute(address) };
        assert_eq!(floor(2.5), 2.0);
        assert!(libm_symbol("fma").is_some());
        assert!(libm_symbol("no_such_function").is_none());
        assert!(libm_symbol("sin").is_none());
    }
}
