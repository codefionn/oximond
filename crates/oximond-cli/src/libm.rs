//! The libm functions the interpreter calls, defined by the executable.
//!
//! The only libm symbols the binary imports are the fifteen below (the rest of
//! `math` is inlined or lives in libc). Loading libm.so.6 for them costs about
//! 0.03 ms, 37,000 instructions and 8 page faults on every start, which is
//! 5% of `startup/empty`. The executable defines them instead; each forwards
//! to the libm function of the same name, which it opens with `dlopen` on the
//! first call. With nothing left to import, the linker drops libm from the
//! needed libraries (`--as-needed`). A script that calls `math.sin` pays one
//! `dlopen` of about 0.03 ms (54,000 instructions, 16 page faults), once, and
//! one `jmp` more on every call: `stdlib/math` runs 1% more instructions.
//! build.rs exports the functions by name (`--export-dynamic-symbol`), so
//! native modules and the C++ bridge that import them reach these and not a
//! second copy of libm.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::atomic::{AtomicPtr, Ordering};

unsafe extern "C" {
    fn dlopen(file: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

/// The `dlopen` handle of libm.so.6, once opened.
static LIBM: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

/// Looks `name` up in libm, opening it first if needed, and caches the address
/// in `slot`. Two threads may both resolve; they store the same address.
#[cold]
#[inline(never)]
fn resolve(slot: &AtomicPtr<c_void>, name: &CStr) -> *mut c_void {
    let mut library = LIBM.load(Ordering::Acquire);
    if library.is_null() {
        // SAFETY: the argument is a NUL-terminated string.
        library = unsafe { dlopen(c"libm.so.6".as_ptr(), RTLD_NOW) };
        assert!(!library.is_null(), "cannot load libm.so.6");
        LIBM.store(library, Ordering::Release);
    }
    // SAFETY: `library` is a live handle and `name` is NUL-terminated.
    let address = unsafe { dlsym(library, name.as_ptr()) };
    assert!(!address.is_null(), "libm.so.6 lacks {name:?}");
    slot.store(address, Ordering::Release);
    address
}

macro_rules! argument {
    ($name:ident) => {
        f64
    };
}

/// On x86-64 each function is one `jmp` through its slot, as a PLT entry is:
/// the slot starts at a resolver that takes the same arguments, looks the
/// function up, stores it, and calls it. A wrapper that tested the slot on
/// every call added 5 instructions to each `math.sin`, 6% of `stdlib/math`.
#[cfg(target_arch = "x86_64")]
macro_rules! forward {
    ($($name:ident($($argument:ident),+) $slot:ident $resolver:ident;)*) => {$(
        static $slot: AtomicPtr<c_void> = AtomicPtr::new($resolver as *mut c_void);

        extern "C" fn $resolver($($argument: f64),+) -> f64 {
            let address = resolve(&$slot, const {
                match CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()) {
                    Ok(name) => name,
                    Err(_) => panic!("a function name"),
                }
            });
            // SAFETY: `address` is the libm function of this name, which has
            // this C signature.
            let function: extern "C" fn($(argument!($argument)),+) -> f64 =
                unsafe { std::mem::transmute(address) };
            function($($argument),+)
        }

        #[unsafe(no_mangle)]
        #[unsafe(naked)]
        pub extern "C" fn $name($($argument: f64),+) -> f64 {
            std::arch::naked_asm!("jmp qword ptr [rip + {slot}]", slot = sym $slot);
        }
    )*};
}

/// Elsewhere each function tests its slot on every call.
#[cfg(not(target_arch = "x86_64"))]
macro_rules! forward {
    ($($name:ident($($argument:ident),+) $slot:ident $resolver:ident;)*) => {$(
        #[unsafe(no_mangle)]
        pub extern "C" fn $name($($argument: f64),+) -> f64 {
            static SLOT: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
            let mut address = SLOT.load(Ordering::Acquire);
            if address.is_null() {
                address = resolve(&SLOT, const {
                    match CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()) {
                        Ok(name) => name,
                        Err(_) => panic!("a function name"),
                    }
                });
            }
            // SAFETY: `address` is the libm function of this name, which has
            // this C signature.
            let function: extern "C" fn($(argument!($argument)),+) -> f64 =
                unsafe { std::mem::transmute(address) };
            function($($argument),+)
        }
    )*};
}

forward! {
    acos(x) ACOS_SLOT resolve_acos;
    asin(x) ASIN_SLOT resolve_asin;
    atan(x) ATAN_SLOT resolve_atan;
    atan2(y, x) ATAN2_SLOT resolve_atan2;
    cos(x) COS_SLOT resolve_cos;
    cosh(x) COSH_SLOT resolve_cosh;
    exp(x) EXP_SLOT resolve_exp;
    log(x) LOG_SLOT resolve_log;
    log10(x) LOG10_SLOT resolve_log10;
    log2(x) LOG2_SLOT resolve_log2;
    pow(x, y) POW_SLOT resolve_pow;
    sin(x) SIN_SLOT resolve_sin;
    sinh(x) SINH_SLOT resolve_sinh;
    tan(x) TAN_SLOT resolve_tan;
    tanh(x) TANH_SLOT resolve_tanh;
}
