//! Loads the C++ bridge as a shared object the first time native code runs.
//!
//! Linking the bridge into the executable pulls in libstdc++.so, which costs about 2.4M
//! instructions of loader work on every start. Scripts that never call into native code do not
//! need it. A static libstdc++ is not an option: C++ modules bring libstdc++.so, and an exception
//! thrown by one runtime and caught by the other leaves both per-thread uncaught-exception counts
//! wrong. Loaded here, the bridge and the modules share one runtime and one set of counters.
//!
//! The shared object is embedded in the executable and opened through a memfd, so there is no
//! install path to find. It resolves `oxi_*` and the Rust `lua_*` entry points against the
//! executable, which exports them for native modules anyway.
use super::*;
use std::io::Write;
use std::os::fd::FromRawFd;
use std::sync::OnceLock;

static BRIDGE_SO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/liboximond_bridge.so"));

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;
const MFD_CLOEXEC: u32 = 1;
/// Asks for an executable memfd where `vm.memfd_noexec` would otherwise seal it. Kernels older
/// than 6.3 reject the flag with EINVAL and make every memfd executable.
const MFD_EXEC: u32 = 0x10;
// LUA_ERRRUN and LUA_ERRMEM.
const ERRRUN: c_int = 2;
const ERRMEM: c_int = 4;

unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn memfd_create(name: *const c_char, flags: u32) -> c_int;
}

type Invoke = unsafe extern "C" fn(*mut lua_State, CFunction, *mut c_int) -> c_int;
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
type InvokeContinuation =
    unsafe extern "C" fn(*mut lua_State, KFunction, c_int, KContext, *mut c_int) -> c_int;
type InvokeHook = unsafe extern "C" fn(*mut lua_State, Hook, *mut lua_Debug) -> c_int;
#[cfg(any(feature = "lua54", feature = "lua55"))]
type InvokeWarning =
    unsafe extern "C" fn(*mut lua_State, WarnFunction, *mut c_void, *const c_char, c_int) -> c_int;
type Allocate = unsafe extern "C" fn(
    *mut lua_State,
    Alloc,
    *mut c_void,
    *mut c_void,
    usize,
    usize,
    *mut *mut c_void,
) -> c_int;

struct Bridge {
    invoke: Invoke,
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    invoke_continuation: InvokeContinuation,
    invoke_hook: InvokeHook,
    #[cfg(any(feature = "lua54", feature = "lua55"))]
    invoke_warning: InvokeWarning,
    allocate: Allocate,
    loadlib: CFunction,
    native_searcher: CFunction,
}

/// Opens the embedded bridge through a memfd. There is no fallback through a file: a path in a
/// shared temporary directory could be swapped for another object between the write and the
/// dlopen.
fn open_bridge() -> Result<*mut c_void, String> {
    // SAFETY: the name is a valid NUL-terminated string and the flags are plain constants.
    let mut fd = unsafe { memfd_create(c"oximond-bridge".as_ptr(), MFD_CLOEXEC | MFD_EXEC) };
    if fd < 0 {
        // SAFETY: as above.
        fd = unsafe { memfd_create(c"oximond-bridge".as_ptr(), MFD_CLOEXEC) };
    }
    if fd < 0 {
        return Err(format!("memfd_create: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: memfd_create returned a new descriptor that nothing else owns.
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    file.write_all(BRIDGE_SO)
        .map_err(|error| format!("writing the bridge: {error}"))?;
    let path = std::ffi::CString::new(format!("/proc/self/fd/{fd}")).unwrap();
    // SAFETY: path is a valid NUL-terminated string that outlives the call.
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    if handle.is_null() {
        // SAFETY: dlopen just failed, so dlerror returns null or a valid NUL-terminated string,
        // read before any other dl call.
        let message = unsafe { dlerror() };
        return Err(if message.is_null() {
            "dlopen failed".to_string()
        } else {
            // SAFETY: message is non-null and NUL-terminated.
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        });
    }
    // The mapping keeps the object alive; the descriptor can go.
    drop(file);
    Ok(handle)
}

fn symbol(handle: *mut c_void, name: &CStr) -> *mut c_void {
    // SAFETY: handle is a non-null dlopen handle that stays open for the life of the process, and
    // name is NUL-terminated.
    let address = unsafe { dlsym(handle, name.as_ptr()) };
    assert!(!address.is_null(), "native bridge lacks {name:?}");
    address
}

/// The loaded bridge, or why it could not be loaded. A failure is reported as a Lua error by
/// each entry point, so a script that cannot run native code still gets a catchable error.
fn bridge() -> Result<&'static Bridge, &'static str> {
    static BRIDGE: OnceLock<Result<Bridge, String>> = OnceLock::new();
    BRIDGE
        .get_or_init(|| {
            let handle = open_bridge()?;
            // SAFETY: each symbol is defined by bridge.cpp with exactly the signature declared
            // above.
            Ok(unsafe {
                Bridge {
                    invoke: std::mem::transmute::<*mut c_void, Invoke>(symbol(
                        handle,
                        c"oximond_invoke",
                    )),
                    #[cfg(any(
                        feature = "lua52",
                        feature = "lua53",
                        feature = "lua54",
                        feature = "lua55"
                    ))]
                    invoke_continuation: std::mem::transmute::<*mut c_void, InvokeContinuation>(
                        symbol(handle, c"oximond_invoke_continuation"),
                    ),
                    invoke_hook: std::mem::transmute::<*mut c_void, InvokeHook>(symbol(
                        handle,
                        c"oximond_invoke_hook",
                    )),
                    #[cfg(any(feature = "lua54", feature = "lua55"))]
                    invoke_warning: std::mem::transmute::<*mut c_void, InvokeWarning>(symbol(
                        handle,
                        c"oximond_invoke_warning",
                    )),
                    allocate: std::mem::transmute::<*mut c_void, Allocate>(symbol(
                        handle,
                        c"oximond_allocate",
                    )),
                    loadlib: std::mem::transmute::<*mut c_void, CFunction>(symbol(
                        handle,
                        c"oximond_package_loadlib",
                    )),
                    native_searcher: std::mem::transmute::<*mut c_void, CFunction>(symbol(
                        handle,
                        c"oximond_package_native_searcher",
                    )),
                }
            })
        })
        .as_ref()
        .map_err(String::as_str)
}

/// Pushes the reason the bridge failed to load as the error object and returns LUA_ERRRUN.
unsafe fn load_failure(l: *mut lua_State, reason: &str) -> c_int {
    let message = format!("cannot load the native bridge: {reason}");
    lua_pushlstring(l, message.as_ptr().cast(), message.len());
    ERRRUN
}

/// Stand-ins installed as `package.loadlib` and the C searchers, so startup does not load the
/// bridge. `oximond_invoke` swaps them for the real functions before calling, because the real
/// ones raise Lua errors as C++ exceptions and those must not unwind through Rust frames. The
/// bodies only run if native code fetches the function with `lua_tocfunction` and calls it, so
/// they are `C-unwind`: an error raised by the real function unwinds through them into that
/// native caller, as it would have from the C++ function itself.
unsafe extern "C-unwind" fn loadlib_stand_in(l: *mut lua_State) -> c_int {
    match bridge() {
        Ok(bridge) => (std::mem::transmute::<CFunction, CUnwindFunction>(bridge.loadlib))(l),
        Err(reason) => raise_load_failure(l, reason),
    }
}

unsafe extern "C-unwind" fn native_searcher_stand_in(l: *mut lua_State) -> c_int {
    match bridge() {
        Ok(bridge) => {
            (std::mem::transmute::<CFunction, CUnwindFunction>(bridge.native_searcher))(l)
        }
        Err(reason) => raise_load_failure(l, reason),
    }
}

type CUnwindFunction = unsafe extern "C-unwind" fn(*mut lua_State) -> c_int;

/// Without a bridge there is no C++ lua_error to raise with, so a direct native call gets the
/// failure as a nil, message pair, as `package.loadlib` reports other load failures.
unsafe fn raise_load_failure(l: *mut lua_State, reason: &str) -> c_int {
    lua_pushnil(l);
    let message = format!("cannot load the native bridge: {reason}");
    lua_pushlstring(l, message.as_ptr().cast(), message.len());
    2
}

// SAFETY: a `C-unwind` function has the same calling convention as the `C` one; the type only
// tells Rust that an exception may pass through. Callers reach these through native code or
// through `oximond_invoke`, which swaps them out first.
#[allow(non_upper_case_globals)]
pub(crate) const oximond_package_loadlib: CFunction =
    unsafe { std::mem::transmute::<CUnwindFunction, CFunction>(loadlib_stand_in) };
#[allow(non_upper_case_globals)]
pub(crate) const oximond_package_native_searcher: CFunction =
    unsafe { std::mem::transmute::<CUnwindFunction, CFunction>(native_searcher_stand_in) };

pub(crate) unsafe fn oximond_invoke(
    state: *mut lua_State,
    mut function: CFunction,
    results: *mut c_int,
) -> c_int {
    let bridge = match bridge() {
        Ok(bridge) => bridge,
        Err(reason) => return load_failure(state, reason),
    };
    if function as usize == oximond_package_loadlib as usize {
        function = bridge.loadlib;
    } else if function as usize == oximond_package_native_searcher as usize {
        function = bridge.native_searcher;
    }
    (bridge.invoke)(state, function, results)
}

#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
pub(crate) unsafe fn oximond_invoke_continuation(
    state: *mut lua_State,
    function: KFunction,
    status: c_int,
    context: KContext,
    results: *mut c_int,
) -> c_int {
    match bridge() {
        Ok(bridge) => (bridge.invoke_continuation)(state, function, status, context, results),
        Err(reason) => load_failure(state, reason),
    }
}

pub(crate) unsafe fn oximond_invoke_hook(
    l: *mut lua_State,
    hook: Hook,
    debug: *mut lua_Debug,
) -> c_int {
    match bridge() {
        Ok(bridge) => (bridge.invoke_hook)(l, hook, debug),
        Err(reason) => load_failure(l, reason),
    }
}

#[cfg(any(feature = "lua54", feature = "lua55"))]
pub(crate) unsafe fn oximond_invoke_warning(
    l: *mut lua_State,
    function: WarnFunction,
    data: *mut c_void,
    message: *const c_char,
    continued: c_int,
) -> c_int {
    match bridge() {
        Ok(bridge) => (bridge.invoke_warning)(l, function, data, message, continued),
        Err(reason) => load_failure(l, reason),
    }
}

/// The bridge only wraps the allocator call in a try/catch for C++ allocators that throw. The
/// default allocator is plain realloc, which does not, so the common path skips the load.
pub(crate) unsafe fn oximond_allocate(
    l: *mut lua_State,
    allocator: Alloc,
    data: *mut c_void,
    pointer: *mut c_void,
    oldsize: usize,
    newsize: usize,
    result: *mut *mut c_void,
) -> c_int {
    if allocator as usize == default_allocator as *const () as usize {
        *result = allocator(data, pointer, oldsize, newsize);
        return if newsize != 0 && (*result).is_null() {
            ERRMEM
        } else {
            0
        };
    }
    match bridge() {
        Ok(bridge) => (bridge.allocate)(l, allocator, data, pointer, oldsize, newsize, result),
        // A custom allocator cannot run without the bridge's try/catch around it.
        Err(_) => ERRMEM,
    }
}

/// Handles only exist once the bridge opened a module, so it is loaded by now.
pub(crate) unsafe fn oximond_close_library(handle: *mut c_void) {
    dlclose(handle);
}
