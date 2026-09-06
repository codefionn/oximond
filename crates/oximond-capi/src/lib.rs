//! Versioned Lua C interface for native embedders and modules.
//!
//! Foreign pointers and reentrant calls are confined to this crate. API calls
//! that can raise Lua errors return a status to bridge.cpp; the C++ wrapper
//! throws only after Rust has returned. C++ callbacks are caught by that same
//! wrapper before control returns to the VM. No longjmp crosses a Rust frame.
#![allow(non_camel_case_types, non_snake_case, improper_ctypes)]

use oximond::bytecode::load::{ChunkLoader, LoadInput, LoadPoll};
use oximond::compiler::{CompilerConfig, CompilerLoader, compile_named};
use oximond::runtime::{
    GcRootBuilder, HostRootsRefresher, Metamethod, StringSet, TableHandle, ThreadHandle,
    UserdataHandle,
};
use oximond::string::WeakLuaString;
use oximond::{
    CompatibilityProfile, ExecutionMode, ExitRequest, HostCoroutineStatus, Lua, LuaString, Runtime,
    RuntimeError, Value, Vm, VmError,
};
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;
use std::rc::{Rc, Weak};

/// The runtime's strings, table parts and call vectors are small blocks
/// made and freed by the million; a host that links this library gets them
/// from per-thread lists instead of the C allocator (see `oximond::alloc`).
/// The `small-blocks` feature, on by default, installs it; a Rust program
/// with its own allocator turns the feature off.
#[cfg(all(feature = "small-blocks", not(windows)))]
#[global_allocator]
static ALLOCATOR: oximond::alloc::SmallBlocks = oximond::alloc::SmallBlocks;

#[cfg(all(test, feature = "bundled-bridge", not(feature = "lazy-bridge")))]
mod native_owner_tests {
    use super::*;

    #[test]
    fn teardown_is_inert_only_while_nothing_waits_for_a_finalizer() {
        let mut native = NativeLua::new(
            Lua::builder()
                .profile(CompatibilityProfile::Lua54)
                .build()
                .unwrap(),
        )
        .unwrap();
        assert!(native.teardown_is_inert());
        native
            .exec("keep = setmetatable({}, {__gc = function() end})")
            .unwrap();
        assert!(!native.teardown_is_inert());
        native.vm_mut().run_close_finalizers();
        assert!(native.teardown_is_inert());
    }

    #[test]
    fn a_fresh_state_ignores_warn_until_it_is_switched_on() {
        for (profile, off) in [(54, true), (55, false), (53, false), (51, false)] {
            // SAFETY: the state is created, used on this thread only, and closed.
            unsafe {
                let l = oximond_newstate_with_options(None, ptr::null_mut(), profile, 0);
                assert!(!l.is_null());
                assert_eq!(!rt(l).warnings_enabled(), off, "profile {profile}");
                if off {
                    rt(l).emit_warning(b"hidden", false).unwrap();
                    assert!(!rt(l).warnings_enabled());
                    rt(l).emit_warning(b"@on", false).unwrap();
                    assert!(rt(l).warnings_enabled());
                }
                lua_close(l);
            }
        }
    }

    #[test]
    fn a_wrapped_finalizer_error_fails_with_lua_errgcmm_and_skips_the_handler() {
        for profile in [CompatibilityProfile::Lua52, CompatibilityProfile::Lua53] {
            let lua = Lua::builder().profile(profile).build().unwrap();
            let mut native = NativeLua::new(lua).unwrap();
            native
                .exec(
                    "handled = false \
                     function handler(message) handled = true return 'handled' end \
                     setmetatable({}, {__gc = function() error('boom', 0) end}) \
                     setmetatable({}, {__gc = function() error('again', 0) end})",
                )
                .unwrap();
            // SAFETY: the state belongs to `native` and is used on this thread.
            unsafe {
                let l = native._state.0;
                let handler = rt(l).get_global("handler");
                let collect = rt(l).get_global("collectgarbage");
                push(l, handler);
                push(l, collect.clone());
                assert_eq!(oxi_pcall(l, 0, 0, -2), 5, "{profile:?}");
                assert!(
                    matches!(pop(l), Value::String(message)
                        if message.as_bytes().starts_with(b"error in __gc metamethod (")),
                    "{profile:?}"
                );
                assert_eq!(rt(l).get_global("handled"), Value::Boolean(false));
                push(l, collect);
                assert_eq!(oxi_pcall(l, 0, 0, 0), 5, "{profile:?}");
                assert!(
                    matches!(pop(l), Value::String(message)
                        if message.as_bytes().starts_with(b"error in __gc metamethod (")),
                    "{profile:?}"
                );
                pop(l);
            }
        }
    }

    /// Test C callback that pushes `true` and returns one result.
    ///
    /// # Safety
    ///
    /// `l` must be a live `lua_State` called back on its owning thread, with room on its stack for one value.
    unsafe extern "C" fn native_true(l: *mut lua_State) -> c_int {
        lua_pushboolean(l, 1);
        1
    }

    /// Test reader that hands its source to the loader in a single piece.
    ///
    /// # Safety
    ///
    /// `data` must point to the `Option<&[u8]>` the test set up, and `length` must be writable.
    /// The returned bytes borrow the test's `source`, which stays alive until the load finishes.
    unsafe extern "C" fn read_once(
        _: *mut lua_State,
        data: *mut c_void,
        length: *mut usize,
    ) -> *const c_char {
        let source = &mut *data.cast::<Option<&[u8]>>();
        match source.take() {
            Some(source) => {
                *length = source.len();
                source.as_ptr().cast()
            }
            None => ptr::null(),
        }
    }

    #[test]
    fn native_integer_push_uses_the_attached_vms_number_model() {
        for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua53] {
            let lua = Lua::builder().profile(profile).build().unwrap();
            let native = NativeLua::new(lua).unwrap();
            unsafe {
                let l = native._state.0;
                lua_pushinteger(l, 17);
                let value = pop(l);
                assert_eq!(
                    value,
                    if profile == CompatibilityProfile::Lua51 {
                        Value::Number(17.0)
                    } else {
                        Value::Integer(17)
                    }
                );
                #[cfg(target_pointer_width = "64")]
                if profile == CompatibilityProfile::Lua53 {
                    let exact = 9_007_199_254_740_993;
                    lua_pushinteger(l, exact);
                    assert_eq!(lua_tointeger(l, -1), exact);
                }
            }
        }
    }

    #[test]
    fn native_load_uses_the_attached_vms_compatibility_profile() {
        for (profile, source) in [
            (CompatibilityProfile::Lua53, b"return 7 & 3".as_slice()),
            (
                CompatibilityProfile::Lua54,
                b"local x <const> = 3; return x".as_slice(),
            ),
            (
                CompatibilityProfile::Lua55,
                b"global x = 3; return x".as_slice(),
            ),
        ] {
            let lua = Lua::builder().profile(profile).build().unwrap();
            let native = NativeLua::new(lua).unwrap();
            unsafe {
                let l = native._state.0;
                let mut source = Some(source);
                assert_eq!(
                    load_source(
                        l,
                        read_once,
                        ptr::from_mut(&mut source).cast(),
                        c"native".as_ptr(),
                        ptr::null()
                    ),
                    0
                );
                let function = pop(l);
                assert_eq!(
                    call(l, function, Vec::new()).unwrap(),
                    vec![Value::Integer(3)]
                );
            }
        }
    }

    #[test]
    fn native_reader_stops_at_a_syntax_error_before_requesting_eof() {
        /// Test reader that returns one syntactically invalid fragment and then ends.
        ///
        /// # Safety
        ///
        /// `data` must point to the test's `usize` call counter, and `length` must be writable.
        /// The returned bytes are a static literal, valid for the whole load.
        unsafe extern "C" fn invalid_source(
            _: *mut lua_State,
            data: *mut c_void,
            length: *mut usize,
        ) -> *const c_char {
            let calls = &mut *data.cast::<usize>();
            *calls += 1;
            if *calls == 1 {
                let source = b"local = 1\n";
                *length = source.len();
                source.as_ptr().cast()
            } else {
                ptr::null()
            }
        }
        let native = NativeLua::new(Lua::builder().build().unwrap()).unwrap();
        let mut calls = 0_usize;
        unsafe {
            let l = native._state.0;
            assert_eq!(
                load_source(
                    l,
                    invalid_source,
                    ptr::from_mut(&mut calls).cast(),
                    c"early".as_ptr(),
                    ptr::null()
                ),
                3,
            );
            assert_eq!(calls, 1);
            assert!(
                matches!(pop(l), Value::String(message) if message.to_lossy().contains("expected"))
            );
        }
    }

    #[test]
    fn native_load_handles_fragmented_text_binary_and_modes() {
        struct Input {
            source: Vec<u8>,
            position: usize,
            width: usize,
        }
        /// Test reader that hands out fragments of `width` bytes at a time.
        ///
        /// # Safety
        ///
        /// `data` must point to the test's `Input`, and `length` must be writable.
        /// The returned pointer borrows `Input::source`, which the test keeps alive for the load.
        unsafe extern "C" fn fragment(
            _: *mut lua_State,
            data: *mut c_void,
            length: *mut usize,
        ) -> *const c_char {
            let input = &mut *data.cast::<Input>();
            let end = input
                .position
                .saturating_add(input.width)
                .min(input.source.len());
            *length = end - input.position;
            let pointer = input.source[input.position..end].as_ptr().cast();
            input.position = end;
            pointer
        }
        let chunk = compile_named("native", b"return 'ok'", CompilerConfig::default()).unwrap();
        let binary = oximond::bytecode::binary::dump(&chunk).unwrap();
        for source in [
            b"return 'ok'".to_vec(),
            binary.clone(),
            [b"\n".as_slice(), &binary].concat(),
        ] {
            for width in [1, 2, 7, 8192] {
                let native = NativeLua::new(Lua::builder().build().unwrap()).unwrap();
                let mut input = Input {
                    source: source.clone(),
                    position: 0,
                    width,
                };
                unsafe {
                    let l = native._state.0;
                    assert_eq!(
                        load_source(
                            l,
                            fragment,
                            ptr::from_mut(&mut input).cast(),
                            c"native".as_ptr(),
                            c"bt".as_ptr()
                        ),
                        0
                    );
                    let function = pop(l);
                    assert_eq!(
                        call(l, function, Vec::new()).unwrap(),
                        vec![Value::from("ok")]
                    );
                }
            }
        }
        for (source, mode) in [(b"return 'ok'".to_vec(), c"b"), (binary, c"t")] {
            let native = NativeLua::new(Lua::builder().build().unwrap()).unwrap();
            let mut input = Input {
                source,
                position: 0,
                width: 1,
            };
            unsafe {
                let l = native._state.0;
                assert_eq!(
                    load_source(
                        l,
                        fragment,
                        ptr::from_mut(&mut input).cast(),
                        c"native".as_ptr(),
                        mode.as_ptr()
                    ),
                    3
                );
                assert!(
                    matches!(pop(l), Value::String(message) if message.to_lossy().contains("attempt to load"))
                );
            }
        }
    }

    #[test]
    fn native_load_of_a_whole_piece_matches_the_streaming_path() {
        struct Input {
            pieces: Vec<Vec<u8>>,
            calls: usize,
        }
        /// Test reader that returns one prepared piece per call and then ends.
        ///
        /// # Safety
        ///
        /// `data` must point to the test's `Input`, and `length` must be writable.
        /// The returned pointer borrows `Input::pieces`, which the test keeps alive for the load.
        unsafe extern "C" fn pieces(
            _: *mut lua_State,
            data: *mut c_void,
            length: *mut usize,
        ) -> *const c_char {
            let input = &mut *data.cast::<Input>();
            input.calls += 1;
            match input.pieces.get(input.calls - 1) {
                Some(piece) => {
                    *length = piece.len();
                    piece.as_ptr().cast()
                }
                None => {
                    *length = 0;
                    ptr::null()
                }
            }
        }
        let load = |parts: &[&[u8]]| {
            let native = NativeLua::new(Lua::builder().build().unwrap()).unwrap();
            let mut input = Input {
                pieces: parts.iter().map(|piece| piece.to_vec()).collect(),
                calls: 0,
            };
            unsafe {
                let l = native._state.0;
                let status = load_source(
                    l,
                    pieces,
                    ptr::from_mut(&mut input).cast(),
                    c"=whole".as_ptr(),
                    ptr::null(),
                );
                let top = pop(l);
                let result = if status == 0 {
                    call(l, top, Vec::new()).map_err(|error| format!("{error:?}"))
                } else {
                    Err(match top {
                        Value::String(message) => message.to_lossy().into_owned(),
                        other => format!("{other:?}"),
                    })
                };
                (status, result, input.calls)
            }
        };
        // One piece, then the end: loaded without the streaming session.
        assert_eq!(
            load(&[b"local x = 20\nreturn x + 1"]),
            (0, Ok(vec![Value::Number(21.0)]), 2)
        );
        // A first piece that is a complete chunk by itself but continues in the next one.
        assert_eq!(
            load(&[b"return 1", b" + 2"]),
            (0, Ok(vec![Value::Number(3.0)]), 3)
        );
        // A leading synthetic newline (luaL_loadfile) still counts as line one.
        let (status, result, calls) = load(&[b"\n", b"error('x')"]);
        assert_eq!((status, calls), (0, 3));
        assert!(format!("{result:?}").contains("whole:2:"), "{result:?}");
        // Invalid first piece: the reader is not asked again, and the message is the streaming one.
        let (status, result, calls) = load(&[b"x = = 1"]);
        assert_eq!((status, calls), (3, 1));
        assert!(result.unwrap_err().contains("whole:1:"));
        // A piece that is complete but fails to lower falls back and reports the same error.
        let (status, _, _) = load(&[b"goto missing"]);
        assert_eq!(status, 3);
        // A runtime-only difference check: the whole-piece chunk keeps its line table.
        let (status, result, calls) = load(&[b"\n\nerror('deep')"]);
        assert_eq!((status, calls), (0, 2));
        assert!(format!("{result:?}").contains("whole:3:"), "{result:?}");
    }

    #[test]
    fn native_callbacks_survive_moving_lua_out_of_the_wrapper() {
        let lua = Lua::builder()
            .profile(CompatibilityProfile::Lua54)
            .build()
            .unwrap();
        let mut native = NativeLua::new(lua).unwrap();
        unsafe {
            let l = native._state.0;
            lua_pushcclosure(l, native_true, 0);
            let function = pop(l);
            native.runtime_mut().set_global("native_true", function);
        }
        let replacement = Lua::builder()
            .profile(CompatibilityProfile::Lua54)
            .build()
            .unwrap();
        let mut escaped = std::mem::replace(&mut *native, replacement);
        drop(native);
        assert_eq!(
            escaped.exec("return native_true()").unwrap(),
            vec![Value::Boolean(true)]
        );
    }
}

const _: () = assert!(
    cfg!(feature = "luajit") as u8
        + cfg!(feature = "lua51") as u8
        + cfg!(feature = "lua52") as u8
        + cfg!(feature = "lua53") as u8
        + cfg!(feature = "lua54") as u8
        + cfg!(feature = "lua55") as u8
        == 1,
    "enable exactly one Lua ABI feature: luajit, lua51, lua52, lua53, lua54, or lua55"
);
const MODERN: bool = cfg!(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
));
const MODERN_INTEGER: bool = cfg!(any(feature = "lua53", feature = "lua54", feature = "lua55"));
const PROFILE: CompatibilityProfile = if cfg!(feature = "lua55") {
    CompatibilityProfile::Lua55
} else if cfg!(feature = "lua54") {
    CompatibilityProfile::Lua54
} else if cfg!(feature = "lua53") {
    CompatibilityProfile::Lua53
} else if cfg!(feature = "lua52") {
    CompatibilityProfile::Lua52
} else if cfg!(feature = "lua51") {
    CompatibilityProfile::Lua51
} else {
    CompatibilityProfile::LuaJit21
};
const REGISTRY: c_int = if cfg!(feature = "lua55") {
    -(c_int::MAX / 2 + 1000)
} else if MODERN {
    -1_001_000
} else {
    -10000
};
const ERROR_HANDLER: c_int = if cfg!(any(feature = "lua52", feature = "lua53")) {
    6
} else {
    5
};
const ENVIRON: c_int = -10001;
const GLOBALS: c_int = -10002;
const UPVALUES: c_int = if MODERN { REGISTRY } else { GLOBALS };
#[cfg(any(feature = "lua53", feature = "lua54", feature = "lua55"))]
pub type LuaInteger = i64;
#[cfg(not(any(feature = "lua53", feature = "lua54", feature = "lua55")))]
pub type LuaInteger = isize;
#[cfg(any(feature = "lua53", feature = "lua54", feature = "lua55"))]
type RawIndex = LuaInteger;
#[cfg(not(any(feature = "lua53", feature = "lua54", feature = "lua55")))]
type RawIndex = c_int;
#[cfg(any(feature = "lua53", feature = "lua54", feature = "lua55"))]
type GetReturn = c_int;
#[cfg(not(any(feature = "lua53", feature = "lua54", feature = "lua55")))]
type GetReturn = ();
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
type PushStringReturn = *const c_char;
#[cfg(not(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
)))]
type PushStringReturn = ();
#[cfg(any(feature = "lua53", feature = "lua54", feature = "lua55"))]
type HookReturn = ();
#[cfg(not(any(feature = "lua53", feature = "lua54", feature = "lua55")))]
type HookReturn = c_int;
fn hook_return(status: c_int) -> HookReturn {
    #[cfg(any(feature = "lua53", feature = "lua54", feature = "lua55"))]
    {
        let _ = status;
    }
    #[cfg(not(any(feature = "lua53", feature = "lua54", feature = "lua55")))]
    {
        status
    }
}
#[cfg(any(feature = "lua54", feature = "lua55"))]
type RawLength = u64;
#[cfg(not(any(feature = "lua54", feature = "lua55")))]
type RawLength = usize;
type CFunction = unsafe extern "C" fn(*mut lua_State) -> c_int;
#[cfg(feature = "lua52")]
type KContext = c_int;
#[cfg(not(feature = "lua52"))]
type KContext = isize;
#[cfg(feature = "lua52")]
type KFunction = CFunction;
#[cfg(not(feature = "lua52"))]
type KFunction = unsafe extern "C" fn(*mut lua_State, c_int, KContext) -> c_int;
#[cfg_attr(
    not(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    )),
    allow(dead_code)
)]
struct Continuation {
    state: *mut lua_State,
    stack: Vec<Value>,
    current: Option<usize>,
    close_slots: Vec<usize>,
    context: KContext,
    function: KFunction,
    child: bool,
    nresults: c_int,
    handler: Option<Value>,
    value: Option<Value>,
    close_handler: Option<Value>,
}
#[cfg(feature = "lazy-bridge")]
mod lazy_bridge;
#[cfg(feature = "lazy-bridge")]
use lazy_bridge::*;

type Alloc = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void;
type Reader = unsafe extern "C" fn(*mut lua_State, *mut c_void, *mut usize) -> *const c_char;

#[cfg(not(feature = "lazy-bridge"))]
unsafe extern "C" {
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    fn oximond_invoke_continuation(
        state: *mut lua_State,
        function: KFunction,
        status: c_int,
        context: KContext,
        results: *mut c_int,
    ) -> c_int;
    fn oximond_allocate(
        l: *mut lua_State,
        allocator: Alloc,
        data: *mut c_void,
        pointer: *mut c_void,
        oldsize: usize,
        newsize: usize,
        result: *mut *mut c_void,
    ) -> c_int;
    fn oximond_invoke(state: *mut lua_State, function: CFunction, results: *mut c_int) -> c_int;
    fn oximond_package_loadlib(state: *mut lua_State) -> c_int;
    fn oximond_package_native_searcher(state: *mut lua_State) -> c_int;
    fn oximond_close_library(handle: *mut c_void);
}

struct CClosure {
    state: *mut lua_State,
    function: CFunction,
    upvalues: TableHandle,
    count: usize,
    index: usize,
    value: Value,
    environment: TableHandle,
}
struct Userdata {
    storage: UserdataStorage,
    size: usize,
}

enum UserdataStorage {
    Owned(Box<[u128]>),
    Host { pointer: *mut c_void, size: usize },
}
impl UserdataStorage {
    fn pointer(&mut self) -> *mut c_void {
        match self {
            Self::Owned(storage) => storage.as_mut_ptr().cast(),
            Self::Host { pointer, .. } => *pointer,
        }
    }
}
struct InternedString {
    bytes: Box<[u8]>,
    /// Every string sharing the buffer, by identity. Equal strings are
    /// separate objects, so one buffer can have thousands of them.
    references: AddressMap<usize, WeakLuaString>,
}
/// Hashes an address or an already computed hash with a multiply and a fold:
/// the keys of the native string tables, which SipHash spent most of an
/// intern call on.
#[derive(Default, Clone, Copy)]
struct AddressHasher(u64);
impl std::hash::Hasher for AddressHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(self.0 ^ u64::from(byte));
        }
    }
    fn write_u64(&mut self, n: u64) {
        // Addresses are aligned, so the low bits of the product alone would
        // collide; the fold brings the high half down.
        let product = (n ^ (n >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = product ^ (product >> 29);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
    // A handle hashes as its slot and then its generation: two words, the
    // first carried into the second, where the byte loop of `write` would
    // run eight rounds.
    fn write_u32(&mut self, n: u32) {
        self.write_u64(self.0.rotate_left(32) ^ u64::from(n));
    }
}
/// A hash map keyed by addresses or precomputed hashes (see [`AddressHasher`]).
type AddressMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<AddressHasher>>;
/// Opaque to C. Each state is used by one host thread at a time, just like Lua.
/// `vm` is reborrowed during callbacks; no State reference survives a C call.
pub struct lua_State {
    _owned_vm: Option<Box<Vm>>,
    vm: *mut Vm,
    /// The main state of the family. A main state points to itself
    /// ([`allocate_state`] sets it), so [`shared`] is one load.
    main: *mut lua_State,
    threads: BTreeMap<ThreadHandle, ThreadState>,
    status: c_int,
    entry_pending: bool,
    /// The tag `lua_pushinteger` gives an integer: [`INTEGER_TAG`] when the
    /// runtime has `tagged_integers`, else [`NUMBER_TAG`]. Copied at creation
    /// (the profile never changes afterwards), so the push does not chase
    /// `main`, `vm` and the runtime for it, and kept as the tag itself, so
    /// the push stores it as it is and tests it once for the payload.
    integer_tag: u8,
    stack: Vec<Value>,
    saved_stacks: Vec<Vec<Value>>,
    /// Emptied stack buffers of finished host calls, reused by the next call so a call
    /// neither allocates its stack nor regrows it on the first pushes.
    spare_stacks: Vec<Vec<Value>>,
    current: Option<usize>,
    pending_continuation: Option<Value>,
    #[allow(dead_code)]
    context: (c_int, KContext),
    close_slots: Vec<usize>,
    #[allow(dead_code)]
    // Boxed because raw-pointer tokens into these records are handed to the VM and must stay valid as the vec grows.
    #[allow(clippy::vec_box)]
    continuations: Vec<Box<Continuation>>,
    // Boxed for the same stable-address token reason as `continuations`.
    #[allow(clippy::vec_box)]
    closures: Vec<Box<CClosure>>,
    userdata: AddressMap<UserdataHandle, Userdata>,
    light: BTreeMap<usize, UserdataHandle>,
    light_reverse: AddressMap<UserdataHandle, usize>,
    strings: AddressMap<usize, InternedString>,
    /// The `strings` key of the newest buffer for each byte content, so
    /// equal strings share one buffer without scanning every entry.
    /// Keyed by the strings' cached hash, which equal bytes share; a hit is
    /// checked against the buffer, and a colliding string just takes the slot.
    string_contents: AddressMap<u64, usize>,
    last_gc_epoch: u64,
    /// `Runtime::gc_freed_userdata` at the last scan of `userdata`.
    last_freed_userdata: u64,
    thread: ThreadHandle,
    panic: Option<CFunction>,
    hook: Option<Hook>,
    hook_value: Option<Value>,
    in_hook: bool,
    hook_mask: c_int,
    hook_count: c_int,
    allocator: Option<Alloc>,
    allocator_data: *mut c_void,
    native_libraries: Vec<*mut c_void>,
    external_strings: Vec<ExternalString>,
    /// Strings the host pushed or used as field names, by the address of the C
    /// string they came from. See [`cached_key`].
    key_cache: Vec<Option<(usize, LuaString)>>,
    /// Short strings whose own storage `lua_tolstring` lent out, by identity, held
    /// until the next collection starts so the pointers outlive a pop, as
    /// the interned copies do. See [`lend`].
    lent: AddressMap<usize, LuaString>,
    /// The identities of recently lent strings, by a hash of the identity:
    /// a string lent again finds itself here and is not held twice.
    lent_recent: Vec<usize>,
    owner: Weak<NativeStateOwner>,
}

#[repr(C)]
struct ExtraState {
    extraspace: [u8; std::mem::size_of::<*mut c_void>()],
    state: lua_State,
    allocation: Option<(Alloc, *mut c_void)>,
}
struct ThreadState(*mut lua_State);
impl Drop for ThreadState {
    fn drop(&mut self) {
        unsafe { free_state_allocation(self.0) }
    }
}
/// Reaches the `ExtraState` allocation that owns the `lua_State` at `l`.
///
/// # Safety
///
/// `l` must be a `lua_State` allocated by `allocate_state`, so subtracting the `state` field offset lands inside its `ExtraState`.
/// The allocation must still be live and uniquely borrowed.
unsafe fn extra_state(l: *mut lua_State) -> *mut ExtraState {
    l.cast::<u8>()
        .sub(std::mem::offset_of!(ExtraState, state))
        .cast()
}
fn allocate_state(state: lua_State) -> *mut lua_State {
    unsafe {
        let allocator = state.allocator;
        let data = state.allocator_data;
        let value = ExtraState {
            extraspace: [0; std::mem::size_of::<*mut c_void>()],
            state,
            allocation: allocator.map(|allocator| (allocator, data)),
        };
        let allocation = if let Some(allocator) = allocator {
            let mut allocation = ptr::null_mut();
            let status = oximond_allocate(
                ptr::null_mut(),
                allocator,
                data,
                ptr::null_mut(),
                8,
                std::mem::size_of::<ExtraState>(),
                &mut allocation,
            );
            if status != 0 {
                return ptr::null_mut();
            }
            let allocation = allocation.cast::<ExtraState>();
            allocation.write(value);
            allocation
        } else {
            Box::into_raw(Box::new(value))
        };
        let l = ptr::addr_of_mut!((*allocation).state);
        // A state made without a main state is the main state of its family.
        if (*l).main.is_null() {
            (*l).main = l;
        }
        l
    }
}
/// Frees the `ExtraState` behind `l`, through the host allocator when one is attached.
///
/// # Safety
///
/// `l` must come from `allocate_state` and be freed exactly once, with no later call touching it.
/// The recorded allocator must accept the final release of the block it returned.
unsafe fn free_state_allocation(l: *mut lua_State) {
    let allocation = extra_state(l);
    if let Some((allocator, data)) = (*allocation).allocation {
        ptr::drop_in_place(allocation);
        let mut result = ptr::null_mut();
        let _ = oximond_allocate(
            ptr::null_mut(),
            allocator,
            data,
            allocation.cast(),
            std::mem::size_of::<ExtraState>(),
            0,
            &mut result,
        );
    } else {
        drop(Box::from_raw(allocation));
    }
}

struct NativeStateOwner(*mut lua_State);

impl Drop for NativeStateOwner {
    fn drop(&mut self) {
        // Host callbacks can survive a safe move of the original Lua or VM.
        // At this point the last VM using the state is being destroyed. Its
        // address can be stale, so releasing native storage never touches it.
        unsafe { release_state(self.0) }
    }
}

struct ExternalString {
    bytes: *mut c_void,
    length: usize,
    allocator: Option<Alloc>,
    data: *mut c_void,
    string: WeakLuaString,
}

/// A Lua interpreter with the C API attached to the same VM.
/// Boxing the interpreter keeps the pointers used by native closures stable.
#[cfg(feature = "bundled-bridge")]
pub struct NativeLua {
    lua: Box<Lua>,
    _state: Rc<NativeStateOwner>,
}

#[cfg(feature = "bundled-bridge")]
impl NativeLua {
    pub fn new(lua: Lua) -> Result<Self, RuntimeError> {
        let mut lua = Box::new(lua);
        let vm = lua.vm_mut();
        let thread = vm.runtime_mut().create_thread()?;
        let owned_tagged = vm.runtime().tagged_integers();
        let l = allocate_state(lua_State {
            _owned_vm: None,
            vm,
            main: ptr::null_mut(),
            threads: BTreeMap::new(),
            status: 0,
            entry_pending: false,
            integer_tag: integer_tag(owned_tagged),
            stack: Vec::new(),
            saved_stacks: Vec::new(),
            spare_stacks: Vec::new(),
            current: None,
            pending_continuation: None,
            context: (0, 0),
            close_slots: Vec::new(),
            continuations: Vec::new(),
            closures: Vec::new(),
            userdata: AddressMap::default(),
            light: BTreeMap::new(),
            light_reverse: AddressMap::default(),
            strings: AddressMap::default(),
            string_contents: AddressMap::default(),
            last_gc_epoch: 0,
            last_freed_userdata: 0,
            thread,
            panic: None,
            hook: None,
            hook_value: None,
            in_hook: false,
            hook_mask: 0,
            hook_count: 0,
            allocator: Some(default_allocator),
            allocator_data: ptr::null_mut(),
            native_libraries: Vec::new(),
            external_strings: Vec::new(),
            key_cache: Vec::new(),
            lent: AddressMap::default(),
            lent_recent: Vec::new(),
            owner: Weak::new(),
        });
        if l.is_null() {
            return Err(RuntimeError::memory_error());
        }
        let owner = Rc::new(NativeStateOwner(l));
        unsafe { (*l).owner = Rc::downgrade(&owner) };
        unsafe { root_main_thread(l) };
        unsafe { install_native_package(l) };
        Ok(Self { lua, _state: owner })
    }
}

#[cfg(feature = "bundled-bridge")]
impl NativeLua {
    /// Whether dropping this interpreter would only free memory. It would
    /// not once a native library is loaded, since closing it runs the
    /// library's destructors, nor while a script holds a file or a pipe,
    /// whose drop flushes buffered writes and waits for the child. A host
    /// that is about to exit the process can leak the interpreter then: the
    /// teardown of a bare start was 77,000 instructions, 9% of the start.
    #[must_use]
    pub fn teardown_is_inert(&self) -> bool {
        // SAFETY: `_state` keeps the state allocated while `self` exists,
        // and `&self` excludes a concurrent writer.
        let state = unsafe { shared(self._state.0) };
        state.native_libraries.is_empty()
            && state.userdata.is_empty()
            && state.external_strings.is_empty()
            && !self.lua.vm().runtime().has_finalizers()
            && !self.lua.vm().runtime().drop_has_effects()
    }
}

#[cfg(feature = "bundled-bridge")]
impl std::ops::Deref for NativeLua {
    type Target = Lua;
    fn deref(&self) -> &Lua {
        &self.lua
    }
}

#[cfg(feature = "bundled-bridge")]
impl std::ops::DerefMut for NativeLua {
    fn deref_mut(&mut self) -> &mut Lua {
        &mut self.lua
    }
}

#[cfg(feature = "bundled-bridge")]
unsafe extern "C" {
    fn realloc(pointer: *mut c_void, size: usize) -> *mut c_void;
    fn free(pointer: *mut c_void);
}

/// The stock allocator: `realloc` to grow and `free` to release, like liblua's `l_alloc`.
///
/// # Safety
///
/// `pointer` must be null or a block this allocator returned earlier, with `size` following the C allocator contract.
/// The libc `realloc` and `free` calls must receive no other pointers.
#[cfg(feature = "bundled-bridge")]
unsafe extern "C" fn default_allocator(
    _: *mut c_void,
    pointer: *mut c_void,
    _: usize,
    size: usize,
) -> *mut c_void {
    if size == 0 {
        free(pointer);
        ptr::null_mut()
    } else {
        realloc(pointer, size)
    }
}

/// The Lua value keeps a safe byte copy; native readers retain the external
/// buffer's pointer and the allocator receives its release after collection.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// `bytes` must be valid for reads of `length + 1` bytes and stay alive and unmodified until `allocator` releases it after collection.
/// `allocator` must be null or an `Alloc` that accepts that release with `data`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushexternalstring(
    l: *mut lua_State,
    bytes: *const c_char,
    length: usize,
    allocator: Option<Alloc>,
    data: *mut c_void,
) -> *const c_char {
    let string = LuaString::from(self::bytes(bytes, length));
    let weak = string.downgrade();
    push(l, Value::String(string));
    shared(l).external_strings.push(ExternalString {
        bytes: bytes.cast_mut().cast(),
        length,
        allocator,
        data,
        string: weak,
    });
    bytes
}
/// Reborrows the `lua_State` at `l` as a mutable reference.
///
/// # Safety
///
/// `l` must be a live `lua_State` that the caller holds uniquely.
/// The reference must not outlive the current C call or coexist with another borrow of the state.
unsafe fn state<'a>(l: *mut lua_State) -> &'a mut lua_State {
    &mut *l
}
/// Returns the main `lua_State` of the family `l` belongs to.
///
/// # Safety
///
/// `l` must be a live `lua_State` made by [`allocate_state`].
unsafe fn main_state(l: *mut lua_State) -> *mut lua_State {
    (*l).main
}
/// Reborrows the main state behind `l`, where the family's shared storage lives.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose family's main state is still allocated.
/// The result stands for the caller's unique borrow of the whole family and must not outlive the C call.
unsafe fn shared<'a>(l: *mut lua_State) -> &'a mut lua_State {
    &mut *main_state(l)
}
/// The environment table of `l`'s thread, or of the main thread on the main state.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose thread belongs to its family's VM; anything else trips the `expect`.
unsafe fn thread_environment(l: *mut lua_State) -> TableHandle {
    let thread = (l != main_state(l)).then_some((*l).thread);
    vm(l)
        .thread_environment_for(thread)
        .expect("valid C API thread")
}
/// Reborrows the VM attached to `l`'s family.
///
/// # Safety
///
/// `l` must be a live `lua_State` and its family's `vm` pointer must address a live `Vm`.
/// No other borrow of that `Vm` may be alive when this is called.
unsafe fn vm<'a>(l: *mut lua_State) -> &'a mut Vm {
    &mut *shared(l).vm
}
/// Reborrows the runtime of the VM attached to `l`'s family.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose family's `vm` pointer addresses a live `Vm`.
/// No other borrow of the VM or its runtime may be alive when this is called.
unsafe fn rt<'a>(l: *mut lua_State) -> &'a mut Runtime {
    vm(l).runtime_mut()
}
/// Registers `l`'s thread as the VM's main thread and seeds the registry entries the C API reads.
///
/// # Safety
///
/// `l` must be a freshly allocated main state created for this VM, called once before any other use of the state.
unsafe fn root_main_thread(l: *mut lua_State) {
    vm(l)
        .set_main_thread((*l).thread)
        .expect("valid main thread");
    rt(l).set_host_roots_refresher(Some(HostRootsRefresher {
        refresh: refresh_host_roots,
        token: l as usize,
    }));
    let registry = rt(l).registry();
    if MODERN {
        let main_index = if cfg!(feature = "lua55") { 3 } else { 1 };
        let _ = rt(l).table_set(
            registry,
            &Value::Integer(main_index),
            Value::Thread((*l).thread),
        );
        let globals = rt(l).global_environment();
        let _ = rt(l).table_set(registry, &Value::Integer(2), Value::Table(globals));
    }
    if let Value::Table(package) = rt(l).get_global("package") {
        for (field, key) in [("loaded", "_LOADED"), ("preload", "_PRELOAD")] {
            if let Ok(value) = rt(l).table_get(package, &Value::from(field)) {
                let _ = rt(l).table_set(registry, &Value::from(key), value);
            }
        }
    }
    let _ = rt(l).table_set(
        registry,
        &Value::from("oximond.capi.thread"),
        Value::Thread((*l).thread),
    );
}
/// Translates a stack index into a position in `l`'s value stack.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose stack nothing else mutably borrows.
/// `index` must be a positive or negative stack index; pseudo-indices yield `None`.
unsafe fn slot(l: *mut lua_State, index: c_int) -> Option<usize> {
    let len = (*l).stack.len();
    let offset = if index > 0 {
        index as isize - 1
    } else if index > REGISTRY {
        len as isize + index as isize
    } else {
        return None;
    };
    usize::try_from(offset).ok().filter(|&i| i < len)
}
/// Borrows the value in a stack slot without cloning it.
/// Pseudo-indices and out-of-range indices yield `None`; callers fall back to [`value`].
///
/// # Safety
///
/// `l` must be a live `lua_State` whose stack nothing else mutably borrows.
/// The reference must be dropped before the next push, pop, or call into the VM.
#[inline]
unsafe fn peek<'a>(l: *mut lua_State, index: c_int) -> Option<&'a Value> {
    let stack = &(*l).stack;
    let len = stack.len();
    let offset = if index > 0 {
        let offset = index as usize - 1;
        if offset >= len {
            return None;
        }
        offset
    } else if index > REGISTRY {
        // The length plus a negative index, as unsigned words: the sum
        // carries out exactly when it is not negative, that is when the
        // index reaches no further down than the bottom, so the carry
        // is the whole bound check.
        let (offset, inside) = len.overflowing_add(index as isize as usize);
        if !inside {
            return None;
        }
        offset
    } else {
        return None;
    };
    Some(&*stack.as_ptr().add(offset))
}
/// [`peek`] at index -1: the value on top, if any.
///
/// # Safety
///
/// As for [`peek`].
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn top_value<'a>(l: *mut lua_State) -> Option<&'a Value> {
    let stack = &(*l).stack;
    let top = stack.len().checked_sub(1)?;
    Some(&*stack.as_ptr().add(top))
}
/// Clones the value a stack, upvalue, or pseudo-index refers to.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
/// For upvalue indices `(*l).current` must name a live entry of the shared closure table.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn value(l: *mut lua_State, index: c_int) -> Option<Value> {
    // Most indices name a stack slot: one bound check and a clone. Hosts
    // read the registry often enough to keep it off the slow path too.
    if index == REGISTRY {
        return Some(Value::Table(rt(l).registry()));
    }
    if let Some(value) = peek(l, index) {
        return Some(value.clone());
    }
    value_slow(l, index)
}
/// [`value`] for a pseudo-index or an index past the stack.
///
/// # Safety
///
/// As for [`value`].
#[inline(never)]
unsafe fn value_slow(l: *mut lua_State, index: c_int) -> Option<Value> {
    match index {
        REGISTRY => Some(Value::Table(rt(l).registry())),
        GLOBALS if !MODERN => Some(Value::Table(thread_environment(l))),
        ENVIRON if !MODERN => Some(Value::Table(environment(l))),
        n if n < UPVALUES => {
            let current = (*l).current?;
            let closure = &shared(l).closures[current];
            let table = closure.upvalues;
            rt(l)
                .table_get(table, &Value::Number(f64::from(UPVALUES - n)))
                .ok()
        }
        _ => slot(l, index).map(|i| state(l).stack[i].clone()),
    }
}
/// The environment of the C closure running on `l`, or the thread environment outside one.
///
/// # Safety
///
/// `l` must be a live `lua_State`; when `(*l).current` is set it must index the shared closures vector.
unsafe fn environment(l: *mut lua_State) -> TableHandle {
    match (*l).current {
        Some(i) => shared(l).closures[i].environment,
        None => thread_environment(l),
    }
}
/// Pushes a value onto `l`'s stack.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `value`'s handles must belong to its VM.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn push(l: *mut lua_State, value: Value) {
    let stack = &mut state(l).stack;
    let len = stack.len();
    if len == stack.capacity() {
        grow_stack(stack);
    }
    // SAFETY: `grow_stack` left room for one more slot, and the slot is past the length.
    stack.as_mut_ptr().add(len).write(value);
    stack.set_len(len + 1);
}
/// The free slot past the top of `l`'s stack, if its buffer has one. Hot
/// pushes write through it and hand a full stack to an out-of-line path
/// with a tail call, so they make no call of their own and save no
/// registers.
///
/// # Safety
///
/// `l` must be a live `lua_State`. The pointer is valid until the stack is
/// next changed; the caller writes a value through it and then raises the
/// length with [`commit_slot`], or leaves the stack alone.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn spare_slot(l: *mut lua_State) -> Option<*mut Value> {
    let stack = &mut state(l).stack;
    let len = stack.len();
    (len != stack.capacity()).then(|| stack.as_mut_ptr().add(len))
}
/// Raises the length of `l`'s stack by one after a write through [`spare_slot`].
///
/// # Safety
///
/// The slot [`spare_slot`] returned must hold a value written since.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn commit_slot(l: *mut lua_State) {
    let stack = &mut state(l).stack;
    stack.set_len(stack.len() + 1);
}
/// Copies an integer or a number into `to` as its tag byte and its payload
/// word: one test of the tag covers both kinds, and the copy stays two
/// narrow loads and stores. A move of the whole 16-byte value is one wide
/// load, which stalls when the source was just written with narrow stores.
///
/// # Safety
///
/// `from` holds an integer or a number, and `to` is writable and holds
/// nothing that needs a drop.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn copy_number_words(to: *mut Value, from: &Value) {
    // `Value` is `repr(C, u8)`: the tag is byte 0 and the payload the word at offset 8.
    let (to, from) = (to.cast::<u8>(), ptr::from_ref(from).cast::<u8>());
    to.write(from.read());
    to.add(8)
        .cast::<u64>()
        .write(from.add(8).cast::<u64>().read());
}
/// Pushes `value` onto a stack whose buffer may be full: the out-of-line
/// half of the pushes that write through [`spare_slot`].
///
/// # Safety
///
/// As for [`push`].
#[cold]
#[inline(never)]
unsafe fn push_cold(l: *mut lua_State, value: Value) {
    push(l, value);
}
/// Makes room for one more value on a full stack. Pushes inline a capacity compare
/// and call this out of line, so the common push is a slot write and a length bump.
#[cold]
#[inline(never)]
fn grow_stack(stack: &mut Vec<Value>) {
    stack.reserve(1);
}
/// Pops the top value of `l`'s stack when it is a plain number or integer, without the
/// drop loop `Vec::truncate` runs. Other values are left alone and `None` is returned.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn pop_number(l: *mut lua_State) -> Option<Value> {
    let stack = &mut state(l).stack;
    let len = stack.len();
    match stack.last() {
        Some(&Value::Number(x)) => {
            // SAFETY: the slot holds a number, which owns nothing to drop.
            stack.set_len(len - 1);
            Some(Value::Number(x))
        }
        Some(&Value::Integer(x)) => {
            stack.set_len(len - 1);
            Some(Value::Integer(x))
        }
        _ => None,
    }
}
/// Pops the top of `l`'s stack, yielding nil when it is empty.
///
/// # Safety
///
/// `l` must be a live `lua_State` and nothing else may borrow its stack vector across the pop.
unsafe fn pop(l: *mut lua_State) -> Value {
    state(l).stack.pop().unwrap_or(Value::Nil)
}
/// Writes a value into a stack slot, an upvalue, or the environment pseudo-index.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `v`'s handles must belong to its VM.
/// Upvalue and environment writes need `(*l).current` to name a live closure entry.
unsafe fn replace_value(l: *mut lua_State, index: c_int, v: Value) {
    if index < UPVALUES {
        if let Some(i) = (*l).current {
            let table = shared(l).closures[i].upvalues;
            let _ = rt(l).table_set(table, &Value::Number(f64::from(UPVALUES - index)), v);
        }
    } else if index == ENVIRON && !MODERN {
        if let (Some(i), Value::Table(table)) = ((*l).current, v) {
            set_c_environment(l, i, table);
        }
    } else if index == GLOBALS && !MODERN {
        if let Value::Table(table) = v {
            let thread = (l != main_state(l)).then_some((*l).thread);
            let _ = vm(l).set_thread_environment_for(thread, table);
        }
    } else if let Some(i) = slot(l, index) {
        state(l).stack[i] = v;
    }
}
/// Reclaims collected native storage and drops the wrappers of dead threads
/// before the VM runs. The host roots themselves are built only when the
/// collector is about to read them (see [`refresh_host_roots`]), not at
/// every call: a host that converts a vector through `lua_call` hundreds of
/// thousands of times would otherwise rebuild them, stack by stack, each
/// time.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose family's main state is alive and whose shared storage maps are intact.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn sync_roots(l: *mut lua_State) {
    // Nothing to reclaim until a collection has run, and no wrapper to drop
    // in a family without coroutines: three compares, with the work out of
    // line, since every call into a C function and out of one comes here.
    let main = main_state(l);
    let runtime = (*(*main).vm).runtime();
    if runtime.gc_freed_userdata() != (*main).last_freed_userdata
        || runtime.gc_collection_epoch() != (*main).last_gc_epoch
        || !(*main).threads.is_empty()
    {
        sync_roots_slow(l);
    }
}
/// [`sync_roots`] once a collection has run or the family has coroutines.
///
/// # Safety
///
/// As for [`sync_roots`].
#[cold]
#[inline(never)]
unsafe fn sync_roots_slow(l: *mut lua_State) {
    // Only collection frees userdata, so skip the scan until it has.
    if rt(l).gc_freed_userdata() != shared(l).last_freed_userdata {
        cleanup_userdata_storage(l);
    }
    let epoch = rt(l).gc_collection_epoch();
    if shared(l).last_gc_epoch != epoch {
        shared(l).last_gc_epoch = epoch;
        cleanup_native_storage(l);
    }
    let main = main_state(l);
    if !(*main).threads.is_empty() {
        let runtime = vm(l).runtime();
        (*main)
            .threads
            .retain(|thread, _| runtime.thread_exists(*thread));
    }
}
/// The collector's hook (see `Runtime::set_host_roots_refresher`): roots
/// the C stacks of every state of the family whose main state is `token`.
fn refresh_host_roots(runtime: &mut Runtime, token: usize) {
    let main = token as *mut lua_State;
    // SAFETY: `root_main_thread` installs the hook with the main state as
    // the token and `close_state` removes it before the state is freed. The
    // collector runs only inside a call into the VM, while the C stacks
    // stand still.
    unsafe {
        (*main)
            .threads
            .retain(|thread, _| runtime.thread_exists(*thread));
        set_thread_roots(runtime, main);
        let children = (*main)
            .threads
            .values()
            .map(|child| child.0)
            .collect::<Vec<_>>();
        for child in children {
            set_thread_roots(runtime, child);
        }
    }
}
/// Roots `l`'s stacks and, on the main state, the light-userdata tokens for the collector.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the family whose VM owns `runtime`.
unsafe fn set_thread_roots(runtime: &mut Runtime, l: *mut lua_State) {
    let mut roots = GcRootBuilder::new();
    roots.values(&(*l).stack);
    for stack in &(*l).saved_stacks {
        roots.values(stack);
    }
    // The VM roots each C closure's mutable capture table. Its environment
    // lives in that table too, so callbacks need only refresh their stacks.
    if (*l).main == l {
        for handle in (*l).light.values() {
            roots.value(&Value::Userdata(*handle));
        }
    }
    let _ = runtime.set_thread_host_roots((*l).thread, roots.finish());
}

/// Thread wrappers are stable while the corresponding Lua thread is alive.
/// The map owns native storage, but does not itself keep Lua threads alive.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the family that owns `thread`, and `thread` must be a thread its VM still knows.
/// The returned wrapper is only usable while the family's main state is alive.
unsafe fn thread_state(l: *mut lua_State, thread: ThreadHandle) -> *mut lua_State {
    let main = main_state(l);
    if thread == (*main).thread {
        return main;
    }
    if !rt(l).thread_exists(thread) {
        return ptr::null_mut();
    }
    let entry = vm(l).host_coroutine_entry(thread).ok().flatten();
    let vm = shared(l).vm;
    if let Some(child) = shared(l).threads.get(&thread) {
        return child.0;
    }
    let child = allocate_state(lua_State {
        _owned_vm: None,
        vm,
        main,
        threads: BTreeMap::new(),
        status: 0,
        entry_pending: entry.is_some(),
        integer_tag: (*main).integer_tag,
        stack: entry.into_iter().collect(),
        saved_stacks: Vec::new(),
        spare_stacks: Vec::new(),
        current: None,
        pending_continuation: None,
        context: (0, 0),
        close_slots: Vec::new(),
        continuations: Vec::new(),
        closures: Vec::new(),
        userdata: AddressMap::default(),
        light: BTreeMap::new(),
        light_reverse: AddressMap::default(),
        strings: AddressMap::default(),
        string_contents: AddressMap::default(),
        last_gc_epoch: 0,
        last_freed_userdata: 0,
        thread,
        panic: None,
        hook: (*main).hook,
        hook_value: None,
        in_hook: false,
        hook_mask: 0,
        hook_count: 0,
        allocator: (*main).allocator,
        allocator_data: (*main).allocator_data,
        native_libraries: Vec::new(),
        external_strings: Vec::new(),
        key_cache: Vec::new(),
        lent: AddressMap::default(),
        lent_recent: Vec::new(),
        owner: Weak::new(),
    });
    if !child.is_null() {
        (*extra_state(child)).extraspace = (*extra_state(main)).extraspace;
        shared(l).threads.insert(thread, ThreadState(child));
    }
    child
}
/// Stores a new environment for the C closure at `closures[index]`.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `index` must name an entry of its family's closure table.
/// `environment` must be a table handle of the same VM.
unsafe fn set_c_environment(l: *mut lua_State, index: usize, environment: TableHandle) {
    shared(l).closures[index].environment = environment;
    let captures = shared(l).closures[index].upvalues;
    let _ = rt(l).table_set(
        captures,
        &Value::from("\0oximond.environment"),
        Value::Table(environment),
    );
}
/// Converts a VM error into the value a failing C call leaves on the stack.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
/// An error carrying an exit request never returns: it runs the state's finalizers and ends the process.
unsafe fn error_value(l: *mut lua_State, error: VmError) -> Value {
    if let Some(request) = error.exit_request() {
        exit_process(l, request);
    }
    match error {
        VmError::HookError(error) => error_value(l, *error),
        VmError::Runtime(RuntimeError::Raised(value)) => value,
        VmError::Runtime(RuntimeError::RaisedNil) => Value::Nil,
        error => Value::from(error.to_string()),
    }
}
/// Ends the process for Lua's `os.exit`, as liblua's `os_exit` does: it
/// closes the state first when asked, and C's `exit` flushes the output.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose finalizers and buffered output may run on this thread.
/// Nothing after this call runs: Rust destructors are skipped and C's `exit` ends the process.
unsafe fn exit_process(l: *mut lua_State, request: ExitRequest) -> ! {
    let mut status = request.status;
    if request.close {
        sync_roots(l);
        if let Some(nested) = vm(l).run_close_finalizers() {
            status = nested.status;
        }
    }
    let _ = rt(l).flush_output();
    let _ = rt(l).flush_error_output();
    let _ = std::io::Write::flush(&mut std::io::stdout());
    std::process::exit(status)
}
/// The status a failed call returns for `error`: `LUA_ERRMEM`, Lua 5.2 and
/// 5.3's `LUA_ERRGCMM` for an error `GCTM` wrapped (only those versions wrap
/// one), or `LUA_ERRRUN`.
fn failure_status(error: &VmError) -> c_int {
    if error.is_memory_error() {
        4
    } else if error.is_wrapped_finalizer_error() {
        5
    } else {
        2
    }
}
/// Pushes the error value and returns the status code a failing C call reports.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// A nested exit request inside `error` ends the process instead of returning.
unsafe fn error_status(l: *mut lua_State, error: VmError) -> c_int {
    let status = failure_status(&error);
    push(l, error_value(l, error));
    status
}
/// Pushes `message` and returns `LUA_ERRRUN`, as a failing C call does.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value on its stack.
unsafe fn failure(l: *mut lua_State, message: impl ToString) -> c_int {
    push(l, Value::from(message.to_string()));
    2
}
/// Borrows `len` raw bytes at `s` as a slice.
///
/// # Safety
///
/// `s` must be valid for reads of `len` bytes with no aliasing writes for `'a`; it may dangle only when `len` is zero.
unsafe fn bytes<'a>(s: *const c_char, len: usize) -> &'a [u8] {
    if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(s.cast(), len)
    }
}
/// Borrows the bytes of the NUL-terminated string at `s`, or nothing when `s` is null.
///
/// # Safety
///
/// `s` must be null or a live NUL-terminated string whose bytes stay unchanged for `'a`.
unsafe fn cstr<'a>(s: *const c_char) -> &'a [u8] {
    if s.is_null() {
        &[]
    } else {
        CStr::from_ptr(s).to_bytes()
    }
}
/// Slots of the key cache. A host's hot path names a few dozen fields, but
/// one that reads definitions names hundreds: Luanti's item and node readers
/// thrashed 64 slots during Mineclonia's startup. 512 took the misses out
/// (2048 measured the same), for 12 KB per state family.
const KEY_CACHE_SLOTS: usize = 512;
/// Longest string the key cache keeps. Longer ones are rarely repeated.
const KEY_CACHE_MAX_LEN: usize = 40;
/// The Lua string for the NUL-terminated `s`, from a cache keyed by the
/// address of `s`. A host passes the same literal for every `lua_getfield`
/// or `lua_pushstring` of a name, so a hit skips the allocation, the copy and
/// the hash of a fresh string, and the reference traffic of dropping it. An
/// address is not an identity (a buffer can hold other text next time), so a
/// hit also compares the bytes. `None` for a string too long to cache.
///
/// The address picks a pair of slots, and a string can sit in either. With
/// one slot per address, two hot literals the linker happened to place at
/// colliding addresses evicted each other on every call: a host binary 80 KB
/// larger moved Luanti's devtest startup from 0 to 3.3% of its instructions
/// in [`cached_key_miss`] and +3.6% in all.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `s` a live NUL-terminated string.
/// The reference is valid until the next call of this function on the family:
/// the caller must use it or clone it first.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn cached_key<'a>(l: *mut lua_State, s: *const c_char) -> Option<&'a LuaString> {
    let address = s as usize;
    let pair = (((address >> 3) ^ (address >> 9)) & (KEY_CACHE_SLOTS / 2 - 1)) * 2;
    let cache = &shared(l).key_cache;
    for slot in [pair, pair + 1] {
        if let Some(Some((held, string))) = cache.get(slot)
            && *held == address
        {
            let stored = string.as_bytes();
            // The stored text has no NUL, so the scan stops at the terminator
            // of `s` at the latest and never reads past it.
            let mut same = true;
            for (i, &byte) in stored.iter().enumerate() {
                if *s.add(i).cast::<u8>() != byte {
                    same = false;
                    break;
                }
            }
            if same && *s.add(stored.len()) == 0 {
                return Some(string);
            }
        }
    }
    cached_key_miss(l, s, pair)
}
/// [`cached_key`] for a string neither slot of `pair` holds: stores it in the
/// first slot and moves the string there to the second, evicting the one the
/// second held.
///
/// # Safety
///
/// As for [`cached_key`]; `pair` must be an even index below [`KEY_CACHE_SLOTS`].
#[cold]
#[inline(never)]
unsafe fn cached_key_miss<'a>(
    l: *mut lua_State,
    s: *const c_char,
    pair: usize,
) -> Option<&'a LuaString> {
    let text = cstr(s);
    if text.len() > KEY_CACHE_MAX_LEN {
        return None;
    }
    let cache = &mut shared(l).key_cache;
    if cache.is_empty() {
        cache.resize(KEY_CACHE_SLOTS, None);
    }
    let address = s as usize;
    // A slot whose address matches but whose text changed is replaced where
    // it stands, so one address never sits in both slots.
    let slot = if cache[pair + 1]
        .as_ref()
        .is_some_and(|(held, _)| *held == address)
    {
        pair + 1
    } else {
        if cache[pair]
            .as_ref()
            .is_some_and(|(held, _)| *held != address)
        {
            cache[pair + 1] = cache[pair].take();
        }
        pair
    };
    // The process-wide constant shares one string with the compiled code
    // that reads or writes the same field, so lookups match by identity.
    cache[slot] = Some((address, oximond::string::shared_key(text)));
    cache[slot].as_ref().map(|(_, string)| string)
}
/// Returns a stable NUL-terminated pointer for a Lua string, sharing buffers between equal short strings.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose family owns the shared string tables.
/// The pointer is valid until the state's cleanup frees the entry and must not be freed by the caller.
unsafe fn intern(l: *mut lua_State, string: LuaString) -> *const c_char {
    let identity = string.identity();
    if let Some(external) = shared(l)
        .external_strings
        .iter()
        .find(|external| external.string.identity() == identity && external.string.is_alive())
    {
        return external.bytes.cast();
    }
    if let Some(data) = shared(l).strings.get(&identity) {
        return data.bytes.as_ptr().cast();
    }
    let shares_buffers = !MODERN || string.len() <= 40;
    if shares_buffers {
        let shared = shared(l);
        if let Some(data) = shared
            .string_contents
            .get(&string.cached_hash())
            .and_then(|key| shared.strings.get_mut(key))
            .filter(|data| data.bytes.split_last().map(|(_, text)| text) == Some(string.as_bytes()))
        {
            // The entry owns its buffer, and this string's reference keeps
            // the entry through cleanup, even if every earlier one died.
            data.references
                .entry(identity)
                .or_insert_with(|| string.downgrade());
            return data.bytes.as_ptr().cast();
        }
    }
    let mut bytes = string.as_bytes().to_vec();
    bytes.push(0);
    let data = InternedString {
        bytes: bytes.into_boxed_slice(),
        references: AddressMap::from_iter([(identity, string.downgrade())]),
    };
    let pointer = data.bytes.as_ptr().cast();
    shared(l).strings.insert(identity, data);
    if shares_buffers {
        shared(l)
            .string_contents
            .insert(string.cached_hash(), identity);
    }
    pointer
}
/// Releases the host-allocated block of a full userdata record.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose allocator is the one that allocated `record`'s block.
/// `record` must be a host-backed storage the userdata map no longer references.
unsafe fn free_userdata(l: *mut lua_State, record: Userdata) {
    if let UserdataStorage::Host { pointer, size } = record.storage
        && let Some(allocator) = shared(l).allocator
    {
        let mut result = ptr::null_mut();
        let data = shared(l).allocator_data;
        let _ = oximond_allocate(l, allocator, data, pointer, size, 0, &mut result);
    }
}
/// Releases the buffer of a collected external string through its allocator.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `external.bytes` must be the `external.length + 1` byte block `external.allocator` allocated with `external.data`.
/// No Lua string may still reference the buffer.
unsafe fn free_external(l: *mut lua_State, external: ExternalString) {
    if let Some(allocator) = external.allocator {
        let mut result = ptr::null_mut();
        let _ = oximond_allocate(
            l,
            allocator,
            external.data,
            external.bytes,
            external.length + 1,
            0,
            &mut result,
        );
    }
}
/// Frees the storage of userdata the collector has reclaimed.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the family owning the shared userdata map, and every freed handle must already be gone from the VM.
unsafe fn cleanup_userdata_storage(l: *mut lua_State) {
    shared(l).last_freed_userdata = rt(l).gc_freed_userdata();
    let dead = shared(l)
        .userdata
        .keys()
        .copied()
        .filter(|handle| rt(l).userdata_environment(*handle).is_err())
        .collect::<Vec<_>>();
    for handle in dead {
        if let Some(record) = shared(l).userdata.remove(&handle) {
            free_userdata(l, record);
        }
    }
}
/// Drops interned buffers and external strings whose Lua strings have been collected.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the family owning the shared string maps, and nothing else may touch those maps during the scan.
unsafe fn cleanup_native_storage(l: *mut lua_State) {
    cleanup_userdata_storage(l);
    // Pointers `lua_tolstring` lent last cycle live until now, as stock
    // Lua's strings do until they are collected.
    shared(l).lent.clear();
    shared(l).lent_recent.fill(0);
    shared(l).strings.retain(|identity, entry| {
        // Equal strings can share this buffer after its first string dies.
        // Keep the key's allocation reserved so a different string cannot
        // acquire the same address and incorrectly find this buffer.
        entry
            .references
            .retain(|reference, weak| reference == identity || weak.is_alive());
        entry.references.values().any(WeakLuaString::is_alive)
    });
    let storage = shared(l);
    let strings = &storage.strings;
    storage
        .string_contents
        .retain(|_, key| strings.contains_key(key));
    let old = std::mem::take(&mut shared(l).external_strings);
    for external in old {
        if external.string.is_alive() {
            shared(l).external_strings.push(external);
        } else {
            free_external(l, external);
        }
    }
}
/// Asks the VM to suspend with `continuation` as the resume point, attaching its parked close handler.
///
/// # Safety
///
/// `vm` must be the live `Vm` that `l`'s family is reborrowed from, and `l` a live state of that family.
/// `continuation` must be a host function previously registered by `park_continuation`.
unsafe fn request_continuation(
    vm: &mut Vm,
    l: *mut lua_State,
    values: Vec<Value>,
    continuation: Value,
) -> VmError {
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    let close = shared(l)
        .continuations
        .iter()
        .find(|record| record.value.as_ref() == Some(&continuation))
        .and_then(|record| record.close_handler.clone());
    let _ = l;
    let result = vm.request_host_yield_with_continuation(values, continuation);
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    if matches!(result, VmError::Yield(_)) {
        if let Some(close) = close {
            if let Err(error) = vm.set_host_continuation_close_handler(close) {
                return error;
            }
        }
    }
    result
}
fn host_call(vm_ref: &mut Vm, token: u64, args: &[Value]) -> Result<Vec<Value>, VmError> {
    host_call_with(vm_ref, token, (args.as_ptr(), args.len()), |vm, results| {
        // A vector from the runtime's pool, which the VM hands back once
        // it has stored the results.
        let mut values = vm.runtime_mut().result_vec();
        values.extend_from_slice(results);
        values
    })
}

/// The [`oximond::vm::HostDirectCallback`] of every C function: [`host_call`]
/// that moves the results from the C stack to where the caller wants them.
///
/// # Safety
///
/// The contract of `HostDirectCallback`: `count` values are readable at
/// `args` and `wanted` initialized values are writable at `out`.
unsafe fn host_call_direct(
    vm_ref: &mut Vm,
    token: u64,
    args: *const Value,
    count: usize,
    out: *mut Value,
    wanted: usize,
) -> Result<(), VmError> {
    host_call_with(vm_ref, token, (args, count), |_, results| {
        let mut results = results.iter_mut();
        for index in 0..wanted {
            let value = results
                .next()
                .map_or(Value::Nil, |value| std::mem::replace(value, Value::Nil));
            // SAFETY: inside the caller's range; the assignment drops the
            // value the slot held.
            unsafe { *out.add(index) = value };
        }
    })
}

#[inline(always)]
fn host_call_with<T>(
    vm_ref: &mut Vm,
    token: u64,
    (args, count): (*const Value, usize),
    finish: impl FnOnce(&mut Vm, &mut [Value]) -> T,
) -> Result<T, VmError> {
    // The boxed closure and state outlive every registered VM callback.
    unsafe {
        let closure = token as *const CClosure;
        let main = (*closure).state;
        let function = (*closure).function;
        let index = (*closure).index;
        let old_vm = (*main).vm;
        (*main).vm = vm_ref;
        let l = match vm_ref.current_coroutine() {
            Some(thread) => thread_state(main, thread),
            None => main,
        };
        if l.is_null() {
            (*main).vm = old_vm;
            return Err(RuntimeError::memory_error().into());
        }
        if (*l).saved_stacks.len() >= 200 {
            (*main).vm = old_vm;
            return Err(RuntimeError::message("C stack overflow").into());
        }
        let old_current = (*l).current.replace(index);
        // Close slots and a pending continuation belong to the caller's C
        // frame. Almost no call has either (only `lua_toclose` and the
        // `k` calls make them), and then they stay where they are: the
        // callee starts with none, as it would after a take.
        let old_close =
            (!(*l).close_slots.is_empty()).then(|| std::mem::take(&mut (*l).close_slots));
        let old_pending = if (*l).pending_continuation.is_some() {
            (*l).pending_continuation.take()
        } else {
            None
        };
        let mut fresh = (*l).spare_stacks.pop().unwrap_or_default();
        // `LUA_MINSTACK` slots, which a C function may push without `lua_checkstack`.
        fresh.reserve(count + 20);
        // The arguments are read here and not again: `finish` may write
        // the results over them.
        fresh.extend_from_slice(std::slice::from_raw_parts(args, count));
        let saved = std::mem::replace(&mut (*l).stack, fresh);
        state(l).saved_stacks.push(saved);
        sync_roots(l);
        let mut count = 0;
        let mut status = oximond_invoke(l, function, &mut count);
        // Only a Lua 5.4 `lua_toclose` marks slots; most calls have none.
        if status != 1 && !(*l).close_slots.is_empty() {
            let error = if status == 0 {
                Value::Nil
            } else {
                (*l).stack.last().cloned().unwrap_or(Value::Nil)
            };
            if let Err(failure) = close_slots(l, 0, error) {
                status = error_status(l, failure);
            }
        }
        let result =
            if (status == 0 || status == 1) && count >= 0 && count as usize <= (*l).stack.len() {
                let first = (*l).stack.len() - count as usize;
                if status == 1 {
                    let mut values = vm_ref.runtime_mut().result_vec();
                    values.extend_from_slice(&state(l).stack[first..]);
                    match (*l).pending_continuation.take() {
                        Some(continuation) => {
                            Err(request_continuation(vm_ref, l, values, continuation))
                        }
                        None => Err(vm_ref.request_host_yield(values)),
                    }
                } else {
                    Ok(finish(vm_ref, &mut state(l).stack[first..]))
                }
            } else {
                let value = pop(l);
                Err(if status == 4 {
                    RuntimeError::memory_error()
                } else {
                    match value {
                        Value::Nil => RuntimeError::RaisedNil,
                        value => RuntimeError::Raised(value),
                    }
                }
                .into())
            };
        let mut finished = std::mem::replace(
            &mut (*l).stack,
            state(l).saved_stacks.pop().expect("saved C stack"),
        );
        // Keep modest buffers for the next call; a huge one is not worth holding.
        if finished.capacity() <= 4096 && (*l).spare_stacks.len() < 8 {
            finished.clear();
            (*l).spare_stacks.push(finished);
        }
        (*l).current = old_current;
        // What the callee left of its own goes; the caller's come back.
        match old_close {
            Some(slots) => (*l).close_slots = slots,
            None => (*l).close_slots.clear(),
        }
        if old_pending.is_some() || (*l).pending_continuation.is_some() {
            (*l).pending_continuation = old_pending;
        }
        // No `sync_roots` here: the storage a collection freed during the
        // call is reclaimed at the next entry into a C function or the VM
        // (`call`), which is before anything can ask for it.
        (*main).vm = old_vm;
        result
    }
}

/// Creates a new state through the stock `lua_newstate` allocator interface.
///
/// # Safety
///
/// `allocator` must be null or an `Alloc` callable with `data` on this thread for the state's whole life.
/// The returned state must be used on the creating thread only and closed with `lua_close`.
#[unsafe(no_mangle)]
#[cfg(not(feature = "lua55"))]
pub unsafe extern "C" fn lua_newstate(
    allocator: Option<Alloc>,
    data: *mut c_void,
) -> *mut lua_State {
    new_state(allocator, data)
}
/// Creates a new state through the stock `lua_newstate` allocator interface; `_seed` is ignored.
///
/// # Safety
///
/// `allocator` must be null or an `Alloc` callable with `data` on this thread for the state's whole life.
/// The returned state must be used on the creating thread only and closed with `lua_close`.
#[unsafe(no_mangle)]
#[cfg(feature = "lua55")]
pub unsafe extern "C" fn lua_newstate(
    allocator: Option<Alloc>,
    data: *mut c_void,
    _seed: u32,
) -> *mut lua_State {
    new_state(allocator, data)
}
/// Creates a state for the compiled-in profile with tracing execution.
///
/// # Safety
///
/// `allocator` must be null or an `Alloc` that accepts the state's allocations and releases with `data` on this thread.
/// The returned state must be used only on the thread that created it.
unsafe fn new_state(allocator: Option<Alloc>, data: *mut c_void) -> *mut lua_State {
    let l = new_state_with_profile(allocator, data, PROFILE, default_execution());
    if MODERN && !l.is_null() {
        cold_libraries(l);
        // Stock `lua_newstate` leaves warnf NULL, so warnings go nowhere
        // until the host installs a function. `luaL_newstate` installs one.
        #[cfg(any(feature = "lua54", feature = "lua55"))]
        rt(l).set_warning_handler(Some(discard_warnings()));
    }
    l
}
/// The execution mode of states made by `lua_newstate` and `luaL_newstate`:
/// tracing, or the interpreter alone when `OXIMOND_EXECUTION` is
/// `interpreter`. The variable lets an embedder's workload be measured
/// without the JIT, with no change to the embedder. Read once per process.
fn default_execution() -> ExecutionMode {
    static MODE: std::sync::OnceLock<ExecutionMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        if std::env::var_os("OXIMOND_EXECUTION").is_some_and(|mode| mode == "interpreter") {
            ExecutionMode::Interpreter
        } else {
            ExecutionMode::Tracing
        }
    })
}
/// A warning handler that drops every warning, as a NULL warnf does.
#[cfg(any(feature = "lua54", feature = "lua55"))]
fn discard_warnings() -> Box<oximond::runtime::WarningHandler> {
    Box::new(|_: &[u8], _: bool| Ok(()))
}
/// Creates a state for an explicit compatibility profile and execution mode.
///
/// # Safety
///
/// `allocator` must be null or an `Alloc` usable with `data` on this thread for the state's whole life.
/// `profile` and `execution` must be documented codes; anything else returns null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oximond_newstate_with_options(
    allocator: Option<Alloc>,
    data: *mut c_void,
    profile: c_int,
    execution: c_int,
) -> *mut lua_State {
    let profile = match profile {
        51 => CompatibilityProfile::Lua51,
        52 => CompatibilityProfile::Lua52,
        53 => CompatibilityProfile::Lua53,
        54 => CompatibilityProfile::Lua54,
        55 => CompatibilityProfile::Lua55,
        21 => CompatibilityProfile::LuaJit21,
        _ => return ptr::null_mut(),
    };
    let execution = match execution {
        0 => ExecutionMode::Interpreter,
        1 => ExecutionMode::Tracing,
        _ => return ptr::null_mut(),
    };
    new_state_with_profile(allocator, data, profile, execution)
}
/// Builds the VM and main `lua_State` for a compatibility profile and execution mode.
///
/// # Safety
///
/// `allocator` must be null or an `Alloc` that accepts this state's allocation pattern with `data`.
/// The returned state must be used only on the thread that created it.
unsafe fn new_state_with_profile(
    allocator: Option<Alloc>,
    data: *mut c_void,
    profile: CompatibilityProfile,
    execution: ExecutionMode,
) -> *mut lua_State {
    let builder = Lua::builder().profile(profile).execution_mode(execution);
    let builder = if !MODERN {
        builder
            .luajit_extension(oximond::LuaJitExtension::Jit)
            .luajit_extension(oximond::LuaJitExtension::Ffi)
    } else {
        builder
    };
    let Ok(lua) = builder.build() else {
        return ptr::null_mut();
    };
    let mut owned_vm = Box::new(lua.into_vm());
    // Without writers the runtime keeps what it writes in memory. liblua's
    // `print` and `io.write` go to stdout, and `luaL_newstate`'s default
    // warning function writes to stderr.
    owned_vm.runtime_mut().set_output_writer(std::io::stdout());
    owned_vm
        .runtime_mut()
        .set_error_output_writer(std::io::stderr());
    // Lua 5.4's `luaL_newstate` starts with warnings off; 5.5's starts with them on.
    if matches!(profile, CompatibilityProfile::Lua54) {
        owned_vm.runtime_mut().set_warnings_enabled(false);
    }
    let Ok(thread) = owned_vm.runtime_mut().create_thread() else {
        return ptr::null_mut();
    };
    owned_vm
        .runtime_mut()
        .set_global("_OXIMOND_VERSION", Value::from(env!("CARGO_PKG_VERSION")));
    let owned_tagged = owned_vm.runtime().tagged_integers();
    let vm = ptr::from_mut(owned_vm.as_mut());
    let l = allocate_state(lua_State {
        _owned_vm: Some(owned_vm),
        vm,
        main: ptr::null_mut(),
        threads: BTreeMap::new(),
        status: 0,
        entry_pending: false,
        integer_tag: integer_tag(owned_tagged),
        stack: Vec::new(),
        saved_stacks: Vec::new(),
        spare_stacks: Vec::new(),
        current: None,
        pending_continuation: None,
        context: (0, 0),
        close_slots: Vec::new(),
        continuations: Vec::new(),
        closures: Vec::new(),
        userdata: AddressMap::default(),
        light: BTreeMap::new(),
        light_reverse: AddressMap::default(),
        strings: AddressMap::default(),
        string_contents: AddressMap::default(),
        last_gc_epoch: 0,
        last_freed_userdata: 0,
        thread,
        panic: None,
        hook: None,
        hook_value: None,
        in_hook: false,
        hook_mask: 0,
        hook_count: 0,
        allocator,
        allocator_data: data,
        native_libraries: Vec::new(),
        external_strings: Vec::new(),
        key_cache: Vec::new(),
        lent: AddressMap::default(),
        lent_recent: Vec::new(),
        owner: Weak::new(),
    });
    if !l.is_null() {
        root_main_thread(l);
        install_native_package(l);
    }
    l
}

/// Installs `package.loadlib` and the native searcher on a fresh state.
///
/// # Safety
///
/// `l` must be a live state just created by `new_state_with_profile`, with its standard libraries present.
unsafe fn install_native_package(l: *mut lua_State) {
    let Value::Table(package) = rt(l).get_global("package") else {
        return;
    };
    lua_pushcclosure(l, oximond_package_loadlib, 0);
    let loader = pop(l);
    rt(l)
        .table_set(package, &Value::from("loadlib"), loader)
        .expect("install package.loadlib");
    if let Ok(Value::Table(loaders)) = rt(l).table_get(package, &Value::from("loaders")) {
        lua_pushcclosure(l, oximond_package_native_searcher, 0);
        let searcher = pop(l);
        for index in [3.0, 4.0] {
            rt(l)
                .table_set(loaders, &Value::Number(index), searcher.clone())
                .expect("install native package searcher");
        }
    }
}

/// Records a native library handle for the state to close.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `handle` a library handle `oximond_close_library` accepts.
/// The handle must stay valid until `lua_close` runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oximond_remember_library(l: *mut lua_State, handle: *mut c_void) {
    shared(l).native_libraries.push(handle);
}

/// Reports whether the state uses Lua 5.2+ module naming.
///
/// # Safety
///
/// `l` must be a live `lua_State` used on its owning thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oximond_modern_module_names(l: *mut lua_State) -> c_int {
    i32::from(matches!(
        vm(l).compatibility_profile(),
        CompatibilityProfile::Lua52
            | CompatibilityProfile::Lua53
            | CompatibilityProfile::Lua54
            | CompatibilityProfile::Lua55
    ))
}

/// Runs the state's finalizers and frees it, like `lua_close`.
///
/// # Safety
///
/// `l` must be null or a live `lua_State` the caller owns; afterwards no pointer to it or its thread wrappers may be used.
/// The finalizers and library destructors it runs must tolerate being called now.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_close(l: *mut lua_State) {
    if l.is_null() {
        return;
    }
    close_state(main_state(l), true);
}

/// Finalizes and releases one state of a family.
///
/// # Safety
///
/// `l` must be null or a live `lua_State` of a family being torn down; after the call `l` and its wrappers dangle.
unsafe fn close_state(l: *mut lua_State, finalize: bool) {
    if l.is_null() {
        return;
    }
    sync_roots(l);
    if finalize {
        let _ = vm(l).run_close_finalizers();
    }
    // C's exit flushes liblua's stdio buffers, but nothing flushes Rust's
    // stdout buffer in a C host, so a partial line would be lost.
    let _ = rt(l).flush_output();
    let _ = rt(l).flush_error_output();
    if l == main_state(l) {
        // The hook names this state, which is about to be freed.
        rt(l).set_host_roots_refresher(None);
        if std::env::var_os("OXIMOND_JIT_TRACE").is_some() {
            // The state's totals, which per-loop reports reset at each
            // recompile.
            let statistics = vm(l).jit_statistics();
            let trace = statistics.trace;
            let kinds = [
                oximond::vm::TraceExitKind::EntryMismatch,
                oximond::vm::TraceExitKind::GuardFail,
                oximond::vm::TraceExitKind::LoopComplete,
                oximond::vm::TraceExitKind::CommittedEffect,
                oximond::vm::TraceExitKind::Poll,
            ];
            let exits: Vec<String> = kinds
                .iter()
                .map(|kind| format!("{kind:?}={}", trace.exits[kind.index()]))
                .collect();
            eprintln!(
                "jit-trace: state totals: native_entries={} iterations={} traces_compiled={} side_entries={} exits {}",
                trace.native_entries,
                trace.iterations,
                trace.traces_compiled,
                trace.side_entries,
                exits.join(" ")
            );
            // What the runs asked the VM for: each is a call out of native
            // code, which a profile shows only as the helper's own share.
            eprintln!(
                "jit-trace: state helper calls: host={} table_lookup={} closure={} hash={} set_string={} array_set={} table_new={}",
                statistics.host_trace_calls,
                trace.table_lookup_helper_calls,
                trace.closure_helper_calls,
                trace.hash_helper_calls,
                trace.set_string_helper_calls,
                trace.array_set_helper_calls,
                trace.table_new_helper_calls,
            );
        }
    }
    release_state(l);
    // `OXIMOND_ATTRIBUTION` reports of a host that never ends the process
    // cleanly are otherwise written once a second.
    oximond::vm::attribution::flush();
}

/// Frees external strings, userdata storage, and native libraries, then the state allocation.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose family nothing will touch again.
/// Its allocator must accept every release recorded in the shared maps.
unsafe fn release_state(l: *mut lua_State) {
    for external in std::mem::take(&mut shared(l).external_strings) {
        free_external(l, external);
    }
    for (_, record) in std::mem::take(&mut shared(l).userdata) {
        free_userdata(l, record);
    }
    for handle in shared(l).native_libraries.drain(..).rev() {
        oximond_close_library(handle);
    }
    free_state_allocation(l);
}
/// Installs the panic function and returns the previous one.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
/// `f` must be null or a `CFunction` upholding the C callback contract, and the returned function must stay valid while the state may call it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_atpanic(l: *mut lua_State, f: Option<CFunction>) -> Option<CFunction> {
    std::mem::replace(&mut shared(l).panic, f)
}
/// Returns the number of values on `l`'s stack.
///
/// # Safety
///
/// `l` must be a live `lua_State` used on its owning thread.
/// The count is stale as soon as anything is pushed or popped.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_gettop(l: *mut lua_State) -> c_int {
    (*l).stack.len() as c_int
}
/// Grows or shrinks `l`'s stack to `index`, padding with nil, like `lua_settop`.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
/// Negative `index` values must not reach below the stack bottom; those are ignored.
#[unsafe(no_mangle)]
#[cfg(not(any(feature = "lua54", feature = "lua55")))]
pub unsafe extern "C" fn lua_settop(l: *mut lua_State, index: c_int) {
    // `lua_pop(L, 1)` of anything but a string, which a host makes after
    // almost every read: a length decrement.
    if index == -2 {
        let stack = &mut state(l).stack;
        // One subtraction both tests for an empty stack and gives the new length.
        if let Some(below) = stack.len().checked_sub(1)
            && !matches!(*stack.as_ptr().add(below), Value::String(_))
        {
            // SAFETY: the value on top owns nothing to drop.
            stack.set_len(below);
            return;
        }
    }
    let size = if index >= 0 {
        index as isize
    } else {
        (*l).stack.len() as isize + index as isize + 1
    };
    if size >= 0 {
        resize_stack(l, size as usize);
    }
}
/// Sets the length of `l`'s stack to `size`, padding with nil. A host pops
/// a value after almost every read (`lua_pop` after `lua_rawgeti`, once per
/// node of a VoxelManip buffer in Luanti), and a pop of values that own
/// nothing (anything but a string) only shortens the stack, without the
/// drop loop `Vec::truncate` runs.
///
/// # Safety
///
/// `l` must be a live `lua_State` and nothing may borrow its stack vector.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn resize_stack(l: *mut lua_State, size: usize) {
    let stack = &mut state(l).stack;
    // `lua_pop(L, 1)` of anything but a string: one tag test.
    if size + 1 == stack.len() && !matches!(stack.last(), Some(Value::String(_))) {
        // SAFETY: the value past `size` owns nothing to drop.
        stack.set_len(size);
        return;
    }
    resize_stack_slow(l, size);
}
/// [`resize_stack`] for anything but a pop of one value that owns nothing.
/// `extern "C"` so the fast path tail-jumps here: see [`rawseti_slow`].
///
/// # Safety
///
/// As for [`resize_stack`].
#[inline(never)]
unsafe extern "C" fn resize_stack_slow(l: *mut lua_State, size: usize) {
    let stack = &mut state(l).stack;
    if size < stack.len() && !stack[size..].iter().any(|v| matches!(v, Value::String(_))) {
        // SAFETY: the values past `size` own nothing to drop.
        unsafe { stack.set_len(size) };
        return;
    }
    stack.resize(size, Value::Nil);
}
/// Pushes a copy of the value at `index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `index` must name one of its values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushvalue(l: *mut lua_State, index: c_int) {
    // A stack slot copied into a stack with room: no call, so this leaf
    // saves no registers. The rest goes out of line with a tail call.
    if let Some(value) = peek(l, index)
        && let Some(slot) = spare_slot(l)
    {
        slot.write(value.clone());
        commit_slot(l);
        return;
    }
    pushvalue_slow(l, index);
}
/// [`lua_pushvalue`] for a pseudo-index, an index past the stack or a full stack.
///
/// # Safety
///
/// As for [`lua_pushvalue`].
#[cold]
#[inline(never)]
unsafe extern "C" fn pushvalue_slow(l: *mut lua_State, index: c_int) {
    let v = value(l, index).unwrap_or(Value::Nil);
    push(l, v);
}
/// Removes the value at `index`, sliding the values above it down.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and no reference into its stack may be held across the removal.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_remove(l: *mut lua_State, index: c_int) {
    if let Some(i) = slot(l, index) {
        state(l).stack.remove(i);
    }
}
/// Moves the top value into `index`, sliding the values above it up.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value, and nothing may borrow its stack across the move.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_insert(l: *mut lua_State, index: c_int) {
    if let Some(i) = slot(l, index) {
        let v = pop(l);
        state(l).stack.insert(i, v);
    }
}
/// Pops the top value into the slot `index` names.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value.
/// `index` must name a stack slot, upvalue, or environment the write accepts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_replace(l: *mut lua_State, index: c_int) {
    let v = state(l).stack.last().cloned().unwrap_or(Value::Nil);
    replace_value(l, index, v);
    pop(l);
}
/// Reserves room for `n` more stack slots, returning one on success.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose stack vector nothing else borrows.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_checkstack(l: *mut lua_State, n: c_int) -> c_int {
    i32::from(n >= 0 && state(l).stack.try_reserve(n as usize).is_ok())
}
/// Returns the type code of the value at `index`, or `LUA_TNONE`.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `index` must be one of its stack or pseudo-indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_type(l: *mut lua_State, index: c_int) -> c_int {
    // A stack slot is read in place, not cloned.
    if let Some(value) = peek(l, index) {
        return type_code(l, value);
    }
    value_slow(l, index).map_or(-1, |value| type_code(l, &value))
}
/// The `lua_type` code of `value`.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the VM `value` belongs to.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn type_code(l: *mut lua_State, value: &Value) -> c_int {
    match value {
        Value::Nil => 0,
        Value::Boolean(_) => 1,
        Value::Userdata(h) if shared(l).light_reverse.contains_key(h) => 2,
        Value::Number(_) | Value::Integer(_) => 3,
        Value::String(_) => 4,
        Value::Table(_) => 5,
        Value::Userdata(_) | Value::File(_) => 7,
        Value::Thread(_) => 8,
        // LuaJIT reports cdata as its internal type 10 (`LUA_TCDATA`).
        Value::Int64(_) | Value::UInt64(_) => 10,
        _ => 6,
    }
}
/// Returns the name of a type code as a static string.
///
/// # Safety
///
/// The state argument is ignored and may be null; the result is a static literal valid while the library is loaded.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_typename(_: *mut lua_State, t: c_int) -> *const c_char {
    match t {
        0 => c"nil",
        1 => c"boolean",
        2 => c"userdata",
        3 => c"number",
        4 => c"string",
        5 => c"table",
        6 => c"function",
        7 => c"userdata",
        8 => c"thread",
        10 => c"cdata",
        _ => c"no value",
    }
    .as_ptr()
}
/// Reports whether the value at `i` converts to a number.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
/// The check itself never runs Lua code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_isnumber(l: *mut lua_State, i: c_int) -> c_int {
    // A slot is judged where it stands. Only a string needs the conversion;
    // any other value is no number, without a clone or an error value made
    // to say so (a host asks this of nil for every optional field).
    match peek(l, i) {
        Some(Value::Number(_) | Value::Integer(_)) => 1,
        Some(text @ Value::String(_)) => {
            i32::from(Runtime::coerce_number(text, "conversion").is_ok())
        }
        Some(_) => 0,
        None => i32::from(
            value_slow(l, i).is_some_and(|v| Runtime::coerce_number(&v, "conversion").is_ok()),
        ),
    }
}
/// Reports whether the value at `i` is a string or a number.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack or pseudo-indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_isstring(l: *mut lua_State, i: c_int) -> c_int {
    i32::from(matches!(lua_type(l, i), 3 | 4))
}
/// Reports whether the value at `i` is a C closure this API created.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_iscfunction(l: *mut lua_State, i: c_int) -> c_int {
    i32::from(lua_tocfunction(l, i).is_some())
}
/// Reports whether the value at `i` is light or full userdata.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_isuserdata(l: *mut lua_State, i: c_int) -> c_int {
    i32::from(matches!(lua_type(l, i), 2 | 7))
}
/// Converts the value at `i` to a number, or zero when it does not convert.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_tonumber(l: *mut lua_State, i: c_int) -> f64 {
    match peek(l, i) {
        Some(&Value::Number(n)) => n,
        Some(held @ (Value::Integer(_) | Value::String(_))) => {
            Runtime::coerce_number(held, "conversion").unwrap_or(0.0)
        }
        // Nothing else converts.
        Some(_) => 0.0,
        None => value_slow(l, i)
            .and_then(|v| Runtime::coerce_number(&v, "conversion").ok())
            .unwrap_or(0.0),
    }
}
/// Converts the value at `i` to an integer under the state's number model.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_tointeger(l: *mut lua_State, i: c_int) -> LuaInteger {
    if MODERN {
        return lua_tointegerx(l, i, ptr::null_mut());
    }
    // The read after a `lua_rawgeti` names the top, which needs no index
    // arithmetic. A host reads numbers far more often than integers: under
    // Lua 5.1 and LuaJIT every number is one.
    let value = if i == -1 { top_value(l) } else { peek(l, i) };
    if let Some(&Value::Number(n)) = value {
        return number_to_integer(n);
    }
    match value {
        Some(&Value::Integer(value)) => value as LuaInteger,
        _ => tointeger_slow(l, i),
    }
}
/// `n as LuaInteger`: truncated toward zero, saturated out of range, and zero
/// for NaN. On x86-64 one truncating conversion settles every number in
/// range. The saturating conversion costs six more instructions, so only the
/// out-of-range answer takes it.
#[allow(clippy::inline_always)]
#[inline(always)]
fn number_to_integer(n: f64) -> LuaInteger {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_cvttsd_si64, _mm_set_sd};
        // SAFETY: SSE2 is part of the x86-64 baseline.
        #[allow(unused_unsafe)]
        let truncated = unsafe { _mm_cvttsd_si64(_mm_set_sd(n)) };
        // NaN and every number out of range give `i64::MIN`, as does -2^63
        // itself, which the saturating conversion maps to the same value.
        if truncated != i64::MIN {
            return truncated as LuaInteger;
        }
    }
    saturate_to_integer(n)
}
/// The saturating half of [`number_to_integer`], out of line so the compiler
/// does not compute it beside the truncation on every call.
#[cold]
#[inline(never)]
fn saturate_to_integer(n: f64) -> LuaInteger {
    n as LuaInteger
}
/// [`lua_tointeger`] for anything but a number on the stack.
/// `extern "C"` so the fast path tail-jumps here: see [`rawseti_slow`].
///
/// # Safety
///
/// As for [`lua_tointeger`].
#[cold]
#[inline(never)]
unsafe extern "C" fn tointeger_slow(l: *mut lua_State, i: c_int) -> LuaInteger {
    match value(l, i) {
        Some(Value::Integer(value)) => value as LuaInteger,
        _ => lua_tonumber(l, i) as LuaInteger,
    }
}
/// Returns one when the value at `i` is truthy.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_toboolean(l: *mut lua_State, i: c_int) -> c_int {
    if let Some(value) = peek(l, i) {
        return i32::from(value.is_truthy());
    }
    i32::from(value_slow(l, i).is_some_and(|v| v.is_truthy()))
}
/// Returns the string at `i` as bytes and a length, converting numbers in place.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `i` must be a stack slot whose number form may replace it.
/// `len` must be null or writable, and the returned pointer follows the interned-string lifetime and is never freed by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_tolstring(
    l: *mut lua_State,
    i: c_int,
    len: *mut usize,
) -> *const c_char {
    // A short string lends the terminated bytes of its own storage. `lend`
    // holds the string until the next collection starts, so the pointer
    // lives as long as an interned copy would: hosts pop a value and use its
    // text after, as stock Lua allows until the string is collected.
    if let Some(Value::String(string)) = peek(l, i)
        && let Some(terminated) = string.inline_bytes_with_nul()
        && shared(l).external_strings.is_empty()
        && lend(l, string)
    {
        if !len.is_null() {
            *len = terminated.len() - 1;
        }
        return terminated.as_ptr().cast();
    }
    tolstring_slow(l, i, len)
}
/// Slots of [`lend`]'s table of recently lent strings.
const LENT_RECENT: usize = 256;
/// The most strings [`lend`] holds in one cycle. C pushes add no GC debt, so
/// a host that pushes and reads fresh strings without running Lua never
/// starts a collection; past this many, `lua_tolstring` interns, as it did
/// before strings were lent.
const LENT_MAX: usize = 1 << 16;
/// Holds `string`, whose storage `lua_tolstring` is about to lend, until the
/// next collection starts (see `cleanup_native_storage`). A string lent
/// again since needs nothing; a recorded identity cannot be reused, as the
/// held clone keeps its allocation.
///
/// The held strings are keyed by identity, so a host that reads the same
/// strings over and over between collections holds each once: the held set
/// is bounded by the distinct strings lent in a cycle, as the interned
/// copies were, and costs no copy of their bytes. A capped list used to
/// send Mineclonia's startup back to interning for 2.0M of its 2.3M reads.
/// [`LENT_MAX`] bounds a cycle that never ends; returns false past it, and
/// the caller interns instead.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn lend(l: *mut lua_State, string: &LuaString) -> bool {
    let identity = string.identity();
    let slot = ((identity >> 4) ^ (identity >> 10)) & (LENT_RECENT - 1);
    let family = shared(l);
    if family.lent_recent.get(slot) == Some(&identity) {
        return true;
    }
    lend_miss(l, string, slot)
}
/// [`lend`] for a string not lent recently.
///
/// # Safety
///
/// As for [`lend`]; `slot` must be below [`LENT_RECENT`].
#[inline(never)]
unsafe fn lend_miss(l: *mut lua_State, string: &LuaString, slot: usize) -> bool {
    let family = shared(l);
    if family.lent_recent.is_empty() {
        family.lent_recent.resize(LENT_RECENT, 0);
    }
    let identity = string.identity();
    if family.lent.len() >= LENT_MAX && !family.lent.contains_key(&identity) {
        return false;
    }
    family
        .lent
        .entry(identity)
        .or_insert_with(|| string.clone());
    family.lent_recent[slot] = identity;
    true
}
/// [`lua_tolstring`] for anything but a short string on the stack.
///
/// # Safety
///
/// As for [`lua_tolstring`].
#[inline(never)]
unsafe fn tolstring_slow(l: *mut lua_State, i: c_int, len: *mut usize) -> *const c_char {
    let string = match value(l, i) {
        Some(Value::String(s)) => Some(s),
        Some(v @ (Value::Integer(_) | Value::Number(_))) => {
            let s = rt(l).format_value(&v).ok();
            if let Some(s) = &s {
                replace_value(l, i, Value::String(s.clone()));
            }
            s
        }
        _ => None,
    };
    if !len.is_null() {
        *len = string.as_ref().map_or(0, |s| s.as_bytes().len());
    }
    string.map_or(ptr::null(), |s| intern(l, s))
}
/// Returns the raw length of the value at `i`.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_objlen(l: *mut lua_State, i: c_int) -> usize {
    if let Some(&Value::Table(t)) = peek(l, i) {
        return rt(l).table_length(t).unwrap_or(0.0) as usize;
    }
    match value(l, i) {
        Some(Value::Userdata(h)) => shared(l).userdata.get(&h).map_or(0, |u| u.size),
        Some(v) => rt(l).raw_length(&v).unwrap_or(0.0) as usize,
        None => 0,
    }
}
/// Returns the memory block of the userdata at `i`.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
/// The returned pointer is valid only while the userdata lives and its storage record is not freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_touserdata(l: *mut lua_State, i: c_int) -> *mut c_void {
    let handle = match peek(l, i) {
        Some(Value::Userdata(h)) => Some(*h),
        Some(_) => None,
        None => match value_slow(l, i) {
            Some(Value::Userdata(h)) => Some(h),
            _ => None,
        },
    };
    if let Some(h) = handle {
        // A handle is a full userdata or a light one, never both, and
        // hosts ask for full ones far more often: that map is read first.
        if let Some(u) = shared(l).userdata.get_mut(&h) {
            return u.storage.pointer();
        }
        return shared(l)
            .light_reverse
            .get(&h)
            .map_or(ptr::null_mut(), |p| *p as *mut c_void);
    }
    ptr::null_mut()
}
/// Returns an identity pointer for the value at `i`, or null.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
/// Only userdata results may be dereferenced, under `lua_touserdata`'s rules; the other identities are for comparison alone.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_topointer(l: *mut lua_State, i: c_int) -> *const c_void {
    match value(l, i) {
        Some(Value::Userdata(_)) => lua_touserdata(l, i),
        Some(Value::Table(h)) => {
            ((h.native_identity() as usize)
                .wrapping_mul(16)
                .wrapping_add(1)) as *const c_void
        }
        Some(Value::Closure(h)) => {
            ((h.native_identity() as usize)
                .wrapping_mul(16)
                .wrapping_add(2)) as *const c_void
        }
        Some(Value::NativeFunction(h)) => {
            ((h.index() as usize).wrapping_mul(16).wrapping_add(3)) as *const c_void
        }
        Some(Value::Thread(thread)) => thread_state(l, thread).cast(),
        _ => ptr::null(),
    }
}
/// Pushes nil.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushnil(l: *mut lua_State) {
    push(l, Value::Nil);
}
/// Pushes a floating-point number.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushnumber(l: *mut lua_State, n: f64) {
    let Some(slot) = spare_slot(l) else {
        return push_cold(l, Value::Number(n));
    };
    slot.write(Value::Number(n));
    commit_slot(l);
}
/// Pushes an integer under the state's number model.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushinteger(l: *mut lua_State, n: LuaInteger) {
    let Some(slot) = spare_slot(l) else {
        return pushinteger_cold(l, n);
    };
    // The tag and the payload are written as the two words they are: a
    // value chosen first is staged on the stack and copied.
    let tag = (*l).integer_tag;
    let payload = if tag == NUMBER_TAG {
        (n as i64 as f64).to_bits()
    } else {
        n as i64 as u64
    };
    let slot = slot.cast::<u8>();
    slot.write(tag);
    slot.add(8).cast::<u64>().write(payload);
    commit_slot(l);
}
/// The tag byte of an integer and of a number on the stack (`Value` is
/// `repr(C, u8)`, so its first byte is the tag).
const INTEGER_TAG: u8 = value_tag(&Value::Integer(0));
const NUMBER_TAG: u8 = value_tag(&Value::Number(0.0));
const fn value_tag(value: &Value) -> u8 {
    // SAFETY: the first byte of a `repr(C, u8)` enum is its tag.
    unsafe { *ptr::from_ref(value).cast::<u8>() }
}
/// [`lua_State::integer_tag`] for a runtime with or without `tagged_integers`.
const fn integer_tag(tagged_integers: bool) -> u8 {
    if tagged_integers {
        INTEGER_TAG
    } else {
        NUMBER_TAG
    }
}
/// [`lua_pushinteger`] onto a full stack.
/// `extern "C"` so the fast path tail-jumps here: see [`rawseti_slow`].
///
/// # Safety
///
/// As for [`lua_pushinteger`].
#[cold]
#[inline(never)]
unsafe extern "C" fn pushinteger_cold(l: *mut lua_State, n: LuaInteger) {
    let value = if (*l).integer_tag == INTEGER_TAG {
        Value::Integer(n as i64)
    } else {
        Value::Number(n as i64 as f64)
    };
    push(l, value);
}
/// Pushes true when `n` is nonzero.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushboolean(l: *mut lua_State, n: c_int) {
    let Some(slot) = spare_slot(l) else {
        return push_cold(l, Value::Boolean(n != 0));
    };
    slot.write(Value::Boolean(n != 0));
    commit_slot(l);
}
/// Pushes a counted string copied from `s`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// `s` must be valid for reads of `n` bytes, and may dangle only when `n` is zero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushlstring(
    l: *mut lua_State,
    s: *const c_char,
    n: usize,
) -> PushStringReturn {
    push(l, Value::String(LuaString::from(bytes(s, n))));
    string_push_return(l)
}
/// Pushes a copy of the NUL-terminated string `s`, or nil when `s` is null.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `s` must be null or a live NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushstring(l: *mut lua_State, s: *const c_char) -> PushStringReturn {
    if s.is_null() {
        lua_pushnil(l);
    } else if let Some(key) = cached_key(l, s) {
        let string = key.clone();
        push(l, Value::String(string));
    } else {
        push(l, Value::String(LuaString::from(cstr(s))));
    }
    string_push_return(l)
}
/// Computes the value a string push returns for the compiled ABI.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose top holds the string just pushed.
unsafe fn string_push_return(l: *mut lua_State) -> PushStringReturn {
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    {
        lua_tolstring(l, -1, ptr::null_mut())
    }
    #[cfg(not(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    )))]
    {
        let _ = l;
    }
}
/// Pushes a light userdata token for the pointer `p`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// `p` is an opaque address the VM never dereferences; equal pointers share one token, so its meaning must stay consistent.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushlightuserdata(l: *mut lua_State, p: *mut c_void) {
    let addr = p as usize;
    let h = match shared(l).light.get(&addr).copied() {
        Some(h) => h,
        None => {
            let h = rt(l).create_userdata().expect("userdata allocation");
            rt(l).set_userdata_light(h).expect("light userdata");
            shared(l).light.insert(addr, h);
            shared(l).light_reverse.insert(h, addr);
            h
        }
    };
    push(l, Value::Userdata(h));
}
/// Allocates a full userdata of `size` bytes and pushes it.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// The returned block belongs to the new userdata and is valid until it is collected.
pub unsafe extern "C" fn lua_newuserdata(l: *mut lua_State, size: usize) -> *mut c_void {
    new_userdata(l, size, usize::from(MODERN))
}
/// Allocates a full userdata with `count` uservalues and pushes it.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `count` must be non-negative and below `i16::MAX` or the assert fires.
/// The returned block is valid until the userdata is collected.
pub unsafe extern "C" fn lua_newuserdatauv(
    l: *mut lua_State,
    size: usize,
    count: c_int,
) -> *mut c_void {
    assert!(
        count >= 0 && count < i16::MAX as c_int,
        "invalid userdata uservalue count"
    );
    new_userdata(l, size, count as usize)
}
/// Allocates a userdata of `size` bytes with `count` uservalues and pushes it.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// The returned pointer is null only when allocation fails, and otherwise belongs to the new userdata.
unsafe fn new_userdata(l: *mut lua_State, size: usize, count: usize) -> *mut c_void {
    let mut pointer = ptr::null_mut();
    let _ = oxi_newuserdata(l, size, count as c_int, &mut pointer);
    pointer
}
/// Allocates full userdata storage through the state's allocator and pushes the userdata.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `result` must be writable.
/// `count` outside `0..i16::MAX` returns an error status instead of allocating.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_newuserdata(
    l: *mut lua_State,
    size: usize,
    count: c_int,
    result: *mut *mut c_void,
) -> c_int {
    if count < 0 || count >= i16::MAX as c_int {
        return failure(l, "invalid userdata uservalue count");
    }
    let storage = if let Some(allocator) = shared(l).allocator {
        let mut pointer = ptr::null_mut();
        let data = shared(l).allocator_data;
        let allocation = size.max(1);
        let status = oximond_allocate(
            l,
            allocator,
            data,
            ptr::null_mut(),
            7,
            allocation,
            &mut pointer,
        );
        if status != 0 || pointer.is_null() {
            push(l, Value::from("not enough memory"));
            return 4;
        }
        UserdataStorage::Host {
            pointer,
            size: allocation,
        }
    } else {
        UserdataStorage::Owned(vec![0u128; size.div_ceil(16).max(1)].into_boxed_slice())
    };
    let h = rt(l).create_userdata().expect("userdata allocation");
    rt(l)
        .set_userdata_external_size(h, size)
        .expect("userdata accounting");
    rt(l)
        .set_userdata_uservalue_count(h, if MODERN { count as usize } else { 0 })
        .expect("userdata uservalues");
    if !MODERN {
        let environment = environment(l);
        rt(l)
            .set_userdata_environment(h, environment)
            .expect("userdata environment");
    }
    let mut record = Userdata { storage, size };
    *result = record.storage.pointer();
    shared(l).userdata.insert(h, record);
    push(l, Value::Userdata(h));
    0
}

/// Pushes the `n`th uservalue of the userdata at `index` and returns its type.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// `index` must address a userdata, or for `n == 1` a file handle in the profiles that support it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getiuservalue(l: *mut lua_State, index: c_int, n: c_int) -> c_int {
    let slot = match value(l, index) {
        Some(Value::Userdata(handle))
            if n > 0 && !shared(l).light_reverse.contains_key(&handle) =>
        {
            rt(l).userdata_uservalue(handle, n as usize).ok().flatten()
        }
        Some(Value::File(handle)) if n == 1 && cfg!(any(feature = "lua52", feature = "lua53")) => {
            rt(l).file_uservalue(handle).ok()
        }
        _ => None,
    };
    if let Some(value) = slot {
        push(l, value);
        lua_type(l, -1)
    } else {
        push(l, Value::Nil);
        -1
    }
}
/// Pops a value into the `n`th uservalue of the userdata at `index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value on its stack.
/// `index` must address a userdata, or for `n == 1` a file handle in the profiles that support it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_setiuservalue(l: *mut lua_State, index: c_int, n: c_int) -> c_int {
    let handle = match value(l, index) {
        Some(Value::Userdata(handle)) if !shared(l).light_reverse.contains_key(&handle) => {
            Some(handle)
        }
        _ => None,
    };
    let file = match value(l, index) {
        Some(Value::File(handle)) if n == 1 && cfg!(any(feature = "lua52", feature = "lua53")) => {
            Some(handle)
        }
        _ => None,
    };
    let value = pop(l);
    if let Some(handle) = handle.filter(|_| n > 0) {
        i32::from(
            rt(l)
                .set_userdata_uservalue(handle, n as usize, value)
                .unwrap_or(false),
        )
    } else if let Some(file) = file {
        i32::from(rt(l).set_file_uservalue(file, value).is_ok())
    } else {
        0
    }
}
/// Pushes the first uservalue of the userdata at `index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `index` must satisfy `lua_getiuservalue`'s requirements for `n == 1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getuservalue(l: *mut lua_State, index: c_int) -> GetReturn {
    lua_getiuservalue(l, index, 1);
    get_return(l)
}
/// Pops a value into the first uservalue of the userdata at `index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value, and `index` must satisfy `lua_setiuservalue`'s requirements for `n == 1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_setuservalue(l: *mut lua_State, index: c_int) {
    lua_setiuservalue(l, index, 1);
}
/// Pops `n` upvalues and pushes a C closure over `function`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least `n` values and room for one more.
/// `function` must uphold the C callback contract and stay callable from the VM for the closure's whole life.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushcclosure(l: *mut lua_State, function: CFunction, n: c_int) {
    let len = (*l).stack.len();
    let count = n.max(0) as usize;
    assert!(count <= len, "invalid C closure upvalue count");
    let captures = state(l).stack.split_off(len - count);
    let upvalues = rt(l).create_table().expect("upvalue table");
    for (i, v) in captures.iter().enumerate() {
        let _ = rt(l).table_set(upvalues, &Value::Number((i + 1) as f64), v.clone());
    }
    let mut closure = Box::new(CClosure {
        state: main_state(l),
        function,
        upvalues,
        count,
        index: shared(l).closures.len(),
        value: Value::Nil,
        environment: environment(l),
    });
    let token = ptr::from_ref(closure.as_ref()) as u64;
    let _ = rt(l).table_set(
        upvalues,
        &Value::from("\0oximond.environment"),
        Value::Table(closure.environment),
    );
    let owner = shared(l)
        .owner
        .upgrade()
        .map(|owner| owner as Rc<dyn std::any::Any>);
    let f = vm(l)
        .register_owned_host_function(host_call, token, vec![Value::Table(upvalues)], owner)
        .expect("C callback registration");
    // SAFETY: `host_call_direct` is `host_call` with the results stored for
    // the caller, and keeps the contract of a direct callback.
    unsafe { vm(l).set_host_direct_callback(&f, host_call_direct) };
    closure.value = f.clone();
    shared(l).closures.push(closure);
    push(l, f);
}
/// Returns the C function behind the closure at `i`, when this API created it.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
/// The result is only valid while the closure stays registered in `l`'s family.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_tocfunction(l: *mut lua_State, i: c_int) -> Option<CFunction> {
    let target = value(l, i)?;
    shared(l)
        .closures
        .iter()
        .find(|c| Runtime::raw_equal(&c.value, &target))
        .map(|c| c.function)
}

/// Pushes a new empty table.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_createtable(l: *mut lua_State, array: c_int, _records: c_int) {
    // The hint only reserves the array part's capacity, which stays empty until stored to.
    // Cap it so a hostile or mistaken hint cannot reserve gigabytes up front.
    let hint = usize::try_from(array).unwrap_or(0).min(MAX_TABLE_HINT);
    let table = rt(l).create_table_sized(hint).expect("table allocation");
    push(l, Value::Table(table));
}
/// Largest array size `lua_createtable` reserves up front (16 Mi values).
const MAX_TABLE_HINT: usize = 1 << 24;
/// Returns the table handle of the value at `i`, if it is a table.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
unsafe fn raw_table(l: *mut lua_State, i: c_int) -> Option<TableHandle> {
    if let Some(&Value::Table(h)) = peek(l, i) {
        return Some(h);
    }
    match value(l, i) {
        Some(Value::Table(h)) => Some(h),
        _ => None,
    }
}
/// Pops a key and pushes the raw table value it maps to.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value and room for one more.
/// `i` must address a table; anything else pushes nil.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_rawget(l: *mut lua_State, i: c_int) -> GetReturn {
    let t = raw_table(l, i);
    let k = pop(l);
    let v = t
        .and_then(|t| rt(l).table_get(t, &k).ok())
        .unwrap_or(Value::Nil);
    push(l, v);
    get_return(l)
}
/// Pushes the raw table value at integer key `n`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `i` must address a table or nil is pushed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_rawgeti(l: *mut lua_State, i: c_int, n: RawIndex) -> GetReturn {
    // A host copying an array out (Luanti's `VoxelManip:set_data` reads every
    // node this way) hits a number in the array part of a table on the stack:
    // that path is a few loads and a push, with everything else out of line so
    // this leaf saves no registers. The number is rebuilt field by field;
    // cloning the slot moves the whole 16-byte Value with one wide load, which
    // stalls when the slot was just written with narrow stores.
    if let Some(table) = peek(l, i).and_then(Value::table_handle_word)
        && let Some(slot) = spare_slot(l)
        // SAFETY: the table is on a stack of this state, which the collector
        // roots (`refresh_host_roots`), so its handle names a live table.
        && let Some(value) = rt(l).table_array_slot_live(table, n as i64)
        && matches!(value, Value::Integer(_) | Value::Number(_))
    {
        // SAFETY: the value is a number, all of it in the two words copied.
        copy_number_words(slot, value);
        commit_slot(l);
        return get_return(l);
    }
    rawgeti_slow(l, i, n)
}
/// [`lua_rawgeti`] for anything but a number in the array part of a table on the stack.
/// `extern "C"` so the fast path tail-jumps here: see [`rawseti_slow`].
///
/// # Safety
///
/// As for [`lua_rawgeti`].
#[cold]
#[inline(never)]
unsafe extern "C" fn rawgeti_slow(l: *mut lua_State, i: c_int, n: RawIndex) -> GetReturn {
    let table = raw_table(l, i);
    let v = match table.and_then(|t| rt(l).table_array_slot(t, n as i64)) {
        Some(v) => v.clone(),
        None => table
            .and_then(|t| rt(l).table_get_integer(t, n as i64).ok())
            .unwrap_or(Value::Nil),
    };
    push(l, v);
    get_return(l)
}
/// Computes the return value the raw-get ABI expects for the pushed result.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose top holds the value just pushed.
unsafe fn get_return(l: *mut lua_State) -> GetReturn {
    #[cfg(any(feature = "lua53", feature = "lua54", feature = "lua55"))]
    {
        lua_type(l, -1)
    }
    #[cfg(not(any(feature = "lua53", feature = "lua54", feature = "lua55")))]
    {
        let _ = l;
    }
}

/// Converts `index` to an equivalent absolute stack index.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose top index fits `c_int`.
/// Pseudo-indices are returned unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_absindex(l: *mut lua_State, index: c_int) -> c_int {
    if index > 0 || index <= REGISTRY {
        index
    } else {
        lua_gettop(l).saturating_add(index).saturating_add(1)
    }
}
/// Copies the value at `from` into the slot `to`.
///
/// # Safety
///
/// `l` must be a live `lua_State`, `from` must name one of its values, and `to` must be a destination the write accepts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_copy(l: *mut lua_State, from: c_int, to: c_int) {
    replace_value(l, to, value(l, from).unwrap_or(Value::Nil));
}
/// Rotates the values from `index` to the top rightward by `count`.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `index` must name a stack slot.
/// Nothing may borrow the stack across the rotation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_rotate(l: *mut lua_State, index: c_int, count: c_int) {
    if let Some(index) = slot(l, index) {
        let length = (*l).stack.len() - index;
        if length != 0 {
            state(l).stack[index..]
                .rotate_right(i64::from(count).rem_euclid(length as i64) as usize);
        }
    }
}
/// Converts the value at `index` to a number, reporting success through `isnum`.
///
/// # Safety
///
/// `l` must be a live `lua_State`, `index` one of its stack indices, and `isnum` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_tonumberx(l: *mut lua_State, index: c_int, isnum: *mut c_int) -> f64 {
    let number = match peek(l, index) {
        Some(&Value::Number(n)) => Some(n),
        _ => value(l, index).and_then(|value| rt(l).to_number(&value)),
    };
    if !isnum.is_null() {
        *isnum = i32::from(number.is_some());
    }
    number.unwrap_or(0.0)
}
/// Converts the value at `index` to an integer, reporting success through `isnum`.
///
/// # Safety
///
/// `l` must be a live `lua_State`, `index` one of its stack indices, and `isnum` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_tointegerx(
    l: *mut lua_State,
    index: c_int,
    isnum: *mut c_int,
) -> LuaInteger {
    // Integers and integral floats in range convert without cloning the slot.
    match peek(l, index) {
        Some(&Value::Integer(integer)) => {
            if !isnum.is_null() {
                *isnum = 1;
            }
            return integer as LuaInteger;
        }
        Some(&Value::Number(n))
            if n.fract() == 0.0 && n >= i64::MIN as f64 && n < -(i64::MIN as f64) =>
        {
            if !isnum.is_null() {
                *isnum = 1;
            }
            return n as LuaInteger;
        }
        _ => {}
    }
    let number = match value(l, index) {
        Some(Value::Integer(integer)) => Some(integer as LuaInteger),
        Some(Value::String(text))
            if MODERN_INTEGER
                && oximond::runtime::parse_lua_integer_exact(text.as_bytes()).is_some() =>
        {
            oximond::runtime::parse_lua_integer_exact(text.as_bytes()).map(|n| n as LuaInteger)
        }
        Some(value) => rt(l).to_number(&value).and_then(|number| {
            if !number.is_finite()
                || (MODERN_INTEGER
                    && (number.fract() != 0.0
                        || number < i64::MIN as f64
                        || number >= -(i64::MIN as f64)))
            {
                None
            } else {
                Some(number as LuaInteger)
            }
        }),
        None => None,
    };
    if !isnum.is_null() {
        *isnum = i32::from(number.is_some());
    }
    number.unwrap_or(0)
}
/// Reports whether the value at `index` is an integer.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `index` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_isinteger(l: *mut lua_State, index: c_int) -> c_int {
    i32::from(matches!(value(l, index), Some(Value::Integer(_))))
}
/// Converts the value at `index` to an unsigned 32-bit number, reporting success through `isnum`.
///
/// # Safety
///
/// `l` must be a live `lua_State`, `index` one of its stack indices, and `isnum` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_tounsignedx(
    l: *mut lua_State,
    index: c_int,
    isnum: *mut c_int,
) -> u32 {
    let number = lua_tonumberx(l, index, isnum);
    if number.is_finite() {
        number.trunc().rem_euclid(4_294_967_296.0) as u32
    } else {
        0
    }
}
/// Pushes an unsigned 32-bit number.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushunsigned(l: *mut lua_State, number: u32) {
    push(l, Value::Number(f64::from(number)));
}
/// Returns the raw length of the value at `index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `index` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_rawlen(l: *mut lua_State, index: c_int) -> RawLength {
    lua_objlen(l, index) as RawLength
}
/// Pushes the raw table value keyed by the light userdata `pointer`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// `pointer` is an opaque key the VM never dereferences.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_rawgetp(
    l: *mut lua_State,
    index: c_int,
    pointer: *const c_void,
) -> GetReturn {
    let index = lua_absindex(l, index);
    lua_pushlightuserdata(l, pointer.cast_mut());
    lua_rawget(l, index)
}
/// Pops a value and stores it at the light userdata key `pointer`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least two values and a table at `index`.
/// `pointer` is an opaque key the VM never dereferences.
#[unsafe(export_name = "oxi_rawsetp")]
pub unsafe extern "C" fn lua_rawsetp(
    l: *mut lua_State,
    index: c_int,
    pointer: *const c_void,
) -> c_int {
    let index = lua_absindex(l, index);
    lua_pushlightuserdata(l, pointer.cast_mut());
    lua_insert(l, -2);
    lua_rawset(l, index)
}
/// Parses `source` as a Lua numeral, pushes it, and returns the bytes consumed.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `source` must be a live NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_stringtonumber(l: *mut lua_State, source: *const c_char) -> usize {
    let bytes = cstr(source);
    if let Some(number) = rt(l).to_number(&Value::bytes(bytes)) {
        let result = if MODERN_INTEGER {
            oximond::runtime::parse_lua_integer_exact(bytes)
                .map_or(Value::Number(number), Value::Integer)
        } else {
            Value::Number(number)
        };
        push(l, result);
        bytes.len() + 1
    } else {
        0
    }
}

/// Returns a static pointer to the compiled Lua version number.
///
/// # Safety
///
/// The state argument is ignored and may be null; the result is static and valid for the life of the process.
#[unsafe(no_mangle)]
#[cfg(any(feature = "lua52", feature = "lua53"))]
pub unsafe extern "C" fn lua_version(_: *mut lua_State) -> *const f64 {
    static VERSION: f64 = if cfg!(feature = "lua52") {
        502.0
    } else {
        503.0
    };
    &VERSION
}
/// Returns the compiled Lua version number.
///
/// # Safety
///
/// The state argument is ignored and may be null, so the call is safe on any thread.
#[unsafe(no_mangle)]
#[cfg(any(feature = "lua54", feature = "lua55"))]
pub unsafe extern "C" fn lua_version(_: *mut lua_State) -> f64 {
    if cfg!(feature = "lua54") {
        504.0
    } else {
        505.0
    }
}
/// Pops a value and a key and stores them raw in the table at `i`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least two values.
/// `i` must address a table, or the call pushes `table expected` and fails instead of raising.
#[unsafe(export_name = "oxi_rawset")]
pub unsafe extern "C" fn lua_rawset(l: *mut lua_State, i: c_int) -> c_int {
    let t = raw_table(l, i);
    let v = pop(l);
    let k = pop(l);
    if let Some(t) = t {
        if let Err(error) = rt(l).table_set(t, &k, v) {
            return failure(l, error);
        }
        return 0;
    }
    failure(l, "table expected")
}
/// Pops a value and stores it at integer key `n` of the table at `i`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value, and `i` must address a table or the call fails with an error status.
#[unsafe(export_name = "oxi_rawseti")]
pub unsafe extern "C" fn lua_rawseti(l: *mut lua_State, i: c_int, n: RawIndex) -> c_int {
    if rawseti_fast(l, i, n) {
        return 0;
    }
    rawseti_slow(l, i, n)
}
/// The C `lua_rawseti`, defined here rather than as a C++ wrapper around
/// `oxi_rawseti` in bridge.cpp, so the common store costs no extra call.
/// Anything the fast path declines goes to bridge.cpp's `oxi_rawseti_raise`,
/// which runs [`lua_rawseti`] and throws its error. That call is the last
/// thing this frame does and the frame owns nothing to drop, so the exception
/// passes it without a landing pad. The lazy bridge, a shared object loaded
/// later, keeps its C++ `lua_rawseti`, since this executable cannot name a
/// function inside it.
///
/// # Safety
///
/// As for [`lua_rawseti`].
#[cfg(not(feature = "lazy-bridge"))]
#[unsafe(export_name = "lua_rawseti")]
pub unsafe extern "C-unwind" fn lua_rawseti_raising(l: *mut lua_State, i: c_int, n: RawIndex) {
    if rawseti_fast(l, i, n) {
        return;
    }
    oxi_rawseti_raise(l, i, n)
}
#[cfg(not(feature = "lazy-bridge"))]
unsafe extern "C-unwind" {
    fn oxi_rawseti_raise(l: *mut lua_State, i: c_int, n: RawIndex);
}
/// Stores the number on top into the array part of the table at `i` and pops
/// it, as a host filling an array does (Luanti's `VoxelManip:get_data` stores
/// every node this way): an overwrite or an append, with the pop a length
/// decrement since a scalar owns nothing. False, with nothing changed, for
/// any other store.
///
/// # Safety
///
/// As for [`lua_rawseti`].
#[inline(always)]
unsafe fn rawseti_fast(l: *mut lua_State, i: c_int, n: RawIndex) -> bool {
    // The handle is loaded as the one word it is, so the stack slot that
    // holds it is dead after this line.
    let Some(table) = peek(l, i).and_then(Value::table_handle_word) else {
        return false;
    };
    let stack = &mut state(l).stack;
    // `peek` found a value, so the stack is not empty.
    let top = &*stack.as_ptr().add(stack.len() - 1);
    // SAFETY: the table is on a stack of this state, which the collector
    // roots (`refresh_host_roots`), so its handle names a live table.
    if !rt(l).table_raw_store_scalar_live(table, n as i64, top) {
        return false;
    }
    // SAFETY: the store took only a scalar, which owns nothing to drop.
    stack.set_len(stack.len() - 1);
    true
}
/// [`lua_rawseti`] for a store its fast path declines.
///
/// `extern "C"`, so it cannot unwind: a call to it from another `extern "C"`
/// function needs no unwind guard and compiles to a tail jump, which spares
/// the fast path the registers a call would save.
///
/// # Safety
///
/// As for [`lua_rawseti`].
#[inline(never)]
unsafe extern "C" fn rawseti_slow(l: *mut lua_State, i: c_int, n: RawIndex) -> c_int {
    let t = raw_table(l, i);
    // An append, as a host filling a fresh array makes.
    if let Some(table) = t {
        let stack = &mut state(l).stack;
        if let Some(top) = stack.last()
            && matches!(
                top,
                Value::Number(_) | Value::Integer(_) | Value::Boolean(_)
            )
            && rt(l).table_raw_set_integer_scalar(table, n as i64, top)
        {
            // SAFETY: the top is a scalar, which owns nothing to drop.
            stack.set_len(stack.len() - 1);
            return 0;
        }
    }
    // Numbers leave the stack field by field, for the reason given in `lua_rawgeti`.
    let v = match pop_number(l) {
        Some(Value::Number(x)) => {
            if let Some(t) = t {
                return match rt(l).table_set_integer_number(t, n as i64, x) {
                    Ok(()) => 0,
                    Err(error) => failure(l, error),
                };
            }
            Value::Number(x)
        }
        Some(v) => v,
        None => pop(l),
    };
    if let Some(t) = t {
        if let Err(error) = rt(l).table_set_integer(t, n as i64, v) {
            return failure(l, error);
        }
        return 0;
    }
    failure(l, "table expected")
}
/// Reports raw equality of the values at `a` and `b`.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `a` and `b` must be two of its stack or pseudo-indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_rawequal(l: *mut lua_State, a: c_int, b: c_int) -> c_int {
    match (value(l, a), value(l, b)) {
        (Some(a), Some(b)) => i32::from(Runtime::raw_equal(&a, &b)),
        _ => 0,
    }
}
/// Pops a key and pushes the next key-value pair of the table at `i`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value and room for two more.
/// `i` must address a table, and the popped key must be one this table's iteration accepts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_next(l: *mut lua_State, i: c_int) -> c_int {
    let t = raw_table(l, i);
    let k = pop(l);
    // The table's traversal cursor answers a walk that steps from the key
    // it gave last without a lookup, as for `next`; the full step settles
    // whatever it declines (a deleted key, a young table) and its errors.
    if let Some(t) = t
        && let Ok((k, v)) = rt(l)
            .table_next_fast(t, &k)
            .map_or_else(|| rt(l).table_next(t, &k), Ok)
        && !matches!(k, Value::Nil)
    {
        push(l, k);
        push(l, v);
        return 1;
    }
    0
}
/// Pushes the raw metatable of the value at `i`, when it has one.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `i` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getmetatable(l: *mut lua_State, i: c_int) -> c_int {
    if let Some(v) = value(l, i)
        && let Ok(Some(t)) = rt(l).raw_metatable(&v)
    {
        push(l, Value::Table(t));
        return 1;
    }
    0
}
/// Pops a table and installs it as the metatable of the value at `i`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value, whose top is a table or nil.
/// `i` must address a valid value or the call fails with an error status.
#[unsafe(export_name = "oxi_setmetatable")]
pub unsafe extern "C" fn lua_setmetatable(l: *mut lua_State, i: c_int) -> c_int {
    let v = value(l, i);
    let m = pop(l);
    let t = match m {
        Value::Table(t) => Some(t),
        _ => None,
    };
    match v {
        Some(value) => match rt(l).raw_set_metatable(&value, t) {
            Ok(()) => 0,
            Err(error) => failure(l, error),
        },
        None => failure(l, "invalid stack index"),
    }
}
/// Pushes the environment of the value at `i`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `i` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getfenv(l: *mut lua_State, i: c_int) {
    let target = value(l, i);
    let env = match target.as_ref() {
        Some(Value::Closure(h)) => rt(l).closure_environment(*h).ok(),
        Some(Value::Userdata(h)) => rt(l).userdata_environment(*h).ok().flatten(),
        Some(Value::Thread(thread)) => {
            let thread = (*thread != shared(l).thread).then_some(*thread);
            vm(l).thread_environment_for(thread).ok()
        }
        Some(v) => shared(l)
            .closures
            .iter()
            .find(|c| Runtime::raw_equal(&c.value, v))
            .map(|c| c.environment),
        None => None,
    }
    .unwrap_or_else(|| thread_environment(l));
    push(l, Value::Table(env));
}
/// Pops a table and installs it as the environment of the value at `i`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value whose top is a table.
/// A non-table top makes the call return zero after the pop.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_setfenv(l: *mut lua_State, i: c_int) -> c_int {
    let v = value(l, i);
    let Value::Table(t) = pop(l) else { return 0 };
    match v {
        Some(Value::Closure(h)) => i32::from(rt(l).set_closure_environment(h, t).is_ok()),
        Some(Value::Userdata(h)) => i32::from(rt(l).set_userdata_environment(h, t).is_ok()),
        Some(Value::Thread(h)) => {
            let thread = (h != shared(l).thread).then_some(h);
            i32::from(vm(l).set_thread_environment_for(thread, t).is_ok())
        }
        Some(v) => {
            if let Some(index) = shared(l)
                .closures
                .iter()
                .position(|c| Runtime::raw_equal(&c.value, &v))
            {
                set_c_environment(l, index, t);
                1
            } else {
                0
            }
        }
        None => 0,
    }
}
/// Calls a value through the VM after refreshing the family's GC roots.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the family `vm` belongs to, with no other outstanding borrow.
/// The call may run Lua code and re-enter C callbacks on `l`.
unsafe fn call(
    l: *mut lua_State,
    function: Value,
    args: Vec<Value>,
) -> Result<Vec<Value>, VmError> {
    sync_roots(l);
    vm(l).call_value(function, args)
}

/// Calls a message handler for an error, optionally without debug hooks.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `handler` a callable value of its VM.
/// The call may run arbitrary Lua code and re-enter C callbacks.
unsafe fn call_message_handler(
    l: *mut lua_State,
    handler: Value,
    error: Value,
    hook_error: bool,
) -> Result<Vec<Value>, VmError> {
    sync_roots(l);
    if hook_error {
        vm(l).call_value_without_hooks(handler, vec![error])
    } else {
        vm(l).call_value(handler, vec![error])
    }
}
/// Calls the function and `nargs` arguments on the stack, reporting failure as a status.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least `nargs + 1` values.
/// The call may run Lua code and re-enter C callbacks before returning.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_call(l: *mut lua_State, nargs: c_int, nresults: c_int) -> c_int {
    oxi_pcall(l, nargs, nresults, 0)
}
/// Calls the function below `nargs` arguments with message handler `errfunc`, returning a status.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least `nargs + 1` values on its stack.
/// The call may run Lua code and invoke `errfunc` and C closures, which must uphold their contracts while `l` is reborrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_pcall(
    l: *mut lua_State,
    nargs: c_int,
    nresults: c_int,
    errfunc: c_int,
) -> c_int {
    let handler = if errfunc != 0 {
        value(l, errfunc)
    } else {
        None
    };
    let len = (*l).stack.len();
    if nargs < 0 || len < nargs as usize + 1 {
        return failure(l, "invalid argument count");
    }
    let start = len - nargs as usize - 1;
    let mut arguments = state(l).stack.split_off(start);
    let function = arguments.remove(0);
    match call(l, function, arguments) {
        Ok(mut results) => {
            if nresults >= 0 {
                results.resize(nresults as usize, Value::Nil);
            }
            state(l).stack.extend(results);
            0
        }
        Err(error) => {
            // `luaD_throw` raises a memory error and `GCTM`'s wrapped error
            // without calling the message handler.
            let mut code = failure_status(&error);
            let hook_error = error.is_hook_error();
            let mut e = error_value(l, error);
            if let Some(handler) = handler.filter(|_| code == 2) {
                match call_message_handler(l, handler, e.clone(), hook_error) {
                    Ok(r) => e = r.first().cloned().unwrap_or(Value::Nil),
                    Err(_) => {
                        e = Value::from("error in error handling");
                        code = ERROR_HANDLER
                    }
                }
            }
            push(l, e);
            code
        }
    }
}
/// Reads `key` from `object`, following `__index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the family `vm` belongs to.
/// Metamethods may run Lua code and re-enter C callbacks on `l`.
/// Converts `index` to an absolute stack index, leaving pseudo-indices alone.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
unsafe fn absolute(l: *mut lua_State, index: c_int) -> c_int {
    if index > 0 || index <= REGISTRY {
        index
    } else {
        (*l).stack.len() as c_int + index + 1
    }
}
/// Pushes the value of the field `key` of the value at `index`, following `__index`:
/// the body of `lua_getfield`. A null `key` reads the field nil.
///
/// # Safety
///
/// `l` must be a live `lua_State`, `key` null or a live NUL-terminated string.
/// The lookup may run Lua metamethods and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_getfield(
    l: *mut lua_State,
    index: c_int,
    key: *const c_char,
) -> c_int {
    if let Some(&Value::Table(table)) = peek(l, index)
        && !key.is_null()
        && let Some(name) = cached_key(l, key)
        && let Some(found) = rt(l).capi_get_string(table, name)
    {
        let found = found.clone();
        push(l, found);
        return 0;
    }
    getfield_slow(l, index, key)
}
/// The general case of [`oxi_getfield`]: push the key, then index like `lua_gettable`.
///
/// # Safety
///
/// As for [`oxi_getfield`].
#[inline(never)]
unsafe fn getfield_slow(l: *mut lua_State, index: c_int, key: *const c_char) -> c_int {
    let index = absolute(l, index);
    lua_pushstring(l, key);
    oxi_gettable(l, index)
}
/// Pops a value into the field `key` of the value at `index`, following `__newindex`:
/// the body of `lua_setfield`. A null `key` stores under nil.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value, `key` null or a live NUL-terminated string.
/// The store may run Lua metamethods and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_setfield(
    l: *mut lua_State,
    index: c_int,
    key: *const c_char,
) -> c_int {
    if let Some(&Value::Table(table)) = peek(l, index)
        && !key.is_null()
        && let Some(name) = cached_key(l, key)
        && let Some(top) = peek(l, -1)
    {
        if matches!(
            top,
            Value::Number(_) | Value::Integer(_) | Value::Boolean(_)
        ) && rt(l).capi_overwrite_scalar(table, name, top)
        {
            // SAFETY: the top is a scalar, which owns nothing to drop.
            let stack = &mut state(l).stack;
            stack.set_len(stack.len() - 1);
            return 0;
        }
        let name = name.clone();
        let stored = pop(l);
        match rt(l).capi_set_string(table, &name, stored) {
            StringSet::Done => return 0,
            StringSet::Failed(error) => return error_status(l, error.into()),
            StringSet::Declined(stored) => push(l, stored),
        }
    }
    setfield_slow(l, index, key)
}
/// The general case of [`oxi_setfield`].
///
/// # Safety
///
/// As for [`oxi_setfield`].
#[inline(never)]
unsafe fn setfield_slow(l: *mut lua_State, index: c_int, key: *const c_char) -> c_int {
    let index = absolute(l, index);
    let object = value(l, index).unwrap_or(Value::Nil);
    let stored = pop(l);
    let name = if key.is_null() {
        Value::Nil
    } else {
        Value::String(
            cached_key(l, key)
                .cloned()
                .unwrap_or_else(|| LuaString::from(cstr(key))),
        )
    };
    match settable(l, object, name, stored) {
        Ok(()) => 0,
        Err(e) => error_status(l, e),
    }
}
unsafe fn gettable(l: *mut lua_State, mut object: Value, key: Value) -> Result<Value, VmError> {
    for _ in 0..100 {
        if let Value::Table(t) = object {
            let v = rt(l).table_get(t, &key)?;
            if !matches!(v, Value::Nil) {
                return Ok(v);
            }
        }
        let meta = metamethod_of(l, &object, Metamethod::Index)?;
        match meta {
            Value::Nil if matches!(object, Value::Table(_)) => return Ok(Value::Nil),
            Value::Nil => {
                return Err(RuntimeError::message("attempt to index non-table value").into());
            }
            Value::Table(_) => object = meta,
            f => {
                return Ok(call(l, f, vec![object, key])?
                    .first()
                    .cloned()
                    .unwrap_or(Value::Nil));
            }
        }
    }
    Err(RuntimeError::message("loop in gettable").into())
}
/// The raw value under `event`'s key in the metatable of `object`, or nil
/// when it has none. It reads with the runtime's own key string and the
/// metatable's cached absence bit, so a lookup that misses in a table with
/// a metatable makes no string and, once the absence is known, no probe.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
unsafe fn metamethod_of(
    l: *mut lua_State,
    object: &Value,
    event: Metamethod,
) -> Result<Value, VmError> {
    let Some(metatable) = rt(l).raw_metatable(object)? else {
        return Ok(Value::Nil);
    };
    if rt(l).metatable_lacks(metatable, event) {
        return Ok(Value::Nil);
    }
    let key = Value::String(rt(l).metamethod_key(event).clone());
    Ok(rt(l).table_get(metatable, &key)?)
}
/// Writes `v` at `key` in `object`, following `__newindex`.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the family `vm` belongs to.
/// Metamethods may run Lua code and re-enter C callbacks on `l`.
unsafe fn settable(
    l: *mut lua_State,
    mut object: Value,
    key: Value,
    v: Value,
) -> Result<(), VmError> {
    for _ in 0..100 {
        if let Value::Table(t) = object
            && !matches!(rt(l).table_get(t, &key)?, Value::Nil)
        {
            rt(l).table_set(t, &key, v)?;
            return Ok(());
        }
        let meta = metamethod_of(l, &object, Metamethod::NewIndex)?;
        match meta {
            Value::Nil => {
                if let Value::Table(t) = object {
                    rt(l).table_set(t, &key, v)?;
                    return Ok(());
                }
                return Err(RuntimeError::message("attempt to index non-table value").into());
            }
            Value::Table(_) => object = meta,
            f => {
                call(l, f, vec![object, key, v])?;
                return Ok(());
            }
        }
    }
    Err(RuntimeError::message("loop in settable").into())
}

/// The state's globals: the registry copy on modern profiles, the thread environment otherwise.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose registry and thread environments are intact.
unsafe fn global_table(l: *mut lua_State) -> Value {
    if MODERN {
        let registry = rt(l).registry();
        rt(l)
            .table_get(registry, &Value::Integer(2))
            .unwrap_or(Value::Nil)
    } else {
        Value::Table(thread_environment(l))
    }
}
/// Pushes the value of the global named `name`, following `__index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `name` a live NUL-terminated string.
/// The lookup may run Lua metamethods and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_getglobal(l: *mut lua_State, name: *const c_char) -> c_int {
    let object = global_table(l);
    match gettable(l, object, Value::bytes(cstr(name))) {
        Ok(value) => {
            push(l, value);
            0
        }
        Err(error) => error_status(l, error),
    }
}
/// Pops a value into the global named `name`, following `__newindex`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value, and `name` a live NUL-terminated string.
/// The store may run Lua metamethods and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_setglobal(l: *mut lua_State, name: *const c_char) -> c_int {
    let object = global_table(l);
    let value = pop(l);
    match settable(l, object, Value::bytes(cstr(name)), value) {
        Ok(()) => 0,
        Err(error) => error_status(l, error),
    }
}
/// Pushes the value at integer key `n` of the value at `index`, following `__index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// The lookup may run Lua metamethods and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_geti(l: *mut lua_State, index: c_int, n: LuaInteger) -> c_int {
    let object = value(l, index).unwrap_or(Value::Nil);
    let key = rt(l).integer_value(n as i64);
    match gettable(l, object, key) {
        Ok(value) => {
            push(l, value);
            0
        }
        Err(error) => error_status(l, error),
    }
}
/// Pops a value into integer key `n` of the value at `index`, following `__newindex`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value.
/// The store may run Lua metamethods and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_seti(l: *mut lua_State, index: c_int, n: LuaInteger) -> c_int {
    let object = value(l, index).unwrap_or(Value::Nil);
    let value = pop(l);
    let key = rt(l).integer_value(n as i64);
    match settable(l, object, key, value) {
        Ok(()) => 0,
        Err(error) => error_status(l, error),
    }
}
/// Runs `expression` as a freshly compiled operator function on `args`.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the family `vm` belongs to, and `expression` must be one of the fixed operator sources this crate compiles.
/// The call may run metamethods and raise Lua errors.
unsafe fn operator_call(
    l: *mut lua_State,
    expression: &str,
    args: Vec<Value>,
) -> Result<Value, VmError> {
    let source = format!("return function(a,b) return {expression} end");
    let chunk = compile_named(
        "=[C]",
        source.as_bytes(),
        CompilerConfig::new(vm(l).compatibility_profile()),
    )
    .map_err(|error| RuntimeError::message(error.to_string()))?;
    let factory = vm(l).load_chunk(&chunk)?;
    let function = call(l, factory, Vec::new())?
        .into_iter()
        .next()
        .ok_or_else(|| RuntimeError::message("operator function missing"))?;
    Ok(call(l, function, args)?
        .into_iter()
        .next()
        .unwrap_or(Value::Nil))
}
/// Applies arithmetic operation `op` to the top one or two stack values.
///
/// # Safety
///
/// `l` must be a live `lua_State` with the operands `op` needs and room for one result.
/// Metamethods may run Lua code and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_arith(l: *mut lua_State, op: c_int) -> c_int {
    let expressions: &[&str] = if cfg!(feature = "lua52") {
        &["a+b", "a-b", "a*b", "a/b", "a%b", "a^b", "-a"]
    } else {
        &[
            "a+b", "a-b", "a*b", "a%b", "a^b", "a/b", "a//b", "a&b", "a|b", "a~b", "a<<b", "a>>b",
            "-a", "~a",
        ]
    };
    let Some(expression) = usize::try_from(op).ok().and_then(|op| expressions.get(op)) else {
        return failure(l, "invalid arithmetic operation");
    };
    let unary = expression.starts_with('-') || expression.starts_with('~');
    let b = pop(l);
    let a = if unary { b.clone() } else { pop(l) };
    match operator_call(l, expression, vec![a, b]) {
        Ok(value) => {
            push(l, value);
            0
        }
        Err(error) => error_status(l, error),
    }
}
/// Pushes the length of the value at `index`, following `__len`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// The operation may run Lua metamethods.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_len(l: *mut lua_State, index: c_int) -> c_int {
    let value = value(l, index).unwrap_or(Value::Nil);
    match operator_call(l, "#a", vec![value]) {
        Ok(value) => {
            push(l, value);
            0
        }
        Err(error) => error_status(l, error),
    }
}
/// Marks or unmarks the table at `index` as read-only.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `index` must address a table, or the call returns an error status.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_enablereadonlytable(
    l: *mut lua_State,
    index: c_int,
    enabled: c_int,
) -> c_int {
    let Some(table) = raw_table(l, index) else {
        return failure(l, "table expected");
    };
    match rt(l).table_set_readonly(table, enabled != 0) {
        Ok(()) => 0,
        Err(error) => failure(l, error),
    }
}
/// Reports whether the table at `index` is read-only.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `index` one of its stack indices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_isreadonlytable(l: *mut lua_State, index: c_int) -> c_int {
    raw_table(l, index)
        .and_then(|table| rt(l).table_is_readonly(table).ok())
        .map_or(0, i32::from)
}
/// Boxes the callback state and registers resume and close handlers for a pending continuation.
///
/// # Safety
///
/// `l` must be a live `lua_State` inside a running C callback, and `function` must be a `KFunction` that stays valid until the parked continuation runs.
/// The registered host functions own the boxed record until then.
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
unsafe fn park_continuation(
    l: *mut lua_State,
    context: KContext,
    function: KFunction,
    child: bool,
    nresults: c_int,
    handler: Option<Value>,
) -> Result<Value, VmError> {
    let mut record = Box::new(Continuation {
        state: l,
        stack: (*l).stack.clone(),
        current: (*l).current,
        close_slots: (*l).close_slots.clone(),
        context,
        function,
        child,
        nresults,
        handler,
        value: None,
        close_handler: None,
    });
    let token = ptr::from_mut(record.as_mut()) as u64;
    let mut captures = record.stack.clone();
    captures.extend(record.handler.iter().cloned());
    if let Some(index) = record.current {
        captures.push(Value::Table(shared(l).closures[index].upvalues));
    }
    let owner = shared(l)
        .owner
        .upgrade()
        .map(|owner| -> Rc<dyn std::any::Any> { owner });
    shared(l).continuations.push(record);
    let value = vm(l).register_owned_host_function(
        resume_native,
        token,
        captures.clone(),
        owner.clone(),
    )?;
    let close = vm(l).register_owned_host_function(close_native, token, captures, owner)?;
    let record = &mut *(token as *mut Continuation);
    record.value = Some(value.clone());
    record.close_handler = Some(close);
    Ok(value)
}

#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
fn resume_native(vm_ref: &mut Vm, token: u64, args: &[Value]) -> Result<Vec<Value>, VmError> {
    unsafe {
        let record = &*(token as *const Continuation);
        let l = record.state;
        let _owner = shared(l).owner.upgrade();
        if let Some(value) = &record.value {
            vm_ref.unregister_host_function(value.clone());
        }
        if let Some(value) = &record.close_handler {
            vm_ref.unregister_host_function(value.clone());
        }
        let main = main_state(l);
        let old_vm = (*main).vm;
        (*main).vm = vm_ref;
        let saved = std::mem::replace(&mut (*l).stack, record.stack.clone());
        (*l).saved_stacks.push(saved);
        let old_current = std::mem::replace(&mut (*l).current, record.current);
        let old_close = std::mem::replace(&mut (*l).close_slots, record.close_slots.clone());
        let old_pending = (*l).pending_continuation.take();
        let old_context = (*l).context;
        let mut resumed = args.to_vec();
        let mut status = 1;
        if record.child {
            let memory = matches!(resumed.first(), Some(Value::Integer(4)));
            let hook_error = matches!(resumed.first(), Some(Value::Integer(2)));
            let success = !memory && !hook_error && resumed.first().is_some_and(Value::is_truthy);
            if !resumed.is_empty() {
                resumed.remove(0);
            }
            if success && record.nresults >= 0 {
                resumed.resize(record.nresults as usize, Value::Nil);
            }
            if !success {
                status = if memory { 4 } else { 2 };
                if let Some(handler) = record.handler.as_ref().filter(|_| !memory) {
                    let error = resumed.first().cloned().unwrap_or(Value::Nil);
                    match call_message_handler(l, handler.clone(), error, hook_error) {
                        Ok(values) => resumed = vec![values.first().cloned().unwrap_or(Value::Nil)],
                        Err(_) => {
                            status = ERROR_HANDLER;
                            resumed = vec![Value::from("error in error handling")];
                        }
                    }
                }
            }
        }
        (*l).stack.extend(resumed);
        (*l).context = (status, record.context);
        sync_roots(l);
        let mut count = 0;
        let mut result =
            oximond_invoke_continuation(l, record.function, status, record.context, &mut count);
        if result != 1 {
            let error = if result == 0 {
                Value::Nil
            } else {
                (*l).stack.last().cloned().unwrap_or(Value::Nil)
            };
            if let Err(failure) = close_slots(l, 0, error) {
                result = error_status(l, failure);
            }
        }
        let returned =
            if (result == 0 || result == 1) && count >= 0 && count as usize <= (*l).stack.len() {
                let values = (&(*l).stack)[(*l).stack.len() - count as usize..].to_vec();
                if result == 1 {
                    match (*l).pending_continuation.take() {
                        Some(next) => Err(request_continuation(vm_ref, l, values, next)),
                        None => Err(vm_ref.request_host_yield(values)),
                    }
                } else {
                    Ok(values)
                }
            } else {
                let error = pop(l);
                Err(if result == 4 {
                    RuntimeError::memory_error()
                } else {
                    RuntimeError::Raised(error)
                }
                .into())
            };
        (*l).stack = (*l).saved_stacks.pop().expect("saved continuation stack");
        (*l).current = old_current;
        (*l).close_slots = old_close;
        (*l).pending_continuation = old_pending;
        (*l).context = old_context;
        sync_roots(l);
        shared(l)
            .continuations
            .retain(|record| ptr::from_ref(record.as_ref()) as u64 != token);
        (*main).vm = old_vm;
        returned
    }
}

/// Calls with continuation `k`, parking it when the call yields.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least `nargs + 1` values, and `k` must stay valid and callable from the VM until the continuation finishes.
/// The call may run Lua code and may suspend, leaving `k` parked for the resume.
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_callk(
    l: *mut lua_State,
    nargs: c_int,
    nresults: c_int,
    ctx: KContext,
    k: Option<KFunction>,
) -> c_int {
    callk(l, nargs, nresults, 0, ctx, k, false)
}
/// Protected call with continuation `k` and message handler `errfunc`, parking `k` on a yield.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least `nargs + 1` values, and `k` and `errfunc` must uphold the callback contract until the continuation completes.
/// The call may run Lua code and may suspend.
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_pcallk(
    l: *mut lua_State,
    nargs: c_int,
    nresults: c_int,
    errfunc: c_int,
    ctx: KContext,
    k: Option<KFunction>,
) -> c_int {
    callk(l, nargs, nresults, errfunc, ctx, k, true)
}
/// Shared call-with-continuation behind `oxi_callk` and `oxi_pcallk`.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the running callback's family, with `nargs + 1` values on its stack.
/// `k` must stay valid until the parked continuation runs or is discarded.
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
unsafe fn callk(
    l: *mut lua_State,
    nargs: c_int,
    nresults: c_int,
    errfunc: c_int,
    ctx: KContext,
    k: Option<KFunction>,
    protected: bool,
) -> c_int {
    let Some(k) = k else {
        return oxi_pcall(l, nargs, nresults, errfunc);
    };
    if (*l).current.is_none() {
        return oxi_pcall(l, nargs, nresults, errfunc);
    }
    let handler = if errfunc != 0 {
        value(l, errfunc)
    } else {
        None
    };
    let len = (*l).stack.len();
    if nargs < 0 || len < nargs as usize + 1 {
        return failure(l, "invalid argument count");
    }
    let mut arguments = (*l).stack.split_off(len - nargs as usize - 1);
    let function = arguments.remove(0);
    sync_roots(l);
    match vm(l).call_value_resumable(function, arguments, protected) {
        Ok(mut results) => {
            if nresults >= 0 {
                results.resize(nresults as usize, Value::Nil);
            }
            (*l).stack.extend(results);
            0
        }
        Err(VmError::Yield(values)) => {
            match park_continuation(l, ctx, k, true, nresults, handler) {
                Ok(continuation) => {
                    (*l).pending_continuation = Some(continuation);
                    (*l).stack = values;
                    1
                }
                Err(error) => {
                    push(l, error_value(l, error));
                    2
                }
            }
        }
        Err(error) => {
            let mut status = failure_status(&error);
            let hook_error = error.is_hook_error();
            let mut value = error_value(l, error);
            if let Some(handler) = handler.filter(|_| status == 2) {
                match call_message_handler(l, handler, value.clone(), hook_error) {
                    Ok(values) => value = values.first().cloned().unwrap_or(Value::Nil),
                    Err(_) => {
                        status = ERROR_HANDLER;
                        value = Value::from("error in error handling");
                    }
                }
            }
            push(l, value);
            status
        }
    }
}
/// Yields `nresults` values with continuation `k`, parking it across the yield.
///
/// # Safety
///
/// `l` must be a live `lua_State` inside a C callback or hook that may yield, with `nresults` values on its stack.
/// `k` must stay valid and callable from the VM until the continuation runs.
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_yieldk(
    l: *mut lua_State,
    nresults: c_int,
    ctx: KContext,
    k: Option<KFunction>,
) -> c_int {
    if (*l).in_hook && k.is_some() {
        return failure(l, "cannot use continuations inside debug hooks");
    }
    let status = oxi_yield(l, nresults);
    if status != 0 {
        return status;
    }
    if let Some(k) = k {
        match park_continuation(l, ctx, k, false, -1, None) {
            Ok(continuation) => (*l).pending_continuation = Some(continuation),
            Err(error) => {
                push(l, error_value(l, error));
                return 2;
            }
        }
    }
    0
}
/// Returns the running call's continuation status and writes its context.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the running callback's family, and `context` must be null or writable.
#[cfg(feature = "lua52")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getctx(l: *mut lua_State, context: *mut c_int) -> c_int {
    if !context.is_null() {
        *context = (*l).context.1;
    }
    (*l).context.0
}
/// Reports whether the running C callback may yield.
///
/// # Safety
///
/// `l` must be a live `lua_State` used on its owning thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_isyieldable(l: *mut lua_State) -> c_int {
    i32::from(vm(l).validate_host_yield().is_ok())
}
/// Pops a key and pushes the indexed value, following `__index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value and room for one more.
/// The lookup may run Lua metamethods and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_gettable(l: *mut lua_State, i: c_int) -> c_int {
    // A table on the stack and a string key it holds, or an array element:
    // the key's slot takes the result, so the pop and the push cancel.
    if let Some(&Value::Table(table)) = peek(l, i) {
        let hit = match peek(l, -1) {
            Some(Value::String(key)) => rt(l).capi_get_string(table, key),
            Some(&Value::Number(x)) if (1.0..9.0e15).contains(&x) && x == x.trunc() => {
                rt(l).capi_get_array(table, x as i64)
            }
            _ => None,
        };
        if let Some(found) = hit {
            let found = found.clone();
            // The top slot held the key, which the assignment drops.
            *state(l).stack.last_mut().unwrap() = found;
            return 0;
        }
    }
    gettable_slow(l, i)
}
/// The general case of [`oxi_gettable`].
///
/// # Safety
///
/// As for [`oxi_gettable`].
#[inline(never)]
unsafe fn gettable_slow(l: *mut lua_State, i: c_int) -> c_int {
    let object = value(l, i).unwrap_or(Value::Nil);
    let key = pop(l);
    match gettable(l, object, key) {
        Ok(v) => {
            push(l, v);
            0
        }
        Err(e) => error_status(l, e),
    }
}
/// Pops a value and a key and stores them, following `__newindex`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least two values.
/// The store may run Lua metamethods and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_settable(l: *mut lua_State, i: c_int) -> c_int {
    let object = value(l, i).unwrap_or(Value::Nil);
    let v = pop(l);
    let key = pop(l);
    match settable(l, object, key, v) {
        Ok(()) => 0,
        Err(e) => error_status(l, e),
    }
}
/// Concatenates the top `n` values into one string.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least `n` values and room for one result.
/// The `__concat` metamethod may run Lua code and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_concat(l: *mut lua_State, n: c_int) -> c_int {
    if n < 0 || n as usize > (*l).stack.len() {
        return failure(l, "invalid concat count");
    }
    if n == 0 {
        push(l, Value::from(""));
        return 0;
    }
    let start = (*l).stack.len() - n as usize;
    let values = state(l).stack.split_off(start);
    let mut combined = values.last().cloned().unwrap();
    for value in values[..values.len() - 1].iter().rev() {
        match operator_call(l, "a .. b", vec![value.clone(), combined]) {
            Ok(value) => combined = value,
            Err(error) => {
                push(l, error_value(l, error));
                return 2;
            }
        }
    }
    push(l, combined);
    0
}
/// Compares the values at `a` and `b` with operation `op`, writing the boolean result.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for the operator temporaries, and `result` must be writable.
/// Comparison metamethods may run Lua code and re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_compare(
    l: *mut lua_State,
    a: c_int,
    b: c_int,
    op: c_int,
    result: *mut c_int,
) -> c_int {
    let (Some(a), Some(b)) = (value(l, a), value(l, b)) else {
        *result = 0;
        return 0;
    };
    let expression = match op {
        0 => "a == b",
        1 => "a < b",
        2 => "a <= b",
        _ => return failure(l, "invalid comparison operation"),
    };
    match operator_call(l, expression, vec![a, b]) {
        Ok(value) => {
            *result = i32::from(value.is_truthy());
            0
        }
        Err(error) => error_status(l, error),
    }
}
/// Loads a chunk read piecewise from `reader`, like `lua_load`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// `reader` must uphold the reader contract with `data`, and `name` must be a live NUL-terminated chunk name.
#[unsafe(no_mangle)]
#[cfg(not(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
)))]
pub unsafe extern "C" fn lua_load(
    l: *mut lua_State,
    reader: Reader,
    data: *mut c_void,
    name: *const c_char,
) -> c_int {
    load_source(l, reader, data, name, ptr::null())
}
/// Loads a chunk from `reader`, restricting text or binary through `mode`.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `reader` must uphold the reader contract with `data`.
/// `name` must be a live NUL-terminated string, and `mode` null or one.
#[unsafe(no_mangle)]
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
pub unsafe extern "C" fn lua_load(
    l: *mut lua_State,
    reader: Reader,
    data: *mut c_void,
    name: *const c_char,
    mode: *const c_char,
) -> c_int {
    load_source(l, reader, data, name, mode)
}
/// Feeds the reader's pieces to the compiler or the bytecode loader.
///
/// # Safety
///
/// `l` must be a live `lua_State`, used only to read the compatibility profile.
/// `reader` must return buffers of `*length` readable bytes that stay valid until its next call, ending the source with a null pointer.
/// `name` must be NUL-terminated, and `mode` null or one.
// The reader may call back into the VM. Keep only the compiler session across
// callbacks, never a borrowed VM or runtime reference.
unsafe fn compile_reader(
    l: *mut lua_State,
    reader: Reader,
    data: *mut c_void,
    name: *const c_char,
    mode: *const c_char,
) -> Result<oximond::bytecode::Chunk, String> {
    const MAGIC: &[u8] = b"\x1bOXIMOND";
    let name = LuaString::from(cstr(name));
    let mode = (!mode.is_null()).then(|| cstr(mode).to_vec());
    let loader = CompilerLoader::with_profile(vm(l).compatibility_profile());
    let mut prefix = Vec::new();
    let mut session = None;
    let mut binary = None::<Vec<u8>>;
    loop {
        let mut count = 0;
        let pointer = reader(l, data, &mut count);
        let ended = pointer.is_null() || count == 0;
        let piece = if ended { &[] } else { bytes(pointer, count) };
        if let Some(source) = &mut binary {
            source.extend_from_slice(piece);
            if ended {
                return oximond::bytecode::binary::load(source).map_err(|error| error.to_string());
            }
            continue;
        }
        if session.is_none() {
            prefix.extend_from_slice(piece);
            // luaL_loadfile can return its synthetic blank line separately.
            let candidate = prefix.strip_prefix(b"\n").unwrap_or(&prefix);
            if !ended && MAGIC.starts_with(candidate) && candidate.len() < MAGIC.len() {
                continue;
            }
            let is_binary = oximond::bytecode::binary::claims(candidate);
            if let Some(mode) = &mode
                && !mode.contains(&if is_binary { b'b' } else { b't' })
            {
                return Err(format!(
                    "attempt to load a {} chunk (mode is '{}')",
                    if is_binary { "binary" } else { "text" },
                    String::from_utf8_lossy(mode),
                ));
            }
            if is_binary {
                if ended {
                    return oximond::bytecode::binary::load(candidate)
                        .map_err(|error| error.to_string());
                }
                binary = Some(candidate.to_vec());
                continue;
            }
            // The common reader hands over the whole source in one piece and then ends
            // (`luaL_loadbuffer`, `luaL_loadstring`). Parse that piece with the whole-source
            // parser, read once more, and keep the result if the reader ends. Anything else
            // starts the streaming session on the piece as before, so a source that fails to
            // parse never makes the reader produce more input than it did. The process-wide
            // chunk cache keeps the result only once the reader ended after the piece.
            let mut more = std::ptr::null();
            if !ended
                && !piece.is_empty()
                && let Some(chunk) = loader.try_load_piece(&name, &prefix, || {
                    more = reader(l, data, &mut count);
                    more.is_null() || count == 0
                })
            {
                if more.is_null() || count == 0 {
                    return Ok(chunk);
                }
                prefix.extend_from_slice(bytes(more, count));
                session = Some(loader.begin(name.clone()));
            } else {
                session = Some(loader.begin(name.clone()));
            }
        }
        let mut active = session.take().expect("text reader session started");
        let input = if prefix.is_empty() {
            LuaString::from(piece)
        } else {
            LuaString::from(std::mem::take(&mut prefix))
        };
        if !input.as_bytes().is_empty() {
            active = match active.resume(LoadInput::Bytes(input)) {
                LoadPoll::NeedInput(next) => next,
                LoadPoll::Loaded(chunk) => return Ok(chunk),
                LoadPoll::Failed(error) => return Err(error.to_lossy().into_owned()),
            };
        }
        if ended {
            return match active.resume(LoadInput::End) {
                LoadPoll::Loaded(chunk) => Ok(chunk),
                LoadPoll::Failed(error) => Err(error.to_lossy().into_owned()),
                LoadPoll::NeedInput(_) => {
                    Err("loader requested input after end of source".to_owned())
                }
            };
        }
        session = Some(active);
    }
}

/// Loads a chunk from the reader and pushes it with its environment installed.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `reader` must uphold the reader contract with `data`.
/// `name` and `mode` carry the requirements of `compile_reader`.
unsafe fn load_source(
    l: *mut lua_State,
    reader: Reader,
    data: *mut c_void,
    name: *const c_char,
    mode: *const c_char,
) -> c_int {
    let chunk = compile_reader(l, reader, data, name, mode);
    match chunk {
        Ok(chunk) => match vm(l).load_owned_chunk(chunk) {
            Ok(v) => {
                if let Value::Closure(handle) = v {
                    let environment = if MODERN {
                        match global_table(l) {
                            Value::Table(table) => table,
                            _ => thread_environment(l),
                        }
                    } else {
                        thread_environment(l)
                    };
                    let _ = rt(l).set_closure_environment(handle, environment);
                    if !vm(l).compatibility_profile().is_lua51_family()
                        && let Ok(snapshot) = rt(l).closure_snapshot(handle)
                    {
                        let chunk = snapshot.chunk();
                        for (index, descriptor) in chunk.upvalues.iter().enumerate() {
                            if descriptor.name.as_bytes() == b"_ENV"
                                && let Some(cell) = snapshot.cells().get(index)
                            {
                                let _ = rt(l).cell_set(*cell, Value::Table(environment));
                            }
                        }
                    }
                }
                push(l, v);
                0
            }
            Err(e) => {
                push(l, error_value(l, e));
                3
            }
        },
        Err(e) => {
            push(l, Value::from(e.to_string()));
            3
        }
    }
}
/// Does nothing: this profile ships its libraries inside the VM.
///
/// # Safety
///
/// The state argument is ignored and may be null, so no precondition applies.
#[cfg(not(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
)))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaL_openlibs(_l: *mut lua_State) {}
/// Pushes the cached library table `name`, restoring it from the registry cache.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose `oximond.capi.libraries` cache was installed by `cold_libraries`.
unsafe fn open_library(l: *mut lua_State, name: &str) -> c_int {
    let mut v = rt(l).get_global(name);
    if matches!(v, Value::Nil) {
        let registry = rt(l).registry();
        if let Ok(Value::Table(cache)) =
            rt(l).table_get(registry, &Value::from("oximond.capi.libraries"))
        {
            v = rt(l)
                .table_get(cache, &Value::from(name))
                .unwrap_or(Value::Nil);
        }
    }
    if name == "string" {
        let registry = rt(l).registry();
        if let Ok(Value::Table(cache)) =
            rt(l).table_get(registry, &Value::from("oximond.capi.libraries"))
            && let Ok(Value::Table(metatable)) =
                rt(l).table_get(cache, &Value::from("string_metatable"))
        {
            rt(l)
                .raw_set_metatable(&Value::from(""), Some(metatable))
                .expect("restore string metatable");
        }
    }
    push(l, v);
    1
}
/// Copies the cached base library back into the globals and pushes them.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache holds the `base` snapshot from `cold_libraries`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_base(l: *mut lua_State) -> c_int {
    let registry = rt(l).registry();
    let globals = rt(l).global_environment();
    if let Ok(Value::Table(cache)) =
        rt(l).table_get(registry, &Value::from("oximond.capi.libraries"))
        && let Ok(Value::Table(base)) = rt(l).table_get(cache, &Value::from("base"))
    {
        for (key, value) in table_snapshot(l, base) {
            let _ = rt(l).table_set(globals, &key, value);
        }
    }
    push(l, Value::Table(globals));
    1
}
/// Pushes the cached `table` library, as `luaopen_table`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_table(l: *mut lua_State) -> c_int {
    open_library(l, "table")
}
/// Pushes the cached `string` library, as `luaopen_string`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_string(l: *mut lua_State) -> c_int {
    open_library(l, "string")
}
/// Pushes the cached `math` library, as `luaopen_math`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_math(l: *mut lua_State) -> c_int {
    open_library(l, "math")
}
/// Pushes the cached `os` library, as `luaopen_os`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_os(l: *mut lua_State) -> c_int {
    open_library(l, "os")
}
/// Pushes the cached `io` library, as `luaopen_io`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_io(l: *mut lua_State) -> c_int {
    open_library(l, "io")
}
/// Pushes the cached `debug` library, as `luaopen_debug`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_debug(l: *mut lua_State) -> c_int {
    open_library(l, "debug")
}
/// Pushes the cached `package` library, as `luaopen_package`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_package(l: *mut lua_State) -> c_int {
    open_library(l, "package")
}
/// Pushes `l`'s thread and reports whether it is the main one.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushthread(l: *mut lua_State) -> c_int {
    push(l, Value::Thread((*l).thread));
    i32::from(l == main_state(l))
}
/// Returns the thread wrapper of the thread value at `i`.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `i` one of its stack indices.
/// The returned pointer is only valid while `l`'s main state is alive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_tothread(l: *mut lua_State, i: c_int) -> *mut lua_State {
    match value(l, i) {
        Some(Value::Thread(thread)) => thread_state(l, thread),
        _ => ptr::null_mut(),
    }
}
/// Returns the status code of the coroutine `l` represents.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose thread is registered in its family's VM.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_status(l: *mut lua_State) -> c_int {
    if (*l).status > 1 {
        return (*l).status;
    }
    if l != main_state(l) && vm(l).host_coroutine_failed((*l).thread).unwrap_or(false) {
        return 2;
    }
    if l != main_state(l)
        && matches!(
            vm(l).host_coroutine_status((*l).thread),
            Ok(HostCoroutineStatus::Suspended)
        )
    {
        1
    } else {
        0
    }
}
/// Creates a host coroutine, writes its wrapper to `result`, and pushes the thread.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `result` must be writable.
/// The wrapper written to `result` may only be used while `l`'s main state is alive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_newthread(l: *mut lua_State, result: *mut *mut lua_State) -> c_int {
    sync_roots(l);
    match vm(l).create_host_coroutine() {
        Ok(thread) => {
            *result = thread_state(l, thread);
            if (*result).is_null() {
                push(l, Value::from("not enough memory"));
                return 4;
            }
            (**result).entry_pending = true;
            if (*l).hook.is_some() {
                let _ = lua_sethook(*result, (*l).hook, (*l).hook_mask, (*l).hook_count);
            }
            push(l, Value::Thread(thread));
            sync_roots(l);
            0
        }
        Err(error) => error_status(l, error),
    }
}
/// Moves `n` values from `from`'s stack to `to`'s.
///
/// # Safety
///
/// `from` and `to` must be live `lua_State`s of the same family, and `n` values must sit on `from`'s stack.
/// Cross-state moves and bad counts trip asserts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_xmove(from: *mut lua_State, to: *mut lua_State, n: c_int) {
    if from == to {
        return;
    }
    assert!(
        main_state(from) == main_state(to),
        "cannot move values between independent states"
    );
    assert!(
        n >= 0 && n as usize <= (*from).stack.len(),
        "invalid move count"
    );
    let start = (*from).stack.len() - n as usize;
    let values = state(from).stack.split_off(start);
    state(to).stack.extend(values);
}
/// Yields `nresults` values from the running C callback or hook.
///
/// # Safety
///
/// `l` must be a live `lua_State` inside a C callback or hook of a coroutine that may yield, with `nresults` values on its stack.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_yield(l: *mut lua_State, nresults: c_int) -> c_int {
    if nresults < 0 || nresults as usize > (*l).stack.len() {
        return failure(l, "invalid yield result count");
    }
    if (*l).current.is_none() && !(*l).in_hook {
        return failure(l, "attempt to yield from outside a C callback");
    }
    if (*l).in_hook && nresults != 0 {
        return failure(l, "attempt to yield values from a debug hook");
    }
    match vm(l).validate_host_yield() {
        Ok(()) => 0,
        Err(error) => error_status(l, error),
    }
}
/// Resumes the coroutine `l` represents with `nargs` arguments.
///
/// # Safety
///
/// `l` must be a live thread `lua_State` — not an independent state — with `nargs` values on its stack, and its wrapper must survive the call.
/// The resume runs Lua code and may re-enter C callbacks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_resume(l: *mut lua_State, nargs: c_int) -> c_int {
    if nargs < 0 || nargs as usize > (*l).stack.len() {
        return failure(l, "invalid argument count");
    }
    sync_roots(l);
    let thread = (*l).thread;
    let status = vm(l).host_coroutine_status(thread);
    // Invalid resumes leave their diagnostic on top and do not change a
    // running or dead coroutine's execution state.
    if !matches!(
        status,
        Ok(HostCoroutineStatus::New | HostCoroutineStatus::Suspended)
    ) {
        let message = if matches!(status, Ok(HostCoroutineStatus::Dead)) {
            (*l).status = 2;
            "cannot resume dead coroutine"
        } else {
            "cannot resume non-suspended coroutine"
        };
        return failure(l, message);
    }
    let start = (*l).stack.len() - nargs as usize;
    let arguments = state(l).stack.split_off(start);
    let result = if matches!(status, Ok(HostCoroutineStatus::New)) && (*l).entry_pending {
        (*l).entry_pending = false;
        let function = pop(l);
        vm(l).start_host_coroutine(thread, function, arguments)
    } else {
        vm(l).resume_host_coroutine(thread, arguments)
    };
    match result {
        Ok(mut values) => {
            let success = matches!(values.first(), Some(Value::Boolean(true)));
            if !values.is_empty() {
                values.remove(0);
            }
            (*l).stack = values;
            (*l).status = if !success {
                2
            } else if matches!(
                vm(l).host_coroutine_status(thread),
                Ok(HostCoroutineStatus::Suspended)
            ) {
                1
            } else {
                0
            };
        }
        Err(error) => {
            (*l).status = failure_status(&error);
            (*l).stack = vec![error_value(l, error)];
        }
    }
    sync_roots(l);
    (*l).status
}
/// Resumes a coroutine, like `lua_resume` for the 5.1 ABI.
///
/// # Safety
///
/// `l` must be a live thread `lua_State` with `nargs` values on its stack.
/// The resume runs Lua code and may re-enter C callbacks.
#[unsafe(no_mangle)]
#[cfg(not(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
)))]
pub unsafe extern "C" fn lua_resume(l: *mut lua_State, nargs: c_int) -> c_int {
    oxi_resume(l, nargs)
}
/// Resumes a coroutine, like `lua_resume` for the 5.2 and 5.3 ABI; `_from` is ignored.
///
/// # Safety
///
/// `l` must be a live thread `lua_State` with `nargs` values on its stack, and `_from` may be null.
/// The resume runs Lua code and may re-enter C callbacks.
#[unsafe(no_mangle)]
#[cfg(any(feature = "lua52", feature = "lua53"))]
pub unsafe extern "C" fn lua_resume(
    l: *mut lua_State,
    _from: *mut lua_State,
    nargs: c_int,
) -> c_int {
    oxi_resume(l, nargs)
}
/// Resumes a coroutine, like `lua_resume` for the 5.4 ABI; `_from` is ignored.
///
/// # Safety
///
/// `l` must be a live thread `lua_State` with `nargs` values on its stack, and `nresults` must be null or writable.
/// The resume runs Lua code and may re-enter C callbacks.
#[unsafe(no_mangle)]
#[cfg(any(feature = "lua54", feature = "lua55"))]
pub unsafe extern "C" fn lua_resume(
    l: *mut lua_State,
    _from: *mut lua_State,
    nargs: c_int,
    nresults: *mut c_int,
) -> c_int {
    let status = oxi_resume(l, nargs);
    if !nresults.is_null() {
        *nresults = lua_gettop(l);
    }
    status
}
/// Runs the GC command `what` with operand `data`, like `lua_gc`.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
/// Commands that step or run collection may execute finalizers and metamethods.
#[unsafe(no_mangle)]
#[cfg(not(any(feature = "lua54", feature = "lua55")))]
pub unsafe extern "C" fn lua_gc(l: *mut lua_State, what: c_int, data: c_int) -> c_int {
    oxi_gc(l, what, data as usize, 0, 0)
}
/// Returns the state's allocator and writes its data pointer.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `data` must be null or writable.
/// The returned allocator is only usable while the state keeps it installed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getallocf(l: *mut lua_State, data: *mut *mut c_void) -> Option<Alloc> {
    if !data.is_null() {
        *data = shared(l).allocator_data
    }
    shared(l).allocator
}
/// Replaces the allocator for the whole state family.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `f` must be null or an `Alloc` usable with `data`.
/// The replacement must accept releasing the blocks the previous allocator handed out.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_setallocf(l: *mut lua_State, f: Option<Alloc>, data: *mut c_void) {
    shared(l).allocator = f;
    shared(l).allocator_data = data;
    let main = main_state(l);
    if (*extra_state(main)).allocation.is_some() {
        (*extra_state(main)).allocation = f.map(|f| (f, data));
    }
    for child in shared(l).threads.values_mut() {
        if (*extra_state(child.0)).allocation.is_some() {
            (*extra_state(child.0)).allocation = f.map(|f| (f, data));
        }
    }
}

#[repr(C)]
pub struct lua_Debug {
    event: c_int,
    name: *const c_char,
    namewhat: *const c_char,
    what: *const c_char,
    source: *const c_char,
    #[cfg(any(feature = "lua54", feature = "lua55"))]
    srclen: usize,
    currentline: c_int,
    #[cfg(not(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    )))]
    nups: c_int,
    linedefined: c_int,
    lastlinedefined: c_int,
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    nups: u8,
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    nparams: u8,
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    isvararg: c_char,
    #[cfg(feature = "lua55")]
    extraargs: u8,
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    istailcall: c_char,
    #[cfg(feature = "lua54")]
    ftransfer: u16,
    #[cfg(feature = "lua54")]
    ntransfer: u16,
    #[cfg(feature = "lua55")]
    ftransfer: c_int,
    #[cfg(feature = "lua55")]
    ntransfer: c_int,
    short_src: [c_char; 60],
    #[cfg(not(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    )))]
    i_ci: c_int,
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    i_ci: *mut c_void,
}
/// Records activation `level` in `ar`, returning one when the level exists.
///
/// # Safety
///
/// `l` must be a live `lua_State` with an inspectable call stack, and `ar` must point to a `lua_Debug` the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getstack(
    l: *mut lua_State,
    level: c_int,
    ar: *mut lua_Debug,
) -> c_int {
    if level < 0 || ar.is_null() || stack_info(l, level as usize).is_none() {
        return 0;
    }
    #[cfg(not(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    )))]
    {
        (*ar).i_ci = level;
    }
    #[cfg(any(
        feature = "lua52",
        feature = "lua53",
        feature = "lua54",
        feature = "lua55"
    ))]
    {
        (*ar).i_ci = (level as usize + 1) as *mut c_void;
    }
    1
}
/// Fills `ar` with the debug information `what` selects.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `ar` must be a caller-owned `lua_Debug` filled by `lua_getstack`, or fresh when `what` starts with `>` and the function is on the stack.
/// `what` must be a live NUL-terminated option string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getinfo(
    l: *mut lua_State,
    what: *const c_char,
    ar: *mut lua_Debug,
) -> c_int {
    if ar.is_null() {
        return 0;
    }
    let options = cstr(what);
    if options.first() == Some(&b'>') {
        let function = pop(l);
        return function_info(l, function, &options[1..], ar);
    }
    let level = if MODERN {
        ((*ar).i_ci as usize).wrapping_sub(1)
    } else {
        (*ar).i_ci as usize
    };
    let Some(info) = stack_info(l, level) else {
        return 0;
    };
    for &option in options {
        match option {
            b'n' => {
                (*ar).name = info.name.clone().map_or(ptr::null(), |s| intern(l, s));
                (*ar).namewhat = intern(l, LuaString::from(info.name_what));
            }
            b'S' => {
                (*ar).source = intern(l, info.source.clone());
                #[cfg(any(feature = "lua54", feature = "lua55"))]
                {
                    (*ar).srclen = info.source.as_bytes().len();
                }
                (*ar).what = intern(l, LuaString::from(info.what));
                (*ar).linedefined = info.definition_line;
                (*ar).lastlinedefined = info.last_definition_line;
                (*ar).short_src = [0; 60];
                for (dest, &src) in (*ar)
                    .short_src
                    .iter_mut()
                    .take(59)
                    .zip(info.short_source.as_bytes())
                {
                    *dest = src as c_char;
                }
            }
            b'l' => (*ar).currentline = info.current_line,
            b'u' => {
                (*ar).nups = info.upvalue_count as _;
                #[cfg(any(
                    feature = "lua52",
                    feature = "lua53",
                    feature = "lua54",
                    feature = "lua55"
                ))]
                {
                    (*ar).nparams = info.parameter_count;
                    (*ar).isvararg = i8::from(info.is_vararg);
                    #[cfg(feature = "lua55")]
                    {
                        (*ar).extraargs = 0;
                    }
                }
            }
            #[cfg(any(
                feature = "lua52",
                feature = "lua53",
                feature = "lua54",
                feature = "lua55"
            ))]
            b't' => (*ar).istailcall = i8::from(info.is_tail_call),
            #[cfg(any(feature = "lua54", feature = "lua55"))]
            b'r' => {
                (*ar).ftransfer = info.first_transfer as _;
                (*ar).ntransfer = info.transfer_count as _;
            }
            b'f' => push(l, info.function.clone().unwrap_or(Value::Nil)),
            b'L' => active_lines(l, info.function.as_ref()),
            _ => return 0,
        }
    }
    1
}
/// Pushes upvalue `n` of the function at `index` and returns its name.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `n` must be positive and within the function's upvalue count.
/// The returned name is interned like `lua_tolstring`'s result and must not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getupvalue(
    l: *mut lua_State,
    index: c_int,
    n: c_int,
) -> *const c_char {
    let Some(target) = value(l, index) else {
        return ptr::null();
    };
    if let Value::Closure(handle) = target {
        if n <= 0 {
            return ptr::null();
        }
        if let Ok(body) = rt(l).closure_snapshot(handle) {
            let chunk = body.chunk();
            if let (Some(cell), Some(descriptor)) = (
                body.cells().get(n as usize - 1),
                chunk.upvalues.get(n as usize - 1),
            ) {
                let name = descriptor.name.clone();
                let value = rt(l).cell_get(*cell).unwrap_or(Value::Nil);
                push(l, value);
                return intern(l, name);
            }
        }
        return ptr::null();
    }
    let record = shared(l)
        .closures
        .iter()
        .find(|c| Runtime::raw_equal(&c.value, &target))
        .map(|c| (c.upvalues, c.count));
    if let Some((table, count)) = record
        && n > 0
        && n as usize <= count
    {
        let v = rt(l)
            .table_get(table, &Value::Number(f64::from(n)))
            .unwrap_or(Value::Nil);
        push(l, v);
        return c"".as_ptr();
    }
    ptr::null()
}
/// Pops a value into upvalue `n` of the function at `index` and returns its name.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value, and `n` must be positive and within the function's upvalue count.
/// The returned name follows `lua_tolstring`'s lifetime rules.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_setupvalue(
    l: *mut lua_State,
    index: c_int,
    n: c_int,
) -> *const c_char {
    let Some(target) = value(l, index) else {
        return ptr::null();
    };
    if let Value::Closure(handle) = target {
        if n <= 0 {
            return ptr::null();
        }
        if let Ok(body) = rt(l).closure_snapshot(handle) {
            let chunk = body.chunk();
            if let (Some(cell), Some(descriptor)) = (
                body.cells().get(n as usize - 1),
                chunk.upvalues.get(n as usize - 1),
            ) {
                let name = descriptor.name.clone();
                let value = pop(l);
                let _ = rt(l).cell_set(*cell, value);
                return intern(l, name);
            }
        }
        return ptr::null();
    }
    let record = shared(l)
        .closures
        .iter()
        .find(|c| Runtime::raw_equal(&c.value, &target))
        .map(|c| (c.upvalues, c.count));
    if let Some((table, count)) = record
        && n > 0
        && n as usize <= count
    {
        let v = pop(l);
        let _ = rt(l).table_set(table, &Value::Number(f64::from(n)), v);
        return c"".as_ptr();
    }
    ptr::null()
}
/// Returns how many trace entries ran native code.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose VM's statistics it may read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oximond_jit_native_runs(l: *mut lua_State) -> u64 {
    // Trace entries that ran native code. An entry whose checks matched no
    // version ran none.
    let trace = vm(l).jit_statistics().trace;
    let mismatches = trace.exits[oximond::vm::TraceExitKind::EntryMismatch.index()];
    trace.native_entries.saturating_sub(mismatches)
}
/// Runs the GC control operation `what` with operands `a`, `b`, and `c`.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
/// Operations that collect or step may run finalizers and metamethods, so the call must be acceptable where Lua code can run.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_gc(
    l: *mut lua_State,
    what: c_int,
    a: usize,
    b: c_int,
    c: c_int,
) -> c_int {
    use oximond::runtime::{BuiltinFunction, GcMode};
    let what = if cfg!(feature = "lua55") {
        match what {
            6 => 9,
            7 => 10,
            8 => 11,
            9 => 12,
            other => other,
        }
    } else {
        what
    };
    match what {
        0 => {
            rt(l).stop_gc();
            0
        }
        1 => {
            rt(l).restart_gc();
            0
        }
        2 => {
            sync_roots(l);
            vm(l).collect_full_with_values(&[]);
            cleanup_native_storage(l);
            0
        }
        3 => (rt(l).gc_count_bytes() / 1024) as c_int,
        4 => (rt(l).gc_count_bytes() % 1024) as c_int,
        5 => {
            let kilobytes = if cfg!(feature = "lua55") {
                a.saturating_add(1023) / 1024
            } else {
                a
            };
            let result = call(
                l,
                Value::BuiltinFunction(BuiltinFunction::CollectGarbage),
                vec![Value::from("step"), Value::Number(kilobytes as f64)],
            );
            match result {
                Ok(values) => i32::from(values.first().is_some_and(Value::is_truthy)),
                Err(_) => -1,
            }
        }
        6 if !cfg!(feature = "lua55") => rt(l).set_gc_pause(a as c_int as f64) as c_int,
        7 if !cfg!(feature = "lua55") => rt(l).set_gc_step_multiplier(a as c_int as f64) as c_int,
        8 if cfg!(feature = "lua52") => {
            rt(l).set_gc_parameter(2, a as c_int as f64).unwrap_or(0.0) as c_int
        }
        9 => i32::from(rt(l).gc_is_running()),
        10 | 11 if cfg!(any(feature = "lua52", feature = "lua54", feature = "lua55")) => {
            let previous = rt(l).set_gc_mode(if what == 10 {
                GcMode::Generational
            } else {
                GcMode::Incremental
            });
            if cfg!(feature = "lua54") {
                if what == 10 {
                    if a != 0 {
                        rt(l).set_gc_parameter(0, a as c_int as f64);
                    }
                    if b != 0 {
                        rt(l).set_gc_parameter(2, f64::from(b));
                    }
                } else {
                    if a != 0 {
                        rt(l).set_gc_pause(a as c_int as f64);
                    }
                    if b != 0 {
                        rt(l).set_gc_step_multiplier(f64::from(b));
                    }
                    if c != 0 {
                        rt(l).set_gc_parameter(5, 2_f64.powi(c));
                    }
                }
            }
            if cfg!(feature = "lua52") {
                return 0;
            }
            if previous == GcMode::Generational {
                if cfg!(feature = "lua55") { 7 } else { 10 }
            } else {
                if cfg!(feature = "lua55") { 8 } else { 11 }
            }
        }
        12 if cfg!(feature = "lua55") => {
            let Some(previous) = rt(l).gc_parameter(a) else {
                return -1;
            };
            if b >= 0 {
                rt(l).set_gc_parameter(a, gc_quantize(b as u32));
            }
            previous as c_int
        }
        _ => -1,
    }
}
fn gc_quantize(value: u32) -> f64 {
    oximond::vm::quantize_lua55_gc_parameter(value)
}
/// Marks the stack slot at `index` to be closed, like `lua_toclose`.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `index` must name a live stack slot above the previously marked one.
/// Any truthy value in the slot must carry a `__close` metamethod.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_toclose(l: *mut lua_State, index: c_int) -> c_int {
    let Some(slot) = slot(l, index) else {
        return failure(l, "invalid close slot");
    };
    if (*l)
        .close_slots
        .last()
        .is_some_and(|&previous| previous >= slot)
    {
        return failure(l, "invalid close slot order");
    }
    let value = state(l).stack[slot].clone();
    if value.is_truthy() {
        let closer = rt(l)
            .raw_metatable(&value)
            .ok()
            .flatten()
            .and_then(|t| rt(l).table_get(t, &Value::from("__close")).ok())
            .unwrap_or(Value::Nil);
        if matches!(closer, Value::Nil) {
            return failure(l, "variable got a non-closable value");
        }
    }
    (*l).close_slots.push(slot);
    0
}
/// Runs `__close` on every marked slot at or above `threshold`, threading `error` through.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose close-slot indices still match its stack.
/// The closers may run Lua code and raise; the first failure is returned and later slots still close.
unsafe fn close_slots(
    l: *mut lua_State,
    threshold: usize,
    mut error: Value,
) -> Result<(), VmError> {
    let mut failed = None;
    while (*l)
        .close_slots
        .last()
        .is_some_and(|&slot| slot >= threshold)
    {
        let slot = (*l).close_slots.pop().unwrap();
        let value = state(l).stack.get(slot).cloned().unwrap_or(Value::Nil);
        if value.is_truthy() {
            let closer = rt(l)
                .raw_metatable(&value)?
                .and_then(|t| rt(l).table_get(t, &Value::from("__close")).ok())
                .unwrap_or(Value::Nil);
            if let Err(failure) = call(l, closer, vec![value, error.clone()]) {
                error = error_value(l, failure.clone());
                failed = Some(failure);
            }
        }
    }
    failed.map_or(Ok(()), Err)
}
/// Closes the top marked variable, like `lua_closeslot`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose top close slot is exactly `index`.
/// The `__close` metamethod it runs may execute Lua code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_closeslot(l: *mut lua_State, index: c_int) -> c_int {
    let Some(slot) = slot(l, index) else {
        return failure(l, "invalid close slot");
    };
    if (*l).close_slots.last() != Some(&slot) {
        return failure(l, "no variable to close at given level");
    }
    let result = close_slots(l, slot, Value::Nil);
    state(l).stack[slot] = Value::Nil;
    match result {
        Ok(()) => 0,
        Err(error) => error_status(l, error),
    }
}
/// Sets the stack top to `index`, closing the marked variables it drops.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `index` must not reach below the stack bottom.
/// Closers may run Lua code; on failure the stack is truncated to `index`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_settop(l: *mut lua_State, index: c_int) -> c_int {
    let size = if index >= 0 {
        index as isize
    } else {
        (*l).stack.len() as isize + index as isize + 1
    };
    if size < 0 {
        return failure(l, "invalid stack index");
    }
    if (*l).close_slots.is_empty() {
        // No pending to-be-closed slot: the call is a plain resize.
        resize_stack(l, size as usize);
        return 0;
    }
    match close_slots(l, size as usize, Value::Nil) {
        Ok(()) => {
            (*l).stack.resize(size as usize, Value::Nil);
            0
        }
        Err(error) => {
            (*l).stack.truncate(size as usize);
            error_status(l, error)
        }
    }
}
/// Resets the coroutine `l` represents and clears its C-side stacks.
///
/// # Safety
///
/// `l` must be a live thread `lua_State` whose wrapper survives the call.
/// The reset may run `__close` metamethods and other Lua code.
unsafe fn reset_thread(l: *mut lua_State) -> c_int {
    let result = vm(l).reset_host_coroutine((*l).thread);
    (*l).stack.clear();
    (*l).pending_continuation = None;
    (*l).close_slots.clear();
    (*l).entry_pending = false;
    match result {
        Ok(values) if values.first().is_some_and(Value::is_truthy) => {
            (*l).status = 0;
            0
        }
        Ok(values) => {
            push(l, values.get(1).cloned().unwrap_or(Value::Nil));
            (*l).status = 2;
            2
        }
        Err(error) => {
            push(l, error_value(l, error));
            (*l).status = 2;
            2
        }
    }
}
/// Resets a coroutine, like `lua_resetthread`.
///
/// # Safety
///
/// `l` must be a live thread `lua_State` of a coroutine, not an independent state.
/// The call may run `__close` metamethods on its thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_resetthread(l: *mut lua_State) -> c_int {
    reset_thread(l)
}
/// Closes a thread by resetting it, like 5.5's `lua_closethread`.
///
/// # Safety
///
/// `l` must be a live thread `lua_State`, and `_from` is ignored and may be null.
/// The call may run `__close` metamethods.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_closethread(l: *mut lua_State, _from: *mut lua_State) -> c_int {
    reset_thread(l)
}
/// Writes the parameter count and vararg flag of `function` into `ar`.
///
/// # Safety
///
/// `l` must be a live `lua_State`, `ar` a caller-owned `lua_Debug`, and `function` a value of `l`'s VM.
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
unsafe fn function_parameters(l: *mut lua_State, function: Option<&Value>, ar: *mut lua_Debug) {
    let chunk = match function {
        Some(Value::Closure(handle)) => rt(l)
            .closure_snapshot(*handle)
            .ok()
            .map(|snapshot| snapshot.chunk()),
        _ => None,
    };
    (*ar).nparams = chunk
        .as_ref()
        .map_or(0, |chunk| chunk.parameter_count as u8);
    (*ar).isvararg = chunk.as_ref().map_or(1, |chunk| i8::from(chunk.is_vararg));
    #[cfg(feature = "lua55")]
    {
        (*ar).extraargs = 0;
    }
}
/// Pushes the active-line set of `function`, or nil.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value, and `function` must be null or a closure of its VM.
unsafe fn active_lines(l: *mut lua_State, function: Option<&Value>) {
    if let Some(Value::Closure(handle)) = function
        && let Ok(snapshot) = rt(l).closure_snapshot(*handle)
    {
        let table = rt(l).create_table().expect("active line table");
        for line in snapshot.chunk().debug.active_lines() {
            let _ = rt(l).table_set(
                table,
                &Value::Integer(i64::from(line)),
                Value::Boolean(true),
            );
        }
        push(l, Value::Table(table));
        return;
    }
    push(l, Value::Nil);
}
/// Fills `ar` with the information `options` selects about `function`.
///
/// # Safety
///
/// `l` must be a live `lua_State`, `ar` a caller-owned `lua_Debug`, and `function` a function value of `l`'s VM.
/// `options` must be a supported `lua_getinfo` option string.
unsafe fn function_info(
    l: *mut lua_State,
    function: Value,
    options: &[u8],
    ar: *mut lua_Debug,
) -> c_int {
    let snapshot = match function {
        Value::Closure(handle) => rt(l).closure_snapshot(handle).ok(),
        _ => None,
    };
    let chunk = snapshot.as_ref().map(|snapshot| snapshot.chunk());
    if !matches!(
        function,
        Value::Closure(_) | Value::NativeFunction(_) | Value::BuiltinFunction(_)
    ) {
        return 0;
    }
    for &option in options {
        match option {
            b'n' => {
                (*ar).name = ptr::null();
                (*ar).namewhat = c"".as_ptr();
            }
            b'S' => {
                let source = chunk.as_ref().map_or_else(
                    || LuaString::from("=[C]"),
                    |chunk| chunk.debug.source_name().clone(),
                );
                (*ar).source = intern(l, source.clone());
                #[cfg(any(feature = "lua54", feature = "lua55"))]
                {
                    (*ar).srclen = source.as_bytes().len();
                }
                (*ar).what = if chunk.is_some() {
                    c"Lua".as_ptr()
                } else {
                    c"C".as_ptr()
                };
                (*ar).linedefined = chunk
                    .as_ref()
                    .map_or(-1, |chunk| chunk.debug.definition_line() as c_int);
                (*ar).lastlinedefined = chunk
                    .as_ref()
                    .map_or(-1, |chunk| chunk.debug.last_definition_line() as c_int);
                (*ar).short_src = [0; 60];
                for (dest, source) in (*ar).short_src.iter_mut().take(59).zip(source.as_bytes()) {
                    *dest = *source as c_char;
                }
            }
            b'l' => (*ar).currentline = -1,
            b'u' => {
                (*ar).nups = snapshot.as_ref().map_or_else(
                    || {
                        shared(l)
                            .closures
                            .iter()
                            .find(|c| Runtime::raw_equal(&c.value, &function))
                            .map_or(0, |c| c.count)
                    },
                    |snapshot| snapshot.cells().len(),
                ) as _;
                #[cfg(any(
                    feature = "lua52",
                    feature = "lua53",
                    feature = "lua54",
                    feature = "lua55"
                ))]
                {
                    function_parameters(l, Some(&function), ar);
                }
            }
            b'f' => push(l, function.clone()),
            b'L' => active_lines(l, Some(&function)),
            #[cfg(any(
                feature = "lua52",
                feature = "lua53",
                feature = "lua54",
                feature = "lua55"
            ))]
            b't' => (*ar).istailcall = 0,
            #[cfg(any(feature = "lua54", feature = "lua55"))]
            b'r' => {
                (*ar).ftransfer = 0;
                (*ar).ntransfer = 0;
            }
            _ => return 0,
        }
    }
    1
}
/// Returns an opaque identity for upvalue `n` of the function at `index`.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `n` positive.
/// The result is only for comparison: it must never be dereferenced or freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_upvalueid(l: *mut lua_State, index: c_int, n: c_int) -> *mut c_void {
    if n <= 0 {
        return ptr::null_mut();
    }
    match value(l, index) {
        Some(Value::Closure(handle)) => rt(l)
            .closure_cell(handle, n as usize - 1)
            .ok()
            .flatten()
            .map_or(ptr::null_mut(), |cell| {
                ((u64::from(cell.index()) + 1) * 16 + 3) as usize as *mut c_void
            }),
        Some(target) => shared(l)
            .closures
            .iter()
            .find(|c| Runtime::raw_equal(&c.value, &target))
            .filter(|c| n as usize <= c.count)
            .map_or(ptr::null_mut(), |c| {
                ((c.upvalues.native_identity() as usize)
                    .wrapping_mul(256)
                    .wrapping_add(n as usize)) as *mut c_void
            }),
        _ => ptr::null_mut(),
    }
}
/// Joins upvalue `an` of `a` to upvalue `bn` of `b` through the debug library.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose values at `a` and `b` are functions of its VM.
/// The join runs `debug.upvaluejoin`, which may execute Lua code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_upvaluejoin(
    l: *mut lua_State,
    a: c_int,
    an: c_int,
    b: c_int,
    bn: c_int,
) {
    use oximond::runtime::BuiltinFunction;
    let args = vec![
        value(l, a).unwrap_or(Value::Nil),
        Value::Integer(i64::from(an)),
        value(l, b).unwrap_or(Value::Nil),
        Value::Integer(i64::from(bn)),
    ];
    let _ = call(
        l,
        Value::BuiltinFunction(BuiltinFunction::DebugUpvalueJoin),
        args,
    );
}
type Hook = unsafe extern "C" fn(*mut lua_State, *mut lua_Debug);
#[cfg(any(feature = "lua54", feature = "lua55"))]
type WarnFunction = unsafe extern "C" fn(*mut c_void, *const c_char, c_int);
#[cfg(not(feature = "lazy-bridge"))]
unsafe extern "C" {
    fn oximond_invoke_hook(l: *mut lua_State, hook: Hook, debug: *mut lua_Debug) -> c_int;
    #[cfg(any(feature = "lua54", feature = "lua55"))]
    fn oximond_invoke_warning(
        l: *mut lua_State,
        function: WarnFunction,
        data: *mut c_void,
        message: *const c_char,
        continued: c_int,
    ) -> c_int;
}
fn hook_callback(vm_ref: &mut Vm, token: u64, args: &[Value]) -> Result<Vec<Value>, VmError> {
    unsafe {
        let main = token as *mut lua_State;
        let old_vm = (*main).vm;
        (*main).vm = vm_ref;
        let l = vm_ref
            .current_coroutine()
            .map_or(main, |thread| thread_state(main, thread));
        if l.is_null() {
            (*main).vm = old_vm;
            return Err(RuntimeError::memory_error().into());
        }
        let Some(hook) = (*l).hook else {
            (*main).vm = old_vm;
            return Ok(Vec::new());
        };
        let old = std::mem::take(&mut (*l).stack);
        (*l).saved_stacks.push(old);
        let mut debug: lua_Debug = std::mem::zeroed();
        debug.event = match args.first() {
            Some(Value::String(s)) => match s.as_bytes() {
                b"call" => 0,
                b"return" => 1,
                b"line" => 2,
                b"count" => 3,
                b"tail call" => 4,
                _ => 0,
            },
            _ => 0,
        };
        debug.currentline = args
            .get(1)
            .and_then(|v| Runtime::coerce_number(v, "hook").ok())
            .map_or(-1, |n| n as c_int);
        #[cfg(any(
            feature = "lua52",
            feature = "lua53",
            feature = "lua54",
            feature = "lua55"
        ))]
        {
            debug.i_ci = 2usize as *mut c_void;
        }
        #[cfg(not(any(
            feature = "lua52",
            feature = "lua53",
            feature = "lua54",
            feature = "lua55"
        )))]
        {
            debug.i_ci = 1;
        }
        let old_hook = std::mem::replace(&mut (*l).in_hook, true);
        let status = oximond_invoke_hook(l, hook, &mut debug);
        (*l).in_hook = old_hook;
        let result = if status == 0 {
            Ok(Vec::new())
        } else if status == 1 {
            Err(vm_ref.request_host_yield(Vec::new()))
        } else {
            let error = pop(l);
            Err(if status == 4 {
                RuntimeError::memory_error().into()
            } else {
                RuntimeError::Raised(error).into()
            })
        };
        (*l).stack = (*l).saved_stacks.pop().unwrap();
        sync_roots(l);
        (*main).vm = old_vm;
        result
    }
}
/// Installs the debug hook `hook` with `mask` and `count` on `l`'s thread.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `hook` must be null or a `Hook` upholding the hook callback contract while installed.
/// Hooks may fire during any VM execution on the state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_sethook(
    l: *mut lua_State,
    hook: Option<Hook>,
    mask: c_int,
    count: c_int,
) -> HookReturn {
    let callback = if hook.is_some() {
        let owner = shared(l)
            .owner
            .upgrade()
            .map(|owner| -> Rc<dyn std::any::Any> { owner });
        match vm(l).register_owned_host_function(
            hook_callback,
            main_state(l) as u64,
            Vec::new(),
            owner,
        ) {
            Ok(callback) => Some(callback),
            Err(_) => return hook_return(0),
        }
    } else {
        None
    };
    let thread = (l != main_state(l)).then_some((*l).thread);
    if vm(l)
        .set_host_debug_hook(thread, callback.clone(), mask, count)
        .is_err()
    {
        if let Some(callback) = callback {
            vm(l).unregister_host_function(callback);
        }
        return hook_return(0);
    }
    if let Some(previous) = (*l).hook_value.take() {
        vm(l).unregister_host_function(previous);
    }
    (*l).hook = hook;
    (*l).hook_mask = if hook.is_some() { mask } else { 0 };
    (*l).hook_count = if hook.is_some() { count } else { 0 };
    (*l).hook_value = callback;
    hook_return(1)
}
/// Returns the hook installed on `l`.
///
/// # Safety
///
/// `l` must be a live `lua_State`.
/// The result is only valid while the hook stays installed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_gethook(l: *mut lua_State) -> Option<Hook> {
    (*l).hook
}
/// Returns the hook event mask installed on `l`.
///
/// # Safety
///
/// `l` must be a live `lua_State` used on its owning thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_gethookmask(l: *mut lua_State) -> c_int {
    (*l).hook_mask
}
/// Returns the hook instruction count installed on `l`.
///
/// # Safety
///
/// `l` must be a live `lua_State` used on its owning thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_gethookcount(l: *mut lua_State) -> c_int {
    (*l).hook_count
}
/// Installs the host warning function `function`, or turns warnings off when it is null.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `function` must uphold the warning callback contract and stay callable while installed.
/// `data` must remain valid for every warning delivered.
#[cfg(any(feature = "lua54", feature = "lua55"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_setwarnf(
    l: *mut lua_State,
    function: Option<WarnFunction>,
    data: *mut c_void,
) {
    let main = main_state(l);
    let weak = shared(l).owner.clone();
    let attached = (*main)._owned_vm.is_none();
    let Some(function) = function else {
        // A NULL warnf turns warnings off, as in stock Lua.
        rt(l).set_warning_handler(Some(discard_warnings()));
        return;
    };
    let handler = {
        Box::new(
            move |bytes: &[u8], continued: bool| -> Result<(), RuntimeError> {
                let _owner = weak.upgrade();
                if attached && _owner.is_none() {
                    return Ok(());
                }
                let mut message = bytes.to_vec();
                message.push(0);
                let status = oximond_invoke_warning(
                    main,
                    function,
                    data,
                    message.as_ptr().cast(),
                    i32::from(continued),
                );
                if status == 0 {
                    Ok(())
                } else {
                    Err(RuntimeError::message("error in host warning callback"))
                }
            },
        ) as Box<dyn FnMut(&[u8], bool) -> Result<(), RuntimeError>>
    };
    rt(l).set_warning_handler(Some(handler));
}
/// Emits a warning through the state's warning handler.
///
/// # Safety
///
/// `l` must be a live `lua_State`, and `message` a live NUL-terminated string.
/// The handler installed by `lua_setwarnf` may run host code.
#[cfg(any(feature = "lua54", feature = "lua55"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_warning(
    l: *mut lua_State,
    message: *const c_char,
    continued: c_int,
) -> c_int {
    match rt(l).emit_warning(cstr(message), continued != 0) {
        Ok(()) => 0,
        Err(error) => failure(l, error),
    }
}
/// Formats the number at `index` into `buffer` and returns the bytes written.
///
/// # Safety
///
/// `l` must be a live `lua_State` and `index` one of its stack indices.
/// `buffer` must be writable for the formatted number plus its NUL terminator, at least `LUAI_MAXNUMBER2STRING` bytes as in stock Lua.
#[cfg(feature = "lua55")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_numbertocstring(
    l: *mut lua_State,
    index: c_int,
    buffer: *mut c_char,
) -> u32 {
    let Some(value @ (Value::Integer(_) | Value::Number(_))) = value(l, index) else {
        return 0;
    };
    let Ok(text) = rt(l).format_value(&value) else {
        return 0;
    };
    ptr::copy_nonoverlapping(
        text.as_bytes().as_ptr().cast(),
        buffer,
        text.as_bytes().len(),
    );
    *buffer.add(text.as_bytes().len()) = 0;
    text.as_bytes().len() as u32 + 1
}
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
fn close_native(vm_ref: &mut Vm, token: u64, args: &[Value]) -> Result<Vec<Value>, VmError> {
    unsafe {
        let record = &*(token as *const Continuation);
        let l = record.state;
        let _owner = shared(l).owner.upgrade();
        if let Some(value) = &record.value {
            vm_ref.unregister_host_function(value.clone());
        }
        if let Some(value) = &record.close_handler {
            vm_ref.unregister_host_function(value.clone());
        }
        let main = main_state(l);
        let old_vm = (*main).vm;
        (*main).vm = vm_ref;
        let saved = std::mem::replace(&mut (*l).stack, record.stack.clone());
        (*l).saved_stacks.push(saved);
        let current = std::mem::replace(&mut (*l).current, record.current);
        let slots = std::mem::replace(&mut (*l).close_slots, record.close_slots.clone());
        sync_roots(l);
        let result = close_slots(l, 0, args.first().cloned().unwrap_or(Value::Nil));
        (*l).stack = (*l).saved_stacks.pop().unwrap();
        (*l).current = current;
        (*l).close_slots = slots;
        sync_roots(l);
        shared(l)
            .continuations
            .retain(|record| ptr::from_ref(record.as_ref()) as u64 != token);
        (*main).vm = old_vm;
        result.map(|()| Vec::new())
    }
}

/// Keeps the C stack limit at its default, as stock Lua's query does.
///
/// # Safety
///
/// Both arguments are ignored and the state may be null, so no precondition applies.
#[cfg(feature = "lua54")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_setcstacklimit(_l: *mut lua_State, _limit: u32) -> c_int {
    200
}

#[cfg(all(test, feature = "bundled-bridge", not(feature = "lazy-bridge")))]
mod versioned_tests;
/// Decodes the activation level stored in a `lua_Debug`'s `i_ci` field.
///
/// # Safety
///
/// `ar` must point to a `lua_Debug` filled by `lua_getstack`, `lua_getinfo`, or this crate's hook callback.
unsafe fn debug_level(ar: *const lua_Debug) -> usize {
    if MODERN {
        ((*ar).i_ci as usize).wrapping_sub(1)
    } else {
        (*ar).i_ci as usize
    }
}
/// Reports whether `l`'s coroutine is suspended while another coroutine runs.
///
/// # Safety
///
/// `l` must be a live `lua_State` of a family whose `vm` pointer is valid.
unsafe fn is_parked(l: *mut lua_State) -> bool {
    l != main_state(l) && vm(l).current_coroutine() != Some((*l).thread)
}
/// Pushes local `n` of the activation `ar` describes, or of the function on the stack when `ar` is null.
///
/// # Safety
///
/// `l` must be a live `lua_State` with room for one value.
/// `ar` must be null or a `lua_Debug` from `lua_getstack`, and `n` a positive local index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_getlocal(
    l: *mut lua_State,
    ar: *const lua_Debug,
    n: c_int,
) -> *const c_char {
    if ar.is_null() {
        let function = value(l, -1).unwrap_or(Value::Nil);
        return vm(l)
            .function_local_name(function, n)
            .ok()
            .flatten()
            .map_or(ptr::null(), |name| intern(l, name));
    }
    let level = debug_level(ar);
    if level == 0 && (*l).current.is_some() {
        if n > 0 && n as usize <= (*l).stack.len() {
            let value = state(l).stack[n as usize - 1].clone();
            push(l, value);
            return if cfg!(any(feature = "lua54", feature = "lua55")) {
                c"(C temporary)".as_ptr()
            } else {
                c"(*temporary)".as_ptr()
            };
        }
        return ptr::null();
    }
    let result = if is_parked(l) {
        vm(l).coroutine_get_local((*l).thread, level, n)
    } else {
        vm(l).host_get_local(level, n)
    };
    match result {
        Ok(Some((name, value))) => {
            push(l, value);
            intern(l, name)
        }
        _ => ptr::null(),
    }
}
/// Pops a value into local `n` of the activation `ar` describes.
///
/// # Safety
///
/// `l` must be a live `lua_State` with at least one value.
/// `ar` must be null — which only pops — or a `lua_Debug` from `lua_getstack`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_setlocal(
    l: *mut lua_State,
    ar: *const lua_Debug,
    n: c_int,
) -> *const c_char {
    let value = pop(l);
    if ar.is_null() {
        return ptr::null();
    }
    let level = debug_level(ar);
    if level == 0 && (*l).current.is_some() {
        if n > 0 && n as usize <= (*l).stack.len() {
            state(l).stack[n as usize - 1] = value;
            return if cfg!(any(feature = "lua54", feature = "lua55")) {
                c"(C temporary)".as_ptr()
            } else {
                c"(*temporary)".as_ptr()
            };
        }
        return ptr::null();
    }
    let result = if is_parked(l) {
        vm(l).coroutine_set_local((*l).thread, level, n, value)
    } else {
        vm(l).host_set_local(level, n, value)
    };
    result
        .ok()
        .flatten()
        .map_or(ptr::null(), |name| intern(l, name))
}

/// Describes activation `level` of `l`'s coroutine, filling in the running C frame.
///
/// # Safety
///
/// `l` must be a live `lua_State` of a family whose `vm` pointer is valid.
/// When level 0 is a running C callback, `(*l).current` must name a live closure entry.
unsafe fn stack_info(l: *mut lua_State, level: usize) -> Option<oximond::vm::VmStackFrame> {
    if is_parked(l) {
        vm(l).coroutine_stack_info((*l).thread, level)
    } else {
        let mut info = vm(l).host_stack_info(level)?;
        if level == 0
            && info.what == "C"
            && let Some(index) = (*l).current
        {
            let closure = &shared(l).closures[index];
            info.function = Some(closure.value.clone());
            info.upvalue_count = closure.count;
        }
        Some(info)
    }
}
const LIBRARIES: &[&str] = &[
    "package",
    "coroutine",
    "table",
    "io",
    "os",
    "string",
    "math",
    "utf8",
    "debug",
];
/// Collects the key-value pairs of `table` by raw iteration.
///
/// # Safety
///
/// `l` must be a live `lua_State` of the VM owning `table`, and `table` must be one of its table handles.
/// The table must not be modified during the iteration.
unsafe fn table_snapshot(l: *mut lua_State, table: TableHandle) -> Vec<(Value, Value)> {
    let mut entries = Vec::new();
    let mut key = Value::Nil;
    while let Ok((next, value)) = rt(l).table_next(table, &key) {
        if matches!(next, Value::Nil) {
            break;
        }
        key = next.clone();
        entries.push((next, value));
    }
    entries
}
/// Stashes the standard libraries in the registry so they can be opened lazily.
///
/// # Safety
///
/// `l` must be a freshly created main `lua_State` whose globals are still the initial set.
unsafe fn cold_libraries(l: *mut lua_State) {
    let registry = rt(l).registry();
    let globals = rt(l).global_environment();
    let stash = rt(l).create_table().expect("native library cache");
    let base = rt(l).create_table().expect("native base cache");
    rt(l)
        .table_set(
            registry,
            &Value::from("oximond.capi.libraries"),
            Value::Table(stash),
        )
        .expect("root library cache");
    rt(l)
        .table_set(stash, &Value::from("base"), Value::Table(base))
        .expect("cache base");
    if let Ok(Some(metatable)) = rt(l).raw_metatable(&Value::from("")) {
        rt(l)
            .table_set(
                stash,
                &Value::from("string_metatable"),
                Value::Table(metatable),
            )
            .expect("cache string metatable");
        rt(l)
            .raw_set_metatable(&Value::from(""), None)
            .expect("clear string metatable");
    }
    for (key, value) in table_snapshot(l, globals) {
        let library = matches!(&key,Value::String(name)if LIBRARIES.iter().any(|library|name.as_bytes()==library.as_bytes()));
        rt(l)
            .table_set(if library { stash } else { base }, &key, value)
            .expect("cache global");
        rt(l)
            .table_set(globals, &key, Value::Nil)
            .expect("clear global");
    }
    if let Ok(Value::Table(loaded)) = rt(l).table_get(registry, &Value::from("_LOADED")) {
        for library in LIBRARIES.iter().copied().chain(["_G"]) {
            rt(l)
                .table_set(loaded, &Value::from(library), Value::Nil)
                .expect("clear loaded library");
        }
    }
}
/// Pushes the cached `coroutine` library, as `luaopen_coroutine`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[cfg(any(
    feature = "lua52",
    feature = "lua53",
    feature = "lua54",
    feature = "lua55"
))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_coroutine(l: *mut lua_State) -> c_int {
    open_library(l, "coroutine")
}
/// Pushes the cached `utf8` library, as `luaopen_utf8`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[cfg(any(feature = "lua53", feature = "lua54", feature = "lua55"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_utf8(l: *mut lua_State) -> c_int {
    open_library(l, "utf8")
}

/// Pushes the cached `bit32` library, as `luaopen_bit32`.
///
/// # Safety
///
/// `l` must be a live `lua_State` whose library cache was seeded by `cold_libraries`.
#[cfg(feature = "lua52")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaopen_bit32(l: *mut lua_State) -> c_int {
    open_library(l, "bit32")
}
