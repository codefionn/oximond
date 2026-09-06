use super::*;
use oximond::runtime::BuiltinFunction;

#[test]
fn canonical_string_cache_keys_keep_their_allocation_identity_until_aliases_die() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        let first = LuaString::from("shared native bytes");
        let alias = LuaString::from("shared native bytes");
        let pointer = intern(l, first.clone());
        assert_eq!(pointer, intern(l, alias.clone()));
        drop(first);
        assert_eq!(oxi_gc(l, 2, 0, 0, 0), 0);
        for (identity, entry) in &shared(l).strings {
            assert!(
                entry.references.contains_key(identity),
                "a cache key must own a weak reference that prevents allocation-address reuse"
            );
        }
        assert_eq!(CStr::from_ptr(pointer).to_bytes(), alias.as_bytes());
        drop(alias);
        assert_eq!(oxi_gc(l, 2, 0, 0, 0), 0);
        assert!(shared(l).strings.is_empty());
        lua_close(l);
    }
}

#[test]
fn extra_space_is_owned_and_inherited_by_threads() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        assert!(!l.is_null());
        assert_eq!(
            std::mem::offset_of!(ExtraState, state),
            std::mem::size_of::<*mut c_void>()
        );
        (*extra_state(l)).extraspace = 123usize.to_ne_bytes();
        let mut child = ptr::null_mut();
        assert_eq!(oxi_newthread(l, &mut child), 0);
        assert_eq!((*extra_state(child)).extraspace, 123usize.to_ne_bytes());
        lua_close(l);
    }
}

#[test]
fn public_registry_globals_and_main_thread_match_the_selected_abi() {
    if !MODERN {
        return;
    }
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        let registry = raw_table(l, REGISTRY).unwrap();
        assert_eq!(
            rt(l).table_get(registry, &Value::Integer(2)).unwrap(),
            Value::Table(rt(l).global_environment())
        );
        assert_eq!(
            rt(l)
                .table_get(
                    registry,
                    &Value::Integer(if cfg!(feature = "lua55") { 3 } else { 1 })
                )
                .unwrap(),
            Value::Thread((*l).thread)
        );
        lua_close(l);
    }
}

#[test]
fn modern_userdata_values_are_traced_and_invalid_slots_consume_the_value() {
    if !MODERN {
        return;
    }
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        lua_newuserdatauv(l, 3, 2);
        lua_createtable(l, 0, 0);
        let table = value(l, -1).unwrap();
        assert_eq!(lua_setiuservalue(l, 1, 2), 1);
        sync_roots(l);
        vm(l).collect_full_with_values(&[]);
        assert_eq!(lua_getiuservalue(l, 1, 2), 5);
        assert_eq!(pop(l), table);
        lua_pushinteger(l, 9);
        assert_eq!(lua_setiuservalue(l, 1, 3), 0);
        assert_eq!(lua_gettop(l), 1);
        assert_eq!(lua_getiuservalue(l, 1, 3), -1);
        assert_eq!(pop(l), Value::Nil);
        lua_close(l);
    }
}

#[test]
fn debug_uservalue_and_c_api_share_the_same_slot() {
    if !MODERN {
        return;
    }
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        lua_newuserdata(l, 1);
        let userdata = value(l, -1).unwrap();
        let stored = if PROFILE == CompatibilityProfile::Lua52 {
            lua_createtable(l, 0, 0);
            value(l, -1).unwrap()
        } else {
            Value::from("via debug")
        };
        let result = vm(l)
            .call_value(
                Value::BuiltinFunction(BuiltinFunction::DebugSetUserValue),
                vec![userdata.clone(), stored.clone()],
            )
            .unwrap();
        assert_eq!(result, vec![userdata.clone()]);
        assert_eq!(
            lua_getiuservalue(l, 1, 1),
            if matches!(stored, Value::Table(_)) {
                5
            } else {
                4
            }
        );
        assert_eq!(pop(l), stored);
        if PROFILE == CompatibilityProfile::Lua52 {
            pop(l);
        }
        lua_pushinteger(l, 37);
        assert_eq!(lua_setiuservalue(l, 1, 1), 1);
        let expected = if PROFILE == CompatibilityProfile::Lua52 {
            Value::Number(37.0)
        } else {
            Value::Integer(37)
        };
        assert_eq!(
            vm(l)
                .call_value(
                    Value::BuiltinFunction(BuiltinFunction::DebugGetUserValue),
                    vec![userdata]
                )
                .unwrap(),
            if PROFILE.capabilities().debug.multiple_uservalues {
                vec![expected, Value::Boolean(true)]
            } else {
                vec![expected]
            }
        );
        lua_close(l);
    }
}

#[test]
fn io_file_uservalue_and_c_api_share_the_same_slot() {
    if !matches!(
        PROFILE,
        CompatibilityProfile::Lua52 | CompatibilityProfile::Lua53
    ) {
        return;
    }
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        let file = Value::File(rt(l).create_temp_file().unwrap());
        push(l, file.clone());
        lua_createtable(l, 0, 0);
        let table = value(l, -1).unwrap();
        assert_eq!(lua_setiuservalue(l, 1, 1), 1);
        assert_eq!(
            vm(l)
                .call_value(
                    Value::BuiltinFunction(BuiltinFunction::DebugGetUserValue),
                    vec![file.clone()]
                )
                .unwrap(),
            vec![table]
        );
        assert_eq!(
            vm(l)
                .call_value(
                    Value::BuiltinFunction(BuiltinFunction::DebugSetUserValue),
                    vec![file, Value::Nil]
                )
                .unwrap(),
            vec![value(l, 1).unwrap()]
        );
        assert_eq!(lua_getiuservalue(l, 1, 1), 0);
        assert_eq!(pop(l), Value::Nil);
        lua_close(l);
    }
}

#[test]
fn integer_strings_retain_every_bit_and_fractional_conversion_matches_the_abi() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        lua_pushstring(l, c"9007199254740993".as_ptr());
        let mut valid = 0;
        let integer = lua_tointegerx(l, -1, &mut valid);
        assert_eq!(valid, 1);
        if MODERN_INTEGER {
            assert_eq!(integer, 9007199254740993);
        }
        pop(l);
        lua_pushnumber(l, 3.5);
        let integer = lua_tointegerx(l, -1, &mut valid);
        assert_eq!(
            (integer, valid),
            if MODERN_INTEGER { (0, 0) } else { (3, 1) }
        );
        pop(l);
        assert_eq!(lua_stringtonumber(l, c"0xFFFFFFFFFFFFFFFF".as_ptr()), 19);
        if MODERN_INTEGER {
            assert_eq!(pop(l), Value::Integer(-1));
        }
        lua_close(l);
    }
}

#[test]
fn modern_constructor_opens_string_metatable_only_when_requested() {
    if !MODERN {
        return;
    }
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        assert_eq!(rt(l).raw_metatable(&Value::from("abc")).unwrap(), None);
        assert_eq!(rt(l).get_global("string"), Value::Nil);
        assert_eq!(luaopen_string(l), 1);
        assert_eq!(lua_type(l, -1), 5);
        assert!(rt(l).raw_metatable(&Value::from("abc")).unwrap().is_some());
        assert_eq!(rt(l).get_global("math"), Value::Nil);
        lua_close(l);
    }
}

#[test]
fn protected_error_handler_failure_returns_the_selected_abis_error_code() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        luaopen_base(l);
        pop(l);
        let error = rt(l).get_global("error");
        push(l, error.clone());
        push(l, error);
        push(l, Value::from("initial error"));
        assert_eq!(oxi_pcall(l, 1, 0, 1), ERROR_HANDLER);
        assert_eq!(value(l, -1), Some(Value::from("error in error handling")));
        lua_close(l);
    }
}

#[test]
fn legacy_globals_replacement_changes_loading_and_keeps_child_environments_separate() {
    if MODERN {
        return;
    }
    unsafe extern "C" fn read_source(
        _: *mut lua_State,
        data: *mut c_void,
        len: *mut usize,
    ) -> *const c_char {
        let source = &mut *data.cast::<Option<&[u8]>>();
        if let Some(bytes) = source.take() {
            *len = bytes.len();
            bytes.as_ptr().cast()
        } else {
            ptr::null()
        }
    }
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        let replacement = rt(l).create_table().unwrap();
        rt(l)
            .table_set(replacement, &Value::from("marker"), Value::Number(47.0))
            .unwrap();
        push(l, Value::Table(replacement));
        lua_replace(l, GLOBALS);
        assert_eq!(value(l, GLOBALS), Some(Value::Table(replacement)));
        let mut source = Some(b"return marker".as_slice());
        assert_eq!(
            load_source(
                l,
                read_source,
                ptr::from_mut(&mut source).cast(),
                c"globals".as_ptr(),
                ptr::null()
            ),
            0
        );
        assert_eq!(oxi_pcall(l, 0, 1, 0), 0);
        assert_eq!(pop(l), Value::Number(47.0));
        let mut child = ptr::null_mut();
        assert_eq!(oxi_newthread(l, &mut child), 0);
        pop(l);
        let child_environment = rt(l).create_table().unwrap();
        rt(l)
            .table_set(
                child_environment,
                &Value::from("marker"),
                Value::Number(73.0),
            )
            .unwrap();
        push(child, Value::Table(child_environment));
        lua_replace(child, GLOBALS);
        assert_eq!(value(l, GLOBALS), Some(Value::Table(replacement)));
        let mut source = Some(b"return marker".as_slice());
        assert_eq!(
            load_source(
                child,
                read_source,
                ptr::from_mut(&mut source).cast(),
                c"child-globals".as_ptr(),
                ptr::null()
            ),
            0
        );
        assert_eq!(oxi_resume(child, 0), 0);
        assert_eq!(pop(child), Value::Number(73.0));
        lua_close(l);
    }
}

#[test]
fn custom_allocator_owns_state_storage_and_reports_constructor_and_thread_failures() {
    #[derive(Default)]
    struct AllocationCounts {
        reject: bool,
        allocated: usize,
        freed: usize,
    }
    unsafe extern "C" fn allocator(
        data: *mut c_void,
        pointer: *mut c_void,
        oldsize: usize,
        newsize: usize,
    ) -> *mut c_void {
        let counts = &mut *data.cast::<AllocationCounts>();
        if newsize > 0 && counts.reject {
            return ptr::null_mut();
        }
        if pointer.is_null() && newsize > 0 {
            counts.allocated += 1;
        } else if !pointer.is_null() && newsize == 0 {
            counts.freed += 1;
        }
        default_allocator(ptr::null_mut(), pointer, oldsize, newsize)
    }
    unsafe {
        let mut counts = AllocationCounts {
            reject: true,
            ..Default::default()
        };
        let data = ptr::from_mut(&mut counts).cast();
        assert!(new_state(Some(allocator), data).is_null());
        assert_eq!(counts.allocated, 0);
        counts.reject = false;
        let l = new_state(Some(allocator), data);
        assert!(!l.is_null());
        assert_eq!(counts.allocated, 1);
        counts.reject = true;
        let mut child = ptr::null_mut();
        assert_eq!(oxi_newthread(l, &mut child), 4);
        assert!(child.is_null());
        assert_eq!(pop(l), Value::from("not enough memory"));
        counts.reject = false;
        assert_eq!(oxi_newthread(l, &mut child), 0);
        assert!(!child.is_null());
        assert_eq!(counts.allocated, 2);
        lua_close(l);
        assert_eq!(counts.freed, counts.allocated);
    }
}

/// Hands `load_source` a whole source in one piece.
unsafe extern "C" fn read_whole(
    _: *mut lua_State,
    data: *mut c_void,
    len: *mut usize,
) -> *const c_char {
    let source = unsafe { &mut *data.cast::<Option<&[u8]>>() };
    if let Some(bytes) = source.take() {
        unsafe { *len = bytes.len() };
        bytes.as_ptr().cast()
    } else {
        ptr::null()
    }
}

/// A state of this build's ABI with the base library open, interpreting
/// (`execution` 0) or tracing (1).
unsafe fn state_in_mode(execution: c_int) -> *mut lua_State {
    let profile = match PROFILE {
        CompatibilityProfile::Lua51 => 51,
        CompatibilityProfile::Lua52 => 52,
        CompatibilityProfile::Lua53 => 53,
        CompatibilityProfile::Lua54 => 54,
        CompatibilityProfile::Lua55 => 55,
        CompatibilityProfile::LuaJit21 => 21,
    };
    unsafe {
        let l = oximond_newstate_with_options(
            Some(default_allocator),
            ptr::null_mut(),
            profile,
            execution,
        );
        assert!(!l.is_null());
        luaopen_base(l);
        assert_eq!(oxi_settop(l, 0), 0);
        l
    }
}

/// Loads and runs `source` as a chunk. Returns its first result.
unsafe fn run_chunk(l: *mut lua_State, source: &str) -> Value {
    let mut data = Some(source.as_bytes());
    unsafe {
        let loaded = load_source(
            l,
            read_whole,
            ptr::from_mut(&mut data).cast(),
            c"=chunk".as_ptr(),
            ptr::null(),
        );
        assert_eq!(loaded, 0, "{:?}", value(l, -1));
        let status = oxi_pcall(l, 0, 1, 0);
        let result = pop(l);
        assert_eq!(status, 0, "{result:?}");
        result
    }
}

/// Calls the global function `name` the way a host does, with `lua_pcall` on
/// an otherwise empty stack. Returns the status and the first result or the
/// error.
unsafe fn call_global(l: *mut lua_State, name: &CStr, arguments: &[Value]) -> (c_int, Value) {
    unsafe {
        assert_eq!(oxi_settop(l, 0), 0);
        oxi_getglobal(l, name.as_ptr());
        for argument in arguments {
            push(l, argument.clone());
        }
        let count = c_int::try_from(arguments.len()).unwrap();
        let status = oxi_pcall(l, count, 1, 0);
        let result = pop(l);
        assert_eq!(oxi_settop(l, 0), 0);
        (status, result)
    }
}

/// Trace entries that ran native code.
unsafe fn trace_runs(l: *mut lua_State) -> u64 {
    let trace = unsafe { vm(l) }.jit_statistics().trace;
    trace
        .native_entries
        .saturating_sub(trace.exits[oximond::vm::TraceExitKind::EntryMismatch.index()])
}

/// Calls `name` with `arguments` 64 times, and in a tracing state
/// (`execution` 1) on until a trace has run natively, for a bounded time.
/// Compiles finish on another thread, so the calls go on while they do. Every
/// call must succeed.
unsafe fn call_until_traced(l: *mut lua_State, execution: c_int, name: &CStr, arguments: &[Value]) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    for calls in 1.. {
        let (status, result) = unsafe { call_global(l, name, arguments) };
        assert_eq!(status, 0, "{result:?}");
        let traced = execution == 0 || unsafe { trace_runs(l) } > 0;
        if traced && calls >= 64 || std::time::Instant::now() > deadline {
            return;
        }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

/// A host may make a table read-only between two calls of a function whose
/// loop has a trace storing into it. The trace overwrites array slots in place
/// without asking the runtime, so its entry must refuse the table and leave the
/// store to the interpreter, which raises the error. The loop skips its first
/// store, which the interpreter runs before any trace entry.
#[test]
fn a_trace_does_not_store_into_a_table_the_host_made_read_only() {
    unsafe fn scenario(execution: c_int) -> (c_int, Value, Value, u64) {
        unsafe {
            let l = state_in_mode(execution);
            run_chunk(
                l,
                "t = {} for i = 1, 200 do t[i] = 0.5 end \
                 function fill(t, v) for i = 1, #t do if i > 1 then t[i] = v end end end",
            );
            let table = rt(l).get_global("t");
            call_until_traced(l, execution, c"fill", &[table.clone(), Value::Number(1.5)]);
            let traced = trace_runs(l);
            push(l, table.clone());
            assert_eq!(oxi_enablereadonlytable(l, -1, 1), 0);
            assert_eq!(oxi_settop(l, 0), 0);
            let (status, error) = call_global(l, c"fill", &[table.clone(), Value::Number(9.0)]);
            push(l, table);
            assert_eq!(oxi_enablereadonlytable(l, -1, 0), 0);
            assert_eq!(oxi_settop(l, 0), 0);
            let sum = run_chunk(l, "local s = 0 for i = 1, #t do s = s + t[i] end return s");
            lua_close(l);
            (status, error, sum, traced)
        }
    }
    let interpreted = unsafe { scenario(0) };
    let tracing = unsafe { scenario(1) };
    assert!(tracing.3 > 0, "the loop never ran a trace");
    assert_ne!(
        interpreted.0, 0,
        "the interpreter stored into a read-only table"
    );
    assert_eq!(
        (tracing.0, &tracing.1, &tracing.2),
        (interpreted.0, &interpreted.1, &interpreted.2)
    );
}

/// A host calls a Lua function again and again (an engine's step callback),
/// with `lua_pcall` on an empty stack, so the function's frame is the bottom
/// one. Its inner loops are short and lie side by side, so no tree covers the
/// loop around them, but over all the calls they run far longer than a compile
/// takes. Judging them by what is left of one call would reject them for good.
#[test]
fn short_loops_of_a_function_a_host_calls_repeatedly_get_traced() {
    unsafe {
        let l = state_in_mode(1);
        run_chunk(
            l,
            "t = {} for i = 1, 100 do t[i] = i * 0.25 end \
             function step(t) \
               local s = 0 \
               for j = 1, 4 do \
                 for i = 1, 100 do s = s + t[i] end \
                 for i = 1, 100 do s = s - t[i] * 0.5 end \
               end \
               return s \
             end",
        );
        let table = rt(l).get_global("t");
        call_until_traced(l, 1, c"step", std::slice::from_ref(&table));
        let (status, result) = call_global(l, c"step", &[table]);
        assert_eq!(status, 0);
        assert_eq!(result, Value::Number(2.0 * 0.25 * 5050.0));
        assert!(
            trace_runs(l) > 0,
            "the inner loops of a host-called function never ran a trace"
        );
        lua_close(l);
    }
}

/// Reads `t[key]` through `lua_rawgeti` on the table at stack index 1 and pops the result.
unsafe fn raw_geti(l: *mut lua_State, key: i64) -> Value {
    unsafe {
        lua_rawgeti(l, 1, key as RawIndex);
        pop(l)
    }
}

/// Stores `v` at `t[key]` through `lua_rawseti` on the table at stack index 1.
unsafe fn raw_seti(l: *mut lua_State, key: i64, v: Value) -> c_int {
    unsafe {
        push(l, v);
        lua_rawseti(l, 1, key as RawIndex)
    }
}

#[test]
fn rawseti_and_rawgeti_cover_array_hash_holes_growth_and_other_values() {
    unsafe {
        let l = state_in_mode(0);
        lua_createtable(l, 4, 0);
        // Array part, in range and appended one past the end.
        for k in 1..=10 {
            assert_eq!(raw_seti(l, k, Value::Number(k as f64 * 2.0)), 0);
        }
        for k in 1..=10 {
            assert_eq!(raw_geti(l, k), Value::Number(k as f64 * 2.0));
        }
        assert_eq!(lua_gettop(l), 1);
        // Overwrite inside the array, then a hole made by storing nil.
        assert_eq!(raw_seti(l, 3, Value::Number(-1.0)), 0);
        assert_eq!(raw_geti(l, 3), Value::Number(-1.0));
        assert_eq!(raw_seti(l, 5, Value::Nil), 0);
        assert_eq!(raw_geti(l, 5), Value::Nil);
        assert_eq!(raw_geti(l, 6), Value::Number(12.0));
        assert_eq!(raw_geti(l, 4), Value::Number(8.0));
        // Refill the hole.
        assert_eq!(raw_seti(l, 5, Value::Number(7.0)), 0);
        assert_eq!(raw_geti(l, 5), Value::Number(7.0));
        // Keys outside the array: a gap, zero, negative, and a far key.
        assert_eq!(raw_seti(l, 20, Value::Number(20.0)), 0);
        assert_eq!(raw_geti(l, 20), Value::Number(20.0));
        assert_eq!(raw_geti(l, 15), Value::Nil);
        assert_eq!(raw_seti(l, 0, Value::Number(0.5)), 0);
        assert_eq!(raw_seti(l, -7, Value::Number(-7.5)), 0);
        assert_eq!(raw_geti(l, 0), Value::Number(0.5));
        assert_eq!(raw_geti(l, -7), Value::Number(-7.5));
        assert_eq!(raw_seti(l, 1 << 40, Value::Number(1.0)), 0);
        assert_eq!(raw_geti(l, 1 << 40), Value::Number(1.0));
        // Fill the gap so the hash entries migrate; every key still reads back.
        for k in 11..=19 {
            assert_eq!(raw_seti(l, k, Value::Number(k as f64)), 0);
        }
        for k in 11..=20 {
            assert_eq!(raw_geti(l, k), Value::Number(k as f64));
        }
        // Growth far past the presized array.
        for k in 21..=2000 {
            assert_eq!(raw_seti(l, k, Value::Number(k as f64)), 0);
        }
        assert_eq!(raw_geti(l, 2000), Value::Number(2000.0));
        assert_eq!(raw_geti(l, 2001), Value::Nil);
        assert_eq!(lua_objlen(l, 1), 2000);
        // Non-number values, including collectable ones.
        let text = Value::from("text");
        assert_eq!(raw_seti(l, 1, text.clone()), 0);
        assert_eq!(raw_geti(l, 1), text);
        let inner = rt(l).create_table().unwrap();
        assert_eq!(raw_seti(l, 2, Value::Table(inner)), 0);
        assert_eq!(raw_geti(l, 2), Value::Table(inner));
        assert_eq!(raw_seti(l, 2001, Value::Boolean(true)), 0);
        assert_eq!(raw_geti(l, 2001), Value::Boolean(true));
        assert_eq!(lua_gettop(l), 1);
        lua_close(l);
    }
}

#[test]
fn rawseti_and_rawgeti_ignore_metamethods_and_reject_non_tables() {
    unsafe {
        let l = state_in_mode(0);
        run_chunk(
            l,
            "calls = 0 \
             t = setmetatable({}, { \
               __index = function() calls = calls + 1 return 99 end, \
               __newindex = function() calls = calls + 1 end }) \
             return 0",
        );
        assert_eq!(oxi_settop(l, 0), 0);
        push(l, rt(l).get_global("t"));
        assert_eq!(raw_geti(l, 1), Value::Nil);
        assert_eq!(raw_seti(l, 1, Value::Number(5.0)), 0);
        assert_eq!(raw_geti(l, 1), Value::Number(5.0));
        let zero = rt(l).integer_value(0);
        assert_eq!(rt(l).get_global("calls"), zero);
        // A non-table target reports an error and leaves the error message on top.
        assert_eq!(oxi_settop(l, 0), 0);
        push(l, Value::Number(1.0));
        assert_eq!(raw_geti(l, 1), Value::Nil);
        assert_eq!(lua_gettop(l), 1);
        assert_eq!(raw_seti(l, 1, Value::Number(1.0)), 2);
        assert_eq!(pop(l), Value::from("table expected"));
        assert_eq!(lua_gettop(l), 1);
        lua_close(l);
    }
}

/// The shapes `lua_rawseti` stores in place (an overwrite of a scalar slot,
/// an append past a full array part with room to spare) and the ones it hands
/// to the general store: a table with a metatable, with a hash part, a read-only
/// table, and a string slot. Every store reads back as the generic path wrote it.
#[test]
fn rawseti_in_place_stores_match_the_general_store_in_every_table_shape() {
    unsafe {
        let l = state_in_mode(0);
        run_chunk(
            l,
            "calls = 0 \
             meta = setmetatable({1, 2, 3}, {__newindex = function() calls = calls + 1 end}) \
             mixed = {1, 2, 3, x = 'y'} \
             strings = {'a', 'b', 'c'} \
             return 0",
        );
        assert_eq!(oxi_settop(l, 0), 0);
        // A raw store ignores `__newindex`, in the array part and past it.
        push(l, rt(l).get_global("meta"));
        assert_eq!(raw_seti(l, 2, Value::Number(20.0)), 0);
        assert_eq!(raw_seti(l, 4, Value::Number(40.0)), 0);
        assert_eq!(raw_seti(l, 5, Value::Boolean(true)), 0);
        assert_eq!(raw_geti(l, 2), Value::Number(20.0));
        assert_eq!(raw_geti(l, 4), Value::Number(40.0));
        assert_eq!(raw_geti(l, 5), Value::Boolean(true));
        assert_eq!(lua_objlen(l, 1), 5);
        assert_eq!(rt(l).get_global("calls"), rt(l).integer_value(0));
        // Appends to a table with a hash part keep the hash entry and the length.
        assert_eq!(oxi_settop(l, 0), 0);
        push(l, rt(l).get_global("mixed"));
        for k in 4..=40 {
            assert_eq!(raw_seti(l, k, Value::Number(k as f64)), 0);
        }
        assert_eq!(lua_objlen(l, 1), 40);
        assert_eq!(raw_geti(l, 40), Value::Number(40.0));
        let mixed = raw_table(l, 1).unwrap();
        assert_eq!(
            rt(l).table_get(mixed, &Value::from("x")),
            Ok(Value::from("y"))
        );
        // A number over a string slot and a string over a number slot.
        assert_eq!(oxi_settop(l, 0), 0);
        push(l, rt(l).get_global("strings"));
        assert_eq!(raw_seti(l, 2, Value::Number(2.0)), 0);
        assert_eq!(raw_seti(l, 2, Value::from("again")), 0);
        assert_eq!(raw_seti(l, 3, Value::Integer(3)), 0);
        assert_eq!(raw_geti(l, 2), Value::from("again"));
        assert_eq!(raw_geti(l, 3), Value::Integer(3));
        // A read-only table refuses an overwrite and an append alike.
        lua_createtable(l, 4, 0);
        for k in 1..=3 {
            assert_eq!(lua_gettop(l), 2);
            lua_pushnumber(l, k as f64);
            assert_eq!(lua_rawseti(l, 2, k as RawIndex), 0);
        }
        assert_eq!(oxi_enablereadonlytable(l, 2, 1), 0);
        for k in [2, 4] {
            lua_pushnumber(l, 9.0);
            assert_eq!(lua_rawseti(l, 2, k as RawIndex), 2, "key {k}");
            assert!(matches!(pop(l), Value::String(_)));
            assert_eq!(lua_gettop(l), 2);
        }
        assert_eq!(oxi_enablereadonlytable(l, 2, 0), 0);
        lua_rawgeti(l, 2, 2);
        assert_eq!(pop(l), Value::Number(2.0));
        assert_eq!(lua_objlen(l, 2), 3);
        lua_close(l);
    }
}

/// A presized table filled by appends past its size hint, refilled in place
/// with scalars of every kind, and stored to at the keys around the array
/// part (zero, negative, one past the next append) that the in-place store
/// hands to the general one.
#[test]
fn rawseti_fills_refills_and_edge_keys_match_the_general_store() {
    unsafe {
        let l = state_in_mode(0);
        lua_createtable(l, 8, 0);
        for k in 1..=20 {
            lua_pushinteger(l, (k * 3) as LuaInteger);
            assert_eq!(lua_rawseti(l, -2, k as RawIndex), 0);
            assert_eq!(lua_gettop(l), 1);
        }
        assert_eq!(lua_objlen(l, 1), 20);
        for k in 1..=20 {
            assert_eq!(raw_geti(l, k), rt(l).integer_value(k * 3));
        }
        // Overwrites with each scalar kind, then a string over a scalar slot
        // and a scalar back over the string.
        assert_eq!(raw_seti(l, 1, Value::Number(1.5)), 0);
        assert_eq!(raw_seti(l, 2, Value::Boolean(false)), 0);
        assert_eq!(raw_seti(l, 3, Value::Integer(-7)), 0);
        assert_eq!(raw_seti(l, 4, Value::from("s")), 0);
        assert_eq!(raw_seti(l, 4, Value::Number(4.0)), 0);
        assert_eq!(raw_geti(l, 1), Value::Number(1.5));
        assert_eq!(raw_geti(l, 2), Value::Boolean(false));
        assert_eq!(raw_geti(l, 3), Value::Integer(-7));
        assert_eq!(raw_geti(l, 4), Value::Number(4.0));
        // Keys outside the array part.
        assert_eq!(raw_seti(l, 0, Value::Number(0.5)), 0);
        assert_eq!(raw_seti(l, -3, Value::Number(-3.0)), 0);
        assert_eq!(raw_seti(l, 22, Value::Number(22.0)), 0);
        assert_eq!(lua_objlen(l, 1), 20);
        assert_eq!(raw_geti(l, 0), Value::Number(0.5));
        assert_eq!(raw_geti(l, -3), Value::Number(-3.0));
        assert_eq!(raw_geti(l, 21), Value::Nil);
        assert_eq!(raw_geti(l, 22), Value::Number(22.0));
        // Filling the hole joins the run.
        assert_eq!(raw_seti(l, 21, Value::Number(21.0)), 0);
        assert_eq!(lua_objlen(l, 1), 22);
        assert_eq!(lua_gettop(l), 1);
        // Values read back through `lua_tointeger` and popped with `lua_settop`.
        lua_rawgeti(l, 1, 5);
        assert_eq!(lua_tointeger(l, -1), 15);
        assert_eq!(oxi_settop(l, -2), 0);
        assert_eq!(lua_gettop(l), 1);
        lua_close(l);
    }
}

#[test]
fn pushes_grow_the_stack_past_its_capacity_and_checkstack_keeps_its_contract() {
    unsafe {
        let l = state_in_mode(0);
        assert_eq!(lua_checkstack(l, 0), 1);
        assert_eq!(lua_checkstack(l, 100), 1);
        assert_eq!(lua_checkstack(l, -1), 0);
        assert_eq!(lua_checkstack(l, c_int::MIN), 0);
        assert_eq!(lua_gettop(l), 0);
        // Far more pushes than any reservation: every push path grows the stack.
        const COUNT: i64 = 20_000;
        for k in 0..COUNT {
            match k % 3 {
                0 => lua_pushnumber(l, k as f64 + 0.5),
                1 => lua_pushinteger(l, k as LuaInteger),
                _ => {
                    lua_pushboolean(l, c_int::from(k % 2 == 0));
                }
            }
        }
        assert_eq!(lua_gettop(l), COUNT as c_int);
        for k in 0..COUNT {
            let value = peek(l, k as c_int + 1).unwrap().clone();
            match k % 3 {
                0 => assert_eq!(value, Value::Number(k as f64 + 0.5)),
                1 => assert_eq!(value, rt(l).integer_value(k)),
                _ => assert_eq!(value, Value::Boolean(k % 2 == 0)),
            }
        }
        // A reservation made first is honoured and does not change the contents.
        assert_eq!(lua_checkstack(l, 50_000), 1);
        lua_pushnumber(l, -1.0);
        assert_eq!(lua_gettop(l), COUNT as c_int + 1);
        assert_eq!(lua_tonumber(l, -1), -1.0);
        assert_eq!(lua_tonumber(l, 1), 0.5);
        lua_close(l);
    }
}

#[test]
fn rawseti_pops_exactly_one_value_of_every_kind_and_settop_matches_its_contract() {
    unsafe {
        let l = state_in_mode(0);
        lua_createtable(l, 0, 0);
        let inner = rt(l).create_table().unwrap();
        let values = [
            Value::Number(1.5),
            rt(l).integer_value(7),
            Value::from("s"),
            Value::Table(inner),
            Value::Nil,
            Value::Boolean(false),
        ];
        for (k, v) in values.iter().enumerate() {
            let before = lua_gettop(l);
            assert_eq!(raw_seti(l, k as i64 + 1, v.clone()), 0);
            assert_eq!(lua_gettop(l), before);
        }
        for (k, v) in values.iter().enumerate() {
            assert_eq!(raw_geti(l, k as i64 + 1), *v);
        }
        // pushinteger then rawseti of a stored integer reads back under the number model.
        lua_pushinteger(l, 41);
        assert_eq!(lua_rawseti(l, 1, 10), 0);
        assert_eq!(raw_geti(l, 10), rt(l).integer_value(41));
        // Non-table targets, by stack index and by empty slot, pop the value and report an error.
        assert_eq!(oxi_settop(l, 0), 0);
        push(l, Value::from("not a table"));
        for v in [Value::Number(2.0), rt(l).integer_value(2), Value::from("x")] {
            push(l, v);
            assert_eq!(lua_rawseti(l, 1, 1), 2);
            assert_eq!(pop(l), Value::from("table expected"));
            assert_eq!(lua_gettop(l), 1);
        }
        // settop grows with nil, shrinks, accepts negative indices and rejects ones below the bottom.
        assert_eq!(oxi_settop(l, 4), 0);
        assert_eq!(lua_gettop(l), 4);
        assert_eq!(peek(l, 4), Some(&Value::Nil));
        assert_eq!(oxi_settop(l, -2), 0);
        assert_eq!(lua_gettop(l), 3);
        assert_eq!(oxi_settop(l, 1), 0);
        assert_eq!(lua_gettop(l), 1);
        assert_ne!(oxi_settop(l, -3), 0);
        assert_eq!(lua_gettop(l), 2);
        assert_eq!(oxi_settop(l, 0), 0);
        assert_eq!(lua_gettop(l), 0);
        // rawseti on an empty stack target keeps the stack sound.
        lua_pushnumber(l, 1.0);
        assert_eq!(lua_rawseti(l, 1, 1), 2);
        lua_close(l);
    }
}

#[test]
fn number_reads_and_pushes_match_the_generic_conversions() {
    unsafe {
        let l = state_in_mode(0);
        lua_pushnumber(l, 2.5);
        lua_pushinteger(l, 7);
        push(l, Value::from("0x10"));
        push(l, Value::from("nope"));
        push(l, Value::Nil);
        lua_createtable(l, 0, 0);
        assert_eq!(lua_tonumber(l, 1), 2.5);
        assert_eq!(lua_tonumber(l, 2), 7.0);
        assert_eq!(lua_tonumber(l, 3), 16.0);
        assert_eq!(lua_tonumber(l, 4), 0.0);
        assert_eq!(lua_tonumber(l, 5), 0.0);
        assert_eq!(lua_tonumber(l, 6), 0.0);
        assert_eq!(lua_tonumber(l, 99), 0.0);
        assert_eq!(lua_tonumber(l, -5), 7.0);
        assert_eq!(lua_isnumber(l, 1), 1);
        assert_eq!(lua_isnumber(l, 2), 1);
        assert_eq!(lua_isnumber(l, 3), 1);
        assert_eq!(lua_isnumber(l, 4), 0);
        assert_eq!(lua_isnumber(l, 5), 0);
        assert_eq!(lua_isnumber(l, 6), 0);
        assert_eq!(lua_isnumber(l, 99), 0);
        // Integer profiles (5.3+) reject a non-integral float; 5.1 and 5.2 truncate.
        assert_eq!(lua_tointeger(l, 1), if MODERN_INTEGER { 0 } else { 2 });
        assert_eq!(lua_tointeger(l, 2), 7);
        assert_eq!(lua_tointeger(l, 3), 16);
        assert_eq!(lua_tointeger(l, 4), 0);
        assert_eq!(lua_objlen(l, 6), 0);
        // lua_createtable's size hint changes capacity only, never length or contents.
        lua_createtable(l, 1000, 3);
        assert_eq!(lua_objlen(l, -1), 0);
        assert_eq!(raw_geti_at(l, -1, 1), Value::Nil);
        lua_close(l);
    }
}

unsafe fn raw_geti_at(l: *mut lua_State, index: c_int, key: i64) -> Value {
    unsafe {
        lua_rawgeti(l, index, key as RawIndex);
        pop(l)
    }
}

/// Micro benchmark for `perf stat`; run with `OXI_CAPI_BENCH=<calls>`:
/// `OXI_CAPI_BENCH=2000000 perf stat -e instructions:u <test binary> --ignored capi_array_micro_benchmark`.
/// Each call count is a fixed number of rawseti, rawgeti and tointeger calls over a
/// VoxelManip-shaped table (createtable with a size hint, then pushinteger and rawseti).
#[test]
#[ignore]
fn capi_array_micro_benchmark() {
    let n: i64 = std::env::var("OXI_CAPI_BENCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let phase = std::env::var("OXI_CAPI_PHASE").unwrap_or_default();
    unsafe {
        let l = state_in_mode(0);
        lua_createtable(l, n as c_int, 0);
        if phase != "get_only" {
            for k in 1..=n {
                lua_pushinteger(l, k as LuaInteger);
                lua_rawseti(l, 1, k as RawIndex);
            }
        } else {
            for k in 1..=n {
                push(l, Value::Number(k as f64));
                lua_rawseti(l, 1, k as RawIndex);
            }
        }
        if phase != "set_only" {
            let mut sum = 0;
            for k in 1..=n {
                lua_rawgeti(l, 1, k as RawIndex);
                sum += lua_tointeger(l, -1);
                oxi_settop(l, 1);
            }
            assert_eq!(sum as i64, n * (n + 1) / 2);
        }
        lua_close(l);
    }
}

#[test]
fn tointeger_fast_reads_agree_with_the_generic_conversion() {
    unsafe {
        let l = state_in_mode(0);
        let cases = [
            Value::Number(5.0),
            Value::Number(-3.0),
            Value::Number(2.5),
            Value::Number(f64::NAN),
            Value::Number(f64::INFINITY),
            Value::Number(1e300),
            Value::Number(9_223_372_036_854_775_808.0),
            Value::Number(-9_223_372_036_854_775_808.0),
            rt(l).integer_value(41),
            Value::from("12"),
            Value::from("1.5"),
            Value::from("x"),
            Value::Boolean(true),
            Value::Nil,
        ];
        for v in cases {
            push(l, v.clone());
            let mut ok = -1;
            let with_flag = lua_tointegerx(l, -1, &mut ok);
            let without_flag = lua_tointegerx(l, -1, ptr::null_mut());
            let plain = lua_tointeger(l, -1);
            assert_eq!(with_flag, without_flag, "{v:?}");
            if MODERN || matches!(v, Value::Number(n) if n.is_finite() && n.abs() < 1e18) {
                assert_eq!(with_flag, plain, "{v:?}");
            }
            if ok == 1 && !matches!(v, Value::Number(n) if n.abs() >= 1e18) {
                // A converted value must agree with the number reading in range.
                assert_eq!(with_flag as f64, lua_tonumber(l, -1).trunc(), "{v:?}");
            }
            pop(l);
        }
        push(l, Value::Number(7.0));
        let mut ok = 0;
        assert_eq!(lua_tointegerx(l, -1, &mut ok), 7);
        assert_eq!(ok, 1);
        // createtable keeps the hint without storing anything and survives a hostile one.
        lua_createtable(l, 1000, 4);
        assert_eq!(lua_rawlen(l, -1), 0);
        lua_createtable(l, -1, -1);
        lua_createtable(l, c_int::MAX, 0);
        lua_close(l);
    }
}

#[test]
fn pushinteger_follows_the_number_model_on_main_and_child_states() {
    unsafe {
        let l = state_in_mode(0);
        let mut child = ptr::null_mut();
        assert_eq!(oxi_newthread(l, &mut child), 0);
        assert!(!child.is_null());
        for state in [l, child] {
            for n in [0, 1, -7, 1 << 40, i64::MIN / 2] {
                lua_pushinteger(state, n as LuaInteger);
                assert_eq!(peek(state, -1), Some(&rt(state).integer_value(n)), "{n}");
                pop(state);
            }
        }
        lua_close(l);
    }
}

/// A fresh table holding `x = depth`, pushed on `l`'s stack only.
unsafe fn push_marked_table(l: *mut lua_State, depth: f64) -> TableHandle {
    unsafe {
        let table = rt(l).create_table().unwrap();
        rt(l)
            .table_set(table, &Value::from("x"), Value::Number(depth))
            .unwrap();
        push(l, Value::Table(table));
        table
    }
}

/// `hold(depth)`: holds a table on its own C stack only, the way Luanti's
/// `read_v3_aux` holds a position, while it calls Lua: `again(depth - 1)`,
/// which calls `hold` once more, or at the bottom `collect()`, which runs
/// full collections. The table must survive them although nothing roots the
/// C stacks at each call any more: the collector asks for them.
unsafe extern "C" fn hold_and_call(l: *mut lua_State) -> c_int {
    unsafe {
        let depth = lua_tonumber(l, 1);
        let table = push_marked_table(l, depth);
        if depth > 0.0 {
            oxi_getglobal(l, c"again".as_ptr());
            push(l, Value::Number(depth - 1.0));
            assert_eq!(oxi_call(l, 1, 1), 0);
        } else {
            oxi_getglobal(l, c"collect".as_ptr());
            assert_eq!(oxi_call(l, 0, 1), 0);
        }
        let below = lua_tonumber(l, -1);
        let mark = match rt(l).table_get(table, &Value::from("x")) {
            Ok(Value::Number(mark)) => mark,
            other => panic!("the held table did not survive: {other:?}"),
        };
        push(l, Value::Number(below + mark));
        1
    }
}

/// A C function's stack values stay alive through collections that Lua it
/// calls runs, at every depth of nested C and Lua calls, and the values on
/// the stack of a coroutine's state too. The C API builds these roots when
/// the collector is about to read them (see `refresh_host_roots`), not at
/// each call, so this checks the hook covers every state of the family.
#[test]
fn c_stack_values_survive_collections_run_by_the_lua_they_call() {
    unsafe fn scenario(execution: c_int) -> (Value, u64) {
        unsafe {
            let l = state_in_mode(execution);
            lua_pushcclosure(l, hold_and_call, 0);
            assert_eq!(oxi_setglobal(l, c"hold".as_ptr()), 0);
            // A coroutine's state holds a table on its stack only.
            let mut child = ptr::null_mut();
            assert_eq!(oxi_newthread(l, &mut child), 0);
            let thread = pop(l);
            rt(l).set_global("keep_thread", thread);
            let held = push_marked_table(child, 42.0);
            run_chunk(
                l,
                "function again(depth) return hold(depth) end \
                 function collect() \
                   local junk = {} for i = 1, 200 do junk[i] = {} end \
                   collectgarbage() collectgarbage() return 0 end \
                 function drive(n) \
                   local s = 0 \
                   for i = 1, n do s = s + hold(i % 3) end \
                   return s end",
            );
            let (status, result) = call_global(l, c"drive", &[Value::Number(120.0)]);
            assert_eq!(status, 0, "{result:?}");
            assert_eq!(
                rt(l).table_get(held, &Value::from("x")).unwrap(),
                Value::Number(42.0),
                "a coroutine state's stack lost its table"
            );
            let traced = trace_runs(l);
            lua_close(l);
            (result, traced)
        }
    }
    let interpreted = unsafe { scenario(0) };
    let tracing = unsafe { scenario(1) };
    assert_eq!(tracing.0, interpreted.0);
    // Each call adds depth + (depth - 1) + ... + 0 for depth = i % 3.
    let expected: f64 = (1..=120).map(|i| f64::from([0, 1, 3][i % 3])).sum();
    assert_eq!(interpreted.0, Value::Number(expected));
}

/// `vector(t)`: converts `t` through the Lua function `read_vector` kept in
/// a global, with `lua_call(L, 1, 3)`, the way Luanti's `read_v3_aux` does,
/// and returns the sum of the three results.
unsafe extern "C" fn read_through_lua(l: *mut lua_State) -> c_int {
    unsafe {
        oxi_getglobal(l, c"read_vector".as_ptr());
        lua_insert(l, -2);
        assert_eq!(oxi_call(l, 1, 3), 0);
        let sum = lua_tonumber(l, -3) + lua_tonumber(l, -2) + lua_tonumber(l, -1);
        assert_eq!(oxi_settop(l, -4), 0);
        push(l, Value::Number(sum));
        1
    }
}

/// A Lua loop calls a C function that calls back into Lua for every
/// element (Luanti's vector conversion), and the Lua it calls has a loop of
/// its own. Each host entry runs on a reused stack (`Vm::run`) and the
/// parked caller is handed back intact: the results match the
/// interpreter's, the called function's loop runs natively on the reused
/// stacks, and a vector whose fields change type midway, gains a metatable
/// or is swapped for another table leaves through the interpreter's paths
/// with the same results. (The loop that calls the C function stays
/// interpreted: a recording gives up at a call of a host function.)
#[test]
fn a_hot_loop_converting_through_a_c_function_matches_the_interpreter() {
    unsafe fn scenario(execution: c_int) -> (Value, u64) {
        unsafe {
            let l = state_in_mode(execution);
            lua_pushcclosure(l, read_through_lua, 0);
            assert_eq!(oxi_setglobal(l, c"vector".as_ptr()), 0);
            run_chunk(
                l,
                "function read_vector(v) \
                   local w = 0 for i = 1, #v do w = w + v[i] end \
                   return v.x + w, v.y, v.z end \
                 function drive(n) \
                   local v = {1, 2, 3, 4, 5, 6, 7, 8, x = 1, y = 2, z = 3} \
                   local s = 0 \
                   for i = 1, n do \
                     v.x = i \
                     if i == 700 then v.y = 2.5 end \
                     if i == 800 then setmetatable(v, {__index = function() return 0 end}) end \
                     if i == 900 then v = {0.5, x = i, y = -1, z = 4} end \
                     s = s + vector(v) \
                   end \
                   return s end",
            );
            // Compiles finish on another thread: call on until a trace has
            // run, for a bounded time.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut last = Value::Nil;
            for calls in 1.. {
                let (status, result) = call_global(l, c"drive", &[Value::Number(1000.0)]);
                assert_eq!(status, 0, "{result:?}");
                last = result;
                let traced = execution == 0 || trace_runs(l) > 0;
                if traced && calls >= 4 || std::time::Instant::now() > deadline {
                    break;
                }
            }
            let traced = trace_runs(l);
            lua_close(l);
            (last, traced)
        }
    }
    let interpreted = unsafe { scenario(0) };
    let tracing = unsafe { scenario(1) };
    assert_eq!(tracing.0, interpreted.0);
    let expected: f64 = (1..=1000)
        .map(|i| {
            let i = f64::from(i);
            match i {
                i if i < 700.0 => i + 36.0 + 5.0,
                i if i < 900.0 => i + 36.0 + 5.5,
                i => i + 0.5 + 3.0,
            }
        })
        .sum();
    assert_eq!(interpreted.0, Value::Number(expected));
    assert!(
        tracing.1 > 0,
        "the loop of the Lua function the C function calls never ran a trace"
    );
}

/// Drops values until `top` remain; the same on every ABI.
unsafe fn settop(l: *mut lua_State, top: c_int) {
    while lua_gettop(l) > top {
        pop(l);
    }
}

unsafe extern "C" fn answer_handler(l: *mut lua_State) -> c_int {
    lua_pushnumber(l, 42.0);
    1
}

static NEWINDEX_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

unsafe extern "C" fn count_newindex(_: *mut lua_State) -> c_int {
    NEWINDEX_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    0
}

/// Reads field `name` of the value at `index` as a number, leaving the stack as it was.
unsafe fn field_number(l: *mut lua_State, index: c_int, name: &CStr) -> Option<f64> {
    let top = lua_gettop(l);
    assert_eq!(oxi_getfield(l, index, name.as_ptr()), 0);
    assert_eq!(lua_gettop(l), top + 1);
    let mut isnum = 0;
    let number = lua_tonumberx(l, -1, &mut isnum);
    settop(l, top);
    (isnum != 0).then_some(number)
}

/// Stores `number` under `name` in the value at `index`, checking the stack effect.
unsafe fn set_number(l: *mut lua_State, index: c_int, name: &CStr, number: f64) {
    let top = lua_gettop(l);
    lua_pushnumber(l, number);
    assert_eq!(oxi_setfield(l, index, name.as_ptr()), 0);
    assert_eq!(lua_gettop(l), top);
}

#[test]
fn field_access_reads_writes_and_keeps_the_stack_balanced() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        lua_createtable(l, 0, 0);
        set_number(l, 1, c"x", 1.0);
        set_number(l, -2, c"y", 2.0);
        // Overwrites of the same type, then of another type.
        set_number(l, 1, c"x", 3.0);
        lua_pushstring(l, c"text".as_ptr());
        assert_eq!(oxi_setfield(l, 1, c"x".as_ptr()), 0);
        assert_eq!(field_number(l, 1, c"x"), None);
        assert_eq!(field_number(l, -1, c"y"), Some(2.0));
        assert_eq!(field_number(l, 1, c"missing"), None);
        // A negative table index counts from the top with the value pushed.
        lua_createtable(l, 0, 0);
        lua_pushnumber(l, 7.0);
        assert_eq!(oxi_setfield(l, -2, c"z".as_ptr()), 0);
        assert_eq!(lua_gettop(l), 2);
        assert_eq!(oxi_getfield(l, 2, c"z".as_ptr()), 0);
        assert_eq!(lua_tonumber(l, -1), 7.0);
        lua_close(l);
    }
}

#[test]
fn field_access_names_do_not_alias_by_address() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        lua_createtable(l, 0, 0);
        // One buffer holds different names in turn, so the cache sees one address.
        let mut buffer = *b"alpha\0";
        set_number(l, 1, CStr::from_ptr(buffer.as_ptr().cast()), 1.0);
        buffer = *b"omega\0";
        set_number(l, 1, CStr::from_ptr(buffer.as_ptr().cast()), 2.0);
        let mut short = *b"om\0\0\0\0";
        assert_eq!(
            field_number(l, 1, CStr::from_ptr(short.as_ptr().cast())),
            None
        );
        short = *b"omega\0";
        assert_eq!(
            field_number(l, 1, CStr::from_ptr(short.as_ptr().cast())),
            Some(2.0)
        );
        assert_eq!(field_number(l, 1, c"alpha"), Some(1.0));
        // A long name is not cached and still works.
        let long = c"a_field_name_that_is_longer_than_the_key_cache_keeps";
        set_number(l, 1, long, 5.0);
        assert_eq!(field_number(l, 1, long), Some(5.0));
        // A pushed string is equal to a table key made from the same text.
        lua_pushstring(l, c"alpha".as_ptr());
        assert_eq!(oxi_gettable(l, 1), 0);
        assert_eq!(lua_tonumber(l, -1), 1.0);
        lua_close(l);
    }
}

#[test]
fn field_access_follows_index_and_newindex() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        // base = {inherited = 5}; middle = setmetatable({}, {__index = base});
        // object = setmetatable({}, {__index = middle}); the stack is base, middle, object.
        lua_createtable(l, 0, 0);
        set_number(l, 1, c"inherited", 5.0);
        lua_createtable(l, 0, 0);
        lua_createtable(l, 0, 0);
        lua_pushvalue(l, 1);
        assert_eq!(oxi_setfield(l, 3, c"__index".as_ptr()), 0);
        assert_eq!(lua_setmetatable(l, 2), 0);
        lua_createtable(l, 0, 0);
        lua_createtable(l, 0, 0);
        lua_pushvalue(l, 2);
        assert_eq!(oxi_setfield(l, 4, c"__index".as_ptr()), 0);
        assert_eq!(lua_setmetatable(l, 3), 0);
        assert_eq!(lua_gettop(l), 3);
        assert_eq!(field_number(l, 3, c"inherited"), Some(5.0));
        assert_eq!(field_number(l, 3, c"nothing"), None);
        // A store goes to the object itself, not to the table it inherits from.
        set_number(l, 3, c"inherited", 6.0);
        assert_eq!(field_number(l, 3, c"inherited"), Some(6.0));
        assert_eq!(field_number(l, 1, c"inherited"), Some(5.0));
        settop(l, 0);

        // A function `__index` answers only for absent keys.
        lua_createtable(l, 0, 0);
        set_number(l, 1, c"own", 1.0);
        lua_createtable(l, 0, 0);
        lua_pushcclosure(l, answer_handler, 0);
        assert_eq!(oxi_setfield(l, 2, c"__index".as_ptr()), 0);
        assert_eq!(lua_setmetatable(l, 1), 0);
        assert_eq!(field_number(l, 1, c"own"), Some(1.0));
        assert_eq!(field_number(l, 1, c"absent"), Some(42.0));
        settop(l, 0);

        // `__newindex` runs for absent keys and never for present ones.
        lua_createtable(l, 0, 0);
        set_number(l, 1, c"present", 1.0);
        lua_createtable(l, 0, 0);
        lua_pushcclosure(l, count_newindex, 0);
        assert_eq!(oxi_setfield(l, 2, c"__newindex".as_ptr()), 0);
        assert_eq!(lua_setmetatable(l, 1), 0);
        let before = NEWINDEX_CALLS.load(std::sync::atomic::Ordering::SeqCst);
        set_number(l, 1, c"present", 2.0);
        assert_eq!(
            NEWINDEX_CALLS.load(std::sync::atomic::Ordering::SeqCst),
            before
        );
        assert_eq!(field_number(l, 1, c"present"), Some(2.0));
        set_number(l, 1, c"absent", 3.0);
        assert_eq!(
            NEWINDEX_CALLS.load(std::sync::atomic::Ordering::SeqCst),
            before + 1
        );
        assert_eq!(field_number(l, 1, c"absent"), None);
        lua_close(l);
    }
}

#[test]
fn field_access_reports_errors_and_reaches_pseudo_indices() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        // Indexing a number fails and leaves the message on top.
        lua_pushnumber(l, 1.0);
        assert_ne!(oxi_getfield(l, 1, c"x".as_ptr()), 0);
        assert_eq!(lua_gettop(l), 2);
        let message = CStr::from_ptr(lua_tolstring(l, -1, ptr::null_mut()));
        assert!(message.to_string_lossy().contains("index"));
        settop(l, 0);
        lua_pushnumber(l, 1.0);
        lua_pushnumber(l, 2.0);
        assert_ne!(oxi_setfield(l, 1, c"x".as_ptr()), 0);
        assert_eq!(lua_gettop(l), 2);
        settop(l, 0);
        // A null name is nil: reading it gives nil, storing under it fails.
        lua_createtable(l, 0, 0);
        assert_eq!(oxi_getfield(l, 1, ptr::null()), 0);
        assert!(matches!(value(l, -1), Some(Value::Nil)));
        settop(l, 1);
        lua_pushnumber(l, 1.0);
        assert_ne!(oxi_setfield(l, 1, ptr::null()), 0);
        settop(l, 0);
        // The registry is a pseudo-index.
        set_number(l, REGISTRY, c"oximond_test_key", 11.0);
        assert_eq!(field_number(l, REGISTRY, c"oximond_test_key"), Some(11.0));
        assert_eq!(lua_gettop(l), 0);
        lua_close(l);
    }
}

#[test]
fn gettable_reads_string_and_array_keys_in_place() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        lua_createtable(l, 0, 0);
        for n in 1..=3i32 {
            lua_pushnumber(l, f64::from(n) * 10.0);
            lua_rawseti(l, 1, n as _);
        }
        set_number(l, 1, c"k", 4.0);
        lua_pushnumber(l, 2.0);
        assert_eq!(oxi_gettable(l, 1), 0);
        assert_eq!((lua_gettop(l), lua_tonumber(l, -1)), (2, 20.0));
        settop(l, 1);
        lua_pushstring(l, c"k".as_ptr());
        assert_eq!(oxi_gettable(l, 1), 0);
        assert_eq!((lua_gettop(l), lua_tonumber(l, -1)), (2, 4.0));
        settop(l, 1);
        // A hole, a fraction and a missing key all read nil.
        for key in [7.0, 1.5, 0.0] {
            lua_pushnumber(l, key);
            assert_eq!(oxi_gettable(l, 1), 0);
            assert!(matches!(value(l, -1), Some(Value::Nil)));
            settop(l, 1);
        }
        lua_close(l);
    }
}

#[test]
fn raw_integer_reads_push_table_and_string_elements() {
    unsafe {
        let l = new_state(Some(default_allocator), ptr::null_mut());
        lua_createtable(l, 4, 0);
        lua_createtable(l, 0, 0);
        lua_rawseti(l, 1, 1 as _);
        lua_pushstring(l, c"element".as_ptr());
        lua_rawseti(l, 1, 2 as _);
        lua_rawgeti(l, 1, 1 as _);
        assert_eq!(lua_type(l, -1), 5);
        lua_rawgeti(l, 1, 2 as _);
        assert_eq!(lua_type(l, -1), 4);
        assert_eq!(
            CStr::from_ptr(lua_tolstring(l, -1, ptr::null_mut())),
            c"element"
        );
        lua_rawgeti(l, 1, 3 as _);
        assert_eq!(lua_type(l, -1), 0);
        assert_eq!(lua_gettop(l), 4);
        lua_close(l);
    }
}

/// `lua_type`, `lua_toboolean` and `lua_touserdata` read a stack slot in
/// place, and still answer for the registry and for an index past the top.
#[test]
fn type_queries_answer_for_slots_and_pseudo_indices() {
    unsafe {
        let l = state_in_mode(0);
        assert_eq!(oxi_settop(l, 0), 0);
        lua_pushboolean(l, 0);
        lua_pushnumber(l, 1.0);
        let block = lua_newuserdata(l, 8);
        assert_eq!(lua_type(l, 1), 1);
        assert_eq!(lua_type(l, -2), 3);
        assert_eq!(lua_type(l, 3), 7);
        assert_eq!(lua_type(l, 4), -1);
        assert_eq!(lua_type(l, REGISTRY), 5);
        assert_eq!(lua_toboolean(l, 1), 0);
        assert_eq!(lua_toboolean(l, 2), 1);
        assert_eq!(lua_toboolean(l, 4), 0);
        assert_eq!(lua_toboolean(l, REGISTRY), 1);
        assert_eq!(lua_touserdata(l, -1), block);
        assert!(lua_touserdata(l, 2).is_null());
        assert!(lua_touserdata(l, 4).is_null());
        assert!(lua_touserdata(l, REGISTRY).is_null());
        lua_close(l);
    }
}

/// `lua_tolstring` lends a short string's own terminated storage when
/// something besides the stack slot holds the string, so the bytes outlive a
/// pop as long as that holder does. A string only the slot holds, and one too
/// long for inline storage, take an interned copy instead. Both report the
/// length, embedded zeros included.
#[test]
fn tolstring_lends_shared_short_strings_and_copies_the_rest() {
    unsafe {
        let l = state_in_mode(0);
        run_chunk(
            l,
            "t = {name = 'stone', zero = 'a\\0b', long = string.rep('x', 30), full = string.rep('y', 22)} \
             return 0",
        );
        assert_eq!(oxi_settop(l, 0), 0);
        push(l, rt(l).get_global("t"));
        let table = raw_table(l, 1).unwrap();
        let field = |name: &str| rt(l).table_get(table, &Value::from(name)).unwrap();
        // A field value: the table holds it, so the pointer is its storage.
        push(l, field("name"));
        let mut len = 0;
        let lent = lua_tolstring(l, -1, &mut len);
        let Value::String(name) = field("name") else {
            unreachable!()
        };
        assert_eq!(
            lent.cast::<u8>(),
            name.inline_bytes_with_nul().unwrap().as_ptr()
        );
        assert_eq!(len, 5);
        pop(l);
        assert_eq!(CStr::from_ptr(lent).to_bytes(), b"stone");
        // Embedded zeros keep their length.
        push(l, field("zero"));
        let lent = lua_tolstring(l, -1, &mut len);
        assert_eq!(len, 3);
        assert_eq!(std::slice::from_raw_parts(lent.cast::<u8>(), 4), b"a\0b\0");
        pop(l);
        // Long strings and inline strings without room for the zero are copied.
        for (name, expected) in [
            ("long", [b'x'; 30].as_slice()),
            ("full", [b'y'; 22].as_slice()),
        ] {
            push(l, field(name));
            let copied = lua_tolstring(l, -1, &mut len);
            assert_eq!(len, expected.len());
            assert_eq!(CStr::from_ptr(copied).to_bytes(), expected);
            pop(l);
        }
        // A string only the stack holds is lent too, and the state holds it
        // past the pop until the next collection.
        push(l, Value::String(LuaString::from(b"fresh".as_slice())));
        let lent = lua_tolstring(l, -1, &mut len);
        let Some(Value::String(fresh)) = peek(l, -1) else {
            unreachable!()
        };
        let fresh = fresh.downgrade();
        pop(l);
        assert!(fresh.is_alive());
        assert_eq!(CStr::from_ptr(lent).to_bytes(), b"fresh");
        // A number converts in place and reads back as its text.
        lua_pushnumber(l, 12.5);
        let text = lua_tolstring(l, -1, &mut len);
        assert_eq!(CStr::from_ptr(text).to_bytes(), b"12.5");
        assert!(matches!(peek(l, -1), Some(Value::String(_))));
        lua_close(l);
    }
}

#[test]
fn a_lent_string_outlives_its_pop_and_the_key_cache_until_a_collection() {
    unsafe {
        let l = state_in_mode(0);
        let literal = std::ffi::CString::new("lit").unwrap();
        // Only the stack slot and the key cache hold the pushed string.
        lua_pushstring(l, literal.as_ptr());
        let mut len = 0;
        let lent = lua_tolstring(l, -1, &mut len);
        let Some(Value::String(pushed)) = peek(l, -1) else {
            unreachable!()
        };
        let pushed = pushed.downgrade();
        assert_eq!(oxi_settop(l, -2), 0);
        // Another literal in the same cache slot evicts it.
        shared(l).key_cache.clear();
        assert!(pushed.is_alive(), "the lent pointer must not dangle");
        assert_eq!(CStr::from_ptr(lent).to_bytes(), b"lit");
        assert_eq!(len, 3);
        // The next collection lets it go.
        vm(l).collect_full_with_values(&[]);
        sync_roots(l);
        assert!(!pushed.is_alive());
        lua_close(l);
    }
}

#[test]
fn every_string_lent_in_a_cycle_is_held_once_without_interning() {
    unsafe {
        let l = state_in_mode(0);
        // More distinct strings than the list once held, each popped.
        let count = 20_000;
        let mut lent = Vec::with_capacity(count);
        for index in 0..count {
            push(l, Value::String(LuaString::from(format!("s{index}"))));
            let Some(Value::String(string)) = peek(l, -1) else {
                unreachable!()
            };
            let weak = string.downgrade();
            let pointer = lua_tolstring(l, -1, ptr::null_mut());
            pop(l);
            lent.push((pointer, weak));
        }
        assert!(
            shared(l).strings.is_empty(),
            "short strings are lent, not interned"
        );
        assert_eq!(shared(l).lent.len(), count);
        // Reading them all again holds nothing more.
        for (_, weak) in &lent {
            push(l, Value::String(weak.upgrade().unwrap()));
            lua_tolstring(l, -1, ptr::null_mut());
            pop(l);
        }
        assert_eq!(shared(l).lent.len(), count);
        for (index, (pointer, weak)) in lent.iter().enumerate() {
            assert!(weak.is_alive());
            assert_eq!(
                CStr::from_ptr(*pointer).to_bytes(),
                format!("s{index}").as_bytes()
            );
        }
        vm(l).collect_full_with_values(&[]);
        sync_roots(l);
        assert!(shared(l).lent.is_empty());
        assert!(lent.iter().all(|(_, weak)| !weak.is_alive()));
        lua_close(l);
    }
}

#[test]
fn a_host_that_never_runs_lua_holds_at_most_lent_max_strings() {
    unsafe {
        let l = state_in_mode(0);
        // C pushes add no GC debt: no collection ends this cycle.
        let count = LENT_MAX + 1000;
        for index in 0..count {
            let text = format!("p{index}");
            lua_pushlstring(l, text.as_ptr().cast(), text.len());
            let pointer = lua_tolstring(l, -1, ptr::null_mut());
            lua_settop(l, -2);
            assert_eq!(CStr::from_ptr(pointer).to_bytes(), text.as_bytes());
        }
        assert_eq!(shared(l).lent.len(), LENT_MAX);
        // Past the cap the strings intern, as before lending.
        assert_eq!(shared(l).strings.len(), count - LENT_MAX);
        vm(l).collect_full_with_values(&[]);
        sync_roots(l);
        assert!(shared(l).lent.is_empty());
        lua_close(l);
    }
}

#[test]
fn field_access_sees_metamethods_added_after_a_miss() {
    unsafe {
        let l = state_in_mode(0);
        run_chunk(l, "mt = {} t = setmetatable({}, mt) sink = {} return 0");
        assert_eq!(oxi_settop(l, 0), 0);
        push(l, rt(l).get_global("t"));
        // No `__index` yet: the miss is nil, and the metatable learns it.
        assert_eq!(oxi_getfield(l, 1, c"x".as_ptr()), 0);
        assert!(matches!(peek(l, -1), Some(Value::Nil)));
        pop(l);
        run_chunk(l, "mt.__index = {x = 5} return 0");
        assert_eq!(oxi_getfield(l, 1, c"x".as_ptr()), 0);
        assert!(matches!(
            peek(l, -1),
            Some(Value::Integer(5) | Value::Number(5.0))
        ));
        pop(l);
        // A store of a new key goes to `__newindex` once it is there.
        lua_pushinteger(l, 7);
        assert_eq!(oxi_setfield(l, 1, c"y".as_ptr()), 0);
        run_chunk(l, "mt.__newindex = sink return 0");
        lua_pushinteger(l, 8);
        assert_eq!(oxi_setfield(l, 1, c"z".as_ptr()), 0);
        let result = run_chunk(
            l,
            "return rawget(t, 'y') == 7 and rawget(t, 'z') == nil and sink.z == 8",
        );
        assert!(matches!(result, Value::Boolean(true)), "{result:?}");
        lua_close(l);
    }
}

#[test]
fn two_literals_whose_addresses_collide_both_stay_in_the_key_cache() {
    unsafe {
        let l = state_in_mode(0);
        // Names at addresses that pick the same pair of cache slots.
        let mut buffer = vec![0_u8; 32768];
        let base = buffer.as_ptr() as usize;
        let pair = |address: usize| ((address >> 3) ^ (address >> 9)) & (KEY_CACHE_SLOTS / 2 - 1);
        let first = (base + 7) & !7;
        let colliding = |after: usize| {
            (after + 8..base + 32768 - 8)
                .step_by(8)
                .find(|address| pair(*address) == pair(first))
                .expect("a colliding address in the buffer")
        };
        let second = colliding(first);
        let third = colliding(second);
        for (address, text) in [
            (first, b"north\0"),
            (second, b"south\0"),
            (third, b"west\0\0"),
        ] {
            let offset = address - base;
            buffer[offset..offset + text.len()].copy_from_slice(text);
        }
        let held = |l: *mut lua_State, addresses: &[usize]| {
            shared(l)
                .key_cache
                .iter()
                .flatten()
                .filter(|(address, _)| addresses.contains(address))
                .count()
        };
        for _ in 0..3 {
            for address in [first, second] {
                lua_pushstring(l, address as *const c_char);
                assert_eq!(oxi_settop(l, -2), 0);
            }
        }
        assert_eq!(
            held(l, &[first, second]),
            2,
            "one literal evicted the other"
        );
        // A third literal in the pair evicts the older one only.
        lua_pushstring(l, third as *const c_char);
        assert_eq!(oxi_settop(l, -2), 0);
        assert_eq!(held(l, &[first, second, third]), 2);
        assert_eq!(held(l, &[second, third]), 2);
        lua_close(l);
    }
}

/// Calls the global `churn(n)`, which makes `n` tables and drops them, on
/// top of whatever `l` holds.
unsafe fn churn(l: *mut lua_State, n: f64) {
    unsafe {
        let top = lua_gettop(l);
        oxi_getglobal(l, c"churn".as_ptr());
        push(l, Value::Number(n));
        assert_eq!(oxi_pcall(l, 1, 0, 0), 0);
        assert_eq!(lua_gettop(l), top);
    }
}

/// `lua_rawseti` and `lua_rawgeti` reach the tables on a C stack without the
/// arena's generation and occupancy checks (`rawseti_fast`), which holds
/// because the collector roots every C stack of the family. Between the
/// calls here Lua makes thousands of tables, which grows and reallocates the
/// table arena, full collections free them so later ones reuse their slots,
/// and incremental steps trace the filling tables in chunks. A table on the
/// main state's stack and one on a coroutine's stack (whose `main` names the
/// main state) keep every value through it all.
#[test]
fn array_transfers_keep_their_tables_across_arena_growth_and_collections() {
    unsafe {
        let l = state_in_mode(0);
        run_chunk(
            l,
            "function churn(n) local t = {} for i = 1, n do t[i] = {} end end",
        );
        let mut child = ptr::null_mut();
        assert_eq!(oxi_newthread(l, &mut child), 0);
        assert_eq!(main_state(child), l);
        // Index 1 of `l` holds the thread, index 2 a table; index 1 of the
        // thread holds the other table.
        lua_createtable(l, 0, 0);
        lua_createtable(child, 0, 0);
        let targets = [(l, 2), (child, 1)];
        let expected = |k: i64, round: i64| k * 7 + round;
        let mut length = 0;
        for round in 0..5 {
            length += 2000;
            for k in 1..=length {
                for (state, index) in targets {
                    lua_pushinteger(state, expected(k, round) as LuaInteger);
                    assert_eq!(lua_rawseti(state, index, k as RawIndex), 0);
                }
                if k % 250 == 0 {
                    assert!(oxi_gc(l, 5, 0, 0, 0) >= 0);
                }
                if k % 1000 == 0 {
                    churn(l, 3000.0);
                }
            }
            churn(l, 20_000.0);
            assert_eq!(oxi_gc(l, 2, 0, 0, 0), 0);
            churn(l, 500.0);
            for (state, index) in targets {
                let top = lua_gettop(state);
                for k in 1..=length {
                    lua_rawgeti(state, index, k as RawIndex);
                    assert_eq!(
                        lua_tointeger(state, -1) as i64,
                        expected(k, round),
                        "key {k} of round {round}"
                    );
                    assert_eq!(oxi_settop(state, -2), 0);
                }
                lua_rawgeti(state, index, (length + 1) as RawIndex);
                assert_eq!(pop(state), Value::Nil);
                assert_eq!(lua_gettop(state), top);
            }
        }
        lua_close(l);
    }
}

/// A C function that stores 5 at key 3 of its first argument through the
/// exported C `lua_rawseti`, which raises its errors, and returns the
/// height of its stack after.
#[cfg(not(feature = "lazy-bridge"))]
unsafe extern "C-unwind" fn store_through_the_raising_rawseti(l: *mut lua_State) -> c_int {
    unsafe {
        lua_pushinteger(l, 5);
        lua_rawseti_raising(l, 1, 3);
        lua_pushinteger(l, lua_gettop(l) as LuaInteger);
        1
    }
}

/// The C `lua_rawseti` takes the in-place store itself, pops the value, and
/// raises what the general store reports through bridge.cpp, so a protected
/// call gets the error as from any other API function.
#[test]
#[cfg(not(feature = "lazy-bridge"))]
fn the_exported_rawseti_stores_in_place_and_raises_errors() {
    type CUnwindFunction = unsafe extern "C-unwind" fn(*mut lua_State) -> c_int;
    unsafe {
        let l = state_in_mode(0);
        // SAFETY: the calling conventions match; `C-unwind` only lets the
        // bridge's exception pass the function, as it would a C++ one.
        let function =
            std::mem::transmute::<CUnwindFunction, CFunction>(store_through_the_raising_rawseti);
        // A table whose key 3 is an array slot (the in-place store), an empty
        // one (the general store), and a number (an error).
        for (prefilled, argument_is_table) in [(4, true), (0, true), (0, false)] {
            assert_eq!(oxi_settop(l, 0), 0);
            if argument_is_table {
                lua_createtable(l, 4, 0);
                for k in 1..=prefilled {
                    lua_pushinteger(l, 1);
                    assert_eq!(lua_rawseti(l, 1, k as RawIndex), 0);
                }
            }
            lua_pushcclosure(l, function, 0);
            if argument_is_table {
                lua_pushvalue(l, 1);
            } else {
                lua_pushnumber(l, 1.0);
            }
            let status = oxi_pcall(l, 1, 1, 0);
            let result = pop(l);
            if argument_is_table {
                assert_eq!(status, 0, "{result:?}");
                assert_eq!(result, rt(l).integer_value(1));
                lua_rawgeti(l, 1, 3);
                assert_eq!(lua_tointeger(l, -1), 5);
            } else {
                assert_eq!(status, 2);
                assert_eq!(result, Value::from("table expected"));
            }
        }
        lua_close(l);
    }
}

/// `lua_pop(L, 1)`, the C API's settop to -2, as each ABI defines it.
unsafe fn pop_one(l: *mut lua_State) {
    unsafe {
        #[cfg(not(any(feature = "lua54", feature = "lua55")))]
        lua_settop(l, -2);
        #[cfg(any(feature = "lua54", feature = "lua55"))]
        assert_eq!(oxi_settop(l, -2), 0);
    }
}

/// The element loop of a host copying an array out (`lua_rawgeti`,
/// `lua_tointeger(L, -1)`, `lua_pop`) and in (`lua_pushinteger`,
/// `lua_rawseti`), on a stack the fast paths take and on the edges they
/// hand on: an empty stack, a string on top, a table at a negative index,
/// and values that are not numbers.
#[test]
fn element_transfers_read_the_top_and_pop_it_on_every_kind_of_stack() {
    unsafe {
        let l = state_in_mode(0);
        assert_eq!(oxi_settop(l, 0), 0);
        // Nothing on top: no integer, and under Lua 5.1 and LuaJIT a pop
        // below the bottom is ignored (5.4 and 5.5 raise it).
        assert_eq!(lua_tointeger(l, -1), 0);
        #[cfg(not(any(feature = "lua54", feature = "lua55")))]
        {
            pop_one(l);
            assert_eq!(lua_gettop(l), 0);
        }
        lua_createtable(l, 4, 0);
        for n in 1..=4 {
            lua_pushinteger(l, (n * 11) as LuaInteger);
            assert_eq!(lua_rawseti(l, -2, n as RawIndex), 0);
        }
        // Integers and a fraction in the array, read through the top. Lua
        // 5.3 and later convert only a float with an integral value.
        push(l, Value::Number(-2.75));
        assert_eq!(lua_rawseti(l, 1, 5), 0);
        assert_eq!(lua_gettop(l), 1);
        let truncated = if MODERN_INTEGER { 0 } else { -2 };
        for (n, expected) in [(1, 11), (4, 44), (5, truncated)] {
            lua_rawgeti(l, 1, n);
            assert_eq!(lua_tointeger(l, -1), expected, "key {n}");
            assert_eq!(lua_tointeger(l, 2), expected, "key {n}");
            pop_one(l);
            assert_eq!(lua_gettop(l), 1);
        }
        // The same table one below the top, by a negative index.
        push(l, Value::Boolean(true));
        lua_rawgeti(l, -2, 2);
        assert_eq!(lua_tointeger(l, -1), 22);
        pop_one(l);
        pop_one(l);
        assert_eq!(lua_gettop(l), 1);
        // A numeric string converts, a non-number gives zero, and popping a
        // string releases it.
        push(l, Value::from("12"));
        assert_eq!(lua_tointeger(l, -1), 12);
        pop_one(l);
        push(l, Value::Boolean(true));
        assert_eq!(lua_tointeger(l, -1), 0);
        pop_one(l);
        // A missing key reads nil, which is no integer.
        lua_rawgeti(l, 1, 9);
        assert_eq!(peek(l, -1), Some(&Value::Nil));
        assert_eq!(lua_tointeger(l, -1), 0);
        pop_one(l);
        assert_eq!(lua_gettop(l), 1);
        lua_close(l);
    }
}
