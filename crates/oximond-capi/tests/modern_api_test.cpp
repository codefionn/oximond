extern "C" {
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

static void require(bool condition, const char *message)
{
    if (!condition) {
        std::fprintf(stderr, "Lua %s API: %s\n", LUA_VERSION, message);
        std::exit(1);
    }
}

static bool reject_allocations;
static void *test_allocator(void *, void *pointer, size_t oldsize, size_t size)
{
    if (size == 0) {
        std::free(pointer);
        return nullptr;
    }
    if (reject_allocations && (pointer == nullptr || size > oldsize))
        return nullptr;
    return std::realloc(pointer, size);
}

static int allocate_payload(lua_State *L)
{
    reject_allocations = true;
    lua_newuserdata(L, 4096);
    return 1;
}

static lua_State *new_allocated_state()
{
#if LUA_VERSION_NUM == 505
    return lua_newstate(test_allocator, nullptr, 123);
#else
    return lua_newstate(test_allocator, nullptr);
#endif
}

static void allocation_failure()
{
    reject_allocations = true;
    require(new_allocated_state() == nullptr, "state allocation failure returns null");
    reject_allocations = false;
    lua_State *L = new_allocated_state();
    require(L != nullptr, "custom allocator state");
    lua_atpanic(L, [](lua_State *state) {
        std::fprintf(stderr, "allocation-test panic: %s\n", lua_tostring(state, -1));
        return 0;
    });
    lua_pushcfunction(L, allocate_payload);
    require(lua_pcall(L, 0, 1, 0) == LUA_ERRMEM, "native userdata allocation failure status");
    reject_allocations = false;
    require(lua_type(L, -1) == LUA_TSTRING, "allocation failure leaves error object");
    lua_close(L);
}

#if LUA_VERSION_NUM == 505
static int released_external;
static void *external_release(void *, void *pointer, size_t oldsize, size_t newsize)
{
    require(newsize == 0 && oldsize >= 1, "external string release sizes");
    ++released_external;
    std::free(pointer);
    return nullptr;
}

static void external_strings(lua_State *L)
{
    for (size_t length : {size_t(3), size_t(100)}) {
        char *bytes = static_cast<char *>(std::malloc(length + 1));
        std::memset(bytes, 'x', length);
        bytes[length] = 0;
        const int before = released_external;
        require(lua_pushexternalstring(L, bytes, length, external_release, nullptr) == bytes,
                "external string returns supplied pointer");
        require(lua_tolstring(L, -1, nullptr) == bytes, "external string retains supplied pointer");
        lua_gc(L, LUA_GCCOLLECT);
        require(released_external == before, "rooted external string is retained");
        lua_pop(L, 1);
        lua_gc(L, LUA_GCCOLLECT);
        require(released_external == before + 1, "unreachable external string is released");
    }
}
#endif

static void run(lua_State *L, const char *source)
{
    int status = luaL_loadstring(L, source);
    if (status == LUA_OK)
        status = lua_pcall(L, 0, LUA_MULTRET, 0);
    if (status != LUA_OK) {
        std::fprintf(stderr, "%s\n", lua_tostring(L, -1));
        require(false, "script failed");
    }
}

static int resume(lua_State *L, lua_State *from, int arguments, int expected)
{
#if LUA_VERSION_NUM >= 504
    int results = -1;
    int status = lua_resume(L, from, arguments, &results);
    require(results == expected, "resume result count");
    return status;
#else
    int status = lua_resume(L, from, arguments);
    require(lua_gettop(L) == expected, "resume stack result count");
    return status;
#endif
}

static int yielded;
static int hook_calls;
static int kill_hook_calls;
static bool in_error_handler;
static int error_handler_hook_calls;

static void kill_hook(lua_State *L, lua_Debug *)
{
    ++kill_hook_calls;
    lua_sethook(L, kill_hook, LUA_MASKLINE, 0);
    lua_pushliteral(L, "hook killed");
    lua_error(L);
}

static int mark_error_handler(lua_State *)
{
    in_error_handler = true;
    return 0;
}

static void ordinary_error_hook(lua_State *, lua_Debug *)
{
    if (in_error_handler)
        ++error_handler_hook_calls;
}

static void error_handler_hooks(lua_State *L)
{
    lua_settop(L, 0);
    run(L, "return function(err) return 'handled:'..err end");
    require(luaL_loadstring(L, "local sum=0; for i=1,100 do sum=sum+i end; return sum") == LUA_OK,
            "load hook interrupted script");
    lua_sethook(L, kill_hook, LUA_MASKCOUNT, 7);
    int status = lua_pcall(L, 0, 1, 1);
    lua_sethook(L, nullptr, 0, 0);
    require(status == LUA_ERRRUN, "hook error keeps runtime status");
    require(std::strcmp(lua_tostring(L, -1), "handled:hook killed") == 0,
            "hook error message handler completes");
    require(kill_hook_calls == 1, "hook error message handler does not repeat kill hook");
    lua_settop(L, 0);

    lua_pushcfunction(L, mark_error_handler);
    lua_setglobal(L, "mark_error_handler");
    run(L, "return function(err)\nmark_error_handler()\nlocal message=err\nreturn message..' handled'\nend");
    require(luaL_loadstring(L, "error('ordinary failure')") == LUA_OK, "load ordinary error script");
    lua_sethook(L, ordinary_error_hook, LUA_MASKLINE, 0);
    status = lua_pcall(L, 0, 1, 1);
    lua_sethook(L, nullptr, 0, 0);
    in_error_handler = false;
    require(status == LUA_ERRRUN && error_handler_hook_calls > 0,
            "ordinary error message handler still runs line hooks");
    lua_settop(L, 0);
}

static void hook(lua_State *L, lua_Debug *record)
{
    ++hook_calls;
    require(lua_getinfo(L, "Sl", record) == 1, "hook debug record");
    require(record->source != nullptr, "hook source metadata");
}

static int change_caller_local(lua_State *L)
{
    lua_Debug frame;
    require(lua_getstack(L, 1, &frame) == 1, "find Lua caller frame");
    for (int index = 1;; ++index) {
        const char *name = lua_getlocal(L, &frame, index);
        if (name == nullptr)
            break;
        const bool target = std::strcmp(name, "scalar") == 0;
        lua_pop(L, 1);
        if (target) {
            lua_pushinteger(L, 47);
            require(lua_setlocal(L, &frame, index) != nullptr, "write Lua caller local");
            return 0;
        }
    }
    require(false, "find active scalar local");
    return 0;
}

/* Reads upvalue 1 of the function argument, adds 100 and stores it back: the
 * closure's declaring function is still running, so the upvalue is open and
 * stands for that function's register. Returns the value it read. */
static int bump_upvalue(lua_State *L)
{
    require(lua_isfunction(L, 1), "bump_upvalue takes a function");
    const char *name = lua_getupvalue(L, 1, 1);
    require(name != nullptr && std::strcmp(name, "n") == 0, "named open upvalue");
    const lua_Integer before = lua_tointeger(L, -1);
    lua_pop(L, 1);
    lua_pushinteger(L, before + 100);
    require(lua_setupvalue(L, 1, 1) != nullptr, "write open upvalue");
    lua_pushinteger(L, before);
    return 1;
}

#if LUA_VERSION_NUM >= 504
static bool call_transfer;
static bool return_transfer;
static void transfer_hook(lua_State *L, lua_Debug *record)
{
    require(lua_getinfo(L, "ru", record) == 1, "hook transfer information");
    if (record->event == LUA_HOOKCALL && record->nparams == 2 &&
        record->ftransfer == 1 && record->ntransfer == 2)
        call_transfer = true;
    if (record->event == LUA_HOOKRET && record->ntransfer == 2)
        return_transfer = true;
}
#endif

#if LUA_VERSION_NUM == 502
static int continued(lua_State *L)
{
    int context = 0;
    require(lua_getctx(L, &context) == LUA_YIELD, "continuation status");
#else
static int continued(lua_State *L, int status, lua_KContext context)
{
    require(status == LUA_YIELD, "continuation status");
#endif
    require(context == 71, "continuation context");
    ++yielded;
    lua_pushinteger(L, luaL_checkinteger(L, -1) + context);
    return 1;
}

static int native_yield(lua_State *L)
{
    lua_pushliteral(L, "native-yield");
    return lua_yieldk(L, 1, 71, continued);
}

static int native_call(lua_State *L)
{
    lua_settop(L, 1);
    lua_callk(L, 0, 1, 71, continued);
    lua_pushinteger(L, luaL_checkinteger(L, -1) + 71);
    return 1;
}

static int native_pcall(lua_State *L)
{
    lua_settop(L, 1);
    int status = lua_pcallk(L, 0, 1, 0, 71, continued);
    require(status == LUA_OK, "immediate pcall status");
    lua_pushinteger(L, luaL_checkinteger(L, -1) + 71);
    return 1;
}

static int dump_writer(lua_State *, const void *bytes, size_t length, void *data)
{
    static_cast<std::string *>(data)->append(static_cast<const char *>(bytes), length);
    return 0;
}

static int invalid_format(lua_State *L)
{
    lua_pushfstring(L, "%q");
    return 1;
}

static void formatting(lua_State *L)
{
    lua_pushfstring(L, "a%cb", 0);
    size_t length = 0;
    const char *text = lua_tolstring(L, -1, &length);
#if LUA_VERSION_NUM == 503
    require(std::string(text, length) == "a<\\0>b", "Lua53 character formatting");
#else
    require(std::string(text, length) == std::string("a\0b", 3), "binary character formatting");
#endif
    lua_pop(L, 1);
    lua_pushfstring(L, "%f", 3.0);
#if LUA_VERSION_NUM >= 503
    require(std::strcmp(lua_tostring(L, -1), "3.0") == 0, "float formatting preserves subtype");
    lua_pop(L, 1);
    lua_pushfstring(L, "%I", static_cast<LUAI_UACINT>(9007199254740993LL));
    require(std::strcmp(lua_tostring(L, -1), "9007199254740993") == 0, "exact integer formatting");
    lua_pop(L, 1);
#if LUA_VERSION_NUM >= 505
    lua_pushfstring(L, "%U", static_cast<unsigned long>(0x20ac));
#else
    lua_pushfstring(L, "%U", static_cast<long>(0x20ac));
#endif
    require(std::strcmp(lua_tostring(L, -1), "\xe2\x82\xac") == 0, "UTF-8 formatting");
#else
    require(std::strcmp(lua_tostring(L, -1), "3") == 0, "Lua52 float formatting");
#endif
    lua_pop(L, 1);
    lua_pushcfunction(L, invalid_format);
#if LUA_VERSION_NUM == 505
    require(lua_pcall(L, 0, 1, 0) == LUA_OK, "Lua55 keeps unknown format options");
    require(std::strcmp(lua_tostring(L, -1), "%q") == 0, "Lua55 unknown format result");
#else
    require(lua_pcall(L, 0, 1, 0) == LUA_ERRRUN, "invalid format raises protected error");
    require(std::strstr(lua_tostring(L, -1), "invalid option") != nullptr, "invalid format diagnostic");
#endif
    lua_settop(L, 0);
}

static void stack_and_values(lua_State *L)
{
    int valid = 1;
    require(lua_gettop(L) == 0, "empty initial stack");
    lua_pushinteger(L, 3);
    lua_pushinteger(L, 4);
    require(lua_absindex(L, -2) == 1, "absolute index");
    lua_copy(L, 1, 2);
    require(lua_tointeger(L, 2) == 3, "copy stack value");
    lua_arith(L, LUA_OPADD);
    require(lua_tointeger(L, -1) == 6, "arithmetic");
    lua_pushinteger(L, 8);
    require(lua_compare(L, -2, -1, LUA_OPLT), "comparison");
#if LUA_VERSION_NUM >= 503
    lua_rotate(L, 1, 1);
    require(lua_tointeger(L, 1) == 8, "rotate stack");
    lua_pushinteger(L, 9007199254740993LL);
    require(lua_isinteger(L, -1), "integer subtype");
    require(lua_tointeger(L, -1) == 9007199254740993LL, "exact integer conversion");
    lua_pop(L, 1);
    lua_pushnumber(L, 1.25);
    valid = 1;
    require(lua_tointegerx(L, -1, &valid) == 0 && !valid, "fractional integer rejection");
#else
    lua_pushunsigned(L, 0xffffffffu);
    require(lua_tounsigned(L, -1) == 0xffffffffu, "unsigned conversion");
#endif
    lua_settop(L, 0);
    valid = 1;
    lua_pushliteral(L, "bad-number");
    require(lua_tonumberx(L, -1, &valid) == 0 && !valid, "invalid number conversion");
    lua_settop(L, 0);
    lua_rawgeti(L, LUA_REGISTRYINDEX, LUA_RIDX_MAINTHREAD);
    require(lua_tothread(L, -1) == L, "registry main thread");
    lua_pop(L, 1);
    lua_rawgeti(L, LUA_REGISTRYINDEX, LUA_RIDX_GLOBALS);
    lua_getglobal(L, "_G");
    require(lua_rawequal(L, -1, -2), "registry globals");
    lua_settop(L, 0);
}

static void tables_and_userdata(lua_State *L)
{
    static int key;
    lua_newtable(L);
    lua_pushinteger(L, 17);
    lua_rawsetp(L, -2, &key);
    lua_rawgetp(L, -1, &key);
    require(lua_tointeger(L, -1) == 17, "pointer table keys");
    lua_pop(L, 1);
#if LUA_VERSION_NUM >= 503
    lua_pushinteger(L, 23);
    lua_seti(L, -2, 9007199254740993LL);
    require(lua_geti(L, -1, 9007199254740993LL) == LUA_TNUMBER, "integer getter return type");
    require(lua_tointeger(L, -1) == 23, "exact integer table key");
    lua_pop(L, 1);
#endif
    lua_settop(L, 0);
#if LUA_VERSION_NUM >= 504
    void *bytes = lua_newuserdatauv(L, 19, 2);
    require(bytes != nullptr && lua_rawlen(L, -1) == 19, "userdata allocation");
    lua_pushliteral(L, "second");
    require(lua_setiuservalue(L, -2, 2) == 1, "set second uservalue");
    lua_gc(L, LUA_GCCOLLECT);
    require(lua_getiuservalue(L, -1, 2) == LUA_TSTRING, "uservalue GC retention");
    require(std::strcmp(lua_tostring(L, -1), "second") == 0, "second uservalue");
    lua_pop(L, 1);
    require(lua_getiuservalue(L, -1, 3) == LUA_TNONE, "invalid uservalue index");
    lua_pop(L, 1);
#else
    lua_newuserdata(L, 19);
    lua_newtable(L);
    lua_pushliteral(L, "retained");
    lua_setfield(L, -2, "key");
    lua_setuservalue(L, -2);
    lua_gc(L, LUA_GCCOLLECT, 0);
    lua_getuservalue(L, -1);
    lua_getfield(L, -1, "key");
    require(std::strcmp(lua_tostring(L, -1), "retained") == 0, "uservalue GC retention");
#endif
    lua_settop(L, 0);
    run(L, "return setmetatable({}, {__len=function() return 31 end})");
    lua_len(L, -1);
    require(lua_tointeger(L, -1) == 31, "length metamethod");
    lua_settop(L, 0);
    luaL_Buffer buffer;
    luaL_buffinit(L, &buffer);
    for (int i = 0; i < 20000; ++i)
        luaL_addchar(&buffer, 'x');
    luaL_pushresult(&buffer);
    require(lua_rawlen(L, -1) == 20000, "auxiliary buffer growth");
    lua_settop(L, 0);
}

static void loading(lua_State *L)
{
    require(luaL_loadbufferx(L, "return 13", 9, "text", "b") == LUA_ERRSYNTAX,
            "binary-only load rejects text");
    lua_pop(L, 1);
    require(luaL_loadstring(L, "return 13") == LUA_OK, "load text");
    std::string bytes;
#if LUA_VERSION_NUM >= 503
    require(lua_dump(L, dump_writer, &bytes, 1) == 0, "dump stripped chunk");
#else
    require(lua_dump(L, dump_writer, &bytes) == 0, "dump chunk");
#endif
    lua_pop(L, 1);
    require(luaL_loadbufferx(L, bytes.data(), bytes.size(), "binary", "b") == LUA_OK,
            "binary load");
    require(lua_pcall(L, 0, 1, 0) == LUA_OK && lua_tointeger(L, -1) == 13,
            "execute dumped chunk");
    lua_settop(L, 0);
    require(luaL_loadstring(L, "return answer") == LUA_OK, "load environment chunk");
    lua_newtable(L);
    lua_pushinteger(L, 29);
    lua_setfield(L, -2, "answer");
    require(lua_setupvalue(L, -2, 1) != nullptr, "set chunk environment upvalue");
    require(lua_pcall(L, 0, 1, 0) == LUA_OK && lua_tointeger(L, -1) == 29,
            "execute replacement environment");
    lua_settop(L, 0);
}

static void coroutines(lua_State *L)
{
    lua_pushcfunction(L, native_yield);
    lua_setglobal(L, "native_yield");
    lua_pushcfunction(L, native_call);
    lua_setglobal(L, "native_call");
    lua_pushcfunction(L, native_pcall);
    lua_setglobal(L, "native_pcall");
    for (const char *source : {
            "return native_yield()",
            "return native_call(function() return coroutine.yield('native-yield') end)",
            "return native_pcall(function() return coroutine.yield('native-yield') end)"}) {
        lua_State *thread = lua_newthread(L);
#if LUA_VERSION_NUM >= 503
        require(*static_cast<void **>(lua_getextraspace(thread)) == L,
                "new thread inherits main extra space");
#endif
        require(luaL_loadstring(thread, source) == LUA_OK, "load coroutine");
        require(resume(thread, L, 0, 1) == LUA_YIELD, "yield through native continuation");
        require(std::strcmp(lua_tostring(thread, -1), "native-yield") == 0, "yielded value");
        lua_settop(thread, 0);
        lua_pushinteger(thread, 9);
        require(resume(thread, L, 1, 1) == LUA_OK, "resume native continuation");
        require(lua_tointeger(thread, -1) == 80, "continuation returned result");
        lua_pop(L, 1);
    }
    require(yielded == 3, "each native continuation ran once");
}

static void debugging(lua_State *L)
{
    lua_pushcfunction(L, bump_upvalue);
    lua_setglobal(L, "bump_upvalue");
    run(L, R"(
        local n = 5
        local function f() return n end
        local seen = bump_upvalue(f)       -- reads the frame's register, writes it
        local after = n
        n = n + 1                          -- the frame writes; the C side reads it
        local again = bump_upvalue(f)
        return seen * 1000000 + after * 1000 + again + (f() == 206 and 0 or 7)
    )");
    require(lua_tointeger(L, -1) == 5 * 1000000 + 105 * 1000 + 106, "C API reaches open upvalues");
    lua_settop(L, 0);
    run(L, R"(
        local function make()
          local n = 7
          return function() return n end
        end
        local f = make()
        return bump_upvalue(f), f()        -- the variable closed when make returned
    )");
    require(lua_tointeger(L, -2) == 7 && lua_tointeger(L, -1) == 107, "C API reaches closed upvalues");
    lua_settop(L, 0);
    lua_pushcfunction(L, change_caller_local);
    lua_setglobal(L, "change_caller_local");
    run(L, "local scalar=11; change_caller_local(); return scalar");
    require(lua_tointeger(L, -1) == 47, "caller resumes with changed local");
    lua_settop(L, 0);
#if LUA_VERSION_NUM >= 504
    lua_sethook(L, transfer_hook, LUA_MASKCALL | LUA_MASKRET, 0);
    run(L, "local function add(a,b) local sum=a+b; return sum,b end; local x,y=add(3,4); return x+y");
    lua_sethook(L, nullptr, 0, 0);
    require(call_transfer && return_transfer, "call and return hook transfer counts");
    lua_settop(L, 0);
#endif
    lua_sethook(L, hook, LUA_MASKCOUNT, 7);
    require(lua_gethook(L) == hook, "installed hook identity");
    require(lua_gethookmask(L) == LUA_MASKCOUNT && lua_gethookcount(L) == 7,
            "installed hook configuration");
    run(L, "local sum=0; for i=1,100 do sum=sum+i end; return sum");
    require(hook_calls > 0 && lua_tointeger(L, -1) == 5050, "count hooks run during Lua execution");
    lua_sethook(L, nullptr, 0, 0);
    lua_settop(L, 0);
    run(L, "local a=11; return function() return a end, function() return a+1 end");
    void *identity = lua_upvalueid(L, 1, 1);
    require(identity != nullptr && identity == lua_upvalueid(L, 2, 1), "shared Lua upvalue identity");
    lua_pushinteger(L, 19);
    require(lua_setupvalue(L, 1, 1) != nullptr, "set Lua upvalue");
    lua_pushvalue(L, 2);
    require(lua_pcall(L, 0, 1, 0) == LUA_OK && lua_tointeger(L, -1) == 20, "shared Lua upvalue mutation");
    lua_settop(L, 0);
    run(L, "local a=11; local b=23; return function() return a end, function() return b end");
    require(lua_upvalueid(L, 1, 1) != lua_upvalueid(L, 2, 1), "separate Lua upvalue identity");
    lua_upvaluejoin(L, 1, 1, 2, 1);
    lua_pushinteger(L, 37);
    lua_setupvalue(L, 2, 1);
    lua_pushvalue(L, 1);
    require(lua_pcall(L, 0, 1, 0) == LUA_OK && lua_tointeger(L, -1) == 37, "joined Lua upvalue mutation");
    lua_settop(L, 0);
}

#if LUA_VERSION_NUM >= 504
static int closed;
static std::string warnings;

static int close_resource(lua_State *L)
{
    require(lua_isuserdata(L, 1), "close callback receives resource");
    ++closed;
    return 0;
}

static void warning(void *data, const char *message, int continued)
{
    require(data == &warnings, "warning user data");
    warnings += message;
    if (!continued)
        warnings += '\n';
}

static void closing_and_warnings(lua_State *L)
{
    lua_setwarnf(L, warning, &warnings);
    lua_warning(L, "first", 1);
    lua_warning(L, "second", 0);
    require(warnings == "firstsecond\n", "warning continuation fragments");
    run(L, "warn('lua',' pieces'); warn('@off')");
    require(warnings == "firstsecond\nlua pieces\n@off\n", "Lua warn uses installed host callback");
    lua_newuserdatauv(L, 1, 0);
    lua_newtable(L);
    lua_pushcfunction(L, close_resource);
    lua_setfield(L, -2, "__close");
    lua_setmetatable(L, -2);
    lua_toclose(L, -1);
    lua_closeslot(L, -1);
    require(closed == 1 && lua_isnil(L, -1), "explicit close clears slot");
    lua_pop(L, 1);
    require(closed == 1, "closed slot is not closed twice");
    lua_newuserdatauv(L, 1, 0);
    lua_newtable(L);
    lua_pushcfunction(L, close_resource);
    lua_setfield(L, -2, "__close");
    lua_setmetatable(L, -2);
    lua_toclose(L, -1);
    lua_pop(L, 1);
    require(closed == 2, "stack pop closes resource");
    lua_State *thread = lua_newthread(L);
    require(luaL_loadstring(thread,
        "local x <close> = setmetatable({}, {__close=function() closed_thread=true end}); coroutine.yield()") == LUA_OK,
        "load closing coroutine");
    require(resume(thread, L, 0, 0) == LUA_YIELD, "closing coroutine suspended");
    require(lua_closethread(thread, L) == LUA_OK, "close suspended coroutine");
    lua_getglobal(L, "closed_thread");
    require(lua_toboolean(L, -1), "coroutine close ran Lua close handler");
    lua_settop(L, 0);
#if LUA_VERSION_NUM == 505
    require(lua_gc(L, LUA_GCPARAM, LUA_GCPMINORMUL, -1) == 20, "Lua55 minor collector parameter");
    require(lua_gc(L, LUA_GCPARAM, LUA_GCPPAUSE, 300) == 250, "Lua55 previous pause parameter");
    require(lua_gc(L, LUA_GCPARAM, LUA_GCPPAUSE, -1) == 300, "Lua55 updated pause parameter");
#else
    require(lua_gc(L, LUA_GCGEN, 20, 100) == LUA_GCINC, "select generational collector");
    require(lua_gc(L, LUA_GCINC, 200, 200, 13) == LUA_GCGEN, "select incremental collector");
#endif
}
#endif

#if LUA_VERSION_NUM >= 504
// Sends "hidden", "@on" and "visible" to a fresh state's warnf and returns
// what reached stderr.
static std::string captured_warnings(lua_State *(*create)(), bool clear_warnf)
{
    std::fflush(stderr);
    FILE *captured = std::tmpfile();
    require(captured != nullptr, "temporary file for stderr");
    const int saved = dup(2);
    require(saved >= 0 && dup2(fileno(captured), 2) >= 0, "redirect stderr");
    lua_State *L = create();
    require(L != nullptr, "new state for default warnings");
    if (clear_warnf)
        lua_setwarnf(L, nullptr, nullptr);
    lua_warning(L, "hidden", 0);
    lua_warning(L, "@on", 0);
    lua_warning(L, "visible", 0);
    lua_close(L);
    std::fflush(stderr);
    require(dup2(saved, 2) >= 0, "restore stderr");
    close(saved);
    std::string text;
    std::rewind(captured);
    char buffer[256];
    for (size_t n; (n = std::fread(buffer, 1, sizeof buffer, captured)) > 0;)
        text.append(buffer, n);
    std::fclose(captured);
    return text;
}

static lua_State *new_auxiliary_state()
{
    return luaL_newstate();
}

// luaL_newstate installs lauxlib's warning function. Lua 5.4's starts with
// warnings off, so only the one after "@on" shows; 5.5's starts with them on.
// A state from lua_newstate has a NULL warnf, as does one whose host set
// NULL, and drops every warning.
static void default_warnings()
{
#if LUA_VERSION_NUM == 504
    const std::string expected = "Lua warning: visible\n";
#else
    const std::string expected = "Lua warning: hidden\nLua warning: visible\n";
#endif
    std::string text = captured_warnings(new_auxiliary_state, false);
    if (text != expected)
        std::fprintf(stderr, "captured warnings: [%s]\n", text.c_str());
    require(text == expected, "luaL_newstate shows warnings");
    text = captured_warnings(new_allocated_state, false);
    if (!text.empty())
        std::fprintf(stderr, "captured warnings: [%s]\n", text.c_str());
    require(text.empty(), "lua_newstate has no warning function");
    text = captured_warnings(new_auxiliary_state, true);
    require(text.empty(), "a NULL warnf drops warnings");
}
#endif

// print and io.write reach the process's stdout, as liblua's do, and
// lua_close flushes a partial line.
static void standard_output()
{
    std::fflush(stdout);
    FILE *captured = std::tmpfile();
    require(captured != nullptr, "temporary file for stdout");
    const int saved = dup(1);
    require(saved >= 0 && dup2(fileno(captured), 1) >= 0, "redirect stdout");
    lua_State *L = luaL_newstate();
    require(L != nullptr, "new state for standard output");
    luaL_openlibs(L);
    require(luaL_dostring(L, "print('a', 1); io.write('b')") == 0, "write to stdout");
    lua_close(L);
    std::fflush(stdout);
    require(dup2(saved, 1) >= 0, "restore stdout");
    close(saved);
    std::string text;
    std::rewind(captured);
    char buffer[256];
    for (size_t n; (n = std::fread(buffer, 1, sizeof buffer, captured)) > 0;)
        text.append(buffer, n);
    std::fclose(captured);
    if (text != "a\t1\nb")
        std::fprintf(stderr, "captured stdout: [%s]\n", text.c_str());
    require(text == "a\t1\nb", "print and io.write reach stdout");
}

#if LUA_VERSION_NUM == 502 || LUA_VERSION_NUM == 503
static int handler_calls;
static int count_handler(lua_State *L)
{
    ++handler_calls;
    return 1;
}

// GCTM raises an error in a finalizer again as LUA_ERRGCMM with luaD_throw,
// which calls no message handler.
static void finalizer_status(lua_State *L)
{
    run(L, "setmetatable({}, {__gc = function() error('boom', 0) end})");
    lua_settop(L, 0);
    lua_pushcfunction(L, count_handler);
    lua_getglobal(L, "collectgarbage");
    require(lua_pcall(L, 0, 0, 1) == LUA_ERRGCMM, "finalizer error status");
    require(std::strcmp(lua_tostring(L, -1), "error in __gc metamethod (boom)") == 0,
            "wrapped finalizer message");
    require(handler_calls == 0, "finalizer error skips the message handler");
    lua_settop(L, 0);
}
#endif

#if LUA_VERSION_NUM == 505
static void selected_libraries()
{
    lua_State *L = luaL_newstate();
    require(L != nullptr, "new selected-library state");
    luaL_openselectedlibs(L, LUA_GLIBK | LUA_LOADLIBK | LUA_STRLIBK, LUA_MATHLIBK);
    lua_getglobal(L, "math");
    require(lua_isnil(L, -1), "preloaded math is not a global");
    lua_pop(L, 1);
    lua_getglobal(L, "coroutine");
    require(lua_isnil(L, -1), "unselected library stays unopened");
    lua_pop(L, 1);
    run(L, "assert(package.loaded.math==nil); assert(type(package.preload.math)=='function'); "
           "local mathlib=require('math'); assert(math==nil); return mathlib.sqrt(81)");
    require(lua_tointeger(L, -1) == 9, "preloaded math opens through require");
    lua_close(L);
}
#endif

int main(int argc, char **argv)
{
    allocation_failure();
    lua_State *L = luaL_newstate();
    require(L != nullptr, "new state");
    lua_getglobal(L, "math");
    require(lua_isnil(L, -1), "new state has no standard libraries before opening");
    lua_pop(L, 1);
    luaL_openlibs(L);
    luaL_checkversion(L);
    if (argc == 2) {
        lua_pushstring(L, argv[1]);
        lua_setglobal(L, "native_path");
        run(L, "local init=assert(package.loadlib(native_path,'luaopen_oximodern')); "
               "local module=init(); local calls=0; "
               "assert(module.invoke(function(value) calls=calls+1; "
               "assert(value=='a\\0b'); return value:reverse() end)=='b\\0a'); "
               "assert(calls==1); return module.abi");
        require(lua_tointeger(L, -1) == LUA_VERSION_NUM, "version-matched native module");
        lua_settop(L, 0);
    }
#if LUA_VERSION_NUM >= 503
    *static_cast<void **>(lua_getextraspace(L)) = L;
#endif
    stack_and_values(L);
    formatting(L);
    tables_and_userdata(L);
    loading(L);
    coroutines(L);
    debugging(L);
    error_handler_hooks(L);
    standard_output();
#if LUA_VERSION_NUM >= 504
    closing_and_warnings(L);
    default_warnings();
#endif
#if LUA_VERSION_NUM == 502 || LUA_VERSION_NUM == 503
    finalizer_status(L);
#endif
#if LUA_VERSION_NUM == 505
    lua_settop(L, 0);
    external_strings(L);
    selected_libraries();
#endif
    lua_close(L);
    std::printf("Lua %s modern API tests passed\n", LUA_VERSION);
    return 0;
}
