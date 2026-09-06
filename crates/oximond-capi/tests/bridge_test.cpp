#include "lua.h"
#include "lauxlib.h"

extern "C" void luaL_openlibs(lua_State *L);
extern "C" void lua_enablereadonlytable(lua_State *L, int index, int enabled);
extern "C" int lua_isreadonlytable(lua_State *L, int index);

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::string error(lua_State *L)
{
    const char *text = lua_tostring(L, -1);
    return text == nullptr ? "(non-string Lua error)" : text;
}

void run(lua_State *L, const char *source)
{
    int status = luaL_loadstring(L, source);
    if (status == 0)
        status = lua_pcall(L, 0, LUA_MULTRET, 0);
    if (status != 0)
        throw std::runtime_error(std::string(source) + ": " + error(L));
}

int add(lua_State *L)
{
    lua_pushnumber(L, luaL_checknumber(L, 1) + luaL_checknumber(L, 2));
    return 1;
}

int reenter(lua_State *L)
{
    lua_getglobal(L, "inner");
    lua_pushvalue(L, 1);
    lua_call(L, 1, 1);
    lua_pushnumber(L, lua_tonumber(L, -1) + 1);
    return 1;
}

int add_upvalue(lua_State *L)
{
    lua_pushnumber(L, lua_tonumber(L, 1) +
                          lua_tonumber(L, lua_upvalueindex(1)));
    return 1;
}

int raise_from_c(lua_State *L)
{
    return luaL_error(L, "C callback failed at %d", 17);
}

int raise_from_nested_lua(lua_State *L)
{
    lua_getglobal(L, "error");
    lua_pushliteral(L, "nested Lua failed");
    lua_call(L, 1, 0);
    return 0;
}

int index_meta(lua_State *L)
{
    lua_pushfstring(L, "index:%s", lua_tostring(L, 2));
    return 1;
}

int newindex_meta(lua_State *L)
{
    lua_pushvalue(L, 2);
    lua_pushvalue(L, 3);
    lua_rawset(L, lua_upvalueindex(1));
    return 0;
}

int gc_count = 0;
int userdata_gc(lua_State *L)
{
    check(lua_touserdata(L, 1) != nullptr, "__gc has no userdata");
    ++gc_count;
    return 0;
}

std::string close_order;
int record_close(lua_State *L)
{
    close_order += luaL_checkstring(L, 1);
    return 0;
}

// lua_close runs every pending userdata finalizer in reverse creation order.
// An error in one is ignored and the rest still run, as in lua5.1. A proxy
// that stores itself in a global from its __gc is still finalized only once.
void close_runs_finalizers_in_reverse_order(lua_State *L)
{
    lua_pushcfunction(L, record_close);
    lua_setglobal(L, "record");
    run(L,
        "local a = newproxy(true); getmetatable(a).__gc = function() record('a') end\n"
        "local b = newproxy(true); getmetatable(b).__gc = function() error('boom') end\n"
        "local c = newproxy(true); getmetatable(c).__gc = function() record('c') end\n"
        "local r = newproxy(true)\n"
        "getmetatable(r).__gc = function(self) revived = self; record('r') end\n"
        "keep = {a, b, c, r}");
    check(close_order.empty(), "finalizers ran before the state closed");
}

int stack_probe(lua_State *L)
{
    lua_Debug frame{};
    check(lua_getstack(L, 0, &frame) == 1, "missing C stack frame");
    check(lua_getinfo(L, "S", &frame) == 1 &&
                  std::strcmp(frame.what, "C") == 0,
          "level 0 is not a C frame");
    check(lua_getstack(L, 1, &frame) == 1, "missing Lua caller frame");
    check(lua_getinfo(L, "S", &frame) == 1 &&
                  std::strcmp(frame.what, "Lua") == 0,
          "level 1 is not a Lua frame");
    lua_pushboolean(L, 1);
    return 1;
}

int thread_flag(lua_State *L)
{
    const int is_main = lua_pushthread(L);
    lua_pop(L, 1);
    lua_pushboolean(L, is_main);
    return 1;
}

int yield_from_c(lua_State *L)
{
    lua_pushliteral(L, "C pause");
    lua_pushnil(L);
    return lua_yield(L, 2);
}

int call_yield_from_c(lua_State *L)
{
    lua_getglobal(L, "c_yield");
    lua_call(L, 0, LUA_MULTRET);
    return lua_gettop(L);
}

lua_State *expected_callback_state = nullptr;
void *expected_shared_userdata = nullptr;

int shared_thread_callback(lua_State *L)
{
    check(L == expected_callback_state, "callback received a different thread state");
    check(lua_pushthread(L) == 0, "child callback identified as main thread");
    check(lua_tothread(L, -1) == L, "callback pushthread lost state identity");
    lua_pop(L, 1);
    lua_getfield(L, LUA_REGISTRYINDEX, "coroutine_shared_table");
    lua_getglobal(L, "coroutine_shared_table");
    check(lua_rawequal(L, -1, -2) == 1,
          "thread global and registry do not share the same table");
    check(lua_rawequal(L, -1, lua_upvalueindex(1)) == 1,
          "C closure table upvalue changed across threads");
    lua_pushinteger(L, 117);
    lua_setfield(L, -2, "value");
    lua_pop(L, 2);
    check(lua_touserdata(L, lua_upvalueindex(2)) == expected_shared_userdata,
          "C closure userdata upvalue changed across threads");
    check(*static_cast<int *>(expected_shared_userdata) == 89,
          "shared userdata contents changed");
    lua_pushinteger(L, 23);
    lua_setfield(L, LUA_REGISTRYINDEX, "coroutine_child_write");
    lua_pushinteger(L, 71);
    return 1;
}

int coroutine_gc_count = 0;
int coroutine_userdata_gc(lua_State *L)
{
    check(lua_touserdata(L, 1) != nullptr, "coroutine __gc has no userdata");
    ++coroutine_gc_count;
    return 0;
}

int rooted_closure(lua_State *L)
{
    if (lua_toboolean(L, 1)) {
        lua_newtable(L);
        lua_pushinteger(L, 22);
        lua_setfield(L, -2, "value");
        lua_replace(L, lua_upvalueindex(1));
    }
    lua_getfield(L, lua_upvalueindex(1), "value");
    const int upvalue = static_cast<int>(lua_tointeger(L, -1));
    lua_pop(L, 1);
    lua_getfield(L, LUA_ENVIRONINDEX, "marker");
    const int marker = static_cast<int>(lua_tointeger(L, -1));
    lua_pop(L, 1);
    lua_pushinteger(L, upvalue + marker);
    return 1;
}

void callbacks_and_upvalues(lua_State *L)
{
    lua_pushcfunction(L, add);
    lua_setglobal(L, "c_add");
    lua_pushcfunction(L, reenter);
    lua_setglobal(L, "reenter");
    lua_pushnumber(L, 40);
    lua_pushcclosure(L, add_upvalue, 1);
    lua_setglobal(L, "plus40");

    run(L, "function inner(x) return c_add(x, 3) end\n"
           "assert(reenter(4) == 8)\n"
           "assert(plus40(2) == 42)");
    check(lua_gettop(L) == 0, "callback stack leaked values");

    lua_getglobal(L, "plus40");
    check(lua_iscfunction(L, -1) == 1, "C closure not recognized");
    check(lua_tocfunction(L, -1) == add_upvalue,
          "lua_tocfunction lost callback address");
    lua_pop(L, 1);
    lua_getglobal(L, "inner");
    check(lua_iscfunction(L, -1) == 0, "Lua closure identified as C");
    lua_pop(L, 1);
}

// Walks the table at the top of the stack with lua_next, clearing every
// third field it meets, and returns the sum of its number values and the
// count of its entries.
static std::pair<double, int> next_walk(lua_State *L)
{
    double sum = 0;
    int count = 0;
    lua_pushnil(L);
    while (lua_next(L, -2) != 0) {
        sum += lua_tonumber(L, -1);
        count += 1;
        lua_pop(L, 1);
        if (count % 3 == 0) {
            lua_pushvalue(L, -1);
            lua_pushnil(L);
            lua_rawset(L, -4);
        }
    }
    return {sum, count};
}

// A C function that drops the only reference to a userdata, collects, and
// makes another: the storage the collection freed is taken back at the next
// entry into a C function, not at the return of this one.
static int churn_userdata(lua_State *L)
{
    int *block = static_cast<int *>(lua_newuserdata(L, sizeof(int)));
    *block = 41;
    lua_pop(L, 1);
    lua_gc(L, LUA_GCCOLLECT, 0);
    block = static_cast<int *>(lua_newuserdata(L, sizeof(int)));
    *block = 42;
    return 1;
}

static int read_userdata(lua_State *L)
{
    int *block = static_cast<int *>(lua_touserdata(L, 1));
    lua_pushinteger(L, block == nullptr ? -1 : *block);
    return 1;
}

// The entry points with a path of their own for the common case, at the
// inputs that take it and at the ones that must leave it.
void fast_path_edges(lua_State *L)
{
    const int base = lua_gettop(L);
    // lua_isnumber and lua_tonumber: numbers, strings that convert and
    // strings that do not, other types, an index past the top, an upvalue
    // pseudo-index outside a C function.
    lua_pushnumber(L, 2.5);
    lua_pushstring(L, " 0x10 ");
    lua_pushstring(L, "12abc");
    lua_pushstring(L, "");
    lua_newtable(L);
    lua_pushboolean(L, 1);
    check(lua_isnumber(L, base + 1) == 1 && lua_tonumber(L, base + 1) == 2.5,
          "a number is a number");
    check(lua_isnumber(L, base + 2) == 1 && lua_tonumber(L, base + 2) == 16,
          "a numeric string converts");
    check(lua_isnumber(L, base + 3) == 0 && lua_tonumber(L, base + 3) == 0,
          "a string with a tail does not convert");
    check(lua_isnumber(L, base + 4) == 0 && lua_tonumber(L, base + 4) == 0,
          "an empty string does not convert");
    check(lua_isnumber(L, base + 5) == 0 && lua_tonumber(L, base + 5) == 0,
          "a table is no number");
    check(lua_isnumber(L, -1) == 0 && lua_tonumber(L, -1) == 0,
          "a boolean is no number");
    check(lua_isnumber(L, base + 40) == 0 && lua_tonumber(L, base + 40) == 0,
          "an index past the top is no number");
    check(lua_isnumber(L, lua_upvalueindex(1)) == 0,
          "an upvalue outside a C function is no number");
    check(lua_type(L, base + 2) == LUA_TSTRING,
          "lua_isnumber left the string a string");
    lua_settop(L, base);

    // lua_pushvalue: stack slots from both ends, pseudo-indices, an index
    // past the top, and a stack that has to grow for the copy.
    lua_pushstring(L, "kept");
    lua_pushnumber(L, 7);
    lua_pushvalue(L, base + 1);
    check(std::strcmp(lua_tostring(L, -1), "kept") == 0, "copy of a slot");
    lua_pushvalue(L, -2);
    check(lua_tonumber(L, -1) == 7, "copy from the top down");
    lua_pushvalue(L, LUA_REGISTRYINDEX);
    check(lua_istable(L, -1), "copy of the registry");
    lua_pushvalue(L, LUA_GLOBALSINDEX);
    check(lua_istable(L, -1) && lua_rawequal(L, -1, LUA_GLOBALSINDEX),
          "copy of the globals table");
    lua_pushvalue(L, base + 60);
    check(lua_isnil(L, -1), "copy of an index past the top is nil");
    lua_settop(L, base + 1);
    check(lua_checkstack(L, 5000), "room for the copies");
    for (int i = 0; i < 5000; i++)
        lua_pushvalue(L, base + 1);
    check(lua_gettop(L) == base + 5001, "every copy landed");
    check(std::strcmp(lua_tostring(L, -1), "kept") == 0 &&
              lua_rawequal(L, base + 1, -1),
          "a copy made while the stack grew");
    lua_settop(L, base);

    // lua_touserdata and lua_type: many full userdata, light ones, the same
    // address as both, values that are neither.
    static int anchors[64];
    void *blocks[64];
    for (int i = 0; i < 64; i++) {
        blocks[i] = lua_newuserdata(L, sizeof(int));
        *static_cast<int *>(blocks[i]) = i;
    }
    for (int i = 0; i < 64; i++) {
        check(lua_touserdata(L, base + 1 + i) == blocks[i] &&
                  *static_cast<int *>(lua_touserdata(L, base + 1 + i)) == i,
              "a full userdata gives its block");
        check(lua_type(L, base + 1 + i) == LUA_TUSERDATA, "a full userdata's type");
    }
    for (int i = 0; i < 64; i++) {
        lua_pushlightuserdata(L, &anchors[i]);
        check(lua_touserdata(L, -1) == &anchors[i], "a light userdata gives its address");
        check(lua_type(L, -1) == LUA_TLIGHTUSERDATA, "a light userdata's type");
        lua_pop(L, 1);
    }
    lua_pushlightuserdata(L, blocks[3]);
    check(lua_touserdata(L, -1) == blocks[3] && lua_type(L, -1) == LUA_TLIGHTUSERDATA &&
              lua_rawequal(L, -1, base + 4) == 0,
          "a light userdata of a full one's address is another value");
    lua_pushnumber(L, 1);
    check(lua_touserdata(L, -1) == nullptr, "a number has no block");
    check(lua_touserdata(L, base + 500) == nullptr, "an index past the top has no block");
    lua_settop(L, base);

    // lua_getfield and lua_gettable of a key that the table lacks, behind
    // every kind of metatable: none, one with no __index, a chain of
    // tables that has the key or does not, a handler, a chain too long to
    // follow in place, and an __index that cannot be indexed.
    run(L, "plain = {} "
           "bare = setmetatable({}, {}) "
           "base = {inherited = 'from base'} "
           "mid = setmetatable({}, {__index = base}) "
           "leaf = setmetatable({own = 1}, {__index = mid}) "
           "handled = setmetatable({}, {__index = function(_, k) return k .. '!' end}) "
           "under = setmetatable({}, {__index = setmetatable({}, "
           "  {__index = function(_, k) return 'deep ' .. k end})}) "
           "long = {tail = 'end'} "
           "for _ = 1, 20 do long = setmetatable({}, {__index = long}) end "
           "broken = setmetatable({}, {__index = 5}) "
           "gone = setmetatable({was = 1}, {__index = base}) gone.was = nil");
    struct Read {
        const char *table;
        const char *key;
        const char *expected; // null for nil
    };
    const Read reads[] = {
        {"plain", "missing", nullptr},     {"bare", "missing", nullptr},
        {"leaf", "inherited", "from base"}, {"leaf", "missing", nullptr},
        {"mid", "missing", nullptr},       {"handled", "missing", "missing!"},
        {"under", "missing", "deep missing"}, {"long", "tail", "end"},
        {"long", "missing", nullptr},      {"gone", "was", nullptr},
        {"gone", "inherited", "from base"},
    };
    for (const Read &read : reads) {
        for (int through_gettable = 0; through_gettable < 2; through_gettable++) {
            lua_getglobal(L, read.table);
            if (through_gettable) {
                lua_pushstring(L, read.key);
                lua_gettable(L, -2);
            } else {
                lua_getfield(L, -1, read.key);
            }
            if (read.expected == nullptr)
                check(lua_isnil(L, -1), "an absent field reads nil");
            else
                check(lua_isstring(L, -1) && std::strcmp(lua_tostring(L, -1), read.expected) == 0,
                      "a field reads through its __index chain");
            lua_settop(L, base);
        }
    }
    lua_getglobal(L, "leaf");
    lua_getfield(L, -1, "own");
    check(lua_tonumber(L, -1) == 1, "an own field wins over the chain");
    lua_settop(L, base);
    check(luaL_loadstring(L, "return broken.missing") == 0 && lua_pcall(L, 0, 1, 0) != 0,
          "an __index that cannot be indexed raises from Lua");
    lua_settop(L, base);
    lua_pushcfunction(L, [](lua_State *state) -> int {
        lua_getglobal(state, "broken");
        lua_getfield(state, -1, "missing");
        return 1;
    });
    check(lua_pcall(L, 0, 1, 0) != 0, "an __index that cannot be indexed raises from C");
    lua_settop(L, base);

    // Userdata storage freed by a collection inside a C function.
    lua_pushcfunction(L, churn_userdata);
    lua_setglobal(L, "churn_userdata");
    lua_pushcfunction(L, read_userdata);
    lua_setglobal(L, "read_userdata");
    run(L, "for i = 1, 200 do "
           "  local u = churn_userdata() "
           "  assert(read_userdata(u) == 42) "
           "end "
           "collectgarbage() "
           "assert(read_userdata(churn_userdata()) == 42)");
    lua_settop(L, base);
}

void next_walks_tables(lua_State *L)
{
    // A hashed table with an array part, a small ordered one, and one whose
    // keys are mixed; each walk's sum and count must be what pairs sees.
    const char *const makers[] = {
        "local t = {} for i = 1, 10 do t[i] = i end "
        "for i = 1, 40 do t['k' .. i] = i * 2 end t[2.5] = 7 return t",
        "return {a = 1, b = 2, c = 3, d = 4}",
        "local t = {5, 6, x = 1, [true] = 9} t[100] = 3 return t",
    };
    for (const char *maker : makers) {
        check(luaL_dostring(L, maker) == 0, "next walk table failed to build");
        lua_pushvalue(L, -1);
        lua_setglobal(L, "walked");
        check(luaL_dostring(L,
                            "local s, n = 0, 0 for _, v in pairs(walked) do "
                            "s = s + (tonumber(v) or 0) n = n + 1 end return s, n") == 0,
              "pairs walk failed");
        const double expected_sum = lua_tonumber(L, -2);
        const int expected_count = static_cast<int>(lua_tointeger(L, -1));
        lua_pop(L, 2);
        const auto [sum, count] = next_walk(L);
        check(count == expected_count, "lua_next met another number of entries");
        check(sum == expected_sum, "lua_next met other values");
        // A second walk sees what the first left: every third entry gone.
        const auto [rest_sum, rest_count] = next_walk(L);
        check(rest_count == expected_count - expected_count / 3,
              "lua_next after clearing met another number of entries");
        (void)rest_sum;
        lua_pop(L, 1);
    }
    // A step from a key other than the one the last step gave.
    check(luaL_dostring(L, "local t = {} for i = 1, 30 do t['k' .. i] = i end return t") == 0,
          "next restart table failed to build");
    int seen = 0;
    lua_pushnil(L);
    while (lua_next(L, -2) != 0) {
        lua_pop(L, 1);
        seen += 1;
    }
    lua_pushstring(L, "k7");
    int after = 0;
    while (lua_next(L, -2) != 0) {
        lua_pop(L, 1);
        after += 1;
    }
    check(seen == 30 && after < 30, "lua_next from a chosen key walked wrongly");
    lua_pop(L, 1);
    check(lua_gettop(L) == 0, "next walks leaked values");
}

void registry_and_binary_strings(lua_State *L)
{
    const char payload[] = {'a', '\0', 'b', '\xff'};
    lua_pushlstring(L, payload, sizeof(payload));
    const int ref = luaL_ref(L, LUA_REGISTRYINDEX);
    check(ref > 0, "registry ref invalid");
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
    size_t length = 0;
    const char *result = lua_tolstring(L, -1, &length);
    check(result != nullptr && length == sizeof(payload) &&
                  std::memcmp(result, payload, length) == 0,
          "binary string changed in registry");
    lua_pop(L, 1);
    luaL_unref(L, LUA_REGISTRYINDEX, ref);
    check(lua_gettop(L) == 0, "registry stack leaked values");
}

void metamethods(lua_State *L)
{
    lua_newtable(L);                 // backing table
    const int backing = lua_gettop(L);
    lua_newtable(L);                 // object
    const int object = lua_gettop(L);
    lua_newtable(L);                 // metatable
    lua_pushcfunction(L, index_meta);
    lua_setfield(L, -2, "__index");
    lua_pushvalue(L, backing);
    lua_pushcclosure(L, newindex_meta, 1);
    lua_setfield(L, -2, "__newindex");
    lua_setmetatable(L, object);

    lua_getfield(L, object, "missing");
    check(std::strcmp(lua_tostring(L, -1), "index:missing") == 0,
          "__index was not called");
    lua_pop(L, 1);
    lua_pushinteger(L, 29);
    lua_setfield(L, object, "saved");
    lua_getfield(L, backing, "saved");
    check(lua_tointeger(L, -1) == 29, "__newindex did not write backing table");
    lua_settop(L, 0);
}

void protected_errors(lua_State *L)
{
    lua_pushcfunction(L, raise_from_c);
    check(lua_pcall(L, 0, 0, 0) == LUA_ERRRUN, "C luaL_error escaped pcall");
    check(error(L).find("C callback failed at 17") != std::string::npos,
          "C error message lost");
    lua_pop(L, 1);

    lua_pushcfunction(L, raise_from_nested_lua);
    check(lua_pcall(L, 0, 0, 0) == LUA_ERRRUN,
          "nested lua_call error escaped pcall");
    check(error(L).find("nested Lua failed") != std::string::npos,
          "nested error message lost");
    lua_pop(L, 1);

    run(L, "local ok, msg = pcall(function() error('Lua failed') end)\n"
           "assert(not ok and string.find(msg, 'Lua failed'))\n"
           "collectgarbage('collect')");
    check(lua_gettop(L) == 0, "protected call leaked values");

    lua_getglobal(L, "pcall");
    lua_getglobal(L, "error");
    lua_pushliteral(L, "C invoked stdlib pcall");
    check(lua_pcall(L, 2, 2, 0) == 0, "calling stdlib pcall from C failed");
    check(lua_toboolean(L, -2) == 0 &&
                  error(L).find("C invoked stdlib pcall") != std::string::npos,
          "stdlib pcall returned wrong status or error");
    lua_pop(L, 2);
}

void thread_environment(lua_State *L)
{
    lua_pushthread(L);
    lua_getfenv(L, -1);
    const int old_env = luaL_ref(L, LUA_REGISTRYINDEX);
    lua_newtable(L);
    lua_pushinteger(L, 83);
    lua_setfield(L, -2, "mark");
    check(lua_setfenv(L, -2) == 1, "lua_setfenv rejected main thread");
    lua_pop(L, 1); // thread

    lua_getglobal(L, "mark");
    check(lua_tointeger(L, -1) == 83,
          "lua_getglobal ignored main thread environment");
    lua_pop(L, 1);
    run(L, "return mark");
    check(lua_tointeger(L, -1) == 83,
          "loaded chunk ignored main thread environment");
    lua_pop(L, 1);

    lua_pushthread(L);
    lua_rawgeti(L, LUA_REGISTRYINDEX, old_env);
    check(lua_setfenv(L, -2) == 1, "could not restore thread environment");
    lua_pop(L, 1);
    luaL_unref(L, LUA_REGISTRYINDEX, old_env);
    check(lua_gettop(L) == 0, "thread environment test leaked values");
}

void userdata_environments(lua_State *L)
{
    for (int mark : {111, 222}) {
        lua_newuserdata(L, 1);
        lua_newtable(L);
        lua_pushinteger(L, mark);
        lua_setfield(L, -2, "mark");
        check(lua_setfenv(L, -2) == 1, "lua_setfenv rejected userdata environment");
    }
    lua_gc(L, LUA_GCCOLLECT, 0);
    for (int index : {1, 2}) {
        lua_getfenv(L, index);
        lua_getfield(L, -1, "mark");
        check(lua_tointeger(L, -1) == index * 111,
              "userdata environment was shared or collected");
        lua_pop(L, 2);
    }
    lua_settop(L, 0);
}

void stack_frames(lua_State *L)
{
    lua_pushcfunction(L, stack_probe);
    lua_setglobal(L, "stack_probe");
    run(L, "function outer() return stack_probe() end\nassert(outer())");
    check(lua_gettop(L) == 0, "stack frame test leaked values");
}

void coroutine_thread_identity(lua_State *L)
{
    lua_pushcfunction(L, thread_flag);
    lua_setglobal(L, "thread_flag");
    run(L, "assert(thread_flag())\n"
           "local co = coroutine.create(function() return thread_flag() end)\n"
           "local ok, is_main = coroutine.resume(co)\n"
           "assert(ok and not is_main)");
    check(lua_gettop(L) == 0, "coroutine identity test leaked values");
}

void thread_stacks(lua_State *L)
{
    lua_State *T = lua_newthread(L);
    check(T != nullptr && T != L, "lua_newthread did not create a child state");
    check(lua_gettop(L) == 1 && lua_tothread(L, 1) == T,
          "lua_newthread did not push its thread object");
    check(lua_gettop(T) == 0 && lua_status(T) == 0,
          "new thread stack or status is wrong");
    lua_pushliteral(L, "main stack");
    lua_pushinteger(T, 19);
    check(lua_gettop(L) == 2 && lua_gettop(T) == 1,
          "main and child stacks are not independent");
    check(lua_tothread(L, 2) == nullptr, "lua_tothread accepted a string");
    check(lua_pushthread(T) == 0 && lua_tothread(T, -1) == T,
          "child lua_pushthread returned the wrong state");
    lua_pop(T, 1);
    lua_pushinteger(L, 31);
    lua_pushinteger(L, 47);
    lua_xmove(L, T, 2);
    check(lua_gettop(L) == 2 && lua_gettop(T) == 3 &&
                  lua_tointeger(T, 2) == 31 && lua_tointeger(T, 3) == 47,
          "lua_xmove changed order or the wrong stack");
    lua_xmove(T, L, 2);
    check(lua_gettop(T) == 1 && lua_tointeger(T, 1) == 19 &&
                  lua_tointeger(L, -2) == 31 && lua_tointeger(L, -1) == 47,
          "lua_xmove back to main changed values");
    lua_xmove(T, T, 1);
    check(lua_gettop(T) == 1 && lua_tointeger(T, 1) == 19,
          "lua_xmove to the same state changed the stack");
    lua_settop(T, 0);
    lua_settop(L, 1);
    const int root = luaL_ref(L, LUA_REGISTRYINDEX);
    lua_gc(L, LUA_GCCOLLECT, 0);
    lua_rawgeti(L, LUA_REGISTRYINDEX, root);
    check(lua_tothread(L, -1) == T, "thread state address changed after GC");
    lua_pop(L, 1);
    luaL_unref(L, LUA_REGISTRYINDEX, root);
    check(lua_gettop(L) == 0, "thread stack test leaked main values");
}

void lua_coroutine_resume(lua_State *L)
{
    lua_State *T = lua_newthread(L);
    check(luaL_loadstring(T,
              "local original = ...; local a,b = coroutine.yield(original+1,nil,'pause'); "
              "return a,b,original") == 0,
          "could not load coroutine body");
    lua_pushinteger(T, 12);
    check(lua_resume(T, 1) == LUA_YIELD && lua_status(T) == LUA_YIELD,
          "Lua coroutine did not suspend");
    check(lua_gettop(T) == 3 && lua_tointeger(T, 1) == 13 &&
                  lua_isnil(T, 2) && std::strcmp(lua_tostring(T, 3), "pause") == 0,
          "Lua yield results are wrong");
    check(lua_gettop(L) == 1, "resume changed the main stack");
    lua_settop(T, 0);
    lua_pushliteral(T, "resumed");
    lua_pushinteger(T, 42);
    check(lua_resume(T, 2) == 0 && lua_status(T) == 0,
          "Lua coroutine did not finish");
    check(lua_gettop(T) == 3 && std::strcmp(lua_tostring(T, 1), "resumed") == 0 &&
                  lua_tointeger(T, 2) == 42 && lua_tointeger(T, 3) == 12,
          "resume arguments or suspended locals were lost");
    lua_settop(T, 0);
    check(lua_resume(T, 0) == LUA_ERRRUN &&
                  lua_isstring(T, -1),
          "dead coroutine resume did not return an error");
    lua_pop(L, 1);

    T = lua_newthread(L);
    check(luaL_loadstring(T, "error('coroutine failed')") == 0,
          "could not load failing coroutine");
    check(lua_resume(T, 0) == LUA_ERRRUN && lua_status(T) == LUA_ERRRUN &&
                  error(T).find("coroutine failed") != std::string::npos,
          "coroutine failure lost its status or error");
    lua_settop(T, 0);
    check(lua_resume(T, 0) == LUA_ERRRUN, "failed coroutine resumed again");
    lua_pop(L, 1);
}

void lua_created_thread_c_resume(lua_State *L)
{
    run(L, "return coroutine.create(function(a) local b=coroutine.yield(a+1); return b+2 end)");
    lua_State *T = lua_tothread(L, -1);
    check(T != nullptr && lua_gettop(T) == 1 && lua_isfunction(T, 1),
          "fresh Lua-created thread has no entry function on its C stack");
    lua_pushinteger(T, 5);
    check(lua_resume(T, 1) == LUA_YIELD && lua_tointeger(T, 1) == 6,
          "C could not start a Lua-created coroutine");
    lua_settop(T, 0);
    lua_pushinteger(T, 10);
    check(lua_resume(T, 1) == 0 && lua_tointeger(T, 1) == 12,
          "C could not finish a Lua-created coroutine");
    lua_settop(T, 0);
    check(lua_resume(T, 0) == LUA_ERRRUN && lua_status(T) == LUA_ERRRUN,
          "invalid dead resume did not set the error status");
    lua_pop(L, 1);

    run(L, "local co=coroutine.create(function() error('Lua-resumed failure') end); "
           "assert(not coroutine.resume(co)); return co");
    T = lua_tothread(L, -1);
    check(T != nullptr && lua_status(T) == LUA_ERRRUN,
          "C cannot observe a Lua-resumed coroutine's failure status");
    lua_pop(L, 1);
}

void c_coroutine_yield(lua_State *L)
{
    lua_pushcfunction(L, yield_from_c);
    lua_setglobal(L, "c_yield");
    lua_pushcfunction(L, call_yield_from_c);
    lua_setglobal(L, "c_nested_yield");
    lua_State *T = lua_newthread(L);
    check(luaL_loadstring(T, "local a,b=c_yield(); return a,b,99") == 0,
          "could not load C yielding coroutine");
    check(lua_resume(T, 0) == LUA_YIELD && lua_status(T) == LUA_YIELD,
          "C callback yield was treated as an error");
    check(lua_gettop(T) == 2 && std::strcmp(lua_tostring(T, 1), "C pause") == 0 &&
                  lua_isnil(T, 2), "C yield results are wrong");
    lua_settop(T, 0);
    lua_pushinteger(T, 5);
    lua_pushinteger(T, 8);
    check(lua_resume(T, 2) == 0 && lua_gettop(T) == 3 &&
                  lua_tointeger(T, 1) == 5 && lua_tointeger(T, 2) == 8 &&
                  lua_tointeger(T, 3) == 99,
          "C yield did not continue the Lua caller with resume arguments");
    lua_pop(L, 1);

    lua_getglobal(L, "c_yield");
    check(lua_pcall(L, 0, 0, 0) == LUA_ERRRUN &&
                  error(L).find("yield") != std::string::npos,
          "main-thread C yield did not report a Lua error");
    lua_pop(L, 1);
    run(L,
        "local co=coroutine.create(function() "
        "local ok,msg=pcall(c_yield); assert(not ok and string.find(msg,'yield')); "
        "return 28 end); local ok,value=coroutine.resume(co); assert(ok and value==28)\n"
        "local co=coroutine.create(function() return c_nested_yield() end); "
        "local ok,msg=coroutine.resume(co); assert(not ok and string.find(msg,'yield'))");
    check(lua_gettop(L) == 0, "C yield test leaked values");
}

void lua_created_thread_shared_objects(lua_State *L)
{
    lua_newtable(L);
    lua_pushvalue(L, -1);
    lua_setglobal(L, "coroutine_shared_table");
    lua_pushvalue(L, -1);
    lua_setfield(L, LUA_REGISTRYINDEX, "coroutine_shared_table");
    expected_shared_userdata = lua_newuserdata(L, sizeof(int));
    *static_cast<int *>(expected_shared_userdata) = 89;
    lua_pushcclosure(L, shared_thread_callback, 2);
    lua_setglobal(L, "shared_thread_callback");
    run(L, "return coroutine.create(function() coroutine.yield('ready'); "
           "return shared_thread_callback() end)");
    lua_State *T = lua_tothread(L, -1);
    check(T != nullptr && T != L, "lua_tothread rejected a Lua-created coroutine");
    const int root = luaL_ref(L, LUA_REGISTRYINDEX);
    check(lua_resume(T, 0) == LUA_YIELD, "Lua-created coroutine could not resume from C");
    lua_settop(T, 0);
    lua_gc(L, LUA_GCCOLLECT, 0);
    lua_rawgeti(L, LUA_REGISTRYINDEX, root);
    check(lua_tothread(L, -1) == T, "Lua-created thread state changed after GC");
    lua_pop(L, 1);
    expected_callback_state = T;
    check(lua_resume(T, 0) == 0 && lua_gettop(T) == 1 && lua_tointeger(T, 1) == 71,
          "Lua-created coroutine callback failed");
    expected_callback_state = nullptr;
    lua_getglobal(L, "coroutine_shared_table");
    lua_getfield(L, -1, "value");
    check(lua_tointeger(L, -1) == 117, "child global write was not visible to main");
    lua_pop(L, 2);
    lua_getfield(L, LUA_REGISTRYINDEX, "coroutine_child_write");
    check(lua_tointeger(L, -1) == 23, "child registry write was not visible to main");
    lua_pop(L, 1);
    luaL_unref(L, LUA_REGISTRYINDEX, root);
    lua_pushnil(L);
    lua_setglobal(L, "shared_thread_callback");
    check(lua_gettop(L) == 0, "shared thread objects test leaked values");
}

void suspended_coroutine_gc_roots(lua_State *L)
{
    lua_State *T = lua_newthread(L);
    const int root = luaL_ref(L, LUA_REGISTRYINDEX);
    check(luaL_loadstring(T,
              "local owned=...; local data={answer=123}; "
              "local text=string.rep('kept',80); coroutine.yield('held'); "
              "return owned,data,text") == 0,
          "could not load GC coroutine");
    void *owned = lua_newuserdata(T, sizeof(int));
    *static_cast<int *>(owned) = 76;
    lua_newtable(T);
    lua_pushcfunction(T, coroutine_userdata_gc);
    lua_setfield(T, -2, "__gc");
    lua_setmetatable(T, -2);
    check(lua_resume(T, 1) == LUA_YIELD, "GC coroutine did not suspend");
    lua_settop(T, 0);
    lua_gc(L, LUA_GCCOLLECT, 0);
    check(coroutine_gc_count == 0, "GC finalized userdata held by suspended coroutine");
    check(lua_resume(T, 0) == 0 && lua_gettop(T) == 3,
          "GC coroutine did not return its suspended locals");
    check(lua_touserdata(T, 1) == owned && *static_cast<int *>(owned) == 76,
          "GC lost suspended userdata");
    lua_getfield(T, 2, "answer");
    check(lua_tointeger(T, -1) == 123, "GC lost suspended table");
    lua_pop(T, 1);
    check(lua_objlen(T, 3) == 320, "GC lost suspended string");
    lua_settop(T, 0);
    luaL_unref(L, LUA_REGISTRYINDEX, root);
    lua_gc(L, LUA_GCCOLLECT, 0);
    check(coroutine_gc_count == 1, "unrooted coroutine userdata was not finalized");
    check(lua_gettop(L) == 0, "coroutine GC test leaked main values");
}

void closure_gc_roots(lua_State *L)
{
    lua_newtable(L);
    lua_pushinteger(L, 21);
    lua_setfield(L, -2, "value");
    lua_pushcclosure(L, rooted_closure, 1);
    lua_newtable(L);
    lua_pushinteger(L, 34);
    lua_setfield(L, -2, "marker");
    check(lua_setfenv(L, -2) == 1, "could not set C closure environment");
    const int ref = luaL_ref(L, LUA_REGISTRYINDEX);
    check(lua_gettop(L) == 0, "closure root setup leaked values");

    const auto invoke = [L, ref](bool replace_upvalue, int expected) {
        lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
        lua_pushboolean(L, replace_upvalue);
        check(lua_pcall(L, 1, 1, 0) == 0, "rooted C closure call failed");
        check(lua_tointeger(L, -1) == expected,
              "C closure upvalue or environment lost after GC");
        lua_pop(L, 1);
    };

    lua_gc(L, LUA_GCCOLLECT, 0);
    invoke(false, 55);
    invoke(true, 56);
    lua_gc(L, LUA_GCCOLLECT, 0);
    invoke(false, 56);
    luaL_unref(L, LUA_REGISTRYINDEX, ref);
    check(lua_gettop(L) == 0, "closure root test leaked values");
}

void binary_loadbuffer(lua_State *L)
{
    run(L, "return string.dump(function() return 91 end)");
    size_t length = 0;
    const char *data = lua_tolstring(L, -1, &length);
    check(data != nullptr && length > 4 && data[0] == '\033',
          "string.dump did not return a binary chunk");
    const std::string chunk(data, length);
    lua_pop(L, 1);
    const int status = luaL_loadbuffer(L, chunk.data(), chunk.size(), "=binary");
    if (status != 0)
        throw std::runtime_error("luaL_loadbuffer rejected string.dump output: " + error(L));
    check(lua_pcall(L, 0, 1, 0) == 0,
          "binary chunk failed when called through C API");
    check(lua_tointeger(L, -1) == 91, "binary chunk returned wrong result");
    lua_pop(L, 1);
}

void userdata_lifetime(lua_State *L)
{
    constexpr size_t size = 33;
    void *memory = lua_newuserdata(L, size);
    check(memory != nullptr, "lua_newuserdata returned null");
    check(reinterpret_cast<std::uintptr_t>(memory) % alignof(std::max_align_t) == 0,
          "userdata storage is under-aligned");
    check(lua_objlen(L, -1) == size, "userdata size changed");
    std::memset(memory, 0x5a, size);
    lua_newtable(L);
    lua_pushcfunction(L, userdata_gc);
    lua_setfield(L, -2, "__gc");
    lua_setmetatable(L, -2);
    lua_pop(L, 1);
    check(lua_gc(L, LUA_GCCOLLECT, 0) >= 0, "explicit lua_gc failed");
    check(gc_count == 1, "explicit GC did not finalize userdata once");
    check(lua_gc(L, LUA_GCCOLLECT, 0) >= 0, "second lua_gc failed");
    check(gc_count == 1, "userdata __gc ran twice on explicit GC");
}

void native_module(lua_State *L, const char *path)
{
    lua_getglobal(L, "package");
    lua_getfield(L, -1, "loadlib");
    lua_pushstring(L, path);
    lua_pushliteral(L, "luaopen_oxitest");
    check(lua_pcall(L, 2, 1, 0) == 0, "package.loadlib raised an error");
    check(lua_isfunction(L, -1), "package.loadlib did not return a loader");
    check(lua_pcall(L, 0, 1, 0) == 0, "native loader failed");
    lua_getfield(L, -1, "answer");
    check(lua_tointeger(L, -1) == 73, "native module returned wrong value");
    lua_settop(L, 0);

    lua_getglobal(L, "package");
    lua_pushstring(L, path);
    lua_setfield(L, -2, "cpath");
    lua_pop(L, 1);
    run(L, "local module = require('oxitest'); assert(module.answer == 73)");
    lua_settop(L, 0);

    lua_getglobal(L, "package");
    lua_getfield(L, -1, "loadlib");
    lua_pushliteral(L, "/missing/oxitest.so");
    lua_pushliteral(L, "luaopen_oxitest");
    check(lua_pcall(L, 2, 3, 0) == 0, "missing library raised an error");
    check(lua_isnil(L, -3), "missing library did not return nil");
    check(lua_isstring(L, -2), "missing library has no diagnostic");
    check(std::strcmp(lua_tostring(L, -1), "open") == 0,
          "missing library has wrong failure stage");
    lua_settop(L, 0);

    lua_getglobal(L, "package");
    lua_getfield(L, -1, "loadlib");
    lua_pushstring(L, path);
    lua_pushliteral(L, "luaopen_missing_symbol");
    check(lua_pcall(L, 2, 3, 0) == 0, "missing symbol raised an error");
    check(lua_isnil(L, -3) && lua_isstring(L, -2),
          "missing symbol did not return a diagnostic");
    check(std::strcmp(lua_tostring(L, -1), "init") == 0,
          "missing symbol has wrong failure stage");
    lua_settop(L, 0);
}

void readonly_tables(lua_State *L)
{
    lua_newtable(L);
    lua_pushinteger(L, 2);
    lua_rawseti(L, -2, 1);
    lua_pushinteger(L, 1);
    lua_rawseti(L, -2, 2);
    lua_pushinteger(L, 1);
    lua_setfield(L, -2, "x");
    lua_pushvalue(L, -1);
    lua_setglobal(L, "protected_table");
    lua_enablereadonlytable(L, -1, 1);
    check(lua_isreadonlytable(L, -1), "readonly table flag was not set");
    run(L,
        "local t=protected_table; local writes={"
        "function() t.x=2 end, function() rawset(t,'x',2) end, "
        "function() setmetatable(t,{}) end, function() table.insert(t,3) end, "
        "function() table.remove(t) end, function() table.sort(t) end}; "
        "for _,write in ipairs(writes) do local ok,err=pcall(write); "
        "assert(not ok and tostring(err):find('readonly',1,true)) end; "
        "assert(t.x==1 and t[1]==2 and t[2]==1 and #t==2)");
    lua_enablereadonlytable(L, 1, 0);
    run(L, "protected_table.x=2; assert(protected_table.x==2)");
    lua_settop(L, 0);
}

int kill_hook_calls = 0;
void kill_hook(lua_State *L, lua_Debug *)
{
    ++kill_hook_calls;
    lua_sethook(L, kill_hook, LUA_MASKLINE, 0);
    lua_pushliteral(L, "hook killed");
    lua_error(L);
}

void hook_error_handler(lua_State *L)
{
    lua_settop(L, 0);
    run(L, "return function(err) return 'handled:'..err end");
    check(luaL_loadstring(L, "local sum=0; for i=1,100 do sum=sum+i end; return sum") == 0,
          "load hook interrupted script");
    lua_sethook(L, kill_hook, LUA_MASKCOUNT, 7);
    int status = lua_pcall(L, 0, 1, 1);
    lua_sethook(L, nullptr, 0, 0);
    check(status == LUA_ERRRUN, "hook error lost runtime status");
    check(std::strcmp(lua_tostring(L, -1), "handled:hook killed") == 0,
          "hook interrupted the message handler");
    check(kill_hook_calls == 1, "hook was reentered during message handling");
    lua_settop(L, 0);
}

} // namespace

int main(int argc, char **argv)
{
    lua_State *L = luaL_newstate();
    if (L == nullptr) {
        std::cerr << "luaL_newstate failed\n";
        return 1;
    }
    try {
        luaL_openlibs(L);
        callbacks_and_upvalues(L);
        registry_and_binary_strings(L);
        next_walks_tables(L);
        fast_path_edges(L);
        metamethods(L);
        protected_errors(L);
        thread_environment(L);
        userdata_environments(L);
        readonly_tables(L);
        hook_error_handler(L);
        stack_frames(L);
        coroutine_thread_identity(L);
        thread_stacks(L);
        lua_coroutine_resume(L);
        lua_created_thread_c_resume(L);
        c_coroutine_yield(L);
        lua_created_thread_shared_objects(L);
        suspended_coroutine_gc_roots(L);
        closure_gc_roots(L);
        binary_loadbuffer(L);
        userdata_lifetime(L);
        close_runs_finalizers_in_reverse_order(L);
        check(argc == 2, "expected native module path");
        native_module(L, argv[1]);
        lua_close(L);
        L = nullptr;
        check(gc_count == 1, "userdata __gc did not run exactly once");
        check(close_order == "rca", "lua_close finalizer order is not r, c, a");
        check(std::count(close_order.begin(), close_order.end(), 'r') == 1,
              "a resurrected proxy was finalized more than once");
        std::cout << "C API bridge integration passed\n";
        return 0;
    } catch (const std::exception &failure) {
        std::cerr << "C API bridge integration failed: " << failure.what() << '\n';
        if (L != nullptr)
            lua_close(L);
        return 1;
    }
}
