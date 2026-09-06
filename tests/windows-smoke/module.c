/* A native module for `scripts/ci.sh windows`. It is linked against the
 * lua51.lib that oximond-capi's build script generates, so it imports the Lua
 * API from lua51.dll, as a module built for LuaJIT on Windows does. */
#include "lua.h"
#include "lauxlib.h"

static int add(lua_State *L)
{
    lua_pushnumber(L, luaL_checknumber(L, 1) + luaL_checknumber(L, 2));
    return 1;
}

/* Raises a Lua error through the module's frame. */
static int fail(lua_State *L)
{
    return luaL_error(L, "failed with %d", (int)luaL_checkinteger(L, 1));
}

/* Calls its argument, so an error unwinds from Lua through the module. */
static int call(lua_State *L)
{
    lua_settop(L, 1);
    lua_call(L, 0, 1);
    return 1;
}

/* Fills its first argument with 1 .. n through lua_rawseti, which Oximond
 * defines in Rust and which raises through the C++ bridge, so a number in
 * place of the table fails as a Lua error. */
static int fill(lua_State *L)
{
    int n = (int)luaL_checkinteger(L, 2);
    for (int i = 1; i <= n; i++) {
        lua_pushinteger(L, i);
        lua_rawseti(L, 1, i);
    }
    return 0;
}

static const luaL_Reg functions[] = {
    {"add", add},
    {"fail", fail},
    {"call", call},
    {"fill", fill},
    {NULL, NULL},
};

__declspec(dllexport) int luaopen_smoke_module(lua_State *L)
{
    luaL_register(L, "smoke_module", functions);
    return 1;
}
