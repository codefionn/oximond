extern "C" {
#include "lua.h"
#include "lauxlib.h"
}

static int invoke(lua_State *L)
{
    luaL_checktype(L, 1, LUA_TFUNCTION);
    lua_pushvalue(L, 1);
    lua_pushlstring(L, "a\0b", 3);
    lua_call(L, 1, 1);
    return 1;
}

extern "C" int luaopen_oximodern(lua_State *L)
{
    const luaL_Reg functions[] = {{"invoke", invoke}, {nullptr, nullptr}};
    luaL_newlib(L, functions);
    lua_pushinteger(L, LUA_VERSION_NUM);
    lua_setfield(L, -2, "abi");
    return 1;
}
