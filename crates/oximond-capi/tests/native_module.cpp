#include "lua.h"

extern "C" int luaopen_oxitest(lua_State *L)
{
    lua_newtable(L);
    lua_pushinteger(L, 73);
    lua_setfield(L, -2, "answer");
    return 1;
}
