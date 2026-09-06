/* Source compatibility for modules using the shared Lua 5.1 API subset.
 * The C API host uses the Lua 5.1 ABI. This header lets the official fixtures
 * register their functions without linking a stock interpreter. */
#ifndef OXIMOND_MODERN_MODULE_H
#define OXIMOND_MODERN_MODULE_H
#include "lua.h"
#include "lauxlib.h"
#define LUAMOD_API LUALIB_API
#define luaL_newlib(L, funcs) \
    (lua_newtable(L), luaL_register(L, NULL, funcs))
LUA_API const char *lua_pushexternalstring(lua_State *L, const char *s,
    size_t len, lua_Alloc allocator, void *ud);
#endif
