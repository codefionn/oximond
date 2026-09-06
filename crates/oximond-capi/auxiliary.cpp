// Compile Lua's unchanged auxiliary library as C++ so errors can unwind to
// bridge.cpp, while keeping the public C symbol names in every Lua version.
extern "C" {
#include "lauxlib.c"
#if LUA_VERSION_NUM >= 502
#include "linit.c"
#endif
}
