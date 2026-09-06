extern "C" {
#include "lua.h"
#include "lauxlib.h"
}

#include <cstdio>
#include <cctype>
#include <cstring>
#include <exception>
#include <string>
#include <sys/stat.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// These operations can execute Lua code (including metamethods). Rust reports
// their errors as a status with the error object at the top of the Lua stack.
// Only this C++ translation unit turns that status into a Lua-style unwind.
extern "C" {
int oxi_call(lua_State *L, int nargs, int nresults);
int oxi_pcall(lua_State *L, int nargs, int nresults, int errfunc);
int oxi_gettable(lua_State *L, int idx);
int oxi_getfield(lua_State *L, int idx, const char *key);
int oxi_setfield(lua_State *L, int idx, const char *key);
int oxi_settable(lua_State *L, int idx);
int oxi_concat(lua_State *L, int n);
int oxi_compare(lua_State *L, int lhs, int rhs, int op, int *result);
int oxi_yield(lua_State *L, int nresults);
int oxi_newthread(lua_State *L, lua_State **result);
int oxi_newuserdata(lua_State *L, size_t size, int uservalues, void **result);
int oxi_rawset(lua_State *L, int idx);
#if LUA_VERSION_NUM >= 503
int oxi_rawseti(lua_State *L, int idx, lua_Integer n);
#else
int oxi_rawseti(lua_State *L, int idx, int n);
#endif
int oxi_setmetatable(lua_State *L, int idx);
int oxi_enablereadonlytable(lua_State *L, int idx, int enabled);
#if LUA_VERSION_NUM >= 502
int oxi_rawsetp(lua_State *L, int idx, const void *p);
#if LUA_VERSION_NUM == 502
using OxiContext = int;
using OxiContinuation = lua_CFunction;
#else
using OxiContext = lua_KContext;
using OxiContinuation = lua_KFunction;
#endif
int oxi_callk(lua_State *L, int nargs, int nresults, OxiContext ctx, OxiContinuation k);
int oxi_pcallk(lua_State *L, int nargs, int nresults, int errfunc,
               OxiContext ctx, OxiContinuation k);
int oxi_yieldk(lua_State *L, int nresults, OxiContext ctx, OxiContinuation k);
int oxi_len(lua_State *L, int idx);
int oxi_arith(lua_State *L, int op);
int oxi_getglobal(lua_State *L, const char *name);
int oxi_setglobal(lua_State *L, const char *name);
#endif
#if LUA_VERSION_NUM >= 503
int oxi_geti(lua_State *L, int idx, lua_Integer n);
int oxi_seti(lua_State *L, int idx, lua_Integer n);
#endif
#if LUA_VERSION_NUM >= 504
int oxi_settop(lua_State *L, int idx);
int oxi_toclose(lua_State *L, int idx);
int oxi_closeslot(lua_State *L, int idx);
int oxi_gc(lua_State *L, int what, size_t a, int b, int c);
int oxi_warning(lua_State *L, const char *message, int continued);
#endif

// Rust calls this instead of calling a lua_CFunction directly. No C++
// exception is allowed to unwind through a Rust frame.
int oximond_invoke(lua_State *L, lua_CFunction function, int *nresults);
void oximond_remember_library(lua_State *L, void *handle);
int oximond_modern_module_names(lua_State *L);
}

namespace {

struct LuaError {
    int status = LUA_ERRRUN;
};
struct LuaYield {
    int nresults;
};

void check_status(int status)
{
    if (status != 0)
        throw LuaError{status};
}

void append_number(std::string &out, const char *format, double value)
{
    char buffer[128];
    const int length = std::snprintf(buffer, sizeof(buffer), format, value);
    if (length > 0)
        out.append(buffer, static_cast<size_t>(length));
}

} // namespace

namespace {

#ifdef _WIN32
// The system's text for the calling thread's last error, as loadlib.c reports it.
std::string last_error_message()
{
    const DWORD error = GetLastError();
    char buffer[512];
    const DWORD length = FormatMessageA(
        FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, error, 0, buffer,
        sizeof(buffer), nullptr);
    if (length == 0) {
        std::snprintf(buffer, sizeof(buffer), "system error %lu\n", error);
        return buffer;
    }
    return std::string(buffer, length);
}

#ifdef OXIMOND_LUA_DLL
// A Windows module imports the Lua API from the DLL its import library names,
// lua51.dll for the 5.1 ABI. build.rs generates that DLL: a thunk per API
// function that jumps through a slot. The loader binds an import to a loaded
// module of the same name, so loading the DLL from a per-user cache directory
// and filling its slots with this executable's functions, before the first
// module loads, routes every module's calls here.
#include "lua_dll_bytes.inc"

void *const lua_dll_table[] = {
#include "lua_dll_table.inc"
};

bool lua_dll_matches(const std::wstring &path)
{
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return false;
    static unsigned char buffer[sizeof(lua_dll_bytes) + 1];
    DWORD read = 0;
    const bool same = ReadFile(file, buffer, sizeof(buffer), &read, nullptr) &&
                      read == sizeof(lua_dll_bytes) &&
                      std::memcmp(buffer, lua_dll_bytes, sizeof(lua_dll_bytes)) == 0;
    CloseHandle(file);
    return same;
}

// Leaves the DLL's bytes at `path`. A file is written under a name of this
// process's own and moved into place, so a concurrent start never loads a
// partial one.
bool lua_dll_write(const std::wstring &path, std::string &message)
{
    if (lua_dll_matches(path))
        return true;
    const std::wstring temporary = path + L"." + std::to_wstring(GetCurrentProcessId());
    HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        message = last_error_message();
        return false;
    }
    DWORD written = 0;
    const bool complete = WriteFile(file, lua_dll_bytes, sizeof(lua_dll_bytes), &written,
                                    nullptr) &&
                          written == sizeof(lua_dll_bytes);
    CloseHandle(file);
    if (complete && MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
        return true;
    message = last_error_message();
    DeleteFileW(temporary.c_str());
    // Another process may have moved an identical file into place first.
    return lua_dll_matches(path);
}

std::string lua_dll_load()
{
    wchar_t temp[MAX_PATH + 1];
    const DWORD length = GetTempPathW(MAX_PATH + 1, temp);
    if (length == 0 || length > MAX_PATH)
        return last_error_message();
    const std::wstring directory = std::wstring(temp, length) + lua_dll_cache;
    if (!CreateDirectoryW(directory.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
        return last_error_message();
    using Bind = void (*)(void *const *);
    for (const wchar_t *name : lua_dll_names) {
        // A Lua DLL that something else loaded under this name keeps its imports.
        HMODULE module = GetModuleHandleW(name);
        if (module == nullptr) {
            const std::wstring path = directory + L"\\" + name;
            std::string message;
            if (!lua_dll_write(path, message))
                return message;
            module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (module == nullptr)
                return last_error_message();
        }
        if (FARPROC bind = GetProcAddress(module, "oximond_lua_bind"))
            reinterpret_cast<Bind>(reinterpret_cast<void *>(bind))(lua_dll_table);
    }
    return {};
}

// Why the Lua DLL could not be prepared, or empty once it is bound.
const std::string &lua_dll_failure()
{
    static const std::string failure = lua_dll_load();
    return failure;
}
#endif

bool native_loader(lua_State *L, const char *path, const char *symbol,
                   std::string &message, const char *&stage)
{
#ifdef OXIMOND_LUA_DLL
    const std::string &lua_dll = lua_dll_failure();
#endif
    HMODULE handle = LoadLibraryExA(path, nullptr, 0);
    if (handle == nullptr) {
        stage = "open";
        message = last_error_message();
#ifdef OXIMOND_LUA_DLL
        if (!lua_dll.empty())
            message += "(preparing the Lua DLL failed: " + lua_dll + ")";
#endif
        return false;
    }
    FARPROC address = GetProcAddress(handle, symbol);
    if (address == nullptr) {
        stage = "init";
        message = last_error_message();
        FreeLibrary(handle);
        return false;
    }
    oximond_remember_library(L, handle);
    lua_pushcfunction(L, reinterpret_cast<lua_CFunction>(address));
    return true;
}
#else
bool native_loader(lua_State *L, const char *path, const char *symbol,
                   std::string &message, const char *&stage)
{
    dlerror();
    void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        stage = "open";
        message = dlerror();
        return false;
    }
    dlerror();
    void *address = dlsym(handle, symbol);
    const char *error = dlerror();
    if (error != nullptr) {
        stage = "init";
        message = error;
        dlclose(handle);
        return false;
    }
    oximond_remember_library(L, handle);
    lua_pushcfunction(L, reinterpret_cast<lua_CFunction>(address));
    return true;
}
#endif

std::string native_symbol(const std::string &module, bool modern)
{
    std::string name = module;
    const size_t suffix = name.find('-');
    if (suffix != std::string::npos) {
        if (modern)
            name.erase(suffix);
        else
            name.erase(0, suffix + 1);
    }
    for (char &character : name) {
        if (character == '.')
            character = '_';
    }
    return "luaopen_" + name;
}

} // namespace

extern "C" int oximond_package_loadlib(lua_State *L)
{
    const char *path = luaL_checkstring(L, 1);
    const char *symbol = luaL_checkstring(L, 2);
    std::string message;
    const char *stage = nullptr;
    if (native_loader(L, path, symbol, message, stage))
        return 1;
    lua_pushnil(L);
    lua_pushlstring(L, message.data(), message.size());
    lua_pushstring(L, stage);
    return 3;
}

extern "C" int oximond_package_native_searcher(lua_State *L)
{
    const char *module = luaL_checkstring(L, 1);
    const bool modern = oximond_modern_module_names(L) != 0;
    const std::string symbol = native_symbol(module, modern);
    std::string filename(module);
    for (char &character : filename) {
        if (character == '.')
            character = '/';
    }

    lua_getglobal(L, "package");
    lua_getfield(L, -1, "cpath");
    const char *cpath = lua_tostring(L, -1);
    if (cpath == nullptr)
        return luaL_error(L, "package.cpath must be a string");
    const std::string paths(cpath);
    lua_pop(L, 2);
    std::string diagnostics;
    size_t start = 0;
    while (start <= paths.size()) {
        const size_t end = paths.find(';', start);
        const std::string pattern = paths.substr(start, end - start);
        if (!pattern.empty()) {
            std::string path;
            for (char character : pattern) {
                if (character == '?')
                    path += filename;
                else
                    path += character;
            }
            struct stat info {};
            if (stat(path.c_str(), &info) == 0) {
                std::string message;
                const char *stage = nullptr;
                if (native_loader(L, path.c_str(), symbol.c_str(), message, stage)) {
                    lua_pushlstring(L, path.data(), path.size());
                    return 2;
                }
                // Lua tries the portion before the version separator first,
                // then the portion after it for the older naming convention.
                const std::string module_name(module);
                const size_t suffix = module_name.find('-');
                if (modern && suffix != std::string::npos &&
                    native_loader(L, path.c_str(),
                        native_symbol(module_name.substr(suffix + 1), modern).c_str(),
                        message, stage)) {
                    lua_pushlstring(L, path.data(), path.size());
                    return 2;
                }
                return luaL_error(L, "error loading module '%s' from file '%s':\n\t%s",
                                  module, path.c_str(), message.c_str());
            }
            diagnostics += "\n\tno file '" + path + "'";
        }
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    lua_pushlstring(L, diagnostics.data(), diagnostics.size());
    return 1;
}

extern "C" void oximond_close_library(void *handle)
{
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

extern "C" int oximond_invoke(lua_State *L, lua_CFunction function, int *nresults)
{
    try {
        *nresults = function(L);
        return 0;
    } catch (const LuaYield &yield) {
        *nresults = yield.nresults;
        return LUA_YIELD;
    } catch (const LuaError &error) {
        // lua_error, or one of the status-returning Rust operations, already
        // left the original error object on the stack.
        return error.status;
    } catch (const std::exception &error) {
        lua_pushstring(L, error.what());
        return LUA_ERRRUN;
    } catch (...) {
        lua_pushliteral(L, "unknown C++ exception in Lua callback");
        return LUA_ERRRUN;
    }
}

extern "C" int lua_error(lua_State *)
{
    throw LuaError{};
}

#if LUA_VERSION_NUM == 503
extern "C" int luaopen_bit32(lua_State *L)
{
    return luaL_error(L, "library 'bit32' has been deprecated");
}
#endif

extern "C" int oximond_allocate(lua_State *, lua_Alloc allocator, void *data,
                                  void *pointer, size_t oldsize, size_t newsize,
                                  void **result)
{
    try {
        *result = allocator(data, pointer, oldsize, newsize);
        return newsize != 0 && *result == nullptr ? LUA_ERRMEM : 0;
    } catch (...) {
        *result = nullptr;
        return LUA_ERRMEM;
    }
}

#if LUA_VERSION_NUM >= 504
extern "C" void *lua_newuserdatauv(lua_State *L, size_t size, int uservalues)
#else
extern "C" void *lua_newuserdata(lua_State *L, size_t size)
#endif
{
    void *result = nullptr;
#if LUA_VERSION_NUM >= 504
    check_status(oxi_newuserdata(L, size, uservalues, &result));
#else
    check_status(oxi_newuserdata(L, size, 1, &result));
#endif
    return result;
}

extern "C" int oximond_invoke_hook(lua_State *L, lua_Hook hook, lua_Debug *record)
{
    try {
        hook(L, record);
        return 0;
    } catch (const LuaYield &yield) {
        if (yield.nresults == 0)
            return LUA_YIELD;
        lua_pushliteral(L, "debug hooks cannot yield values");
        return LUA_ERRRUN;
    } catch (const LuaError &error) {
        return error.status;
    } catch (const std::exception &error) {
        lua_pushstring(L, error.what());
        return LUA_ERRRUN;
    } catch (...) {
        lua_pushliteral(L, "unknown C++ exception in Lua hook");
        return LUA_ERRRUN;
    }
}

#if LUA_VERSION_NUM >= 504
extern "C" int oximond_invoke_warning(lua_State *, lua_WarnFunction warning,
                                        void *data, const char *message, int continued)
{
    try {
        warning(data, message, continued);
        return LUA_OK;
    } catch (...) {
        // Warning callbacks have no Lua state argument. A Runtime can also
        // outlive the NativeLua wrapper after a move, so do not use a cached
        // VM address to construct an error here.
        return LUA_ERRRUN;
    }
}
#endif

#if LUA_VERSION_NUM == 501
extern "C" int lua_yield(lua_State *L, int nresults)
{
    check_status(oxi_yield(L, nresults));
    throw LuaYield{nresults};
}
#else
extern "C" int oximond_invoke_continuation(lua_State *L, OxiContinuation k,
                                             int status, OxiContext context,
                                             int *nresults)
{
    try {
#if LUA_VERSION_NUM == 502
        *nresults = k(L);
#else
        *nresults = k(L, status, context);
#endif
        return LUA_OK;
    } catch (const LuaYield &yield) {
        *nresults = yield.nresults;
        return LUA_YIELD;
    } catch (const LuaError &error) {
        return error.status;
    } catch (const std::exception &error) {
        lua_pushstring(L, error.what());
        return LUA_ERRRUN;
    } catch (...) {
        lua_pushliteral(L, "unknown C++ exception in Lua continuation");
        return LUA_ERRRUN;
    }
}

extern "C" int lua_yieldk(lua_State *L, int nresults, OxiContext context,
                            OxiContinuation k)
{
    check_status(oxi_yieldk(L, nresults, context, k));
    throw LuaYield{nresults};
}

extern "C" void lua_callk(lua_State *L, int nargs, int nresults,
                            OxiContext context, OxiContinuation k)
{
    const int status = oxi_callk(L, nargs, nresults, context, k);
    if (status == LUA_YIELD)
        throw LuaYield{lua_gettop(L)};
    check_status(status);
}

extern "C" int lua_pcallk(lua_State *L, int nargs, int nresults, int errfunc,
                            OxiContext context, OxiContinuation k)
{
    const int status = oxi_pcallk(L, nargs, nresults, errfunc, context, k);
    if (status == LUA_YIELD)
        throw LuaYield{lua_gettop(L)};
    return status;
}
#endif

extern "C" lua_State *lua_newthread(lua_State *L)
{
    lua_State *result = nullptr;
    check_status(oxi_newthread(L, &result));
    return result;
}

#if LUA_VERSION_NUM == 501
extern "C" void lua_call(lua_State *L, int nargs, int nresults)
{
    check_status(oxi_call(L, nargs, nresults));
}

extern "C" int lua_pcall(lua_State *L, int nargs, int nresults, int errfunc)
{
    return oxi_pcall(L, nargs, nresults, errfunc);
}

extern "C" int lua_cpcall(lua_State *L, lua_CFunction function, void *userdata)
{
    lua_pushcfunction(L, function);
    lua_pushlightuserdata(L, userdata);
    return lua_pcall(L, 1, 0, 0);
}
#endif

extern "C" int lua_dump(lua_State *L, lua_Writer writer, void *userdata
#if LUA_VERSION_NUM >= 503
                          , int strip
#endif
                          )
{
    const int top = lua_gettop(L);
    if (top == 0 || !lua_isfunction(L, -1))
        return 1;

    lua_getglobal(L, "string");
    lua_getfield(L, -1, "dump");
    lua_pushvalue(L, top);
#if LUA_VERSION_NUM >= 503
    lua_pushboolean(L, strip);
    const int status = lua_pcall(L, 2, 1, 0);
#else
    const int status = lua_pcall(L, 1, 1, 0);
#endif
    if (status != 0) {
        lua_settop(L, top);
        return status;
    }

    size_t length = 0;
    const char *bytes = lua_tolstring(L, -1, &length);
    const int result = bytes == nullptr ? 1 : writer(L, bytes, length, userdata);
    lua_settop(L, top);
    return result;
}

#if LUA_VERSION_NUM >= 503
using GetResult = int;
#else
using GetResult = void;
#endif

extern "C" GetResult lua_gettable(lua_State *L, int idx)
{
    check_status(oxi_gettable(L, idx));
#if LUA_VERSION_NUM >= 503
    return lua_type(L, -1);
#endif
}

extern "C" GetResult lua_getfield(lua_State *L, int idx, const char *key)
{
    check_status(oxi_getfield(L, idx, key));
#if LUA_VERSION_NUM >= 503
    return lua_type(L, -1);
#endif
}

extern "C" void lua_settable(lua_State *L, int idx)
{
    check_status(oxi_settable(L, idx));
}

extern "C" void lua_setfield(lua_State *L, int idx, const char *key)
{
    check_status(oxi_setfield(L, idx, key));
}

extern "C" void lua_rawset(lua_State *L, int idx)
{
    check_status(oxi_rawset(L, idx));
}

// Outside the lazy bridge, src/lib.rs defines lua_rawseti itself and calls
// this for a store its fast path declines.
#ifdef OXIMOND_LAZY_BRIDGE
extern "C" void lua_rawseti(lua_State *L, int idx,
#else
extern "C" void oxi_rawseti_raise(lua_State *L, int idx,
#endif
#if LUA_VERSION_NUM >= 503
                              lua_Integer n
#else
                              int n
#endif
                              )
{
    check_status(oxi_rawseti(L, idx, n));
}

extern "C" int lua_setmetatable(lua_State *L, int idx)
{
    check_status(oxi_setmetatable(L, idx));
    return 1;
}

extern "C" void lua_enablereadonlytable(lua_State *L, int idx, int enabled)
{
    check_status(oxi_enablereadonlytable(L, idx, enabled));
}

static int open_bundled(lua_State *L, const char *name)
{
    lua_getglobal(L, "require");
    lua_pushstring(L, name);
    lua_call(L, 1, 1);
    lua_pushvalue(L, -1);
    lua_setglobal(L, name);
    return 1;
}

extern "C" int luaopen_cjson(lua_State *L)
{
    return open_bundled(L, "cjson");
}

extern "C" int luaopen_struct(lua_State *L)
{
    return open_bundled(L, "struct");
}

extern "C" int luaopen_cmsgpack(lua_State *L)
{
    return open_bundled(L, "cmsgpack");
}

#if LUA_VERSION_NUM >= 502
extern "C" void lua_rawsetp(lua_State *L, int idx, const void *p)
{
    check_status(oxi_rawsetp(L, idx, p));
}
#endif

extern "C" void lua_concat(lua_State *L, int n)
{
    check_status(oxi_concat(L, n));
}

#if LUA_VERSION_NUM == 501
extern "C" int lua_equal(lua_State *L, int lhs, int rhs)
{
    int result = 0;
    check_status(oxi_compare(L, lhs, rhs, 0, &result));
    return result;
}
#else
extern "C" int lua_compare(lua_State *L, int lhs, int rhs, int op)
{
    int result = 0;
    check_status(oxi_compare(L, lhs, rhs, op, &result));
    return result;
}

extern "C" void lua_arith(lua_State *L, int op)
{
    check_status(oxi_arith(L, op));
}

extern "C" void lua_len(lua_State *L, int idx)
{
    check_status(oxi_len(L, idx));
}

extern "C" GetResult lua_getglobal(lua_State *L, const char *name)
{
    check_status(oxi_getglobal(L, name));
#if LUA_VERSION_NUM >= 503
    return lua_type(L, -1);
#endif
}

extern "C" void lua_setglobal(lua_State *L, const char *name)
{
    check_status(oxi_setglobal(L, name));
}
#endif

#if LUA_VERSION_NUM >= 503
extern "C" int lua_geti(lua_State *L, int idx, lua_Integer n)
{
    check_status(oxi_geti(L, idx, n));
    return lua_type(L, -1);
}

extern "C" void lua_seti(lua_State *L, int idx, lua_Integer n)
{
    check_status(oxi_seti(L, idx, n));
}
#endif

#if LUA_VERSION_NUM >= 504
extern "C" void lua_settop(lua_State *L, int idx)
{
    check_status(oxi_settop(L, idx));
}

extern "C" void lua_toclose(lua_State *L, int idx)
{
    check_status(oxi_toclose(L, idx));
}

extern "C" void lua_closeslot(lua_State *L, int idx)
{
    check_status(oxi_closeslot(L, idx));
}

extern "C" void lua_warning(lua_State *L, const char *message, int continued)
{
    check_status(oxi_warning(L, message, continued));
}

extern "C" int lua_gc(lua_State *L, int what, ...)
{
    size_t a = 0;
    int b = 0, c = 0;
    va_list arguments;
    va_start(arguments, what);
    switch (what) {
#if LUA_VERSION_NUM == 505
    case LUA_GCSTEP:
        a = va_arg(arguments, size_t);
        break;
    case LUA_GCPARAM:
        a = static_cast<size_t>(va_arg(arguments, int));
        b = va_arg(arguments, int);
        break;
#else
    case LUA_GCSTEP:
    case LUA_GCSETPAUSE:
    case LUA_GCSETSTEPMUL:
        a = static_cast<size_t>(va_arg(arguments, int));
        break;
    case LUA_GCGEN:
        a = static_cast<size_t>(va_arg(arguments, int));
        b = va_arg(arguments, int);
        break;
    case LUA_GCINC:
        a = static_cast<size_t>(va_arg(arguments, int));
        b = va_arg(arguments, int);
        c = va_arg(arguments, int);
        break;
#endif
    default:
        break;
    }
    va_end(arguments);
    return oxi_gc(L, what, a, b, c);
}
#endif

#if LUA_VERSION_NUM == 501
extern "C" int lua_lessthan(lua_State *L, int lhs, int rhs)
{
    int result = 0;
    check_status(oxi_compare(L, lhs, rhs, 1, &result));
    return result;
}
#endif

extern "C" const char *lua_pushvfstring(lua_State *L, const char *format,
                                          va_list args)
{
    std::string output;
    for (const char *cursor = format; *cursor != '\0'; ++cursor) {
#if LUA_VERSION_NUM >= 502 && LUA_VERSION_NUM < 505
        if (*cursor == '%' && cursor[1] == '\0') {
            luaL_error(L, "invalid option to 'lua_pushfstring'");
            return nullptr;
        }
#endif
        if (*cursor != '%' || cursor[1] == '\0') {
            output.push_back(*cursor);
            continue;
        }

        switch (*++cursor) {
        case '%':
            output.push_back('%');
            break;
        case 's': {
            const char *value = va_arg(args, const char *);
            output.append(value != nullptr ? value : "(null)");
            break;
        }
        case 'c': {
            const char value = static_cast<char>(va_arg(args, int));
#if LUA_VERSION_NUM == 503
            if (std::isprint(static_cast<unsigned char>(value)))
                output.push_back(value);
            else
                output += "<\\" + std::to_string(static_cast<unsigned char>(value)) + ">";
#elif LUA_VERSION_NUM >= 502
            output.push_back(value);
#else
            if (value != '\0')
                output.push_back(value);
#endif
            break;
        }
        case 'd': {
            char buffer[32];
            const int length = std::snprintf(buffer, sizeof(buffer), "%d",
                                             va_arg(args, int));
            if (length > 0)
                output.append(buffer, static_cast<size_t>(length));
            break;
        }
        case 'f': {
#if LUA_VERSION_NUM >= 503
            lua_pushnumber(L, va_arg(args, double));
            size_t length = 0;
            const char *number = lua_tolstring(L, -1, &length);
            if (number != nullptr)
                output.append(number, length);
            lua_pop(L, 1);
#else
            append_number(output, LUA_NUMBER_FMT, va_arg(args, double));
#endif
            break;
        }
#if LUA_VERSION_NUM >= 503
        case 'I':
            output += std::to_string(va_arg(args, LUAI_UACINT));
            break;
        case 'U': {
#if LUA_VERSION_NUM >= 505
            const unsigned long codepoint = va_arg(args, unsigned long);
#else
            const unsigned long codepoint = static_cast<unsigned long>(va_arg(args, long));
#endif
            if (codepoint >
#if LUA_VERSION_NUM >= 504
                0x7fffffffUL
#else
                0x10ffffUL
#endif
                ) {
                luaL_error(L, "UTF-8 value out of range in 'lua_pushfstring'");
                return nullptr;
            }
            if (codepoint <= 0x7f) {
                output.push_back(static_cast<char>(codepoint));
            } else {
                char encoded[6];
                size_t count = 0;
                unsigned long remaining = codepoint;
                unsigned int mask = 0x3f;
                do {
                    encoded[count++] = static_cast<char>(0x80 | (remaining & 0x3f));
                    remaining >>= 6;
                    mask >>= 1;
                } while (remaining > mask);
                output.push_back(static_cast<char>((~mask << 1) | remaining));
                while (count > 0)
                    output.push_back(encoded[--count]);
            }
            break;
        }
#endif
        case 'p': {
            char buffer[4 * sizeof(void *) + 8];
            const int length = std::snprintf(buffer, sizeof(buffer), "%p",
                                             va_arg(args, void *));
            if (length > 0)
                output.append(buffer, static_cast<size_t>(length));
            break;
        }
        default:
#if LUA_VERSION_NUM >= 502 && LUA_VERSION_NUM < 505
            luaL_error(L, "invalid option '%%%c' to 'lua_pushfstring'", *cursor);
            return nullptr;
#else
            output.push_back('%');
            output.push_back(*cursor);
            break;
#endif
        }
    }

    lua_pushlstring(L, output.data(), output.size());
    return lua_tolstring(L, -1, nullptr);
}

extern "C" const char *lua_pushfstring(lua_State *L, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    const char *result = lua_pushvfstring(L, format, args);
    va_end(args);
    return result;
}
