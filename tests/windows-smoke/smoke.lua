-- Windows smoke test, run by `scripts/ci.sh windows` under wine (or on
-- Windows) with the interpreter, and with tracing on each native backend.
-- Every run must print the same lines.

-- Hot loops the tracing modes compile: float and integer arithmetic, a helper
-- call (math.floor), table reads and writes, and a nested loop with a side.
local x, n = 0.0, 0
for i = 1, 300000 do
    x = x + i * 0.5 - math.floor(i / 3)
    n = n + i % 7
end
print("arith", x, n)

local t = {}
for i = 1, 200000 do
    t[i] = i * 2
end
local s = 0
for _, v in ipairs(t) do
    s = s + v
end
print("tables", #t, s)

local rows = {}
for b = 1, 8 do
    rows[b] = {0, 0, 0, 0, b, 0}
end
local acc = 0
for d = 1, 20000 do
    for _, row in pairs(rows) do
        acc = acc + row[5]
    end
    if d % 1000 == 0 then
        acc = acc - 1
    end
end
print("pairs", acc)

local words = {}
for i = 1, 2000 do
    words[#words + 1] = string.format("%d:%s", i, string.rep("x", i % 5))
end
print("strings", #table.concat(words, ","))

-- The C calls of the ffi module: integer, pointer, float and variadic
-- arguments, through ffi.C (the C runtime and kernel32) and an explicit DLL.
local ffi = require("ffi")
ffi.cdef([[
int abs(int);
size_t strlen(const char *);
double pow(double, double);
float sqrtf(float);
int sprintf(char *buf, const char *fmt, ...);
int MulDiv(int a, int b, int c);
long long llabs(long long);
]])
print("ffi", ffi.os, ffi.arch, ffi.C.abs(-42), tonumber(ffi.C.strlen("hello")), ffi.C.pow(2, 10))
print("ffi float", string.format("%.4f", ffi.C.sqrtf(2)))
local buf = ffi.new("char[64]")
-- Eight arguments: a double in the fourth register, the last four on the
-- stack. A Lua number passes a double to `...` and an int cdata its value, as
-- in LuaJIT.
local written = ffi.C.sprintf(buf, "%s %.2f %s %.3f %s %d", "a", 1.5, "b", 2.25, "c", ffi.new("int", 42))
print("ffi varargs", written, ffi.string(buf))
print("ffi kernel32", ffi.C.MulDiv(10, 20, 3), ffi.load("kernel32").MulDiv(6, 7, 2), tonumber(ffi.C.llabs(-5)))
print("ffi missing", (pcall(ffi.load, "oximond_no_such_library")))

-- A native module that imports the Lua API from lua51.dll, built from
-- module.c. scripts/ci.sh puts it in the working directory, where the default
-- package.cpath finds it.
local found, module = pcall(require, "smoke_module")
if found then
    local _, failed = pcall(function()
        module.fail(7)
    end)
    local _, inner = pcall(module.call, function()
        error("inner", 0)
    end)
    -- The position names this file, whose path depends on the checkout.
    failed = failed:gsub("^.*:%d+: ", "")
    print("module", module.add(2, 3), failed, inner, module.call(function()
        return "back"
    end))
    -- A hot loop whose trace calls the module itself (a host call from
    -- native code, through the platform's calling convention).
    local add = module.add
    local s = 0
    for i = 1, 20000 do
        local x = i * 3 % 7 + i % 5
        local y = x * 2 - i % 3 + (x % 4) * 3 - (i % 6)
        s = add(s, y) + x % 2
        s = s - x + i % 7 - (y % 3) + (s % 11)
    end
    print("module loop", s)
    local filled = {}
    module.fill(filled, 100)
    local sum = 0
    for i = 1, #filled do
        sum = sum + filled[i]
    end
    print("module fill", #filled, sum, (pcall(module.fill, 1, 1)))
else
    print("module missing")
end
