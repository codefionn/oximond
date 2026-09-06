-- oximond-difftest repro profile=luajit21
local raw_print = print
local jit = type(jit) == "table"
local unpack = table.unpack or unpack
-- Addresses differ between runs; LuaJIT ends a traceback with one.
local function print(...)
  local n = select("#", ...)
  local t = {...}
  for i = 1, n do
    t[i] = (tostring(t[i]):gsub("at 0x%x+", "at 0xADDR"):gsub("table: 0x%x+", "table: ADDR"))
  end
  raw_print(unpack(t, 1, n))
end

print("== tb1: nested calls, handlers, tail calls, coroutines, metamethods")
do
local function lvl3() error("boom") end
local function lvl2() lvl3() end
local up = function() lvl2() end
local t = {}
function t.field() up() end
function t:meth() t.field() end
function globalfn() t:meth() end
print(select(2, xpcall(globalfn, debug.traceback)))
print(select(2, xpcall(globalfn, function(m) return debug.traceback(m, 1) end)))
print(debug.traceback("msg"))
print(debug.traceback("msg", 2))
local function tail() return lvl3() end
print(select(2, xpcall(tail, debug.traceback)))
local co = coroutine.create(function() local x = 1; lvl2() end)
local ok, e = coroutine.resume(co)
print(debug.traceback(co))
print(debug.traceback(co, "hi"))
print(debug.traceback(co, nil, 1))
local mt = setmetatable({}, {__index = function(t, k) error("idx") end, __add = function() error("add") end})
print(select(2, xpcall(function() return mt.x end, debug.traceback)))
print(select(2, xpcall(function() local y = mt + 1; return y end, debug.traceback)))
print(select(2, pcall(function() local ok, tb = xpcall(function() error("in") end, function(m) return debug.traceback(m) end); error(tb, 0) end)))
print(debug.traceback(12))
print(type(debug.traceback({})))
print(pcall(string.rep))
if not jit then print(select(2, xpcall(string.rep, debug.traceback))) end
print(select(2, xpcall(function() string.rep() end, debug.traceback)))
print(select(2, xpcall(function() local a = nil; a() end, debug.traceback)))
if not jit then print(select(2, xpcall(function() table.sort({3,2,1}, function(a,b) error("cmp") end) end, debug.traceback))) end

end

print("== tb2: levels, naming, deep stacks")
do
print(debug.traceback())
print(debug.traceback(""))
print(debug.traceback(nil))
print(debug.traceback("m", 0))
print(debug.traceback("m", 1))
print(debug.traceback("m", 3))
print(debug.traceback("m", -1))
print(debug.traceback(coroutine.create(function() end)))
print(debug.traceback(coroutine.create(function() end), "x"))
function g() print(debug.traceback("g")) end
g()
local function loc() print(debug.traceback("loc")) end
loc()
local t = {f = function() print(debug.traceback("fld")) end}
t.f()
t:f()
local co = coroutine.wrap(function() print(debug.traceback("co")); coroutine.yield(1); error("x") end)
co()
local co2 = coroutine.create(function() local x = 1; coroutine.yield(1) end)
coroutine.resume(co2)
print(debug.traceback(co2))
print(debug.traceback(co2, "m", 1))
local co3 = coroutine.create(function() error("dead") end)
print(coroutine.resume(co3))
print(debug.traceback(co3))
local function rec(n) if n == 0 then print(debug.traceback("deep")) else rec(n-1) end end
rec(30)
local function rec2(n) if n == 0 then print(debug.traceback("deep")) else local x = rec2(n-1); return x end end
rec2(8)

end

print("== tb3: message and level arguments")
do
local function out(r) if type(r) == "string" then print((r:gsub("\n", "|"))) else print(type(r)) end end
local function show(...) local ok, e = pcall(function(...) return (debug.traceback(...)) end, ...) out(e) end
show()
show("")
show(nil)
show(nil, 1)
show("m", -1)
show("m", 100)
show("m", 2.0)
show("m", "1")
show(12)
show(true)
show({})
show(print)
local co = coroutine.create(function() coroutine.yield() end)
show(co)
show(co, "m", 0)
show(co, nil)
coroutine.resume(co)
show(co)
show(co, "m", 1)
show(co, "m", 5)
show(co, "m", -1)

end

print("== tb4: runtime errors, metamethods, iterators, coroutines")
do
local function show(f, ...) print(select(2, xpcall(f, debug.traceback, ...))) end
-- runtime error, not native
show(function() local a; return a.b end)
show(function() return 1 + {} end)
-- error with table value, level 2
local function lv2() error("lvl2", 2) end
show(function() lv2() end)
-- tail call chain
local function t3() error("t3") end
local function t2() return t3() end
local function t1() return t2() end
show(function() return t1() end)
show(function() local r = t1(); return r end)
-- global tail call
function gtail() error("gt") end
show(function() return gtail() end)
-- call through pcall inside
show(function() local ok, e = pcall(function() error("inner") end); error(debug.traceback(e), 0) end)
-- __call, __index function, __newindex, __eq, __lt, __concat, __len, __unm
local mt = {}
mt.__call = function(self) error("call") end
mt.__index = function(t, k) error("idx") end
mt.__newindex = function(t, k, v) error("nidx") end
mt.__eq = function(a, b) error("eq") end
mt.__lt = function(a, b) error("lt") end
mt.__le = function(a, b) error("le") end
mt.__concat = function(a, b) error("cat") end
mt.__len = function(a) error("len") end
mt.__unm = function(a) error("unm") end
mt.__mul = function(a, b) error("mul") end
mt.__tostring = function(a) error("ts") end
local o = setmetatable({}, mt)
local o2 = setmetatable({}, mt)
show(function() o() end)
show(function() return o.x end)
show(function() o.x = 1 end)
show(function() return o == o2 end)
show(function() return o < o2 end)
show(function() return o <= o2 end)
show(function() return o .. "x" end)
show(function() return #o end)
show(function() return -o end)
show(function() return o * 2 end)
show(function() return tostring(o) end)
-- iterators
show(function() for k in function() error("iter") end do end end)
-- sort / gsub callbacks
if not jit then show(function() table.sort({3, 2, 1}, function(a, b) error("cmp") end) end) end
if not jit then show(function() return (string.gsub("abc", ".", function(c) error("gs") end)) end) end
-- method and field
local obj = {}
function obj:m() error("meth") end
function obj.f() error("fld") end
show(function() obj:m() end)
show(function() obj.f() end)
-- coroutines
local co = coroutine.create(function() local function inner() error("in co") end inner() end)
local ok, e = coroutine.resume(co)
print(e)
print(debug.traceback(co))
print(debug.traceback(co, "msg"))
show(function()
  local c = coroutine.wrap(function() error("wrapped") end)
  c()
end)
local co2 = coroutine.create(function()
  print(select(2, xpcall(function() error("xp in co") end, debug.traceback)))
  print(debug.traceback("tb in co"))
end)
coroutine.resume(co2)
-- varargs, main chunk fn
local function va(...) error("va") end
show(function() va(1, 2, 3) end)
-- string.format('%s') calling tostring
if not jit then show(function() return string.format("%s", o) end) end
-- load'ed chunk
show((loadstring or load)("error('chunk')", "=mychunk"))
show((loadstring or load)("local x = ...; error('chunk2')", "@file.lua"))
show((loadstring or load)("error('chunk3')"))
-- pcall(error)
print(select(2, xpcall(function() error("direct") end, debug.traceback)))
print(select(2, xpcall(function() return pcall(error, "x") end, debug.traceback)))

end

print("== tb5: tail calls of natives, nested coroutines, handler failures")
do
local function show(f, ...) print(select(2, xpcall(f, debug.traceback, ...))) end
local function boom() error("boom") end
local mt = {__tostring = function() error("ts") end, __index = function(t, k) return boom() end,
  __add = function(a, b) return boom() end, __call = function(self, ...) return boom() end}
local o = setmetatable({}, mt)
-- tail call of natives
show(function() return pcall(boom) end)
show(function() local ok, e = pcall(boom); return debug.traceback(e, 1) end)
show(function() return tostring(o) end)
show(function() local s = tostring(o); return s end)
show(function() return o.x end)
show(function() local v = o.x; return v end)
show(function() return o + 1 end)
show(function() return o(1) end)
show(function() local v = o(1); return v end)
show(function() return select(2, pcall(boom)) end)
if not jit then show(function() return table.sort({3,2,1}, function(a, b) return boom() end) end) end
if not jit then show(function() return (string.gsub("a", ".", function() return boom() end)) end) end
if not jit then show(function() return string.gsub("a", ".", function() return boom() end) end) end
-- pcall inside coroutine
local co = coroutine.create(function()
  local ok, e = pcall(function() boom() end)
  print(e)
  print(debug.traceback("inco"))
  local function y() coroutine.yield() end
  y()
end)
coroutine.resume(co)
print(debug.traceback(co))
-- nested coroutines
local outer = coroutine.wrap(function()
  local inner = coroutine.wrap(function() print(debug.traceback("nested")) end)
  inner()
end)
outer()
-- error in handler
print(xpcall(boom, function(m) error("handler fail") end))
print(xpcall(boom, function(m) return debug.traceback("h:" .. m, 2) end))
print(xpcall(boom, function(m) local function deeper() return debug.traceback("d", 1) end return (deeper()) end))
-- debug.traceback in various call forms
local function a() return (debug.traceback("a")) end
local function b() local x = a(); return x end
print(b())
print((function() return a() end)())
local t = {a = a}
print(t.a())
print(t:a())
-- method on string
print(("x"):rep(2) and debug.traceback("s"))
-- level beyond / inside xpcall levels
print(select(2, xpcall(function() return (debug.traceback("lv2", 2)) end, print)))
-- loaded chunk
local f = (loadstring or load)("local function inner() error('li') end inner()", "=lc")
show(f)
-- string.format %d with bad arg
if not jit then show(function() return string.format("%d", "x") end) end
if not jit then show(function() local s = string.format("%d", "x"); return s end) end
-- ipairs/pairs iterators
show(function() for i, v in ipairs({1}) do boom() end end)
show(function() for k, v in pairs({1}) do boom() end end)
show(function() for k, v in next, {1} do boom() end end)
-- require-d module error
package.preload.mymod = function() boom() end
show(function() require("mymod") end)
-- dofile/loadfile skip; select
show(function() select(-5) end)
-- setmetatable __index chain native
show(function() local t = setmetatable({}, {__index = rawget}) return t.x end)

end

print("== tb-meta: native called directly as a metamethod, native tail-called by a handler")
do
local function show(f) print(select(2, xpcall(f, debug.traceback))) end
show(function() local s = setmetatable({}, {__index = string.rep}).x; return s end)
show(function() local t = setmetatable({}, {__add = string.rep}); local s = t + 1; return s end)
show(function() local t = setmetatable({}, {__newindex = string.rep}); t.x = 1; return 1 end)
show(function() local s = setmetatable({}, {__index = function(t, k) return string.rep() end}).x; return s end)
show(function() local t = setmetatable({}, {__newindex = function(t, k) return string.rep() end}); t.x = 1; return 1 end)
show(function() local t = setmetatable({}, {__add = function(a, b) return string.rep() end}); local s = t + 1; return s end)
show(function() local t = setmetatable({}, {__concat = function(a, b) return string.rep() end}); local s = t .. "x"; return s end)
end

print("== tb-direct-native: a builtin that xpcall or pcall calls directly")
print(select(2, xpcall(debug.traceback, print)))
print(select(2, pcall(debug.traceback, "m")))
