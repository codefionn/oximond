-- oximond-difftest repro profile=luajit21
-- Native functions called as metamethods or by a generic for, and wrapper
-- errors, as stock Lua names and locates them.
local raw_print = print
local unpack = table.unpack or unpack
local function print(...)
  local n = select("#", ...)
  local t = {...}
  for i = 1, n do
    t[i] = (tostring(t[i]):gsub("table: 0x%x+", "table: ADDR"))
  end
  raw_print(unpack(t, 1, n))
end
do
local function try(name, f) print(name, pcall(f)) end
try("index", function() local t = setmetatable({}, {__index = string.rep}); return t.x end)
try("index2", function() local t = setmetatable({}, {__index = string.rep}); local v = t.x; return v end)
try("newindex", function() local t = setmetatable({}, {__newindex = setmetatable}); t.x = 1 end)
try("add", function() local t = setmetatable({}, {__add = string.rep}); return t + 1 end)
try("add2", function() local t = setmetatable({}, {__add = string.rep}); local v = t + 1; return v end)
try("concat", function() local t = setmetatable({}, {__concat = string.rep}); return t .. "x" end)
try("eq", function() local mt = {__eq = string.rep}; local a, b = setmetatable({}, mt), setmetatable({}, mt); return a == b end)
try("lt", function() local mt = {__lt = string.rep}; local a, b = setmetatable({}, mt), setmetatable({}, mt); return a < b end)
try("call", function() local t = setmetatable({}, {__call = string.rep}); return t() end)
try("call2", function() local t = setmetatable({}, {__call = string.rep}); local v = t(); return v end)
try("tostring", function() return tostring(setmetatable({}, {__tostring = string.rep})) end)
try("tostring2", function() local v = tostring(setmetatable({}, {__tostring = string.rep})); return v end)
try("len", function() local t = setmetatable({}, {__len = string.rep}); return #t end)
try("unm", function() local t = setmetatable({}, {__unm = string.rep}); return -t end)
try("idxf", function() local t = setmetatable({}, {__index = function(t,k) return string.rep() end}); return t.x end)
try("addf", function() local t = setmetatable({}, {__add = function(a,b) return string.rep() end}); return t + 1 end)
try("tsf", function() return tostring(setmetatable({}, {__tostring = function(a) return string.rep() end})) end)
try("tsf2", function() local v = tostring(setmetatable({}, {__tostring = function(a) return string.rep() end})); return v end)
try("callf", function() local t = setmetatable({}, {__call = function(a) return string.rep() end}); return t() end)

end
do
local t = setmetatable({}, {__call = string.rep})
print(pcall(function() t() end))
print(pcall(function() t("x") end))
print(pcall(function() t(2, 3) end))
local s = setmetatable({}, {__call = function(self, ...) return string.rep(...) end})
print(pcall(function() s() end))
local u = setmetatable({}, {__call = string.sub})
print(pcall(function() u({}) end))
print(pcall(function() u(1, {}) end))
print(pcall(function() u:foo() end))
local v = setmetatable({}, {__call = setmetatable({}, {__call = string.rep})})
local chains = _VERSION ~= "Lua 5.1"  and type(jit) ~= "table"
if chains then print(pcall(function() v() end)) end
if chains then print(pcall(function() v(2) end)) end
if chains then print(pcall(function() local a = v(2) end)) end
if chains then print(pcall(function() return v() end)) end
print(pcall(t))
print(pcall(function() return t() end))

end
do
print(pcall(function() for k in next, nil do end end))
print(pcall(function() for k in nil do end end))
print(pcall(function() for k in 5 do end end))
print(pcall(function() for k in string.rep do end end))
print(pcall(function() local it = string.rep; for k in it do end end))
print(pcall(function() for k in function() error("x") end do end end))
print(pcall(function() for k in function() error("x", 2) end do end end))

end
do
local c = coroutine.wrap(function() error("x") end)
print(pcall(c))
local c2 = coroutine.wrap(function() error("x", 0) end)
print(pcall(c2))
local c3 = coroutine.wrap(function() error({}) end)
print(type(select(2, pcall(c3))))
local c4 = coroutine.wrap(function() error("x") end)
print(pcall(function() c4() end))
local c5 = coroutine.wrap(function() local y = nil + 1 end)
print(pcall(function() local v = c5() return v end))
local c6 = coroutine.wrap(function() end)
c6()
print(pcall(function() c6() end))

end
