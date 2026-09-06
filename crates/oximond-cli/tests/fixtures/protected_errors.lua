local src = [==[
-- captured locals written by the protected function, with C and Lua handlers
local a, b, c, d, e = 0, 0, 0, 0, 0
local function bump() a = a + 1 end
print(pcall(function() a = 2 bump() error("p") end))
print("pcall", a)
print(xpcall(function() b = 2 error("x") end, function(m) b = b + 10 return m end))
print("lua handler", b)
print(xpcall(function() c = 2 error("y") end, tostring))
print("c handler", c)
print(xpcall(function() d = 2 return d end, function(m) return m end))
print("ok path", d)
print(xpcall(function() e = 5 error({}) end, function(m) e = e + 1 return type(m) end))
print("table error", e)
local f = 1
local function inner() f = f + 1 error("deep") end
print(xpcall(function() inner() end, function(m) f = f * 10 return m end))
print("nested", f)
-- error levels
local function lvl(n, v) return function() error(v, n) end end
local function call(n, v) local function mid() lvl(n, v)() end mid() end
for _, v in ipairs({ "s", 42 }) do
  for n = 0, 3 do
    print("pcall", n, v, pcall(call, n, v))
    print("xpcall", n, v, xpcall(function() call(n, v) end, function(m) return m end))
    if n < 3 then print("direct", n, v, pcall(error, v, n)) end
  end
end
print(select("#", pcall(error)))
-- error levels through native callers
-- LuaJIT counts levels differently through require at level 3 and does not
-- yet call __tostring from string.format; those cases are skipped.
local isjit = type(jit) == "table"
local function viaNative(name, callf, maxlevel)
  for n = 1, maxlevel or 3 do
    local function boom() error("m", n) end
    print(name, n, pcall(callf, boom))
  end
end
viaNative("require", function(boom)
  package.preload.pe_mod = boom
  package.loaded.pe_mod = nil
  return require("pe_mod")
end, isjit and 2 or 3)
viaNative("tostring", function(boom) local s = tostring(setmetatable({}, { __tostring = boom })); return s end)
if not isjit then viaNative("format", function(boom) return string.format("%s", setmetatable({}, { __tostring = boom })) end) end
viaNative("sort", function(boom) return table.sort({ 3, 2, 1 }, function() boom() end) end)
viaNative("gsub", function(boom) return string.gsub("a", "a", function() boom() end) end)
if newproxy and not isjit then
  viaNative("gc", function(boom)
    local u = newproxy(true)
    getmetatable(u).__gc = boom
    u = nil
    collectgarbage()
  end)
end
-- error levels through tail calls, which Lua 5.1 counts as levels
local function tlvl(n, m) error(m, n) end
local function ttail(n, m) return tlvl(n, m) end
local function tcaller(n, m) local r = ttail(n, m) return r end
for _, v in ipairs({ "msg", 42 }) do
  for n = 1, 4 do print("tail", n, v, pcall(tcaller, n, v)) end
end
-- hot loop
local total, g = 0, 0
for i = 1, 400 do
  local ok = xpcall(function() g = g + 1 error("z") end, function(m) g = g + 1 return m end)
  if not ok then total = total + g end
end
print("loop", total, g)
]==]
assert((loadstring or load)(src, "=t"))()
