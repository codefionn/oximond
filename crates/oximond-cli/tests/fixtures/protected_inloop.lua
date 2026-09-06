local pcall, select, tostring, type = pcall, select, tostring, type
local function show(...)
  local t = {}
  for i = 1, select('#', ...) do t[#t + 1] = tostring((select(i, ...))) end
  return table.concat(t, ",")
end
local function zero() end
local function one(a) return a end
local function many(a, b, c) return a, b, c, a + b + c end
local function va(...) return select('#', ...), ... end
local function nonstring(i)
  if i == 1 then error(nil) elseif i == 2 then error(42) elseif i == 3 then error(false)
  else error({ code = i }) end
end
for i = 1, 3 do
  print("zero", pcall(zero))
  print("one", pcall(one, i))
  print("many", pcall(many, 1, 2, i))
  print("va", pcall(va, 1, nil, i))
  print("short", pcall(one))
  local a, b = pcall(many, 1, 2, i)
  print("two", a, b)
  local t = { pcall(many, 1, 2, i) }
  print("multret", #t, show(pcall(many, 1, 2, i)))
  print("sel", select('#', pcall(zero)), (pcall(one, i)))
end
for i = 1, 4 do
  local ok, e = pcall(nonstring, i)
  print("nonstring", ok, type(e), type(e) == "table" and e.code or tostring(e))
end
-- nested pcall
local function inner(x) return pcall(error, "in" .. x) end
for i = 1, 3 do
  print("nested", pcall(inner, i))
  print("nested2", pcall(pcall, error, "x"))
  print("nested3", pcall(function() return pcall(function() error("deep" .. i, 0) end) end))
end
-- error positions and runtime errors
for i = 1, 3 do
  print("pos", pcall(function() error("msg" .. i) end))
  print("lvl2", pcall(function() error("msg", 2) end))
  print("rt", pcall(function() local t = nil; return t.x end))
  print("arith", pcall(function(a) return a + {} end, i))
end
-- non-callable and missing function
print("nocall", pcall(nil))
print("nocall2", pcall(42))
print("none", pcall(pcall))
-- stack overflow inside pcall
local function rec(n) return 1 + rec(n + 1) end
local ok, e = pcall(rec, 1)
print("overflow", ok, type(e), (tostring(e):find("stack overflow", 1, true)) ~= nil)
local depth = 0
local function deep() depth = depth + 1; local ok = pcall(deep); return ok end
print("deep", deep(), depth > 100)
-- error inside a hook
if debug and debug.sethook then
  local n = 0
  debug.sethook(function() n = n + 1 end, "c")
  for i = 1, 3 do pcall(one, i) end
  debug.sethook()
  print("hookcalls", n > 0)
  local fired = false
  debug.sethook(function() if not fired then fired = true; error("hooked", 0) end end, "c")
  local ok, e = pcall(function() return pcall(one, 1) end)
  debug.sethook()
  print("hookerr", ok, e)
end
-- traceback through pcall
for i = 1, 2 do
  print("tb", select(2, pcall(function() return debug.traceback("T", 1) end)):gsub("0x%x+", "P"))
  local function lvl() return debug.getinfo(2, "n") end
  local info = select(2, pcall(function() return (debug.getinfo(1, "S").what) end))
  print("info", info)
end
-- xpcall unchanged
for i = 1, 2 do
  print("xp", xpcall(function(a) error("E" .. a, 0) end, function(m) return "H:" .. m end, i))
  print("xp2", xpcall(one, show, i))
end
-- close variables and varargs callers
local function caller(...)
  local a, b = pcall(one, ...)
  return a, b, select('#', ...)
end
print("vacaller", caller(7, 8, 9))
print("vacaller0", caller())
-- __call
local callable = setmetatable({ bias = 10 }, { __call = function(self, a, b, c) return self.bias, a, b, c end })
local vcall = setmetatable({}, { __call = function(self, ...) return select('#', ...), ... end })
local zcall = setmetatable({}, { __call = function() end })
local tail = setmetatable({}, { __call = function(self, x) return one(x) end })
local chain = setmetatable({}, { __call = callable })
for i = 1, 3 do
  print("c0", callable())
  print("c1", callable(i))
  print("c3", callable(i, i + 1, i + 2))
  print("c5", callable(1, 2, 3, 4, 5))
  print("v", vcall(), vcall(i, nil, nil))
  print("z", zcall(1, 2), (zcall()))
  print("tail", tail(i))
  print("pc", pcall(callable, i))
  print("chain", (select(2, pcall(chain, i))))
  local x = callable(i)
  print("local", x)
  local t = { callable(i, 1) }
  print("tbl", #t, t[1], t[2], t[3])
end
print("nocall", pcall(function() return setmetatable({}, {})() end))
print("badcall", pcall(function() return setmetatable({}, { __call = 5 })() end))
print("tb2", (select(2, pcall(setmetatable({}, { __call = function() return debug.traceback("C", 1) end }))):gsub("0x%x+", "P")))
print("native", type(select(2, pcall(setmetatable({}, { __call = tostring })))))
print("err", pcall(setmetatable({}, { __call = function(s, m) error(m, 0) end }), "boom"))
print("rec", (function()
  local o; o = setmetatable({}, { __call = function(self, n) if n == 0 then return "done" end return o(n - 1) end })
  return o(50)
end)())
local co = coroutine.create(function()
  local ok, e = pcall(function() return coroutine.yield(1) end)
  return ok, e
end)
print("yield", coroutine.resume(co))
print("yield2", coroutine.resume(co, "back"))
if newproxy then
  local u = newproxy(true)
  getmetatable(u).__call = function(self, a, b) return type(self), a, b end
  for i = 1, 3 do print("ud", u(i, i + 1)) end
end
