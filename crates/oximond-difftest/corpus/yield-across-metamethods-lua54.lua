-- oximond-difftest repro profile=lua54 from=lua54 yield across metamethods
-- Yield across metamethods and iterators. Each case runs in a coroutine and
-- prints every value the coroutine yields, then its final result.
local y = coroutine.yield
local function clean(v) return (tostring(v):gsub("^[^:\n]+:%d+: ", "")) end

local function drive(name, f, ...)
  local co = coroutine.create(f)
  local out, n = {}, 0
  local ok, a = coroutine.resume(co, ...)
  while coroutine.status(co) == "suspended" and n < 50 do
    n = n + 1
    out[#out + 1] = tostring(ok) .. ":" .. clean(a)
    ok, a = coroutine.resume(co, "r" .. n)
  end
  out[#out + 1] = tostring(ok) .. ":" .. clean(a)
  print(name, table.concat(out, " | "))
end

local mt = {}
local function new() return setmetatable({}, mt) end
mt.__index = function(t, k) return y("index " .. tostring(k)) end
mt.__newindex = function(t, k, v) y("newindex " .. tostring(k)) rawset(t, k, v) end
mt.__add = function(a, b) return y("add") end
mt.__sub = function(a, b) return y("sub") end
mt.__mul = function(a, b) return y("mul") end
mt.__div = function(a, b) return y("div") end
mt.__mod = function(a, b) return y("mod") end
mt.__pow = function(a, b) return y("pow") end
mt.__unm = function(a) return y("unm") end
mt.__concat = function(a, b) return y("concat") end
mt.__eq = function(a, b) return y("eq") end
mt.__lt = function(a, b) return y("lt") end
mt.__le = function(a, b) return y("le") end
mt.__call = function(self, ...) return y("call", ...) end

drive("index", function() return new().foo end)
drive("index2", function() local t = new() return t.a, t.b end)
drive("newindex", function() local t = new() t.x = 1 return rawget(t, "x") end)
drive("add", function() return new() + 1 end)
drive("add-right", function() return 1 + new() end)
drive("sub-mul", function() local o = new() return o - 1, o * 2 end)
drive("div-mod-pow", function() local o = new() return o / 1, o % 1, 2 ^ o end)
drive("unm", function() return -new() end)
drive("concat", function() return new() .. "a" end)
drive("concat-chain", function() return "a" .. new() .. "b" .. "c" end)
drive("concat-nums", function() return 1 .. 2 .. new() .. 3 .. 4 end)
drive("eq", function() return new() == new() end)
drive("ne", function() return new() ~= new() end)
drive("eq-if", function() if new() == new() then return "T" end return "F" end)
drive("lt", function() return new() < new() end)
drive("gt", function() return new() > new() end)
drive("le", function() return new() <= new() end)
drive("lt-if", function() if new() < new() then return "T" end return "F" end)
drive("le-not", function() return not (new() <= new()) end)
drive("call", function() return new()("a", "b") end)
drive("call-nontail", function() local r = new()(1) return r end)
drive("callable-hop", function()
  local c = setmetatable({}, {__call = function(self, a) return y("hop", a) end})
  return c(5)
end)
drive("index-chain", function()
  local c = setmetatable({}, {__index = setmetatable({}, {__index = function(t, k) return y("chain") end})})
  return c.x
end)
drive("newindex-chain", function()
  local c = setmetatable({}, {__newindex = setmetatable({}, {__newindex = function(t, k, v) y("ni") end})})
  c.x = 1
  return "ok"
end)

-- generic-for iterators, a Lua function that yields once, many times, and a
-- coroutine.wrap iterator
drive("for-once", function()
  for a in function(s, c) if not c then return y("once") end end do return a end
end)
drive("for-many", function()
  local n, got = 0, {}
  for a in function() n = n + 1 if n < 4 then return y("it" .. n) end end do got[#got + 1] = a end
  return table.concat(got, ",")
end)
drive("for-wrap", function()
  local got = {}
  for a in coroutine.wrap(function() y("w1") y("w2") return nil end) do got[#got + 1] = tostring(a) end
  return #got
end)

-- results and errors after resume
drive("multi-result", function()
  local t = setmetatable({}, {__index = function() return y("a"), 2 end})
  return t.x, t.y
end)
drive("error-after", function()
  local o = setmetatable({}, {__add = function() y("a") error("boom", 0) end})
  return o + 1
end)
drive("error-after-pcall", function()
  local o = setmetatable({}, {__add = function() y("a") error("boom", 0) end})
  return pcall(function() return o + 1 end)
end)
drive("error-table", function()
  local o = setmetatable({}, {__index = function() y("a") error({}) end})
  return (pcall(function() return o.x end))
end)
drive("type-error-after", function()
  local o = setmetatable({}, {__add = function() y("a") return nil + 1 end})
  return pcall(function() return o + 1 end)
end)
drive("pcall-around", function()
  return pcall(function() return new().foo end)
end)
drive("xpcall-around", function()
  return xpcall(function() return new() + 1 end, function(m) return "H:" .. m end)
end)
drive("nested-coroutine", function()
  local o = setmetatable({}, {__add = function()
    local inner = coroutine.wrap(function() return y("inner") end)
    return inner()
  end})
  return o + 1
end)

-- wrap while suspended inside a metamethod
local w = coroutine.wrap(function() local t = new() return t.k end)
print(w())
print(w("fin"))
print(pcall(w))
collectgarbage()

-- tables and strings survive garbage collection while suspended
co = coroutine.wrap(function()
  local keep = {}
  local o = setmetatable({}, {__index = function(t, k)
    local junk = {k, tostring(k) .. "x", {k}}
    y(junk[2])
    return junk[3][1]
  end})
  for i = 1, 40 do keep[i] = o[i] end
  return #keep, keep[40]
end)
local v = co()
local seen = 0
while type(v) == "string" do seen = seen + 1 if seen % 10 == 0 then collectgarbage() end v = co(seen) end
print(seen, v)

-- hot loops: every instruction kind yields many times
local function hot(name, body, count, expect)
  local c = coroutine.wrap(body)
  local n, last = 0, c()
  while last == "y" do n = n + 1 last = c(n) end
  print(name, n, last)
end
local hmt = {
  __add = function(a, b) return y("y") end,
  __index = function(t, k) return y("y") end,
  __newindex = function(t, k, v) y("y") end,
  __concat = function(a, b) return y("y") end,
  __lt = function(a, b) return y("y") end,
  __eq = function(a, b) return y("y") end,
  __call = function(self, x) return y("y") end,
}
hot("hot-add", function() local o, s = setmetatable({}, hmt), 0 for i = 1, 600 do s = s + (o + i) end return s end)
hot("hot-index", function() local o, s = setmetatable({}, hmt), 0 for i = 1, 600 do s = s + o[i] end return s end)
hot("hot-newindex", function() local o = setmetatable({}, hmt) for i = 1, 600 do o[i] = i end return "ni" end)
hot("hot-concat", function() local o, s = setmetatable({}, hmt), 0 for i = 1, 600 do local r = o .. "x" s = s + 1 end return s end)
hot("hot-lt", function() local o, s = setmetatable({}, hmt), 0 for i = 1, 600 do if o < o then s = s + 1 end end return s end)
hot("hot-eq", function() local a, b, s = setmetatable({}, hmt), setmetatable({}, hmt), 0 for i = 1, 600 do if a == b then s = s + 1 end end return s end)
hot("hot-call", function() local o, s = setmetatable({}, hmt), 0 for i = 1, 600 do s = s + o(i) end return s end)
hot("hot-for", function()
  local s, n = 0, 0
  for v in function() n = n + 1 if n <= 600 then return y("y") end end do s = s + 1 end
  return s
end)
hot("hot-mixed", function()
  local o, s = setmetatable({}, hmt), 0
  for i = 1, 600 do
    if i % 3 == 0 then s = s + o[i] elseif i % 3 == 1 then s = s + (o + 1) else s = s + i end
  end
  return s
end)

-- long warm-up first so a trace exists, then yields from inside it
local function warm(name, body)
  local c = coroutine.wrap(body)
  local n, last = 0, c()
  while last == "w" do n = n + 1 last = c(n) end
  print(name, n, last)
end
local counter = 0
local wmt = {
  __add = function(a, b) counter = counter + 1 if counter % 700 == 0 then return y("w") end return counter end,
  __index = function(t, k) counter = counter + 1 if counter % 700 == 0 then return y("w") end return k end,
  __lt = function(a, b) counter = counter + 1 if counter % 700 == 0 then return y("w") end return true end,
  __concat = function(a, b) counter = counter + 1 if counter % 700 == 0 then return y("w") end return "c" end,
}
warm("warm-add", function() local o, s = setmetatable({}, wmt), 0 for i = 1, 5000 do s = s + (o + i) end return s end)
warm("warm-index", function() local o, s = setmetatable({}, wmt), 0 for i = 1, 5000 do s = s + o[i] end return s end)
warm("warm-lt", function() local o, s = setmetatable({}, wmt), 0 for i = 1, 5000 do if o < o then s = s + 1 end end return s end)
warm("warm-concat", function() local o, s = setmetatable({}, wmt), 0 for i = 1, 5000 do local r = o .. i s = s + 1 end return s end)

-- stock refuses these
drive("sort-comparator", function()
  table.sort({3, 1, 2}, function(a, b) local r = y("c") return a < b end)
  return "sorted"
end)
drive("gsub-callback", function()
  return (string.gsub("abc", "b", function() local r = y("g") return r end))
end)
drive("gsub-table-index", function()
  return (string.gsub("abc", "b", setmetatable({}, {__index = function() return y("i") end})))
end)
print(select("#", pcall(coroutine.wrap(function()
  return pcall(table.sort, {3, 1, 2}, function(a, b) local r = y("c") return a < b end)
end))))
