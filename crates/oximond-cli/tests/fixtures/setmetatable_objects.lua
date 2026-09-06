-- Objects built with setmetatable({...}, Class) followed by field stores.
-- Output must not depend on iteration order.

local function keys(t)
  local ks = {}
  for k in pairs(t) do ks[#ks + 1] = tostring(k) end
  table.sort(ks)
  return table.concat(ks, ",")
end

local function msg(m)
  return (tostring(m):gsub("^[^:]*:%d+: ", ""))
end

local log = {}
local function note(...)
  local n = select("#", ...)
  local p = {}
  for i = 1, n do p[i] = tostring((select(i, ...))) end
  log[#log + 1] = table.concat(p, " ")
  print(log[#log])
end

-- 1. Plain class, no __newindex: stores are raw, pairs sees stored keys only.
-- Each constructor is straight-line code followed directly by its stores, so
-- the constructor shape scan sees them. Three ways to name setmetatable.
local Plain = {}
Plain.__index = Plain

local function newGlobal(i)
  local o = setmetatable({ x = i, y = 0 }, Plain)
  o.z = i + 1
  o.w = nil
  o.x = o.x + 1
  return o
end

local function newLocal(i)
  local setmetatable = setmetatable
  local o = setmetatable({ x = i, y = 0 }, Plain)
  o.z = i + 1
  o.w = nil
  o.x = o.x + 1
  return o
end

local newUpvalue
do
  local setmetatable = setmetatable
  local function mkup(i)
    local o = setmetatable({ x = i, y = 0 }, Plain)
    o.z = i + 1
    o.w = nil
    o.x = o.x + 1
    return o
  end
  newUpvalue = mkup
end

-- A local that is named setmetatable but is not the real one must not be
-- treated as the real one.
local keep = {}
local newImpostor
do
  local function setmetatable(t, mt) keep[#keep + 1] = t; return {} end
  function newImpostor(i)
    local o = setmetatable({ x = i, y = 0 }, Plain)
    o.z = i + 1
    o.w = nil
    o.x = (o.x or 0) + 1
    return o
  end
end

-- Same, with the impostor in a register of the constructor itself.
local keep2 = {}
local function newLocalImpostor(i)
  local setmetatable = function(t, mt) keep2[#keep2 + 1] = t; return {} end
  local o = setmetatable({ x = i, y = 0 }, Plain)
  o.z = i + 1
  o.w = nil
  o.x = (o.x or 0) + 1
  return o
end

local makers = { newGlobal, newLocal, newUpvalue }
for round = 1, 300 do
  for kind = 1, 3 do
    local o = makers[kind](round)
    if round == 1 or round == 300 then
      note("plain", kind, round, keys(o), o.x, o.y, o.z, o.w)
    end
    assert(o.z == round + 1 and o.x == round + 1)
    assert(getmetatable(o) == Plain)
  end
  local imp2 = newLocalImpostor(round)
  if round == 1 or round == 300 then
    note("local-impostor", round, keys(imp2), keys(keep2[#keep2]), getmetatable(imp2), getmetatable(keep2[#keep2]))
  end
  local imp = newImpostor(round)
  if round == 1 or round == 300 then
    note("impostor", round, keys(imp), keys(keep[#keep]), getmetatable(imp), getmetatable(keep[#keep]))
  end
end

-- 2. __newindex added later, removed later, on a shared class.
local Cls = {}
Cls.__index = Cls
local seen = {}
local function mk(i)
  local o = setmetatable({ a = i }, Cls)
  o.b = i
  o.a = i + 1
  return o
end
for phase = 1, 4 do
  if phase == 2 then
    Cls.__newindex = function(t, k, v) seen[#seen + 1] = k; rawset(t, k, v) end
  elseif phase == 3 then
    Cls.__newindex = nil
  elseif phase == 4 then
    Cls.__newindex = function(t, k, v) seen[#seen + 1] = "f" .. k end
  end
  for i = 1, 200 do
    local o = mk(i)
    if i == 200 then
      note("late", phase, keys(o), #seen)
    end
  end
  seen = {}
end

-- 3. __newindex as a table.
local sink = {}
local TC = { __newindex = sink }
for i = 1, 200 do
  local o = setmetatable({ a = 1 }, TC)
  o.b = i
  o.a = i
  if i == 200 then note("tbl", keys(o), keys(sink), sink.b, o.a) end
end
TC.__newindex = nil
for i = 1, 200 do
  local o = setmetatable({ a = 1 }, TC)
  o.b = i
  if i == 200 then note("tbl-removed", keys(o), keys(sink)) end
end
-- chain of tables
local deep = {}
local mid = setmetatable({}, { __newindex = deep })
local top = setmetatable({}, { __newindex = mid })
for i = 1, 100 do top["k" .. (i % 3)] = i end
note("chain", keys(top), keys(mid), keys(deep), deep.k1)

-- 4. A stored nil is absent: __newindex fires again.
local nilfires = 0
local NC = { __newindex = function(t, k, v) nilfires = nilfires + 1; rawset(t, k, v) end }
local o = setmetatable({ a = 1 }, NC)
o.a = nil
o.a = 2
o.a = 3
note("nilkey", nilfires, o.a)
local o2 = setmetatable({}, NC)
for i = 1, 50 do o2.q = nil; o2.q = i end
note("nilkey2", nilfires, o2.q)

-- 5. rawset into objects, with and without __newindex.
local RC = { __newindex = function() error("blocked") end }
local r = setmetatable({ a = 1 }, RC)
rawset(r, "b", 2)
local rok, rerr = pcall(function() r.c = 3 end)
note("rawset", keys(r), r.b, rok, msg(rerr))
RC.__newindex = nil
r.c = 3
note("rawset-after", keys(r))
-- adding __newindex through rawset on the metatable
local M = {}
local m1 = setmetatable({}, M)
m1.x = 1
rawset(M, "__newindex", function(t, k, v) rawset(t, k, v * 2) end)
m1.y = 5
note("rawset-meta", keys(m1), m1.x, m1.y)
M.__newindex = nil
m1.z = 7
note("rawset-meta-removed", m1.z)

-- 6. A table that is its own metatable.
local self = {}
self.__index = self
setmetatable(self, self)
self.a = 1
self.__newindex = function(t, k, v) rawset(t, k, "n" .. tostring(v)) end
self.b = 2
note("self", keys(self), self.a, self.b)
self.__newindex = nil
self.c = 3
note("self-removed", self.c)

-- 7. pairs sees exactly the stored keys, including after deletes and refills.
for i = 1, 100 do
  local p = setmetatable({ a = 1, b = 2, c = 3 }, Plain)
  p.b = nil
  if i % 2 == 0 then p.d = 4 end
  if i % 3 == 0 then p.b = 5 end
  local n = 0
  for k, v in pairs(p) do
    assert(v ~= nil)
    n = n + 1
  end
  local want = 2 + (i % 2 == 0 and 1 or 0) + (i % 3 == 0 and 1 or 0)
  assert(n == want, "pairs count " .. n .. " want " .. want)
  if i == 6 then note("pairs", keys(p), n) end
end

-- 8. getmetatable with and without __metatable; protected setmetatable.
local free = setmetatable({}, { tag = 1 })
note("getmt", getmetatable(free).tag)
local locked = setmetatable({}, { __metatable = "locked" })
note("getmt-locked", getmetatable(locked))
local ok, e = pcall(setmetatable, locked, {})
note("protected", ok, msg(e))
local fm = {}
local obj = setmetatable({}, fm)
note("unprotected", getmetatable(obj) == fm)
fm.__metatable = "now"
note("late-protect", getmetatable(obj))
ok, e = pcall(setmetatable, obj, {})
note("late-protect-set", ok, msg(e))
fm.__metatable = nil
note("unprotect", getmetatable(obj) == fm, (pcall(setmetatable, obj, nil)))
note("cleared", getmetatable(obj))
-- a false __metatable still protects, and getmetatable returns false
local pm = { __metatable = false }
local po = setmetatable({}, pm)
local fok, ferr = pcall(setmetatable, po, {})
note("false-protect", getmetatable(po), fok, msg(ferr))
