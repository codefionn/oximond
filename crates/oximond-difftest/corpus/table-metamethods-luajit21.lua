-- oximond-difftest repro profile=luajit21
-- table.concat, unpack, insert, remove, move and ipairs follow __index,
-- __newindex and __len in Lua 5.3 and later; 5.1 and LuaJIT read raw.
local unpack = table.unpack or unpack
local log = {}
local function proxy(n, store)
  store = store or {}
  return setmetatable({}, {
    __index = function(_, k) log[#log+1] = "get" .. tostring(k); return store[k] end,
    __newindex = function(_, k, v) log[#log+1] = "set" .. tostring(k) .. "=" .. tostring(v); store[k] = v end,
    __len = function() log[#log+1] = "len"; return n end,
  }), store
end
local function flush(name, ...)
  print(name, ..., table.concat(log, ","))
  log = {}
end
local p = proxy(3, {10, 20, 30})
flush("concat", pcall(table.concat, p, "-"))
flush("concat-ij", pcall(table.concat, p, "-", 2, 3))
flush("unpack", pcall(unpack, p))
flush("unpack-ij", pcall(unpack, p, 2, 3))
local p = proxy(2, {1, 2})
flush("insert", pcall(table.insert, p, 99))
flush("insert-pos", pcall(table.insert, p, 1, 99))
local p = proxy(3, {1, 2, 3})
flush("remove", pcall(table.remove, p))
local p = proxy(3, {1, 2, 3})
flush("remove-pos", pcall(table.remove, p, 1))
local p = proxy(3, {1, 2, 3})
flush("move", pcall(table.move or function() return "nomove" end, p, 1, 3, 2))
local p, store = proxy(3, {1, 2, 3})
for i, v in ipairs(p) do log[#log+1] = i .. ":" .. v end
flush("ipairs")
local q = setmetatable({}, {__index = function(t, k) if k <= 3 then return k * 2 end end})
for i, v in ipairs(q) do log[#log+1] = i .. ":" .. v end
flush("ipairs-index-only")
flush("unpack-index-only", pcall(unpack, q, 1, 3))
flush("concat-index-only", pcall(table.concat, q, ",", 1, 3))
local r = setmetatable({1, 2, 3}, {__len = function() return 2 end})
flush("concat-len-only", pcall(table.concat, r, ","))
flush("unpack-len-only", pcall(unpack, r))
flush("insert-len-only", pcall(table.insert, r, 9))
print(rawget(r, 3), rawget(r, 4))
-- hot loops reach the tracing engine
local n = 0
local store = {}
local h = setmetatable({}, {
  __index = function(_, k) n = n + 1; return store[k] end,
  __newindex = function(_, k, v) n = n + 1; store[k] = v end,
  __len = function() return #store end,
})
for i = 1, 300 do table.insert(h, i) end
print(#store, n)
local s = 0
for round = 1, 10 do
  for i, v in ipairs(h) do s = s + v end
end
print(s, n)
for i = 1, 100 do table.remove(h) end
print(#store, n)
local t = 0
for i = 1, 200 do t = t + select("#", unpack(h, 1, 3)) end
for i = 1, 200 do t = t + #table.concat(h, ",", 1, 3) end
print(t, n)
-- tables whose metatable has no element metamethods stay raw
local c = setmetatable({}, {__tostring = function() return "c" end})
for i = 1, 100 do table.insert(c, i) end
print(#c, table.concat(c, ",", 1, 5))
-- an __index class keeps working with element functions
local C = {}
C.__index = C
local o = setmetatable({}, C)
for i = 1, 100 do table.insert(o, i * 2) end
print(#o, o[100], table.remove(o), #o)
-- argument checks
print(pcall(table.insert, h, 1, 2, 3))
print(pcall(table.insert, h))
print(pcall(table.concat, setmetatable({{}}, {__index = function() end}), ",", 1, 1))
local q = setmetatable({}, {__len = function() return 2.0 end, __newindex = function(t, k, v) rawset(t, k, v) end})
print(pcall(table.insert, q, 1), rawget(q, 3))
