local function put(t, v) t.x = v end
local log = {}
local t = {x = 0}
for i = 1, 30 do put(t, i) end
assert(t.x == 30)
setmetatable(t, {__newindex = function(tt, k, v)
  log[#log + 1] = k .. "=" .. tostring(v)
  rawset(tt, k, v)
end})
put(t, 31)
assert(t.x == 31 and #log == 0, "a present field is a raw store")
rawset(t, "x", nil)
put(t, 32)
assert(#log == 1 and log[1] == "x=32" and t.x == 32, "an absent field calls __newindex")
put(t, 33)
assert(#log == 1 and t.x == 33)
put(t, nil)
assert(rawget(t, "x") == nil)
put(t, 34)
assert(#log == 2 and log[2] == "x=34")

-- __newindex added later to a table whose field is cached.
local u = {x = 1}
for i = 1, 30 do put(u, i) end
local calls = 0
setmetatable(u, {__newindex = function() calls = calls + 1 end})
put(u, 99)
assert(u.x == 99 and calls == 0)
u.x = nil
put(u, 100)
assert(rawget(u, "x") == nil and calls == 1)

-- A hashed table with a metatable.
local h = {x = 1}
for i = 1, 40 do h["f" .. i] = i end
for i = 1, 30 do put(h, i) end
local seen = 0
setmetatable(h, {__newindex = function(tt, k, v) seen = seen + 1; rawset(tt, k, v) end})
put(h, 5)
assert(h.x == 5 and seen == 0)
h.fresh = 1
assert(seen == 1)
