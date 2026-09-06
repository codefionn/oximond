local function get(t) return t.x end
local function warm(t, expected)
  for i = 1, 40 do assert(get(t) == expected) end
end

-- rawset on the receiver adds the key.
local base = {x = "base"}
local o = setmetatable({}, {__index = base})
warm(o, "base")
rawset(o, "x", "own")
warm(o, "own")
o.x = nil
warm(o, "base")

-- Changing the receiver's metatable.
local other = setmetatable({}, {__index = {x = "other"}})
warm(o, "base")
setmetatable(o, getmetatable(other))
warm(o, "other")
setmetatable(o, nil)
warm(o, nil)
setmetatable(o, {__index = {x = "again"}})
warm(o, "again")

-- Replacing __index of the metatable, through a hot field store as well.
local mt = {__index = {x = "first"}}
local p = setmetatable({}, mt)
warm(p, "first")
mt.__index = {x = "second"}
warm(p, "second")
local function setindex(m, v) m.__index = v end
for i = 1, 40 do setindex(mt, {x = "loop" .. i}); assert(get(p) == "loop" .. i) end
mt.__index = nil
warm(p, nil)
mt.__index = {x = "third"}
warm(p, "third")
rawset(mt, "__index", nil)
warm(p, nil)

-- Mutating the table __index names: change, remove, add.
local holder = {x = 1}
local q = setmetatable({}, {__index = holder})
warm(q, 1)
holder.x = 2
warm(q, 2)
for i = 1, 40 do holder.x = i; assert(get(q) == i) end
holder.x = nil
warm(q, nil)
holder.x = "again"
warm(q, "again")
holder.other = true
warm(q, "again")
for i = 1, 40 do holder["k" .. i] = i end
warm(q, "again")
holder.x = nil
for i = 1, 40 do holder["k" .. i] = nil end
warm(q, nil)

-- Setting the metatable of a table in the middle.
local deep = {x = "deep"}
local mid = {}
local r = setmetatable({}, {__index = mid})
warm(r, nil)
setmetatable(mid, {__index = deep})
warm(r, "deep")
setmetatable(mid, nil)
warm(r, nil)
setmetatable(mid, {__index = deep})
warm(r, "deep")
getmetatable(mid).__index = {x = "moved"}
warm(r, "moved")
