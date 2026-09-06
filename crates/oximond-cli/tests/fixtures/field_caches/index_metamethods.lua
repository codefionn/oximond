local function get(t) return t.x end
local t = {}
for i = 1, 20 do assert(get(t) == nil) end
setmetatable(t, {__index = function(_, k) return "idx-" .. k end})
assert(get(t) == "idx-x")
t.x = 1
for i = 1, 30 do assert(get(t) == 1) end
t.x = nil
assert(get(t) == "idx-x")

local base = {x = "base"}
local c = setmetatable({}, {__index = base})
for i = 1, 30 do assert(get(c) == "base") end
base.x = "changed"
assert(get(c) == "changed")
c.x = "own"
for i = 1, 30 do assert(get(c) == "own") end
getmetatable(c).__index = nil
assert(get(c) == "own")
c.x = nil
assert(get(c) == nil)

-- A present field never consults __index, however the metatable changes.
local d = setmetatable({x = "d"}, {__index = function() return "wrong" end})
for i = 1, 30 do assert(get(d) == "d") end
getmetatable(d).__index = {x = "also wrong"}
assert(get(d) == "d")
setmetatable(d, nil)
assert(get(d) == "d")
-- Added later, to a table whose field is cached, then the field goes away.
local e = {x = "e", y = 1}
for i = 1, 30 do assert(get(e) == "e") end
setmetatable(e, {__index = function(_, k) return "late-" .. k end})
assert(get(e) == "e")
e.x = nil
assert(get(e) == "late-x")
