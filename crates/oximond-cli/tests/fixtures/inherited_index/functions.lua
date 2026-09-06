local function get(t) return t.x end
local function warm(t, expected)
  for i = 1, 40 do assert(get(t) == expected) end
end

-- A function in the chain is called every time, never cached.
local calls = 0
local f = setmetatable({}, {__index = function(_, k) calls = calls + 1; return k .. calls end})
for i = 1, 40 do assert(get(f) == "x" .. i) end

-- A function at the end of a chain of tables.
local tail = setmetatable({}, {__index = function(t, k) calls = calls + 1; return "tail" end})
local mid = setmetatable({}, {__index = tail})
local top = setmetatable({}, {__index = mid})
warm(top, "tail")
-- A table with the key put in front of the function ends the walk there.
mid.x = "mid"
warm(top, "mid")
mid.x = nil
warm(top, "tail")

-- A function becomes a table and back.
local mt = {__index = function() return "fn" end}
local o = setmetatable({}, mt)
warm(o, "fn")
mt.__index = {x = "table"}
warm(o, "table")
mt.__index = function() return "fn2" end
warm(o, "fn2")
mt.__index = {x = "table2"}
warm(o, "table2")

-- The function sees the receiver and the key, not the holder.
local seen
local g = setmetatable({}, {__index = setmetatable({}, {__index = function(t, k) seen = t; return 7 end})})
for i = 1, 40 do assert(get(g) == 7) end
assert(seen ~= g and getmetatable(g).__index == getmetatable(g).__index)

-- A non-function, non-table __index is an error exactly as before.
local bad = setmetatable({}, {__index = 5})
for i = 1, 3 do assert(not pcall(get, bad)) end
-- A self-referencing chain is an error, not a hang.
local loop = {}
setmetatable(loop, {__index = loop})
for i = 1, 3 do assert(not pcall(get, loop)) end
