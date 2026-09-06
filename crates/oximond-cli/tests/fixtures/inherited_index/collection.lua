local function get(t) return t.x end
local function warm(t, expected)
  for i = 1, 40 do assert(get(t) == expected) end
end

-- A weak __index table that dies stops answering.
local mt = setmetatable({}, {__mode = "v"})
mt.__index = {x = "weak"}
local keep = mt.__index
local o = setmetatable({}, mt)
warm(o, "weak")
keep = nil
collectgarbage()
collectgarbage()
assert(mt.__index == nil)
warm(o, nil)

-- Chains are rebuilt after collections that free other tables.
local class = {x = "kept"}
class.__index = class
local inst = setmetatable({}, class)
for round = 1, 20 do
  for i = 1, 200 do local junk = {i, setmetatable({}, class)} end
  collectgarbage()
  warm(inst, "kept")
end

-- A freed table's slot reused by another does not answer for the old one.
local function make(value)
  local m = {__index = {x = value}}
  return setmetatable({}, m)
end
local one = make("a")
warm(one, "a")
one = nil
collectgarbage()
local two = make("b")
warm(two, "b")
