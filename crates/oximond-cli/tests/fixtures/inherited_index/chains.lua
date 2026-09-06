-- Chains of depth 1, 2 and 3 answer the same read, each warmed up first.
local function get(t) return t.x end
local function warm(t, expected)
  for i = 1, 40 do assert(get(t) == expected) end
end

local d1 = setmetatable({}, {__index = {x = "one"}})
warm(d1, "one")

local d2 = setmetatable({}, {__index = setmetatable({}, {__index = {x = "two"}})})
warm(d2, "two")

local d3base = {x = "three"}
local d3mid2 = setmetatable({}, {__index = d3base})
local d3mid1 = setmetatable({}, {__index = d3mid2})
local d3 = setmetatable({}, {__index = d3mid1})
warm(d3, "three")

-- Objects of one class share the answer; their own fields still win.
local Class = {x = "class", y = "y"}
Class.__index = Class
local a, b = setmetatable({}, Class), setmetatable({x = "own"}, Class)
warm(a, "class")
warm(b, "own")
assert(a.y == "y" and b.y == "y")

-- A key found nowhere reads nil at every depth.
local function missing(t) return t.nope end
for i = 1, 40 do
  assert(missing(d1) == nil and missing(d2) == nil and missing(d3) == nil)
end

-- Mid-chain table shadows the base after warm-up.
d3mid2.x = "shadow2"
warm(d3, "shadow2")
d3mid1.x = "shadow1"
warm(d3, "shadow1")
d3mid1.x = nil
warm(d3, "shadow2")
d3mid2.x = nil
warm(d3, "three")
d3base.x = "changed"
warm(d3, "changed")
d3base.x = nil
warm(d3, nil)
d3base.x = "back"
warm(d3, "back")
