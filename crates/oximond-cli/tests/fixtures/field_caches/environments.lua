-- Gives `f` an environment of its own: `_ENV` is the function's first upvalue
-- and is shared with the main chunk unless it is joined to a fresh cell.
local setenv = setfenv or function(f, env)
  debug.upvaluejoin(f, 1, function() return env end, 1)
end
gx = "g1"
local function rd() return gx end
local function wr(v) gx = v end
for i = 1, 40 do assert(rd() == "g1") end
wr("g2")
assert(rd() == "g2")
for i = 1, 40 do wr("loop" .. i) end
assert(rd() == "loop40")
wr("g2")

local env = setmetatable({}, {__index = _G})
env.gx = "e1"
setenv(rd, env)
assert(rd() == "e1")
for i = 1, 30 do assert(rd() == "e1") end
env.gx = nil
assert(rd() == "g2", "falls through to _G")
env.gx = "e2"
assert(rd() == "e2")
setenv(rd, _G)
assert(rd() == "g2")

local big = {gx = "big"}
for i = 1, 100 do big["p" .. i] = i end
setenv(rd, big)
for i = 1, 20 do assert(rd() == "big") end
for i = 1, 100 do big["p" .. i] = nil end
assert(rd() == "big")
big.gx = nil
assert(rd() == nil)
setenv(rd, _G)
gx = nil
assert(rd() == nil)
gx = "last"
assert(rd() == "last")

-- Growing the globals table moves its buckets.
local function late() return late_global end
late_global = "before"
for i = 1, 30 do assert(late() == "before") end
for i = 1, 2000 do _G["pad_" .. i] = i end
assert(late() == "before")
for i = 1, 2000 do _G["pad_" .. i] = nil end
assert(late() == "before")
late_global = nil
assert(late() == nil)
