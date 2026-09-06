-- Sites are filled and invalidated inside one activation of a function, which
-- holds the chunk's site slice across the loop: every fill is seen at once,
-- and a stale or empty site never answers.
local function run()
  local small = {a = 1, b = 2}                    -- ordered
  local shaped = (function(i) return {x = i, y = i} end)(5)
  local wide = {}
  for i = 1, 40 do wide["f" .. i] = i end
  wide.x = 7                                      -- hashed
  local sum = 0
  for i = 1, 300 do
    sum = sum + small.a + shaped.x + (wide.x or 0) + (small.c or 0)
    if i == 50 then shaped.z = 1 end              -- layout change mid-loop
    if i == 100 then wide.x = nil end             -- field removed
    if i == 150 then wide.x = 3 end
    if i == 200 then small.b = nil; small.c = 4 end
    if i == 250 then shaped = {y = 1, x = 9} end  -- another layout
  end
  return sum
end

local expected = 0
local sx, wx, sc = 5, 7, 0
for i = 1, 300 do
  expected = expected + 1 + sx + (wx or 0) + sc
  if i == 100 then wx = nil end
  if i == 150 then wx = 3 end
  if i == 200 then sc = 4 end
  if i == 250 then sx = 9 end
end
assert(run() == expected)
assert(run() == expected)

-- A global read site that starts empty and sees the global change, vanish and
-- return inside one loop.
G = 1
local function globals()
  local total = 0
  for i = 1, 200 do
    total = total + (G or 100)
    if i == 20 then G = 2 end
    if i == 60 then G = nil end
    if i == 80 then G = 3 end
  end
  return total
end
local mirror, g = 0, 1
for i = 1, 200 do
  mirror = mirror + (g or 100)
  if i == 20 then g = 2 end
  if i == 60 then g = nil end
  if i == 80 then g = 3 end
end
assert(globals() == mirror)
G = 1
assert(globals() == mirror)
