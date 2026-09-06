-- One read site and one write site see the same table change layout between hits.
local function get(t) return t.x end
local function put(t, v) t.x = v end
local t = {x = 1, y = 2, z = 3}
for i = 1, 30 do assert(get(t) == 1) end
t.w = 9; assert(get(t) == 1)
t.a = 8; assert(get(t) == 1)
for i = 1, 30 do put(t, i) end
assert(t.x == 30 and get(t) == 30)
t.x = nil; assert(get(t) == nil)
t.x = 4; assert(get(t) == 4)
for i = 1, 40 do t["k" .. i] = i end
for i = 1, 30 do assert(get(t) == 4) end
for i = 41, 3000 do t["k" .. i] = i end
assert(get(t) == 4)
put(t, 5); assert(get(t) == 5)
for i = 1, 3000 do t["k" .. i] = nil end
assert(get(t) == 5)
t.x = nil; assert(get(t) == nil)
put(t, 6); assert(get(t) == 6 and rawget(t, "x") == 6)
local n = 0
for _ in pairs(t) do n = n + 1 end
assert(n == 5, n)

-- Tables built alike share a layout; each keeps its own values.
local function pair(i) return {x = i, y = -i} end
local a, b, c = pair(1), pair(2), pair(3)
for round = 1, 40 do
  assert(get(a) == 1 and get(c) == 3)
  assert(get(b) == (round == 1 and 2 or 19 + round))
  put(b, 20 + round)
end
assert(get(b) == 60)
b.extra = true
assert(get(b) == 60 and get(a) == 1)
b.x = nil
assert(get(b) == nil and get(c) == 3)
