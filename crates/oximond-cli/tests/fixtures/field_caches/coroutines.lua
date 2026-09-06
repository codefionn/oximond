local function get(t) return t.x end
local function put(t, v) t.x = v end
local ta = {x = "a"}
local tb = {}
for i = 1, 40 do tb["k" .. i] = i end
tb.x = "b"
local function worker(t, n)
  for i = 1, n do
    assert(get(t) == t.x)
    put(t, t.x)
    coroutine.yield(get(t))
  end
end
local c1 = coroutine.wrap(function() worker(ta, 50) end)
local c2 = coroutine.wrap(function() worker(tb, 50) end)
for i = 1, 50 do
  assert(c1() == "a")
  assert(c2() == "b")
  assert(get(ta) == "a" and get(tb) == "b")
end
