local function get(t) return t.x end
local function put(t, v) t.x = v end
local function wide(i)
  local t = {}
  for k = 1, 40 do t["f" .. k] = k end
  t.x = i
  return t
end
for round = 1, 200 do
  local t = wide(round)
  for i = 1, 5 do assert(get(t) == round) end
  put(t, -round)
  assert(get(t) == -round)
  t = nil
  if round % 7 == 0 then collectgarbage() end
end
local function narrow(i) return {x = i, y = i} end
for round = 1, 300 do
  local a, b = narrow(round), narrow(-round)
  assert(get(a) == round and get(b) == -round)
  put(a, 0)
  assert(get(a) == 0 and get(b) == -round)
  if round % 11 == 0 then collectgarbage() end
end
-- The only table the site knew is collected, then others take its place.
local site_owner = wide("first")
for i = 1, 20 do assert(get(site_owner) == "first") end
site_owner = nil
collectgarbage()
collectgarbage()
local keep = {}
for i = 1, 400 do keep[i] = wide(i) end
for i = 1, 400 do assert(get(keep[i]) == i) end
