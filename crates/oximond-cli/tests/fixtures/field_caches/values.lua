local function put(t, v) t.v = v end
local t = {v = 0}
for i = 1, 100 do put(t, "s" .. i) end
assert(t.v == "s100")
put(t, 1.5); assert(t.v == 1.5)
put(t, true); assert(t.v == true)
put(t, false); assert(t.v == false)
put(t, {1}); assert(t.v[1] == 1)
put(t, print); assert(t.v == print)
collectgarbage()
put(t, "end"); assert(t.v == "end")

-- An old table takes young tables through a cached store.
local holder = {child = false}
local function setchild(h, c) h.child = c end
collectgarbage()
collectgarbage()
for i = 1, 300 do
  setchild(holder, {n = i})
  local junk = {}
  for j = 1, 50 do junk[j] = {j} end
  if i % 13 == 0 then collectgarbage() end
end
local live = holder.child
assert(live.n == 300)
for i = 1, 200 do local junk = {i, {i}} end
collectgarbage()
assert(holder.child == live and live.n == 300)

-- A hashed holder.
local wide = {child = false}
for i = 1, 40 do wide["f" .. i] = i end
collectgarbage()
for i = 1, 300 do
  setchild(wide, {n = i})
  local junk = {}
  for j = 1, 50 do junk[j] = {j} end
  if i % 17 == 0 then collectgarbage() end
end
collectgarbage()
assert(wide.child.n == 300)
