local function get(t) return t.x end
local base = {x = 1}
local o = setmetatable({}, {__index = base})
for i = 1, 40 do assert(get(o) == 1) end
local co = coroutine.wrap(function()
  for i = 1, 5 do
    coroutine.yield(get(o))
  end
  base.x = 2
  coroutine.yield(get(o))
  getmetatable(o).__index = {x = 3}
  return get(o)
end)
for i = 1, 5 do assert(co() == 1) end
assert(co() == 2)
assert(co() == 3)
assert(get(o) == 3)

-- A hook sees the same values.
local count = 0
debug.sethook(function() count = count + 1 end, "", 1)
for i = 1, 10 do assert(get(o) == 3) end
debug.sethook()
assert(count > 0)
