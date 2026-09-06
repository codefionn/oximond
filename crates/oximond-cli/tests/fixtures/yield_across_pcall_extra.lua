local y = coroutine.yield
-- isyieldable inside pcall
local co = coroutine.wrap(function()
  print("iy", coroutine.isyieldable(), pcall(coroutine.isyieldable))
  print(pcall(function() return coroutine.isyieldable() end))
  return "done"
end)
print(co())

-- hot loop with garbage between yields
co = coroutine.wrap(function()
  local sum = 0
  for i = 1, 5000 do
    local ok, v = pcall(function(n) local r = y(n); local t = {r, tostring(r)}; return t[1] end, i)
    sum = sum + v
    if i % 1000 == 0 then collectgarbage() end
  end
  return "sum", sum
end)
local v = co()
local n = 0
while type(v) == "number" do
  n = n + 1
  v = co(v * 2)
end
print(n, v, co == nil)

-- yields from a hot loop inside the protected function, with xpcall and an error at the end
co = coroutine.wrap(function()
  local acc = 0
  local ok, e = xpcall(function()
    for i = 1, 3000 do
      local a = y(i)
      acc = acc + (a or 0)
      for j = 1, 20 do acc = acc + j end
    end
    error("end-of-loop", 0)
  end, function(m) return "H:" .. m end)
  return ok, e, acc
end)
local r, e, acc = co()
local count = 0
while type(r) == "number" do count = count + 1; r, e, acc = co(1) end
print(count, r, e, acc)
