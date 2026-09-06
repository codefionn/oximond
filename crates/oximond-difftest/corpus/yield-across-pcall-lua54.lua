-- oximond-difftest repro profile=lua54 from=lua54 seed 2
local co = coroutine.create(function()
  local ok, v = pcall(function() return coroutine.yield("p") end)
  return "after", ok, v
end)
print("o1#", coroutine.resume(co))
print("o2#", coroutine.resume(co, 5))
