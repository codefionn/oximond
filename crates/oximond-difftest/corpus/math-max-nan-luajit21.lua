-- oximond-difftest repro profile=luajit21 from=luajit21 seeds 509, 1199, 2346
-- LuaJIT's math.max/min with a NaN argument.
local nan = 0/0
print("o1#", tostring(math.max(1.5, nan) ~= math.max(1.5, nan)))
print("o2#", tostring(math.max(nan, 92) ~= math.max(nan, 92)))
print("o3#", tostring(math.min(nan, 1) ~= math.min(nan, 1)))
