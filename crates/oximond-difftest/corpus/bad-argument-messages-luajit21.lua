-- oximond-difftest repro profile=luajit21 from=luajit21 baseline cases on candidate c9f8733 (76 cases)
-- Bad-argument errors from statement calls: stock names the function, counts
-- the method receiver as #1's self and adds the caller's position.
print("o1#", pcall(function() local w = ("x"):rep({}) end))
print("o2#", pcall(function() setmetatable(1, {}) end))
print("o3#", pcall(function() local w = string.rep() end))
