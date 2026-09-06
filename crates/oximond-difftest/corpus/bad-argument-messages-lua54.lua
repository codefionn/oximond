-- oximond-difftest repro profile=lua54 from=lua54 campaign 1, P observations
-- The calls are statements, not tail calls: a tail call reports correctly.
print("o1#", pcall(function() setmetatable(1, {}) end))
print("o2#", pcall(function() local w = ("x"):rep({}) end))
print("o3#", pcall(function() string.rep() end))
print("o4#", pcall(function() local w = string.char(256) end))
