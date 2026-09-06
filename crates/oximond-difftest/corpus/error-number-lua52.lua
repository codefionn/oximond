-- oximond-difftest repro profile=lua52
local fail = assert(load("error(42)", "@case.lua"))
print(pcall(fail))
print(xpcall(fail, function(err) return "H:" .. tostring(err) end))
print(pcall(function() error(42, 0) end))
