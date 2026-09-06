-- oximond-difftest repro profile=lua51
-- Natives that need the VM (print, table functions) as metamethod handlers.
local a = setmetatable({}, {__add = print, __tostring = function() return "A" end})
print(pcall(function() return a + 1 end))
local ci = setmetatable({1, 2, 3}, {__index = table.concat})
print(pcall(function() return ci[","] end))
local ct = setmetatable({}, {__index = table.unpack})
print(pcall(function() return ct[1] end))
