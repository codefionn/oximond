-- oximond-difftest repro profile=lua51
-- Runaway recursion through a table function or print is a catchable error.
local t = setmetatable({}, {__newindex = function(t, k, v) table.insert(t, k, v) end})
print(pcall(function() t[1] = 5 end))
local o = setmetatable({}, {__tostring = function(s) print(s) return "x" end})
print(pcall(print, o))
local c = setmetatable({}, {__index = function(t) return table.concat(t) end, __len = function() return 1 end})
print(pcall(function() return table.concat(c) end))
