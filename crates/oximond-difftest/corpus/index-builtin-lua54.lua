-- oximond-difftest repro profile=lua54
-- A builtin as __index or __newindex is called with the table and key.
local t = setmetatable({}, {__index = rawget})
print(t.x, t[1])
local s = setmetatable({}, {__index = select})
print((pcall(function() return s[2] end)))
local w = setmetatable({}, {__newindex = rawset})
w.x = 5
print(rawget(w, "x"))
local ts = setmetatable({}, {__index = tostring})
print(type(ts.x))
local loaded = setmetatable({}, {__index = load})
print(pcall(function() return loaded.x end))
print(pcall(function() return loaded[1] end))
local newloaded = setmetatable({}, {__newindex = load})
print(pcall(function() newloaded.x = 1 end))
local sorted = {3, 1, 2}
print((pcall(table.sort, sorted, load)))
print(pcall(table.sort, {"a", "b", "c"}, load))
