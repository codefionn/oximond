-- oximond-difftest repro profile=lua51 from=lua54 seed 55, lua55 seeds 163 and 176 (reduced by hand)
-- The protected function writes an upvalue, then errors. With a Lua
-- message handler the write is lost; with print as the handler it stays.
local s = 1
xpcall(function() s = 2 error("x") end, function(m) return m end)
print("o1#", s)
local t = 1
xpcall(function() t = 2 error("x") end, print)
print("o2#", t)
