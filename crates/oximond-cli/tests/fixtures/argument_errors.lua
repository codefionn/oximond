-- Bad-argument errors named by their call: a statement call names the
-- function and gives the position. A tail call (LuaJIT) is named by the call
-- that reached the tail-calling function, and gets '?' with no position only
-- when that was not a Lua call (pcall). Positions print as L<line>.
local function show(...)
  local ok, message = ...
  message = tostring(message):gsub("^[^:]*:(%d+): ", "L%1: ")
  print(ok, message)
end
show(pcall(function() local w = ("x"):rep({}) end))
show(pcall(function() local w = ("x"):rep() end))
show(pcall(function() local s = "x"; local w = s:rep({}) end))
show(pcall(function() return ("x"):rep({}) end))
show(pcall(function() return string.rep() end))
show(pcall(function() setmetatable(1, {}) end))
show(pcall(function() return setmetatable(1, {}) end))
show(pcall(function() local t = {f = string.rep}; t.f() end))
show(pcall(function() local t = {f = string.rep}; t:f() end))
show(pcall(function() local f = string.rep; f() end))
show(pcall(function() string.char(256) end))
show(pcall(function() string.char(-1) end))
show(pcall(function() string.char(65, 300) end))
show(pcall(function() return string.char(256) end))
show(pcall(string.rep))
show(pcall(function() local a = tostring() end))
show(pcall(function() math.floor({}) end))
show(pcall(function() io.stdout:write({}) end))
show(pcall(function() local x = select(0) end))
show(pcall(function() local x = string.rep("x", {}) end))
show(pcall(function() local x = ("x"):sub({}) end))
local inner_tail = function() return string.rep("x", {}) end
local M = {}
function M.f() return string.rep("x", {}) end
function M:m() return string.rep("x", {}) end
function M.g(s) return s:rep({}) end
function M:b() return setmetatable(1) end
local function nested() return inner_tail() end
show(pcall(function() local r = inner_tail() end))
show(pcall(function() local r = M.f() end))
show(pcall(function() local r = M:m() end))
show(pcall(function() local r = M.g("x") end))
show(pcall(function() local r = M:b() end))
show(pcall(function() local r = M.b(M) end))
show(pcall(function() local r = nested() end))
show(pcall(function() return nested() end))
show(pcall(nested))
show(pcall(function()
  local r = (function() return math.floor("a") end)()
end))
show(pcall(coroutine.wrap(function() return string.rep("x", {}) end)))
