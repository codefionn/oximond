-- raw messages: LuaJIT adds a position only when a Lua frame calls yield
-- itself, so a tail call yields a bare message
local y = coroutine.yield
local function try(name, f)
  local c = coroutine.create(f)
  local ok, m = coroutine.resume(c)
  print(name, ok, (tostring(m):gsub("^[^:]+%.lua", "FILE")))
end
try("gsub-tail", function() return (string.gsub("ab", "a", function(c) return y(c) end)) end)
try("gsub-call", function() return (string.gsub("ab", "a", function(c) local r = y(c) return r end)) end)
try("sort-tail", function() table.sort({3, 2, 1}, function(a, b) return y() end) end)
try("sort-call", function() table.sort({3, 2, 1}, function(a, b) local r = y() return a < b end) end)
