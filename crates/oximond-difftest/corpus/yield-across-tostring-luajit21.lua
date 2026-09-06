-- oximond-difftest repro profile=luajit21 from=luajit21 yield across metamethods
-- LuaJIT runs __tostring from tostring as a plain call, so it may yield.
local y = coroutine.yield
local function clean(v) return (tostring(v):gsub("^[^:\n]+:%d+: ", "")) end
local function drive(name, f)
  local c = coroutine.create(f)
  local out = {}
  local ok, a = coroutine.resume(c)
  local n = 0
  while coroutine.status(c) == "suspended" and n < 20 do
    n = n + 1
    out[#out + 1] = tostring(ok) .. ":" .. clean(a)
    ok, a = coroutine.resume(c, "r" .. n)
  end
  out[#out + 1] = tostring(ok) .. ":" .. clean(a)
  print(name, table.concat(out, " | "))
end
local o = setmetatable({}, {__tostring = function() return y("ts") end})
drive("tostring", function() return tostring(o) end)
drive("pcall-tostring", function() return pcall(tostring, o) end)
drive("tostring-twice", function() return tostring(o), tostring(o) end)
drive("tostring-error", function()
  local e = setmetatable({}, {__tostring = function() y("e") error("bad", 0) end})
  return pcall(tostring, e)
end)
drive("sort", function() table.sort({3, 2, 1}, function(a, b) local r = y("s") return a < b end) end)
drive("gsub", function() return (string.gsub("abc", "b", function() local r = y("g") return r end)) end)

-- LuaJIT names the Lua line that called yield; 5.4 and 5.5 give no position
local c = coroutine.create(function() table.sort({3, 2, 1}, function(a, b) local r = y("s") return a < b end) end)
local ok, m = coroutine.resume(c)
print(ok, m:match("^[^:]+:%d+: attempt") ~= nil, (m:gsub("^[^:]+:%d+: ", "")))
