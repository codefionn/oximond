-- pcall and xpcall call a native directly; the native calls back into Lua
-- (or a native) and the callback raises. Each call must return false plus the
-- stock message instead of escaping.
local unpack = unpack or table.unpack
local function show(label, ...)
  local t, n = {...}, select("#", ...)
  for i = 1, n do
    local v = t[i]
    t[i] = (type(v) == "function" or type(v) == "table") and type(v) or tostring(v)
  end
  print(label, unpack(t, 1, n))
end
local function boom() error("x") end
local function boomt() error({}, 0) end
local function h(m) return "H:" .. tostring(m) end
local direct_xpcall = _VERSION ~= "Lua 5.1"
local function both(label, f, ...)
  show(label .. " pcall", pcall(f, ...))
  if direct_xpcall then
    show(label .. " xpcall", xpcall(f, h, ...))
  end
end
-- gsub
both("gsub ok", string.gsub, "abc", "b", function(c) return c:upper() end)
both("gsub lua", string.gsub, "a", "a", boom)
both("gsub native", string.gsub, "a", "a", string.rep)
both("gsub later", string.gsub, "aaa", "a", function(c) error("late") end)
both("gsub level0", string.gsub, "a", "a", function() error("y", 0) end)
both("gsub tbl", string.gsub, "a", "a", setmetatable({}, {__index = function() error("idx") end}))
both("gsub badret", string.gsub, "a", "a", function() return {} end)
both("gsub nested", string.gsub, "a", "a", function() return (string.gsub("b", "b", boom)) end)
both("gsub inner pcall", string.gsub, "a", "a", function() return tostring(pcall(boom)) end)
-- sort
both("sort ok", table.sort, {3, 1, 2}, function(a, b) return a < b end)
both("sort lua", table.sort, {3, 1, 2}, boom)
both("sort native", table.sort, {3, 1, 2}, string.rep)
both("sort lt", table.sort, {setmetatable({}, {__lt = boom}), setmetatable({}, {__lt = boom})})
-- tostring / format
local bad = setmetatable({}, {__tostring = boom})
both("tostring ok", tostring, setmetatable({}, {__tostring = function() return "ok" end}))
both("tostring lua", tostring, bad)
if not jit then both("format ok", string.format, "%s", setmetatable({}, {__tostring = function() return "ok" end}))
both("format lua", string.format, "%s", bad) end
-- load reader
local function ponly(label, f, ...) show(label .. " pcall", pcall(f, ...)) end
ponly("load ok", load, function() return nil end)
ponly("load lua", load, boom)
-- load with a native reader: stock names string.rep, oximond names ? (pre-existing)
ponly("load tbl", load, boomt)
if direct_xpcall then show("load xpcall ok", xpcall(load, h, function() return nil end)) end
both("next", next, {}, "nokey")
both("select", select, 0)
both("setmetatable", setmetatable, 1)
both("tostring nil", tostring)
both("string.rep", string.rep)
both("coroutine.wrap err", coroutine.wrap(boom))
both("string.find idx", string.find, "a", setmetatable({}, {__tostring = boom}))
ponly("tostring tbl", tostring, setmetatable({}, {__tostring = boomt}))
both("error", error, "direct")
ponly("error tbl", error, {}, 0)
-- table.foreach and foreachi exist in 5.1 and LuaJIT only
if table.foreach then
  both("foreach lua", table.foreach, {10}, boom)
  both("foreach native", table.foreach, {10}, string.rep)
  both("foreachi lua", table.foreachi, {10}, boom)
  both("foreachi native", table.foreachi, {10}, string.rep)
end
-- errors after earlier callbacks already succeeded
local calls = 0
both("gsub second", string.gsub, "aaa", "a", function(c)
  calls = calls + 1
  if calls % 2 == 0 then error("second") end
  return c
end)
both("gsub second badret", string.gsub, "aa", "a", function(c)
  calls = calls + 1
  if calls % 2 == 0 then return {} end
end)
both("sort second", table.sort, {3, 2, 1}, function(a, b)
  calls = calls + 1
  if calls % 3 == 0 then error("later") end
  return a < b
end)
-- gsub with a native __index replacement keeps the callback name
both("gsub native index", string.gsub, "a", "a", setmetatable({}, {__index = string.rep}))
-- dofile and require
both("dofile err", dofile, "pn_errfile.lua")
package.path = "./?.lua"
show("require err pcall", pcall(require, "pn_errmod"))
if direct_xpcall then show("require err xpcall", xpcall(require, h, "pn_errmod2")) end
-- pcall as a metamethod reaches ProtectedIndex and ConcatRange
local meta = setmetatable({}, {
  __concat = pcall,
  __index = pcall,
  __call = function() return string.gsub("a", "a", boom) end,
})
show("concat pcall", meta .. "s")
show("index pcall", meta.k)
show("nested pcall", pcall(pcall, string.gsub, "a", "a", boom))
show("nested sort", pcall(pcall, table.sort, {3, 2, 1}, boom))
-- gmatch iterator called directly
both("gmatch ok", string.gmatch("ab", "."))
