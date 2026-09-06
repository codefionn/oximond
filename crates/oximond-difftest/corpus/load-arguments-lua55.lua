-- oximond-difftest repro profile=lua55
-- Lua 5.1's load takes a reader function and loadstring a string; later
-- profiles and LuaJIT take either from load. table.sort and xpcall call
-- load as a comparator and as a protected function.
local function show(ok, v, ...)
  if type(v) == "function" then v = "function" end
  print(ok, v, ...)
end
show(pcall(load, "return 1"))
show(pcall(load))
show(pcall(load, nil))
show(pcall(load, {}))
show(xpcall(load, function(m) return "H:" .. m end, "return 1"))
show(pcall(table.sort, {"a", "b", "c"}, load))
show(pcall(table.sort, {3, 1, 2}, load))
show(pcall(load(function() return nil end)))
local parts, i = {"return ", "4"}, 0
print(load(function() i = i + 1 return parts[i] end)())
if loadstring then
  show(pcall(loadstring, function() end))
  show(pcall(loadstring))
  show(pcall(loadstring, {}))
  show(xpcall(loadstring, function(m) return "H:" .. m end, {}))
  show(pcall(table.sort, {"a", "b", "c"}, loadstring))
  print(loadstring("return 7")())
end
