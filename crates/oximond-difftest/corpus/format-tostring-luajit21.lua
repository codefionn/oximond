-- oximond-difftest repro profile=luajit21
-- string.format("%s") calls __tostring; LuaJIT also formats any other value.
local function T(f) return setmetatable({}, {__tostring = f}) end
local obj = T(function() return "OBJ" end)
print(string.format("%s", obj))
print(string.format("[%s][%5s][%-5s]", obj, obj, obj))
print(string.format("%s %s", obj, 1))
print(string.format("%.2s", obj))
local function show(ok, ...)
  local t = {n = select("#", ...), ...}
  for i = 1, t.n do t[i] = tostring(t[i]):gsub("table: [^ ]*", "TBL") end
  print(ok, table.concat(t, " "))
end
show(pcall(string.format, "%s", T(function() return 1 end)))
show(pcall(string.format, "%s", T(function() return {} end)))
show(pcall(string.format, "%s", T(function() error("boom", 0) end)))
show(pcall(string.format, "%s", T(type)))
show(pcall(string.format, "%s", T(string.upper)))
show(pcall(string.format, "%s", T(math.floor)))
show(pcall(string.format, "%s", T("notfunc")))
show(pcall(string.format, "%s", T(function() return "a", "b" end)))
show(pcall(string.format, "%d", obj))
show(pcall(string.format, "%s"))
print(string.format("%s %s %s %s", nil, true, 12, 1.5))
print(#string.format("%s", {}) > 7)
