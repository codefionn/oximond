-- oximond-difftest repro profile=luajit21
-- __tostring through tostring and print: results, wrong result types, and
-- a native function as the metamethod.
local function T(f) return setmetatable({}, {__tostring = f}) end
local function show(ok, ...)
  local t = {n = select("#", ...), ...}
  for i = 1, t.n do
    local v = t[i]
    if type(v) == "string" then v = v:gsub("table: [^ ]*", "TBL") end
    if type(v) == "table" then v = "tbl" end
    t[i] = tostring(v)
  end
  print(ok, table.concat(t, " "))
end
show(pcall(tostring, T(function() return "OBJ" end)))
show(pcall(tostring, T(function() return 1 end)))
show(pcall(tostring, T(function() return 1.5 end)))
show(pcall(tostring, T(function() return {} end)))
show(pcall(tostring, T(function() return nil end)))
show(pcall(tostring, T(function() return true end)))
show(pcall(tostring, T(type)))
show(pcall(tostring, T(string.upper)))
show(pcall(tostring, T(math.floor)))
show(pcall(tostring, T("str")))
show(pcall(tostring, T(5)))
show(pcall(tostring, T(function() error("boom", 0) end)))
print(T(function() return "OBJ" end))
print(1, T(function() return "A" end), "x", T(function() return 7 end), nil, true)
print(select("#", print(T(function() return "R" end))))
show(pcall(print, T(function() return {} end)))
show(pcall(print, T(function() error("boom", 0) end)))
show(pcall(print, T(string.upper)))
local t = setmetatable({}, {__tostring = function(self) return "T:" .. #self end})
t[1] = 1
print(t, #t)
for i = 1, 300 do
  if i % 100 == 0 then print(i, T(function() return "L" .. i end)) end
end
