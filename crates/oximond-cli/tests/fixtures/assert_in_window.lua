local function show(...)
  local parts = {}
  for i = 1, select("#", ...) do parts[#parts + 1] = tostring(select(i, ...)) end
  print(table.concat(parts, " "))
end
local function checked(value, ...)
  local a, b, c = assert(value, ...)
  return a, b, c
end
for i = 1, 3 do show("kept", checked(i, "message", i * 2, "dropped")) end
show("one", (assert("x")))
show("padded", checked(true))
show("count", select("#", assert(1, nil, nil)))
show("false", pcall(checked, false, "said so"))
show("nil", pcall(checked, nil))
show("default", pcall(checked, false))
show("none", pcall(function() return assert() end))
local ok, object = pcall(checked, false, { code = 7 })
show("object", ok, type(object), object.code)
local sum = 0
for i = 1, 200 do
  local ok, value = pcall(checked, i % 50 ~= 0 and i, "stop " .. i)
  if ok then sum = sum + value else show("stopped", value) end
end
show("sum", sum)
local calls = 0
debug.sethook(function() calls = calls + 1 end, "c")
show("hooked", assert(5, 6))
debug.sethook()
show("hook saw calls", calls >= 2)
local saved = assert
assert = function(...) return "replaced", ... end
show("global", assert(false, 1))
assert = saved
show("restored", assert(1, 2))
