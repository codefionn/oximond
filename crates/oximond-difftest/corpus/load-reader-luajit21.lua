-- oximond-difftest repro profile=luajit21
-- load with a reader function: through xpcall, with a native reader, and
-- with a reader that returns the wrong type. Native readers name themselves
-- in an argument error as the profile's stock Lua does.
local parts = {"return ", "1+", "2"}
local i = 0
local function reader() i = i + 1; return parts[i] end
local function h(m) return "H:" .. tostring(m) end
local function show(ok, f, ...)
  if type(f) == "function" then print(ok, "fn", f()) else print(ok, f, ...) end
end
show(pcall(load, reader))
i = 0
show(pcall(load, reader, "=name"))
i = 0
show(xpcall(load, h, reader))
i = 0
show(xpcall(load, h, reader, "=name"))
show(pcall(load, function() return {} end))
show(pcall(load, rawlen or tostring))
show(pcall(load, tostring))
show(pcall(load, string.rep))
show(pcall(load, math.floor))
show(pcall(load, setmetatable))
show(pcall(load, ipairs))
show(pcall(load, string.upper))
show(xpcall(load, h))
