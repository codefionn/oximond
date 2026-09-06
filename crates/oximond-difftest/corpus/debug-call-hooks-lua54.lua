-- oximond-difftest repro profile=lua54 seed=11 generator=7
-- Extracted atomic probe from the seeded campaign.
local function show(v)
  local t = type(v)
  if t == "string" then
    return '"' .. string.gsub(v, '[%c"\\\128-\255]', function(c) return string.format('\\%03d', string.byte(c)) end) .. '"'
  end
  if t == "number" then if math.type(v) == "integer" then return "i" .. tostring(v) end if v ~= v then return "fnan" end return "f" .. string.format("%.17g", v) end
  if t == "nil" or t == "boolean" then return tostring(v) end
  return t
end
local function out(tag, ...)
  local n = select("#", ...)
  local parts = {}
  for k = 1, n do parts[k] = show((select(k, ...))) end
  print(tag .. "#" .. n .. " " .. table.concat(parts, " "))
end
do
local target = assert(load([=[local function child(value)
  return value + 1
end
local function parent(value)
  local result = child(value)
  return result + 1
end
local result = parent(-8)
return result]=], "@difftest-debug"))
local trace = {}
local function hook(event)
  local info = debug.getinfo(2, "S")
  if info and info.source == "@difftest-debug" then
    trace[#trace + 1] = event
  end
end
debug.sethook(hook, "cr")
local ok, result = pcall(target)
debug.sethook()
out("D3", "calls", false, ok, ok and result or type(result), table.concat(trace, ","), debug.gethook() == nil)
end
