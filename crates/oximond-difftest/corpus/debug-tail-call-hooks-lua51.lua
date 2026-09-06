-- oximond-difftest repro profile=lua51 seed=6 generator=7
-- Extracted atomic probe from the seeded campaign.
local function show(v)
  local t = type(v)
  if t == "string" then
    return '"' .. string.gsub(v, '[%c"\\\128-\255]', function(c) return string.format('\\%03d', string.byte(c)) end) .. '"'
  end
  if t == "number" then if v ~= v then return "nnan" end if v == 0 then return "n0" end return "n" .. string.format("%.17g", v) end
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
local target = assert(loadstring([=[local function child(value)
  return value + 1
end
local function parent(value)
  return child(value)
end
local result = parent(1)
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
out("D3", "calls", true, ok, ok and result or type(result), table.concat(trace, ","), debug.gethook() == nil)
end
