-- oximond-difftest repro profile=lua54 seed=5 generator=7
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
local target = assert(load([=[local values = {3, 2, 12}
local total = 0
for key, value in pairs(values) do
  total = total + key + value
end
return total]=], "@difftest-debug"))
local trace = {}
local function hook(event, line)
  local info = debug.getinfo(2, "S")
  if info and info.source == "@difftest-debug" then
    trace[#trace + 1] = line
  end
end
debug.sethook(hook, "l")
local ok, result = pcall(target)
debug.sethook()
out("D3", "lines", 3, 4, 5, ok, ok and result or type(result), table.concat(trace, ","), debug.gethook() == nil)
end
