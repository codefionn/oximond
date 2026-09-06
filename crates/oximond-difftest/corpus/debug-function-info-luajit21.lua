-- oximond-difftest repro profile=luajit21 seed=1 generator=7
-- Extracted atomic probe from the seeded campaign.
local function show(v)
  local t = type(v)
  if t == "string" then
    return '"' .. string.gsub(v, '[%c"\\\128-\255]', function(c) return string.format('\\%03d', string.byte(c)) end) .. '"'
  end
  if t == "number" then if v ~= v then return "nnan" end return "n" .. string.format("%.17g", v) end
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
local maker = assert(loadstring([=[local function subject(value, ...)
  local identity = debug.getinfo(1, "f").func == subject
  return identity, value, select("#", ...)
end
return subject]=], "@difftest-debug"))
local subject = maker()
local info = debug.getinfo(subject, "Suf")
local identity, value, count = subject(0, false, "extra")
out("D3", "info", info.what, info.source == "@difftest-debug", info.linedefined, info.lastlinedefined, info.func == subject, info.nups, info.nparams, info.isvararg, identity, value, count)
end
