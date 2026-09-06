-- oximond-difftest repro profile=lua51 seed=2 generator=7
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
  local marker = {code = 15}
  local errors = {"semantic error", 15, false, marker}
  for i = 1, 4 do
    local payload = errors[i]
    local ok, err = pcall(function() error(payload, 0) end)
    out("M5", ok, type(err), err == payload, err == marker)
    local handled, result = xpcall(function() error(payload, 0) end,
      function(e) return {kind = type(e), same = e == payload, marker = e == marker} end)
    out("M5", handled, type(result), result.kind, result.same, result.marker)
  end
  local ok, err = pcall(function() error(nil, 0) end)
  out("M5", ok, type(err), err == nil)
  local handled, result = xpcall(function() error(nil, 0) end, function(e) return type(e) end)
  out("M5", handled, result)
  local attempts = 0
  local recovered, recovery = xpcall(function() error("first", 0) end, function(e)
    attempts = attempts + 1
    if attempts == 1 then error(marker, 0) end
    return e == marker and "recovered" or "other"
  end)
  out("M5", recovered, type(recovery), recovery == "recovered", attempts)
end
