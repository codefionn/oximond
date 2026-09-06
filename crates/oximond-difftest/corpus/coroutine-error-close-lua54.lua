-- oximond-difftest repro profile=lua54 seed=0 generator=7
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
  local log = {}
  local function resource(name)
    return setmetatable({}, {__close = function(self, err)
      log[#log + 1] = name .. ":" .. type(err)
    end})
  end
  local co = coroutine.create(function()
    local first <close> = resource("first")
    local second <close> = resource("second")
    coroutine.yield(16, nil)
    return 7
  end)
  out("C2", coroutine.resume(co))
  out("C2", #log, coroutine.status(co))
  out("C2", coroutine.close(co))
  out("C2", table.concat(log, ","), coroutine.status(co))
  out("C2", coroutine.close(co))
  local failed = coroutine.create(function()
    local item <close> = resource("failed")
    error("runtime close error", 0)
  end)
  out("C2", coroutine.resume(failed))
  out("C2", #log)
  out("C2", coroutine.close(failed))
  out("C2", table.concat(log, ","), coroutine.status(failed))
end
