-- oximond-difftest repro profile=lua54 seed=1 generator=7
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
  local calls = 0
  local mt = {__eq = function(a, b) calls = calls + 1 return a.n == b.n end,
    __lt = function(a, b) calls = calls + 1 return a.n < b.n end}
  local a, b, c = setmetatable({n = 5}, mt), setmetatable({n = 5}, mt), setmetatable({n = 5 + 1}, mt)
  local equal = a == b
  local different = a == c
  local identical = a == a
  out("M5", equal, different, identical, rawequal(a, b), rawequal(a, a))
  local less = a < c
  local greater = c > a
  local less_ok, less_equal = pcall(function() return a <= b end)
  local greater_ok, greater_equal = pcall(function() return c >= a end)
  if not less_ok then less_equal = type(less_equal) end
  if not greater_ok then greater_equal = type(greater_equal) end
  out("M5", less, greater, less_ok, less_equal, greater_ok, greater_equal, calls)
  mt.__le = function(left, right) return left.n <= right.n end
  out("M5", a <= b, c <= a, a >= b)
  mt.__metatable = "protected"
  out("M5", getmetatable(a), getmetatable(b))
  local ok, err = pcall(setmetatable, a, {})
  out("M5", ok, type(err), rawget(a, "n"))
  local other = setmetatable({n = 5}, {__eq = function() return false end})
  local left = a == other
  local right = other == a
  out("M5", left, right)
  mt.__len = function(self) return self.n end
  out("M5", #a, #b)
end
