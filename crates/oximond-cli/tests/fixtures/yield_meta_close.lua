local y = coroutine.yield
local mt = {__add = function() return y("add") end, __index = function() return y("idx") end}
local function new() return setmetatable({}, mt) end

-- close while suspended inside a metamethod
local co = coroutine.create(function() return new() + 1 end)
print(coroutine.resume(co))
print(coroutine.status(co), coroutine.close(co), coroutine.status(co))
print(coroutine.resume(co))

-- close runs pending to-be-closed variables of the frames below the metamethod
co = coroutine.create(function()
  local guard <close> = setmetatable({}, {__close = function() print("guard closed") end})
  local o = new()
  return o.field
end)
print(coroutine.resume(co))
print(coroutine.close(co))
print(coroutine.status(co))

-- an error after resume closes the variables as it unwinds to the pcall
co = coroutine.wrap(function()
  return pcall(function()
    local guard <close> = setmetatable({}, {__close = function(_, e) print("closing", e) end})
    local o = setmetatable({}, {__add = function() y("a") error("late", 0) end})
    return o + 1
  end)
end)
print(co())
print(co())

-- yield inside __close, once at normal exit and once through break
co = coroutine.wrap(function()
  do
    local c <close> = setmetatable({}, {__close = function() y("closing-normal") end})
  end
  for i = 1, 2 do
    local c <close> = setmetatable({}, {__close = function() y("closing-" .. i) end})
    if i == 1 then goto continue end
    break
    ::continue::
  end
  local function f()
    local c <close> = setmetatable({}, {__close = function() y("closing-return") end})
    return "ret"
  end
  return f()
end)
for _ = 1, 5 do print(co()) end

-- yield inside __close while an error unwinds, in pcall
co = coroutine.wrap(function()
  return pcall(function()
    local c <close> = setmetatable({}, {__close = function(_, e) y("unwind-close") end})
    error("oops", 0)
  end)
end)
print(co())
print(co())

-- length, bitwise and __pairs yield in 5.3 and later
local m2 = {
  __len = function() return y("len") end,
  __band = function() return y("band") end,
  __bor = function() return y("bor") end,
  __bxor = function() return y("bxor") end,
  __shl = function() return y("shl") end,
  __shr = function() return y("shr") end,
  __bnot = function() return y("bnot") end,
  __idiv = function() return y("idiv") end,
  __pairs = function(t) y("pairs") return next, {10, 20}, nil end,
}
local function clean(v) return (tostring(v):gsub("^[^:\n]+:%d+: ", "")) end
local function drive(name, f)
  local c = coroutine.create(f)
  local out = {}
  local ok, a = coroutine.resume(c)
  local n = 0
  while coroutine.status(c) == "suspended" and n < 20 do
    n = n + 1
    out[#out + 1] = tostring(ok) .. ":" .. clean(a)
    ok, a = coroutine.resume(c, "r" .. n)
  end
  out[#out + 1] = tostring(ok) .. ":" .. clean(a)
  print(name, table.concat(out, " | "))
end
local function o() return setmetatable({}, m2) end
drive("len", function() return #o() end)
drive("band", function() return o() & 1 end)
drive("bor-bxor", function() local x = o() return x | 1, x ~ 1 end)
drive("shifts", function() local x = o() return x << 1, x >> 1 end)
drive("bnot", function() return ~o() end)
drive("idiv", function() return o() // 1 end)
drive("pairs", function() local s = 0 for k, v in pairs(o()) do s = s + v end return s end)

-- refused: tostring, string.format, sort call __tostring from C
local ts = setmetatable({}, {__tostring = function() return y("ts") end})
drive("tostring", function() return tostring(ts) end)
drive("format", function() return string.format("%s", ts) end)
drive("sort", function() table.sort({3, 2, 1}, function(a, b) local r = y("s") return a < b end) end)

local c = coroutine.create(function() table.sort({3, 2, 1}, function(a, b) local r = y("s") return a < b end) end)
local ok, m = coroutine.resume(c)
print(ok, m:match(":%d+:") == nil, m)
