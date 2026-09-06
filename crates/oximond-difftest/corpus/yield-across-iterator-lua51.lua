-- oximond-difftest repro profile=lua51 from=lua51 yield across generic-for iterators
local y = coroutine.yield
local mt = {
  __index = function() return y("i") end,
  __newindex = function() y("n") end,
  __add = function() return y("a") end,
  __unm = function() return y("u") end,
  __concat = function() return y("c") end,
  __eq = function() return y("e") end,
  __lt = function() return y("l") end,
  __le = function() return y("le") end,
  __call = function() return y("call") end,
}
local function new() return setmetatable({}, mt) end
local cases = {
  function() return new().x end,
  function() new().x = 1 end,
  function() return new() + 1 end,
  function() return -new() end,
  function() return new() .. "a" end,
  function() return new() == new() end,
  function() return new() < new() end,
  function() return new() <= new() end,
  function() return pcall(function() return new().x end) end,
  function() return new()() end,
}
for i, f in ipairs(cases) do
  local co = coroutine.create(f)
  print(i, coroutine.resume(co))
end
print(pcall(coroutine.wrap(function() local t = new() return t.x end)))

-- generic-for iterators run through a C call in 5.1: always refused
local function iter(_, i) i = i + 1 if i < 3 then return y("it") end end
print(coroutine.resume(coroutine.create(function() for _ in iter, nil, 0 do end end)))
local callable = setmetatable({}, {__call = function(_, _, i) return y("ci") end})
print(coroutine.resume(coroutine.create(function() for _ in callable, nil, 0 do end end)))
local hot = coroutine.create(function()
  local n = 0
  for i = 1, 2000 do
    n = n + i
    if i == 1999 then for _ in iter, nil, 0 do end end
  end
  return n
end)
print(coroutine.resume(hot))
-- a builtin yield (or a __call to it) used as the iterator is refused too
print(coroutine.resume(coroutine.create(function()
  local n = 0
  for a in coroutine.yield, "s", "c" do n = n + 1 if n > 2 then break end end
  return n
end)))
local cy = setmetatable({}, {__call = coroutine.yield})
print(coroutine.resume(coroutine.create(function()
  local n = 0
  for a in cy, "s", "c" do n = n + 1 if n > 2 then break end end
  return n
end)))
