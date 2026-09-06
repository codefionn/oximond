-- Resume and yield semantics the direct switch path must keep. Every line
-- printed here is compared with what the stock interpreter prints.
local unpack = table.unpack or unpack
local function show(...)
  local n = select("#", ...)
  local parts = {}
  for i = 1, n do parts[i] = tostring((select(i, ...))) end
  print(table.concat(parts, " "))
end
local function try(f, ...)
  local ok, err = pcall(f, ...)
  if ok then return "ok" end
  return "error: " .. tostring(err)
end

-- Resuming a dead, running or normal coroutine, and a non-coroutine.
local co
co = coroutine.create(function()
  show("running", coroutine.status(co), select(2, coroutine.resume(co)))
  local inner = coroutine.create(function()
    show("normal", coroutine.status(co))
    return 1
  end)
  coroutine.resume(inner)
  coroutine.yield(1)
end)
show(coroutine.resume(co))
show(coroutine.status(co))
show(coroutine.resume(co))
show(coroutine.status(co))
show(coroutine.resume(co))
show(try(coroutine.resume, 1))
show(try(coroutine.resume))
show(try(coroutine.status, {}))

-- Yield from the main thread, and yield with no coroutine around.
show(try(coroutine.yield, 1))

-- Many values through resume and yield, with varargs and 0 and 250 values.
local gen = coroutine.create(function(...)
  local n = select("#", ...)
  local got = {...}
  while true do
    local more = {coroutine.yield(n, unpack(got, 1, n))}
    n = #more
    got = more
  end
end)
local function counts(...) return select("#", ...), (select(2, ...)) end
show(counts(coroutine.resume(gen)))
show(counts(coroutine.resume(gen, 1, 2, 3)))
local big = {}
for i = 1, 250 do big[i] = i end
local r = {coroutine.resume(gen, unpack(big))}
show(#r, r[1], r[2], r[3], r[252])
local sum = 0
for i = 3, #r do sum = sum + r[i] end
show(sum)
show(counts(coroutine.resume(gen, nil, nil)))
show(counts(coroutine.resume(gen)))

-- wrap: values, errors and a dead wrapper.
local w = coroutine.wrap(function(a, b)
  local c = coroutine.yield(a + b)
  local d, e = coroutine.yield(c * 2)
  return d, e, "done"
end)
show(w(1, 2))
show(w(10))
show(w("x", "y"))
show(try(w))
local ew = coroutine.wrap(function() error("boom") end)
show(try(ew))
local tw = coroutine.wrap(function() error({code = 7}) end)
local ok, err = pcall(tw)
show(ok, type(err), err.code)
local nw = coroutine.wrap(function() error() end)
show(pcall(nw))
local sw = coroutine.wrap(function() local x = nil; return x.y end)
show(pcall(sw))

-- Error inside a coroutine, and its traceback.
local bad = coroutine.create(function()
  local function deep() error("deep failure") end
  deep()
end)
local ok, msg = coroutine.resume(bad)
show(ok, msg)
show(coroutine.status(bad))
local tb = debug.traceback(bad)
show(type(tb), tb:find("stack traceback:", 1, true) ~= nil, tb:find("deep", 1, true) ~= nil)
show(coroutine.resume(bad))
local tb2 = debug.traceback(bad, "msg")
show(tb2:sub(1, 4), tb2:find("stack traceback:", 1, true) ~= nil)

-- A suspended coroutine's traceback shows where it stopped.
local parked = coroutine.create(function()
  local function inner() coroutine.yield("parked") end
  inner()
end)
show(coroutine.resume(parked))
local ptb = debug.traceback(parked)
show(ptb:find("inner", 1, true) ~= nil, ptb:find("yield", 1, true) ~= nil)

-- Yield across pcall, xpcall and a Lua callback (refused in Lua 5.1).
local yp = coroutine.create(function()
  local ok, v = pcall(function()
    local got = coroutine.yield("in pcall")
    error("after " .. tostring(got))
  end)
  local ok2, v2 = xpcall(function()
    return coroutine.yield("in xpcall")
  end, function(e) return "handled " .. tostring(e) end)
  return ok, v, ok2, v2
end)
show(coroutine.resume(yp))
show(coroutine.resume(yp, "A"))
show(coroutine.resume(yp, "B"))

-- A generator through a generic for.
local function range(n)
  return coroutine.wrap(function() for i = 1, n do coroutine.yield(i, i * i) end end)
end
local acc = {}
for i, sq in range(5) do acc[#acc + 1] = i .. ":" .. sq end
show(table.concat(acc, ","))

-- A yield nested in plain Lua calls, resumed from several places.
local function leaf(x) return coroutine.yield(x) end
local function mid(x) return leaf(x) + 1 end
local nest = coroutine.create(function(x) return mid(mid(x)) end)
show(coroutine.resume(nest, 1))
show(coroutine.resume(nest, 10))
show(coroutine.resume(nest, 20))
show(coroutine.status(nest))

-- Resume from inside another coroutine; yields go to the nearest resumer.
local outer = coroutine.create(function()
  local child = coroutine.create(function()
    coroutine.yield("child 1")
    coroutine.yield("child 2")
  end)
  for _ = 1, 2 do
    local _, v = coroutine.resume(child)
    coroutine.yield("outer saw " .. v)
  end
  return "outer done"
end)
for _ = 1, 3 do show(coroutine.resume(outer)) end

-- Deep recursion inside a coroutine overflows the stack the same way.
local function recurse(n) return 1 + recurse(n + 1) end
local ok, msg = coroutine.resume(coroutine.create(function() return recurse(1) end))
show(ok, tostring(msg):find("stack overflow", 1, true) ~= nil)
-- And the coroutine that overflowed is dead while others still run.
local still = coroutine.wrap(function() coroutine.yield("alive") end)
show(still())

-- Values kept only by a suspended coroutine's locals stay alive.
local weak = setmetatable({}, {__mode = "v"})
local keeper = coroutine.create(function()
  local mine = {tag = "kept"}
  weak[1] = mine
  coroutine.yield()
  return mine.tag
end)
coroutine.resume(keeper)
collectgarbage(); collectgarbage()
show(weak[1] ~= nil and weak[1].tag)
show(coroutine.resume(keeper))
collectgarbage(); collectgarbage()
show(weak[1])

-- An unreachable suspended coroutine and what its stack held are collected.
local gone = setmetatable({}, {__mode = "v"})
do
  local lost = coroutine.create(function()
    local held = {}
    gone[1] = held
    coroutine.yield()
  end)
  coroutine.resume(lost)
end
collectgarbage(); collectgarbage()
show(gone[1])

-- Finalizers of values held on a suspended stack run after it is dropped.
if _VERSION ~= "Lua 5.1" then
  local order = {}
  do
    local sleeper = coroutine.create(function()
      local a = setmetatable({}, {__gc = function() order[#order + 1] = "a" end})
      coroutine.yield()
      return a
    end)
    coroutine.resume(sleeper)
    collectgarbage(); collectgarbage()
    order[#order + 1] = "held"
  end
  collectgarbage(); collectgarbage()
  show(table.concat(order, ","))
end

-- The stack of a suspended coroutine seen through debug.getlocal.
local locals = coroutine.create(function()
  local first, second = "one", {2}
  coroutine.yield()
  return first, second
end)
coroutine.resume(locals)
collectgarbage()
show(debug.getlocal(locals, 1, 1))
show((debug.getlocal(locals, 1, 2)))

-- A hook set on one coroutine fires there and nowhere else.
local fired = 0
local hooked = coroutine.create(function()
  local a = 1
  coroutine.yield()
  local b = 2
  return a + b
end)
debug.sethook(hooked, function() fired = fired + 1 end, "l")
coroutine.resume(hooked)
local mid_count = fired
local plain = coroutine.create(function() local x = 1; coroutine.yield(); return x end)
coroutine.resume(plain)
show(mid_count > 0, fired == mid_count)
coroutine.resume(hooked)
show(fired > mid_count)

-- coroutine.running and isyieldable.
show(select("#", coroutine.running()) >= 1)
if coroutine.isyieldable then
  show(coroutine.isyieldable())
  show(coroutine.wrap(function() return coroutine.isyieldable() end)())
end

-- A long ping-pong keeps its totals.
local total = 0
local threads = {}
for slot = 1, 4 do
  threads[slot] = coroutine.create(function(first)
    local v = first
    while true do v = coroutine.yield((v * 3 + 7) % 997) end
  end)
  local _, v = coroutine.resume(threads[slot], slot)
  total = total + v
end
for i = 1, 4000 do
  local _, v = coroutine.resume(threads[(i - 1) % 4 + 1], i % 997)
  total = total + v
end
show(total)
