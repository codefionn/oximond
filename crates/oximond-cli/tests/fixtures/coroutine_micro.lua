-- Hot resume and yield loops: the values of every shape, yields from nested
-- calls, errors, dead and running threads, wrap, yield across pcall, and
-- threads that change what the driver reads. Every line printed here is
-- compared with what the stock interpreter prints.
local unpack = table.unpack or unpack
local resume, yield, create, status = coroutine.resume, coroutine.yield, coroutine.create, coroutine.status
local function show(...)
  local n = select("#", ...)
  local parts = {}
  for i = 1, n do parts[i] = tostring((select(i, ...))) end
  print(table.concat(parts, " "))
end
local ROUNDS = 2500

-- Values both ways: the thread yields 0, 1, 3 and 6 values and takes 0, 1, 2
-- and 6 back.
local function shape(name, body, ...)
  local co = create(body)
  local total, counts = 0, 0
  local args = {...}
  for i = 1, ROUNDS do
    local r = {resume(co, unpack(args, 1, #args))}
    counts = counts + #r
    for k = 2, #r do total = total + (tonumber(r[k]) or 0) end
    args[1] = i
  end
  show(name, total, counts, status(co))
end
shape("yield0", function() local a; while true do a = yield() end end)
shape("yield1", function(a) while true do a = yield(a * 2 + 1) end end, 1)
shape("yield3", function(a, b)
  while true do a, b = yield(a, b, a + 1); b = (b or 0) + 1 end
end, 1, 2)
shape("yield6", function(a) while true do a = yield(a, a + 1, a + 2, a + 3, a + 4, a + 5) end end, 1)
shape("yield4", function(a) while true do a = yield(a, a, a, a) end end, 2)
shape("take0", function() while true do yield(7) end end)
shape("take6", function(a, ...) while true do a = select("#", ...) + yield(a) end end, 3, 1, 2, 3, 4, 5, 6)

-- The same loop at the top of the thread and in a function it calls.
do
  local function deeper(x) return yield(x + 1) end
  local co = create(function(a)
    while true do a = deeper(a * 3 % 1000) end
  end)
  local total = 0
  for i = 1, ROUNDS do
    local ok, v = resume(co, i)
    total = total + v
  end
  show("nested", total, status(co))
  local walk = create(function()
    local function tree(depth, n)
      if depth == 0 then yield(n) else tree(depth - 1, n * 2); tree(depth - 1, n * 2 + 1) end
    end
    tree(10, 1)
    return "done"
  end)
  local leaves, last = 0, 0
  while true do
    local ok, v = resume(walk)
    if v == "done" then break end
    leaves, last = leaves + 1, v
  end
  show("tree", leaves, last, status(walk), resume(walk))
end

-- Arithmetic on what the driver passes: numeric strings, nil, a table with a
-- metamethod.
do
  local co = create(function(a) while true do a = yield(a * 3 + 7) % 997 end end)
  local total = 0
  for i = 1, ROUNDS do
    local ok, v = resume(co, i % 5 == 0 and tostring(i) or i)
    total = total + v
  end
  show("strings", total)
  local meta = setmetatable({}, {__mul = function(self, n) self.calls = (self.calls or 0) + 1; return n end})
  local co2 = create(function(a) while true do a = yield(a * 3) end end)
  local total2 = 0
  for i = 1, ROUNDS do
    local ok, v = resume(co2, i % 100 == 0 and meta or i)
    total2 = total2 + (tonumber(v) or 0)
  end
  show("meta", total2, meta.calls)
  local co3 = create(function(a) while true do a = yield(a + 1) end end)
  local seen = {}
  for i = 1, ROUNDS do
    local passed = i
    if i == ROUNDS - 3 then passed = nil end
    local ok, v = resume(co3, passed)
    seen[#seen + 1] = ok and "t" or "f"
    if not ok then show("nilarith", i, (tostring(v):gsub("^.-:%d+: ", ""))); break end
  end
  show("nilarith", status(co3), resume(co3))
end

-- Errors in a thread that has been resumed many times.
do
  local count = 0
  local co = create(function(a)
    while true do
      count = count + 1
      if count == ROUNDS - 5 then error("boom " .. a) end
      a = yield(a + 1)
    end
  end)
  local last
  for i = 1, ROUNDS do
    local ok, v = resume(co, i)
    if not ok then show("error", i, (tostring(v):gsub("^.-:%d+: ", "")), status(co)); break end
    last = v
  end
  show("after error", last, resume(co))
  local tco = create(function() local n = 0; while true do n = n + 1; if n > 1500 then error({code = n}) end; yield(n) end end)
  local ok, v
  repeat ok, v = resume(tco) until not ok
  show("table error", type(v), v.code, status(tco))
  local fco = create(function(a) for i = 1, ROUNDS do a = yield(a + i) end return "finished", a end)
  local rounds
  for i = 1, ROUNDS + 5 do
    local r = {resume(fco, i)}
    if status(fco) == "dead" then rounds = i; show("finish", i, unpack(r)); break end
  end
  show("dead", resume(fco))
  show("dead again", resume(fco, 1, 2, 3))
end

-- A thread resuming itself, and one that is normal.
do
  local self_co
  self_co = create(function()
    for i = 1, 3 do
      local ok = resume(self_co)
      show("self", i, ok, status(self_co))
      yield(i)
    end
  end)
  for i = 1, 4 do show("outer", resume(self_co)) end
  local outer
  outer = create(function()
    local inner = create(function() show("inner sees", status(outer)); yield() end)
    resume(inner)
    show("outer sees", status(inner))
    yield()
  end)
  resume(outer)
  show(status(outer))
end

-- wrap: values, a dead wrapper and an error.
do
  local gen = coroutine.wrap(function(a)
    while true do a = yield(a % 13 * 2 + 1) end
  end)
  local total = gen(1)
  for i = 1, ROUNDS do total = total + gen(i) end
  show("wrap", total)
  local fin = coroutine.wrap(function() for i = 1, ROUNDS do yield(i) end return "end" end)
  local sum, got = 0, nil
  for i = 1, ROUNDS + 1 do got = fin(); if got == "end" then break end; sum = sum + got end
  show("wrap done", sum, got, pcall(fin))
  local bad = coroutine.wrap(function() local n = 0; while true do n = n + 1; if n == 2000 then error("wrapped " .. n) end; yield(n) end end)
  local ok, err
  for i = 1, ROUNDS do ok, err = pcall(bad); if not ok then break end end
  show("wrap error", ok, (tostring(err):gsub("^.-:%d+: ", "")))
end

-- Yield across pcall, in the profiles that allow it.
if jit ~= nil or tonumber(_VERSION:match("%d+%.%d+")) >= 5.2 then
  local co = create(function(a)
    while true do
      local ok, v = pcall(yield, a + 1)
      a = v
    end
  end)
  local total = 0
  for i = 1, ROUNDS do
    local ok, v = resume(co, i)
    total = total + v
  end
  show("pcall yield", total)
  local eco = create(function(a)
    while true do
      local ok, err = pcall(function() a = yield(a + 1); if a % 500 == 0 then error("inner " .. a) end end)
      if not ok then a = 0 end
    end
  end)
  local sum = 0
  for i = 1, ROUNDS do local ok, v = resume(eco, i); sum = sum + v end
  show("pcall error", sum)
else
  show("pcall yield", "skipped")
end

-- Threads that change what the driver reads: a global, a captured local and
-- a table the driver indexes.
do
  counter = 0
  local shared = 0
  local log = {}
  local co = create(function()
    while true do
      counter = counter + 1
      shared = shared + 2
      log[#log + 1] = counter
      yield()
    end
  end)
  local total = 0
  for i = 1, ROUNDS do
    resume(co)
    total = total + counter + shared + #log
  end
  show("writes", total, counter, shared, #log)
  -- A thread that writes only now and then.
  local flag = 0
  local gate = {value = 5}
  local rare = create(function(a)
    local i = 0
    while true do
      i = i + 1
      if i % 400 == 0 then flag = flag + 1; gate.value = gate.value + 1 end
      a = yield(a + 1)
    end
  end)
  local seen = 0
  for i = 1, ROUNDS do
    local ok, v = resume(rare, i)
    seen = seen + v + flag + gate.value
  end
  show("rare writes", seen, flag, gate.value)
end
