-- Threads whose bodies do more than a few instructions between two yields:
-- a branch, a global read, a call. The fast loop resumes such a thread in
-- place, so every scenario runs its thread several times, and each checks
-- what a nested resume would give: values, statuses, errors and their
-- positions, dead threads, closes, hooks and collections.
local create, resume, yield, wrap = coroutine.create, coroutine.resume, coroutine.yield, coroutine.wrap
local status, running = coroutine.status, coroutine.running

local function show(...)
  local parts = {}
  for i = 1, select("#", ...) do parts[#parts + 1] = tostring(select(i, ...)) end
  print(table.concat(parts, " "))
end

-- Values both ways, of every count, with more and fewer than the call takes.
do
  local co = create(function(a, b)
    local round = 0
    while true do
      round = round + 1
      if round % 3 == 0 then
        a, b = yield()
      elseif round % 3 == 1 then
        a, b = yield(a, b, "extra", round, nil, false)
      else
        a = yield(tostring(a) .. "!")
        b = nil
      end
    end
  end)
  for i = 1, 9 do
    show("values", i, resume(co, i, "s" .. i, i * 2))
    local ok, one = resume(co)
    show("fewer", i, ok, one)
  end
end

-- The statuses and the running thread at every point of a switch.
do
  local outer, inner
  inner = create(function()
    for i = 1, 4 do
      show("inner", status(inner), status(outer), running() == inner)
      if coroutine.isyieldable then show("yieldable", coroutine.isyieldable()) end
      yield(i)
    end
    return "inner done"
  end)
  outer = create(function()
    for i = 1, 6 do
      show("outer", status(outer), status(inner), resume(inner))
      yield(i)
    end
  end)
  for i = 1, 6 do
    show("main", status(outer), status(inner), resume(outer))
  end
  show("after", status(outer), status(inner), resume(inner))
end

-- A thread that ends, and resumes of the dead thread after it.
do
  local co = create(function(n)
    for i = 1, n do
      if i > 100 then error("unreachable") end
      yield(i)
    end
    return "done", n
  end)
  for i = 1, 7 do show("ending", i, status(co), resume(co, 4)) end
end

-- An error in the thread: `resume` returns it with its position, and a
-- wrapper raises it in the caller.
do
  local co = create(function()
    for i = 1, 3 do
      if i == 3 then error("failed at " .. i) end
      yield(i)
    end
  end)
  for i = 1, 5 do show("error", i, resume(co)) end
  local step = wrap(function()
    for i = 1, 3 do
      if i == 3 then local t = nil; return t.field end
      yield(i)
    end
  end)
  for i = 1, 5 do show("wrapped error", i, pcall(step)) end
  local object = wrap(function()
    for i = 1, 2 do
      if i == 2 then error({ code = 7 }) end
      yield(i)
    end
  end)
  show("object", object())
  local ok, err = pcall(object)
  show("object error", ok, type(err), type(err) == "table" and err.code)
  show("object dead", pcall(object))
end

-- A yield across pcall, from a nested call and from a metamethod-free
-- helper several frames deep.
do
  local function deep(n, value)
    if n == 0 then return yield(value) end
    return deep(n - 1, value) + 1
  end
  local co = create(function()
    local sum = 0
    while true do
      local ok, got = pcall(function() return deep(sum % 4, sum) end)
      if not ok then error(got, 0) end
      sum = sum + got
    end
  end)
  for i = 1, 12 do show("deep", i, resume(co, i)) end
  local guarded = create(function()
    for i = 1, 4 do
      show("pcall in thread", pcall(function()
        local got = yield(i)
        if got == "raise" then error("raised " .. i, 0) end
        return got
      end))
    end
    return "guarded done"
  end)
  show("guarded", resume(guarded))
  show("guarded", resume(guarded, "plain"))
  show("guarded", resume(guarded, "raise"))
  show("guarded", resume(guarded, "plain"))
  show("guarded", resume(guarded, "raise"))
  show("guarded", resume(guarded))
end

-- A thread cannot resume the thread that resumed it, nor itself. The
-- profiles word the refusal differently, so only its `false` is printed.
do
  local outer
  local inner = create(function()
    for i = 1, 3 do
      show("resume resumer", (resume(outer)))
      show("resume self", (resume(running())))
      yield()
    end
  end)
  outer = create(function()
    for i = 1, 3 do
      resume(inner)
      yield()
    end
  end)
  for i = 1, 3 do resume(outer) end
end

-- A generator with a branch in its loop, in a generic `for`, and one that
-- reads globals.
do
  local function evens(n)
    return wrap(function()
      for i = 1, n do
        if i % 2 == 0 then yield(i, "even") end
      end
    end)
  end
  local total, count = 0, 0
  for round = 1, 20 do
    for value, kind in evens(40) do
      total = total + value
      if kind == "even" then count = count + 1 end
    end
  end
  show("generator", total, count)
  scale = 3
  local scaled = wrap(function(value)
    while true do value = yield(value * scale + math.floor(value / 2)) end
  end)
  local sum = 0
  for i = 1, 200 do sum = sum + scaled(i) end
  show("globals", sum)
end

-- Collections while a thread runs in place: what the resumer and the
-- thread hold must survive, strings and tables included.
do
  local keep = setmetatable({}, { __mode = "v" })
  local co = create(function(first)
    local held = { first }
    while true do
      if #held % 2 == 0 then collectgarbage() end
      local name = "item" .. #held
      held[#held + 1] = yield(name, { #held })
    end
  end)
  local seen = {}
  for i = 1, 20 do
    local mine = { i }
    keep[i] = mine
    local ok, name, box = resume(co, mine)
    collectgarbage()
    seen[#seen + 1] = name .. ":" .. box[1] .. ":" .. mine[1]
  end
  show("collected", table.concat(seen, " "))
end

-- A hook set while a thread runs in place sees the calls that follow, and
-- the switch goes on working under it and after it.
do
  local calls = 0
  local co = create(function()
    for i = 1, 6 do
      if i == 2 then debug.sethook(function() calls = calls + 1 end, "c") end
      if i == 4 then debug.sethook() end
      yield(i)
    end
  end)
  for i = 1, 7 do show("hooked", i, resume(co)) end
  show("hook calls", calls > 0)
  debug.sethook()
end

-- Tracebacks from inside the thread name the thread's own frames.
do
  local co = create(function()
    for i = 1, 3 do
      local trace = debug.traceback("trace " .. i)
      yield((trace:gsub("\n.*", "")), select(2, trace:gsub("\n", "\n")) > 0)
    end
  end)
  for i = 1, 3 do show("traceback", resume(co)) end
end

-- A thread that yields from a loop it was resumed into many times: the
-- driver's own locals stay what they were across every switch.
do
  local co = create(function(value)
    local history = 0
    while true do
      if value > 500 then history = history - value else history = history + value end
      value = yield(history % 997)
    end
  end)
  local a, b, c, total = 1, "two", { 3 }, 0
  for i = 1, 3000 do
    local ok, value = resume(co, i % 997)
    total = total + value
    if a ~= 1 or b ~= "two" or c[1] ~= 3 then error("driver locals changed") end
  end
  show("driver", total)
end
