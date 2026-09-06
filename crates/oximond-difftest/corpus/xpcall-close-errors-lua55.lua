-- oximond-difftest repro profile=lua55
-- xpcall transforms errors before each remaining close callback.
local log = {}
local original, transformed, replacement = {}, {}, {}
local ok, result = xpcall(function()
  local outer <close> = setmetatable({}, {__close = function(_, err)
    assert(err == replacement)
    log[#log + 1] = "outer"
  end})
  local inner <close> = setmetatable({}, {__close = function(_, err)
    assert(err == transformed)
    log[#log + 1] = "inner"
    error(replacement, 0)
  end})
  error(original, 0)
end, function(err)
  collectgarbage("collect")
  if err == original then
    log[#log + 1] = "original"
    return transformed
  end
  assert(err == replacement)
  log[#log + 1] = "replacement"
  return replacement
end)
assert(not ok and result == replacement)
print(table.concat(log, ","))

-- A failing handler is retried while the target's resources remain open.
log = {}
local count = 0
ok, result = xpcall(function()
  local item <close> = setmetatable({}, {__close = function(_, err)
    assert(err == transformed)
    log[#log + 1] = "close"
  end})
  error(original, 0)
end, function(err)
  count = count + 1
  log[#log + 1] = "handler" .. count
  if count == 1 then error(replacement, 0) end
  assert(err == replacement)
  return transformed
end)
assert(not ok and result == transformed)
print(table.concat(log, ","))

-- Nested protected calls must keep separate pending close stacks.
log = {}
ok, result = xpcall(function()
  local item <close> = setmetatable({}, {__close = function(_, err)
    assert(err == transformed)
    log[#log + 1] = "outer close"
  end})
  error(original, 0)
end, function(err)
  assert(err == original)
  local nested_ok, nested_result = xpcall(function()
    local item <close> = setmetatable({}, {__close = function(_, err)
      assert(err == replacement)
      log[#log + 1] = "inner close"
    end})
    error(original, 0)
  end, function(err)
    assert(err == original)
    log[#log + 1] = "inner handler"
    return replacement
  end)
  assert(not nested_ok and nested_result == replacement)
  log[#log + 1] = "outer handler"
  return transformed
end)
assert(not ok and result == transformed)
print(table.concat(log, ","))

-- Handler retries precede closing the failed handler's own locals.
log = {}
count = 0
ok, result = xpcall(function()
  local item <close> = setmetatable({}, {__close = function(_, err)
    assert(err == transformed)
    log[#log + 1] = "target close"
  end})
  error(original, 0)
end, function(err)
  count = count + 1
  local attempt = count
  local item <close> = setmetatable({}, {__close = function(_, err)
    assert(err == (attempt == 1 and transformed or nil))
    log[#log + 1] = "handler close" .. attempt
  end})
  collectgarbage("collect")
  log[#log + 1] = "handler" .. attempt
  if attempt == 1 then error(replacement, 0) end
  return transformed
end)
assert(not ok and result == transformed)
print(table.concat(log, ","))

-- Message handlers reject yields, but error-time close callbacks can yield.
local thread = coroutine.create(function()
  local status, err = xpcall(function()
    local item <close> = setmetatable({}, {__close = function()
      coroutine.yield("close yield")
    end})
    error(original, 0)
  end, function(err)
    assert(err == original)
    local status = pcall(coroutine.yield, "handler yield")
    assert(not status)
    return transformed
  end)
  assert(not status and err == transformed)
  return "done"
end)
local resumed, value = coroutine.resume(thread)
assert(resumed and value == "close yield" and coroutine.status(thread) == "suspended")
resumed, value = coroutine.resume(thread)
assert(resumed and value == "done" and coroutine.status(thread) == "dead")
print(value)
