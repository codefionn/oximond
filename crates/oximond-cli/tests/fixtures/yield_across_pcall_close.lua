local y = coroutine.yield
local log = {}
local function note(s) log[#log + 1] = s end
local function closer(name)
  return setmetatable({}, {__close = function() note(name) end})
end

-- close with pending to-be-closed variables inside and outside pcall
local co = coroutine.create(function()
  local outer <close> = closer("outer")
  local ok, e = pcall(function()
    local inner <close> = closer("inner")
    y("c")
    note("not reached")
  end)
  note("pcall returned")
end)
print("c1", coroutine.resume(co))
print("c2", coroutine.close(co))
print("c3", coroutine.status(co), table.concat(log, ","))

-- an erroring __close reports the error and pcall does not catch it
log = {}
co = coroutine.create(function()
  local ok = pcall(function()
    local bad <close> = setmetatable({}, {__close = function() error("close-err", 0) end})
    y("d")
  end)
  note("pcall returned")
end)
print("d1", coroutine.resume(co))
print("d2", coroutine.close(co))
print("d3", coroutine.status(co), table.concat(log, ","))

-- xpcall suspended, handler is not called by close
log = {}
co = coroutine.create(function()
  xpcall(function() y("x") end, function(m) note("handler") return m end)
  note("xpcall returned")
end)
print("x1", coroutine.resume(co))
print("x2", coroutine.close(co))
print("x3", coroutine.status(co), table.concat(log, ","))
