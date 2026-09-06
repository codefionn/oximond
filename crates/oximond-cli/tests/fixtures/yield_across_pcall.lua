local y = coroutine.yield
local function show(tag, ...) print(tag, ...) end

-- pcall: yield, resume, normal return
local co = coroutine.create(function(a)
  local r = {pcall(function(x) local v, w = y(x + 1, "y1"); return v, w, "done" end, a)}
  show("inner", (unpack or table.unpack)(r))
  return "end", #r
end)
show("p1", coroutine.resume(co, 1))
show("p2", coroutine.resume(co, "r1", "r2"))
show("p3", coroutine.status(co), coroutine.resume(co))

-- xpcall
co = coroutine.create(function()
  return xpcall(function() return (y("xp")) end, print)
end)
show("x1", coroutine.resume(co))
show("x2", coroutine.resume(co, 42))

-- error after resume is caught
co = coroutine.create(function()
  local ok, e = pcall(function() y(1); error("boom", 0) end)
  show("caught", ok, e)
  local ok2, e2 = pcall(function() y(2); error({code = 7}) end)
  show("caught2", ok2, type(e2), e2.code)
  return "fin"
end)
show("e1", coroutine.resume(co))
show("e2", coroutine.resume(co))
show("e3", coroutine.resume(co))

-- xpcall handler after resume, traceback first line
co = coroutine.create(function()
  return xpcall(function() y("h"); error("bad", 0) end, function(m)
    return debug.traceback(m):match("^[^\n]*") end)
end)
show("h1", coroutine.resume(co))
show("h2", coroutine.resume(co))

-- error with position after resume
co = coroutine.create(function()
  return pcall(function() y(); error("positioned") end)
end)
coroutine.resume(co)
show("pos", coroutine.resume(co))

-- nested pcall
co = coroutine.create(function()
  return pcall(function()
    local a, b = pcall(function()
      local c, d = pcall(function() return y("deep") end)
      return c, d, "mid"
    end)
    return a, b, "outer"
  end)
end)
show("n1", coroutine.resume(co))
show("n2", coroutine.resume(co, "back"))

-- nested: inner error caught by outer after resume
co = coroutine.create(function()
  return pcall(function()
    pcall(function() y("a") end)
    y("b")
    error("outer-err", 0)
  end)
end)
show("m1", coroutine.resume(co))
show("m2", coroutine.resume(co))
show("m3", coroutine.resume(co))

-- wrap error propagation
local w = coroutine.wrap(function()
  pcall(function() y("w1") end)
  pcall(function() y("w2") error("ignored") end)
  error("wrapped", 0)
end)
show("w1", w())
show("w2", w())
show("w3", pcall(w))
show("w4", pcall(w))

