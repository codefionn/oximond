local compile = loadstring or load
local out = {}
local function P(...) 
  local t = {}
  for i = 1, select('#', ...) do t[#t+1] = tostring((select(i, ...))) end
  out[#out+1] = table.concat(t, " ")
end
local function msg(f, ...)
  local ok, e = pcall(f, ...)
  if ok then return "ok" end
  return (tostring(e):gsub("^[^:]*:%d+: ", ""))
end
-- ipairs
do
  local s = {}
  for i, v in ipairs({10, 20, 30, nil, 50}) do s[#s+1] = i .. "=" .. v end
  P("ipairs", table.concat(s, ","))
  s = {}
  for i in ipairs({10, 20, 30}) do s[#s+1] = i end
  P("ipairs1", table.concat(s, ","))
  s = {}
  for i, v, x in ipairs({10, 20}) do s[#s+1] = tostring(i) .. tostring(v) .. tostring(x) end
  P("ipairs3", table.concat(s, ","))
  local mt = setmetatable({}, {__index = function(t, k) if k <= 3 then return k * 2 end end})
  s = {}
  for i, v in ipairs(mt) do s[#s+1] = i .. ":" .. v end
  P("ipairs_index", table.concat(s, ","))
end
-- pairs / next
do
  local t = {1, 2, 3, a = "x"}
  local n, sum = 0, 0
  for k, v in pairs(t) do n = n + 1; sum = sum + (type(v) == "number" and v or 100) end
  P("pairs", n, sum)
  n = 0
  for k in next, t do n = n + 1 end
  P("next1", n)
  local keys = {}
  for k, v, extra in next, {5, 6} do keys[#keys+1] = tostring(k) .. tostring(v) .. tostring(extra) end
  P("next3", table.concat(keys, ","))
  P("next_bad", msg(function() for k in next, {}, "missing" do end end))
  P("next_nontable", msg(function() for k in next, 5 do end end))
  -- modify while traversing (assign nil allowed)
  local m = {a = 1, b = 2, c = 3}
  local cnt = 0
  for k in pairs(m) do m[k] = nil; cnt = cnt + 1 end
  P("clear", cnt, next(m) == nil)
end
-- closure iterator
do
  local function range(n)
    local i = 0
    return function() i = i + 1; if i <= n then return i, i * i end end
  end
  local s = {}
  for a, b in range(4) do s[#s+1] = a .. "/" .. b end
  P("closure", table.concat(s, ","))
  local function stateless(st, c) if c < st then return c + 1, "v" .. c end end
  s = {}
  for i, v in stateless, 3, 0 do s[#s+1] = i .. v end
  P("stateless", table.concat(s, ","))
  -- first result false continues
  local items = {false, false, nil}
  local idx = 0
  local function fi() idx = idx + 1; return items[idx] end
  local cnt = 0
  for v in fi do cnt = cnt + 1; if cnt > 5 then break end end
  P("false_first", cnt)
  -- callable table
  local callable = setmetatable({}, {__call = function(self, st, c) c = c + 1; if c <= 3 then return c end end})
  s = {}
  for i in callable, nil, 0 do s[#s+1] = i end
  P("callable", table.concat(s, ","))
  -- natives
  s = {}
  for w in string.gmatch("one two three", "%a+") do s[#s+1] = w end
  P("gmatch", table.concat(s, ","))
  s = {}
  for a, b in string.gmatch("k1=v1,k2=v2", "(%w+)=(%w+)") do s[#s+1] = a .. b end
  P("gmatch2", table.concat(s, ","))
end
-- errors
do
  P("nil_iter", msg(function() for x in nil do end end))
  P("num_iter", msg(function() for x in 5 do end end))
  P("nil_iter2", msg(function() local t = {} for x in t.missing do end end))
  P("string_iter", msg(function() for x in "abc" do end end))
  P("err_in_iter", msg(function() for x in function() error("boom") end do end end))
  P("err_in_iter_nonstr", select(2, pcall(function() for x in function() error({}) end do end end)) ~= nil)
  P("tb", (select(2, xpcall(function() for x in nil do end end, function(m) return debug.traceback(m, 1):gsub("\n.*", "") end))))
end
-- captured loop variables
do
  local fs = {}
  for i, v in ipairs({"a", "b", "c"}) do fs[#fs+1] = function() return i .. v end end
  P("captured", fs[1](), fs[2](), fs[3]())
  local gs = {}
  for k, v in pairs({x = 1}) do
    local inner = k
    gs[#gs+1] = function() inner = inner .. "!"; return inner .. v end
  end
  P("captured2", gs[1](), gs[1]())
  local hs = {}
  for i, v in ipairs({1, 2, 3}) do
    local j = i
    hs[#hs+1] = function() j = j + 10; return j end
  end
  P("captured3", hs[1](), hs[1](), hs[2](), hs[3]())
  -- break with captured
  local ks = {}
  for i in ipairs({1, 2, 3, 4}) do
    ks[#ks+1] = function() return i end
    if i == 2 then break end
  end
  P("capbreak", #ks, ks[1](), ks[2]())
end
-- assignment to loop variable
do
  -- Lua 5.5 makes loop variables constant, so the assignments are compiled
  -- on their own.
  local assign = compile([[
    local P = ...
    local s = {}
    for i, v in ipairs({1, 2, 3}) do
      i = i * 100
      v = "x"
      s[#s+1] = i .. v
    end
    P("assign", table.concat(s, ","))
    s = {}
    for k, v in pairs({10, 20, 30}) do k = nil; s[#s+1] = tostring(v) end
    P("assign_nilkey", table.concat(s, ","))
    local order = {}
    local function it(st, c) if c < 3 then return c + 1 end end
    for i in it, nil, 0 do i = i + 5; order[#order+1] = i end
    P("assign_closure", table.concat(order, ","))]])
  if assign then assign(P) else P("assign", "const loop variables") end
end
-- break / goto / nested / returns / varargs
do
  local s = {}
  for i, v in ipairs({1, 2, 3, 4, 5}) do
    if v == 4 then break end
    s[#s+1] = v
  end
  P("break", table.concat(s, ","))
  local function find(t, x) for i, v in ipairs(t) do if v == x then return i end end return nil end
  P("return", find({5, 6, 7}, 7), find({5}, 9))
  s = {}
  for i = 1, 3 do for j, v in ipairs({"a", "b"}) do s[#s+1] = i .. j .. v end end
  P("nested", table.concat(s, ","))
  s = {}
  local go = compile([[
    local s = {}
    for i, v in ipairs({1, 2, 3}) do
      for j, w in ipairs({10, 20}) do
        if w == 20 then goto continue end
        s[#s+1] = v + w
        ::continue::
      end
    end
    return table.concat(s, ",")]])
  P("goto", go and go() or "no goto")
  local function va(...) local s = {} for i, v in ipairs({...}) do s[#s+1] = v end return table.concat(s, ",") end
  P("vararg", va(1, 2, 3))
  -- many variables
  local function many() local i = 0 return function() i = i + 1 if i <= 2 then return i, i, i, i, i, i, i, i end end end
  s = {}
  for a, b, c, d, e, f, g, h, extra in many() do s[#s+1] = a + b + c + d + e + f + g + h .. tostring(extra) end
  P("many", table.concat(s, ","))
  -- loop variable nil in middle
  s = {}
  local function mid() local i = 0 return function() i = i + 1 if i <= 2 then return i, nil, i end end end
  for a, b, c in mid() do s[#s+1] = tostring(a) .. tostring(b) .. tostring(c) end
  P("mid", table.concat(s, ","))
  -- empty body, empty iteration
  local c = 0
  for _ in pairs({}) do c = c + 1 end
  for _ in ipairs({}) do c = c + 1 end
  P("empty", c)
  -- upvalue iterator state, and 4th value (5.4 closing value ignored in 5.1)
  local sum = 0
  for i, v in ipairs({1, 2, 3}) do sum = sum + v end
  P("sum", sum)
end
-- coroutines
do
  local co = coroutine.wrap(function()
    local function gen() for i = 1, 3 do coroutine.yield(i) end end
    local results = {}
    for i, v in ipairs({"a", "b"}) do
      local inner = coroutine.wrap(function() coroutine.yield(v .. 1); coroutine.yield(v .. 2) end)
      for x in inner do results[#results+1] = x end
    end
    coroutine.yield(table.concat(results, ","))
    return "done"
  end)
  P("co_wrap", co(), co())
  -- yield inside a Lua iterator
  local co2 = coroutine.create(function()
    local function it(st, c)
      c = c + 1
      if c <= 3 then
        local got = coroutine.yield("y" .. c)
        return c, got
      end
    end
    local s = {}
    for i, got in it, nil, 0 do s[#s+1] = i .. tostring(got) end
    return table.concat(s, ",")
  end)
  local r = {}
  local ok, v = coroutine.resume(co2)
  r[#r+1] = tostring(ok) .. tostring(v)
  for i = 1, 3 do ok, v = coroutine.resume(co2, "r" .. i); r[#r+1] = tostring(ok) .. tostring(v) end
  P("co_yield_iter", table.concat(r, "|"))
  -- yield as the iterator itself
  local co3 = coroutine.create(function() for a, b in coroutine.yield do P("never") end return "end" end)
  local ok1, v1 = coroutine.resume(co3)
  P("co_yield_builtin", ok1, v1)
  local ok2, v2 = coroutine.resume(co3, nil)
  P("co_yield_builtin2", ok2, v2)
  local co4 = coroutine.create(function() local s = {} for a, b in coroutine.yield do s[#s+1] = tostring(a) .. tostring(b) end return table.concat(s, ",") end)
  coroutine.resume(co4)
  coroutine.resume(co4, 1, 2)
  coroutine.resume(co4, 3)
  P("co_yield_builtin3", coroutine.resume(co4))
  -- pcall around yield in iterator
  local co5 = coroutine.wrap(function()
    local ok, e = pcall(function() for x in function() coroutine.yield("in"); return nil end do end end)
    return tostring(ok) .. tostring(e)
  end)
  P("co_pcall", pcall(function() return co5(), co5() end))
end
-- GC during iteration
do
  local t = {}
  for i = 1, 200 do t[i] = {i} end
  local s = 0
  for i, v in ipairs(t) do
    local junk = {}
    for j = 1, 20 do junk[j] = tostring(j) .. "x" end
    s = s + v[1]
    if i % 50 == 0 then collectgarbage() end
  end
  P("gc", s)
  s = 0
  for k, v in pairs(t) do s = s + v[1]; if k % 50 == 0 then collectgarbage() end end
  P("gc2", s)
end
-- string keys iteration order is implementation specific; use sorted
do
  local t = {z = 1, y = 2, x = 3}
  local ks = {}
  for k in pairs(t) do ks[#ks+1] = k end
  table.sort(ks)
  P("sorted", table.concat(ks, ","))
end
print(table.concat(out, "\n"))
