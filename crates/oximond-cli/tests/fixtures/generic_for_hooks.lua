-- Hook events, call names and debug.getinfo around generic-for loops. The
-- expected output is the pre-ForInCall loop's output, not stock Lua, except
-- `lines2`, which matches stock 5.1/5.4/LuaJIT (the old code dropped the first
-- line event after debug.sethook). Keep this header 4 lines: tests pin lines.
local out = {}
local function P(...)
  local t = {}
  for i = 1, select('#', ...) do t[#t+1] = tostring((select(i, ...))) end
  out[#out+1] = table.concat(t, " ")
end
-- hooks: line events
do
  local lines = {}
  local function body()
    local t = {1, 2}
    for i, v in ipairs(t) do
      local x = i + v
    end
    for k, v in pairs({a = 1}) do
      local y = v
    end
    for x in function() return nil end do
    end
    local z = 1
  end
  debug.sethook(function(ev, line) lines[#lines+1] = line end, "l")
  body()
  debug.sethook()
  P("lines", table.concat(lines, ","))
  local calls = {}
  debug.sethook(function(ev) local i = debug.getinfo(2, "nS"); calls[#calls+1] = ev .. ":" .. tostring(i.name) .. ":" .. i.what end, "cr")
  for i in function() return nil end do end
  for i, v in ipairs({1}) do end
  for k, v in pairs({1}) do end
  debug.sethook()
  P("calls", table.concat(calls, ","))
  local counts = 0
  debug.sethook(function() counts = counts + 1 end, "", 1)
  for i, v in ipairs({1, 2, 3}) do end
  debug.sethook()
  P("count>0", counts > 0)
  -- getinfo currentline in iterator, name of iterator
  local info = {}
  local function it(st, c)
    local i = debug.getinfo(1, "n")
    info[#info+1] = tostring(i.name) .. "/" .. tostring(i.namewhat)
    local j = debug.getinfo(2, "l")
    info[#info+1] = j.currentline
    if c < 1 then return c + 1 end
  end
  for i in it, nil, 0 do end
  P("getinfo", table.concat(info, ","))
end
-- getlocal in body
do
  for i, v in ipairs({"p"}) do
    local names = {}
    local idx = 1
    while true do
      local n = debug.getlocal(1, idx)
      if not n then break end
      names[#names+1] = n
      idx = idx + 1
    end
    P("locals", table.concat(names, ","):sub(-30))
  end
end

-- an error raised by the hook-visible iterator reports the same traceback
do
  local function bad() for x in 5 do end end
  local ok, tb = xpcall(bad, debug.traceback)
  P("traceback", (tb:gsub("\n%s*%[C%]: in function 'xpcall'.*", "")))
end
-- the count hook still fires inside loops, and line hooks see every body step
do
  local lines = {}
  debug.sethook(function(ev, line) lines[#lines+1] = line end, "l")
  local n = 0
  for i, v in ipairs({1, 2, 3}) do
    n = n + v
  end
  for k in pairs({}) do n = n + 1 end
  debug.sethook()
  P("lines2", table.concat(lines, ","), n)
end
-- a yield in a hooked loop's iterator
do
  local co = coroutine.wrap(function()
    for i in function(_, c) c = c + 1 if c <= 2 then coroutine.yield(c) return c end end, nil, 0 do end
    return "end"
  end)
  P("yield_hooked", pcall(function() return co(), co(), co() end))
end
-- a header that spans several lines puts the iterator call and the loop test
-- on the line where the expression list starts (stock Lua adds one more line
-- event on the first pass; the old code did not, and neither does this)
do
  local lines = {}
  local function run()
    for k in next,
      {10, 20} do
      local a = k
    end
  end
  debug.sethook(function(ev, line) lines[#lines+1] = line end, "l")
  run()
  debug.sethook()
  P("lines", table.concat(lines, ","))
end
print(table.concat(out, "\n"))
