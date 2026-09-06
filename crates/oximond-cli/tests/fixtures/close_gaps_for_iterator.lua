local function run(name, f)
  local co = coroutine.create(f)
  local out = {}
  for i = 1, 4 do
    local r = {coroutine.resume(co, "r"..i)}
    local t = {}
    for j = 1, #r do t[#t+1] = tostring(r[j]) end
    out[#out+1] = coroutine.status(co) .. ":" .. table.concat(t, ",")
    if coroutine.status(co) == "dead" then break end
  end
  print(name, (table.concat(out, " ; "):gsub("(%a+): [%da-fx]+","%1: ADDR")))
end
local Y = coroutine.yield
run("iter", function() local n=0 for a in function() return Y("it") end do n=n+1 if n>2 then break end end return n end)
run("iter_tbl", function() for a in setmetatable({}, {__call=function() return Y("itc") end}) do return a end return "x" end)
run("iter_pcall", function() for a, b in pcall, function() return Y("ip") end do return tostring(a)..tostring(b) end return "none" end)
run("iter_pcall_error", function() for a, b in pcall, function() error("e") end do return tostring(a)..tostring(b) end return "none" end)
run("iter_next", function() local n=0 for k in next, {1,2} do n=n+1 end return n end)
run("iter_nested_call", function() local function f() return Y("nested") end for a in function() return f() end do return a end return "x" end)
run("iter_tail", function() for a in function() return Y("tail") end do return a end return "x" end)
run("call_yield", function() local t = setmetatable({}, {__call=function(self, x) return Y("call") end}) return t(1) end)
run("call_yield_builtin", function() local t = setmetatable({}, {__call=Y}) return t(1) end)
run("call_in_pcall", function() local t = setmetatable({}, {__call=function(self, x) return Y("call") end}) return pcall(t, 1) end)
print("-- for loops")
for a, b in pcall, function() error("e0") end do print("A", a, b) break end
for a, b in pcall, error, "x" do print("B", a, b) break end
for a, b in pcall, function() error("e") end do print("C", a, b) break end
for a, b in pcall, function() return 1,2 end do print("D", a, b) break end
for a, b in pcall, print do print("E", a, b) break end
local function it() return pcall(error, "f") end
for a, b in it do print("F", a, b) break end
local n = 0
for a in function() n = n + 1 if n < 3 then return false end end do print("F", a) end
for k, v in pairs({[false]=1}) do print("P", k, v) end
for i = 1, 3 do local fs = {} for k in (function() local n=0 return function() n=n+1 if n==1 then return false elseif n==2 then return 7 end end end)() do fs[#fs+1] = function() return k end end print(#fs, fs[1](), fs[2]()) end
for i = 1, 300 do for a in pairs({[false]=1, x=2}) do n = (n or 0) + 1 end end print(n)
