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
local function yl() return setmetatable({}, {__len=function() return Y("len") end, __index=function(t,k) return Y("idx") end, __newindex=function(t,k,v) Y("nidx") end}) end
run("ipairs", function() local n=0 for i,v in ipairs(yl()) do n=n+1 end return n end)
run("insert", function() table.insert(yl(), 1) return "ok" end)
run("insert2", function() table.insert(yl(), 1, 2) return "ok" end)
run("concat", function() return table.concat(yl()) end)
run("unpack", function() return table.unpack(yl(), 1, 2) end)
run("unpack_len", function() return table.unpack(yl()) end)
run("remove", function() return table.remove(yl()) end)
run("move", function() return table.move(yl(), 1, 2, 1) end)
run("sort", function() table.sort(yl()) return "ok" end)
run("gsub_tbl", function() return string.gsub("abc", "%w", setmetatable({}, {__index=function() return Y("g") end})) end)
run("strarith", function() return "x" + setmetatable({}, {__add=function() return Y("a") end}) end)
run("strarith2", function() return "10" + setmetatable({}, {__add=function() return Y("a") end}) end)
run("strconcat", function() return "x" .. setmetatable({}, {__concat=function() return Y("c") end}) end)
run("tostring", function() return tostring(setmetatable({}, {__tostring=function() return Y("t") end})) end)
run("pcallmeta", function() return pcall(function() return setmetatable({}, {__index=function() return Y("p") end}).x end) end)
run("next_pairs", function() local n=0 for k,v in pairs(setmetatable({}, {__pairs=function(t) return Y("pairs") end})) do n=n+1 end return n end)
run("select", function() return select("#", Y("s")) end)
run("rawequal", function() return rawequal(setmetatable({}, {__eq=function() return Y("e") end}), 1) end)
run("xpcall", function() return xpcall(Y, print, "xp") end)
run("string_rep_tostr", function() return string.format("%s", setmetatable({}, {__tostring=function() return Y("f") end})) end)
run("tbl_unpack_idx2", function() return table.unpack(setmetatable({}, {__index=function(t,k) return Y("u") end}), 1, 3) end)
print("-- sort proxy")
do
local store = {5,3,9,1,7}
local log = 0
local p = setmetatable({}, {__len=function() return #store end, __index=function(t,k) log=log+1 return store[k] end, __newindex=function(t,k,v) store[k]=v end})
table.sort(p)
print(table.concat(store, ","))
table.sort(p, function(a,b) return a>b end)
print(table.concat(store, ","))
print(pcall(table.sort, setmetatable({}, {__len=function() error("lenerr") end})))
print(pcall(table.sort, setmetatable({}, {__len=function() return "x" end})))
local co = coroutine.wrap(function() table.sort(setmetatable({}, {__len=function() return coroutine.yield(1) end})) end)
print(pcall(co))
end
