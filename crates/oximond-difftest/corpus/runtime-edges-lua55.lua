-- oximond-difftest repro profile=lua55 from=generator 3 yield_boundaries and stdlib_edges families
-- One line per edge case; see difftest-edges-stock-vs-oximond-9236910.txt.
local function show(...) local t = {} for i = 1, select("#", ...) do t[i] = tostring((select(i, ...))) end return table.concat(t, " | ") end
local tagno = 0
local function yb(name, body)
  tagno = tagno + 1
  local co = coroutine.create(body)
  local r = {show(coroutine.resume(co, 1))}
  for i = 1, 4 do r[#r + 1] = coroutine.status(co) .. " " .. show(coroutine.resume(co, 10 + i)) end
  print("y" .. tagno .. "# " .. name .. ": " .. table.concat(r, " ; "))
end
yb("index", function(a) local o = setmetatable({}, {__index = function(t, k) return coroutine.yield(k) end}) return "index", o.field end)
yb("newindex", function(a) local log = {} local o = setmetatable({}, {__newindex = function(t, k, v) log[#log + 1] = coroutine.yield(v) end}) o.f = a o.g = a + 1 return "newindex", log[1], log[2] end)
yb("arith", function(a) local o = setmetatable({}, {__add = function(x, y) return coroutine.yield("add") end, __unm = function(x) return coroutine.yield("unm") end}) local r = o + a local s = -o return "arith", r, s end)
yb("compare", function(a) local mt = {} mt.__lt = function() return coroutine.yield("lt") end mt.__le = function() return coroutine.yield("le") end mt.__eq = function() return coroutine.yield("eq") end local o, p = setmetatable({}, mt), setmetatable({}, mt) return "compare", o < p, o <= p, o == p end)
yb("concat", function(a) local o = setmetatable({}, {__concat = function() return coroutine.yield("concat") end}) return "concat", o .. "s" end)
yb("call", function(a) local o = setmetatable({}, {__call = function(self, x) return coroutine.yield(x) end}) return "call", o(a) end)
yb("iter", function(a) local n = 0 for v in function(_, c) c = (c or 0) + 1 if c > 3 then return nil end coroutine.yield(c) return c end do n = n + v end return "iter", n end)
yb("pcallmeta", function(a) local o = setmetatable({}, {__index = function(t, k) return select(2, pcall(coroutine.yield, k)) end}) return "pcallmeta", o.key end)
yb("sort", function(a) local t = {3, 1, 2} table.sort(t, function(x, y) coroutine.yield("sort") return x < y end) return "sort", t[1] end)
yb("gsub", function(a) local s = string.gsub("ab", "%a", function(c) return coroutine.yield(c) end) return "gsub", s end)
yb("tostring", function(a) local o = setmetatable({}, {__tostring = function() return coroutine.yield("ts") end}) return "tostring", tostring(o) end)
yb("load", function(a) local f = load(function() return coroutine.yield("reader") end) return "load", type(f) end)
yb("len", function(a) local o = setmetatable({}, {__len = function() return coroutine.yield("len") end}) return "len", #o end)
local function sx(name, f) tagno = tagno + 1 print("s" .. tagno .. "# " .. name .. ": " .. show(pcall(f))) end
sx("gsub_bad_capture", function() return string.gsub("abc", "(a)", "%2") end)
sx("gsub_bad_escape", function() return string.gsub("abc", "a", "%") end)
sx("format_s_tostring", function() return string.format("%s|%5s", setmetatable({}, {__tostring = function() return "TS" end}), "x") end)
sx("dofile_missing", function() return dofile("/nonexistent/oximond-difftest.lua") end)
sx("gmatch_malformed", function() local n = 0 for w in string.gmatch("abc", "[a") do n = n + 1 end return n end)
sx("gmatch_caret", function() local n = 0 for w in string.gmatch("aaa", "^a") do n = n + 1 end return n end)
sx("find_malformed", function() return string.find("abc", "%") end)
sx("rep_negative", function() return string.rep("ab", -1) end)
sx("xpcall_load_reader", function() local parts = {"return ", "4", "2"} local i = 0 local ok, f = xpcall(load, function(m) return "H" end, function() i = i + 1 return parts[i] end) return ok, type(f), f and f() end)
if table.foreach then
  sx("table_foreach", function() return table.foreach({10, 20, 30}, function(k, v) if k == 2 then return v * 3 end end) end)
  sx("table_foreachi", function() return table.foreachi({10, 20}, function(i, v) return nil end) end)
  sx("table_getn", function() return table.getn({1, 2, 3}) end)
end
if math.type then
  local function proxy() return setmetatable({}, {__index = function(t, i) return i * 10 end, __len = function() return 3 end}) end
  sx("table_concat_meta", function() return table.concat(proxy(), ",") end)
  sx("table_unpack_meta", function() return table.unpack(proxy()) end)
  sx("table_move_meta", function() return table.concat(table.move(proxy(), 1, 3, 1, {}), ",") end)
  sx("table_insert_meta", function() local store = {} local p = setmetatable({}, {__index = store, __newindex = function(t, k, v) rawset(store, k, v) end, __len = function() return #store end}) table.insert(p, 5) table.insert(p, 1, 6) return store[1], store[2], #store end)
  sx("table_remove_meta", function() local store = {7, 8, 9} local p = setmetatable({}, {__index = store, __newindex = function(t, k, v) rawset(store, k, v) end, __len = function() return #store end}) return table.remove(p), table.remove(p, 1), store[1], store[2] end)
end
