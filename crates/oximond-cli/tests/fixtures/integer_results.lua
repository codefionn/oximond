-- Prints the tostring and math.type of every result of library calls whose
-- integer/float subtype Lua 5.3+ specifies. The integer_results test diffs
-- this output between Oximond and the stock lua5.4/lua5.5 interpreters, so
-- every case must be deterministic and spelled the same by both versions.

local function describe(value)
  local subtype = math.type(value)
  if subtype then
    return tostring(value) .. ":" .. subtype
  end
  return tostring(value) .. ":" .. type(value)
end

local function show(label, ...)
  local results = table.pack(...)
  local parts = {}
  for index = 1, results.n do
    parts[#parts + 1] = describe(results[index])
  end
  print(label, table.concat(parts, " "))
end

local function check(label, call)
  local results = table.pack(pcall(call))
  if results[1] then
    show(label, table.unpack(results, 2, results.n))
  else
    print(label, "error")
  end
end

-- math
check("floor float", function() return math.floor(3.7) end)
check("floor negative", function() return math.floor(-3.7) end)
check("floor integer", function() return math.floor(3) end)
check("floor integral float", function() return math.floor(4.0) end)
check("floor negative zero", function() return math.floor(-0.0) end)
check("floor 2^62", function() return math.floor(2 ^ 62) end)
check("floor 2^63", function() return math.floor(2 ^ 63) end)
check("floor -2^63", function() return math.floor(-2 ^ 63) end)
check("floor huge", function() return math.floor(1e100) end)
check("floor inf", function() return math.floor(math.huge) end)
check("floor string", function() return math.floor("3.7") end)
check("ceil float", function() return math.ceil(3.2) end)
check("ceil negative", function() return math.ceil(-3.2) end)
check("ceil to zero", function() return math.ceil(-0.5) end)
check("ceil integer", function() return math.ceil(3) end)
check("ceil 2^63", function() return math.ceil(2 ^ 63) end)
check("ceil -inf", function() return math.ceil(-math.huge) end)
check("ceil string", function() return math.ceil("3.2") end)
check("max float int", function() return math.max(1.5, 2) end)
check("max int float", function() return math.max(2, 1.5) end)
check("max ints", function() return math.max(1, 2) end)
check("max tie int first", function() return math.max(2, 2.0) end)
check("max tie float first", function() return math.max(2.0, 2) end)
check("max single int", function() return math.max(7) end)
check("max single float", function() return math.max(7.0) end)
check("max many", function() return math.max(3, 1.5, 9, 2.5) end)
check("max huge int", function() return math.max(math.maxinteger, 2 ^ 53) end)
check("min float int", function() return math.min(1.5, 2) end)
check("min int float", function() return math.min(2, 1.5) end)
check("min ints", function() return math.min(1, 2) end)
check("min tie int first", function() return math.min(2, 2.0) end)
check("min tie float first", function() return math.min(2.0, 2) end)
check("min many", function() return math.min(3, 1.5, -9, 2.5) end)
check("min mininteger", function() return math.min(math.mininteger, -2 ^ 63) end)
check("random range", function() return math.random(7, 7) end)
check("random upper", function() return math.random(1) end)
check("random negative range", function() return math.random(-3, -3) end)
check("random float bounds", function() return math.random(4.0, 4.0) end)
check("random type m n", function() return math.type(math.random(1, 100)) end)
check("random type m", function() return math.type(math.random(100)) end)
check("random type none", function() return math.type(math.random()) end)
check("random zero", function() return math.type(math.random(0)) end)
check("random wide range", function()
  local value = math.random(1, 2 ^ 40)
  return math.type(value), value >= 1 and value <= 2 ^ 40
end)
check("random full range", function()
  return math.type(math.random(math.mininteger, math.maxinteger))
end)
check("random 2^53 bounds", function() return math.random(2 ^ 53, 2 ^ 53) end)
check("random string bound", function() return math.random("3", "3") end)
-- Drop the chunk position: stock Lua abbreviates long script paths.
local function argument_error(call)
  local message = select(2, pcall(call))
  return (message:gsub("^.*:%d+: ", ""))
end
check("random fraction", function()
  return argument_error(function() return math.random(1.5) end)
end)
check("random empty", function()
  return argument_error(function() return math.random(2, 1) end)
end)
check("abs", function() return math.abs(-3), math.abs(-3.5), math.abs(-0.0) end)
check("fmod", function() return math.fmod(7, 3), math.fmod(-7, 3), math.fmod(7.0, 3) end)
check("modf", function() return math.modf(3.7) end)
check("modf integer", function() return math.modf(3) end)
check("tointeger", function() return math.tointeger(3.0), math.tointeger(3.5) end)
check("limits", function() return math.maxinteger, math.mininteger end)
check("math.type floor", function() return math.type(math.floor(3.7)) end)

-- string
check("len", function() return string.len("abc"), string.len("") end)
check("len method", function() return ("hello"):len() end)
check("length operator", function() return #"abcd" end)
check("byte", function() return string.byte("a") end)
check("byte range", function() return string.byte("abc", 1, -1) end)
check("byte negative", function() return string.byte("abc", -1) end)
check("find", function() return string.find("abc", "b") end)
check("find plain", function() return string.find("a.c", ".", 1, true) end)
check("find init", function() return string.find("abcabc", "b", 3) end)
check("find empty", function() return string.find("abc", "") end)
check("find captures", function() return string.find("abc", "(b)()") end)
check("match position", function() return string.match("abc", "()b()") end)
check("gmatch position", function()
  local positions = {}
  for position in string.gmatch("a1b2", "()%a") do
    positions[#positions + 1] = position
  end
  return table.unpack(positions)
end)
check("gsub count", function() return string.gsub("abc", "b", "x") end)
check("gsub count limit", function() return string.gsub("aaa", "a", "b", 2) end)
check("gsub position capture", function() return string.gsub("abc", "()", "%1") end)
check("gsub position table", function()
  return string.gsub("ab", "()", { [1] = "one", [2] = "two" })
end)
check("gsub position function", function()
  return string.gsub("ab", "()%a", function(position) return math.type(position) end)
end)
check("unpack", function() return string.unpack("i4", string.pack("i4", 7)) end)
check("unpack float", function() return string.unpack("d", string.pack("d", 2.0)) end)
check("packsize", function() return string.packsize("i4i8") end)
check("format d", function() return string.format("%d", 3.0) end)

-- utf8
check("utf8.len", function() return utf8.len("h\u{e9}llo") end)
check("utf8.len invalid", function() return utf8.len("a\255b") end)
check("utf8.codepoint", function() return utf8.codepoint("h\u{e9}", 1, -1) end)
-- Lua 5.5 also returns the character's final byte; compare the start only.
check("utf8.offset", function() return (utf8.offset("h\u{e9}llo", 3)) end)
check("utf8.offset multibyte", function() return utf8.offset("h\u{e9}llo", 2) end)
check("utf8.offset end", function() return utf8.offset("h\u{e9}", 3) end)
check("utf8.offset negative", function() return utf8.offset("h\u{e9}", -1) end)
check("utf8.offset current", function() return utf8.offset("h\u{e9}", 0, 3) end)
check("utf8.offset past", function() return utf8.offset("h\u{e9}", 4) end)
check("utf8.codes", function()
  local results = {}
  for position, code in utf8.codes("h\u{e9}") do
    results[#results + 1] = position
    results[#results + 1] = code
  end
  return table.unpack(results)
end)

-- base and table
check("select count", function() return select("#", 1, 2) end)
check("select count empty", function() return select("#") end)
check("select count nils", function() return select("#", nil, nil, nil) end)
check("select negative", function() return select(-1, 1, 2) end)
check("select varargs count", function()
  local function count(...) return select("#", ...) end
  return count(1, 2, 3)
end)
check("table.pack n", function() return table.pack(1, 2).n end)
check("table length", function() return #{ 10, 20, 30 } end)
check("rawlen", function() return rawlen({ 1, 2 }), rawlen("abc") end)
check("ipairs", function()
  local results = {}
  for index in ipairs({ "a", "b" }) do
    results[#results + 1] = index
  end
  return table.unpack(results)
end)
check("pairs array keys", function()
  local results = {}
  for key in pairs({ "a", "b" }) do
    results[#results + 1] = key
  end
  table.sort(results)
  return table.unpack(results)
end)
check("next", function() return next({ "a" }) end)
check("table.remove", function() return table.remove({ 5, 6 }) end)
check("tonumber", function()
  return tonumber("10"), tonumber("0x10"), tonumber("10", 16), tonumber("1e2"), tonumber(" 7 ")
end)
check("numeric for", function()
  local results = {}
  for index = 1, 3 do
    results[#results + 1] = index
  end
  return table.unpack(results)
end)
check("collectgarbage collect", function() return collectgarbage("collect") end)
check("collectgarbage count type", function() return math.type(collectgarbage("count")) end)
check("collectgarbage step type", function() return type(collectgarbage("step")) end)
check("collectgarbage step zero type", function() return type(collectgarbage("step", 0)) end)

-- string arithmetic coercion (strings convert the way the lexer reads them)
check("string constant arithmetic", function()
  return "10" + 1, "3.0" + 1, -"2", "10" // 3, "7" % 2, "2" ^ 2, "7" / 2
end)
check("string register arithmetic", function()
  local ten, three, half = "10", " 3 ", "0.5"
  return ten + three, ten - three, ten * three, ten // three, ten % three, ten + half, -three
end)
check("string mixed operands", function()
  local value = "4"
  return 1 + value, value * 2.0, 10 // value, value % -3, -value
end)
check("string hex and exponent", function()
  return "0x10" + 0, "0xffffffffffffffff" + 0, "1e2" + 0, "-0x8000000000000000" + 0
end)
check("string integer limits", function()
  return "9223372036854775807" + 0, "-9223372036854775808" + 0, math.type("9223372036854775808" + 0)
end)
check("string wraparound", function()
  local max = "9223372036854775807"
  return max + 1, -"-9223372036854775808"
end)
check("string floor division by zero", function() return "1" // "0" end)
check("string modulo by zero", function() local zero = "0" return 1 % zero end)
check("string float division by zero", function() return "1" // "0.0", "1" / "0" end)
check("string bitwise", function() return "3" | 0 end)
check("string bitwise not", function() local value = "5" return ~value end)
check("invalid string arithmetic", function() return "1x" + 1 end)
check("signed sign string", function() return "-+5" + 0 end)

-- os
check("os.time table", function()
  return os.time({ year = 2020, month = 1, day = 1, hour = 0 })
end)
check("os.time normalizes", function()
  return os.time({ year = 2020, month = 13, day = 1, hour = 12 })
end)
check("os.time now type", function() return math.type(os.time()) end)
check("os.date table", function()
  local date = os.date("!*t", 86400 * 40 + 3661)
  return date.year, date.month, date.day, date.hour, date.min, date.sec, date.wday, date.yday
end)
check("os.difftime", function() return os.difftime(10, 5) end)
check("os.clock type", function() return math.type(os.clock()) end)

-- io
check("file seek", function()
  local file = io.tmpfile()
  file:write("hello")
  local ending = file:seek("cur")
  local start = file:seek("set", 1)
  file:close()
  return ending, start
end)
check("file read formats", function()
  local file = io.tmpfile()
  file:write("42 3.5 0x10 -7 1e2 abc\nline\nlast\n")
  file:seek("set")
  local results = table.pack(file:read("n", "n", "n", "n", "n"))
  local failed = file:read("n")
  local count = file:read(2)
  local rest = file:read("l")
  local line = file:read("L")
  local starred = file:read("*l")
  local ending = file:read("a")
  local eof = file:read("n")
  file:close()
  return results[1], results[2], results[3], results[4], results[5], failed, count, rest,
    line, starred, ending, eof
end)

-- lines iterators pass their formats to read on each call
local function escaped(text)
  return (string.gsub(text, "\n", "\\n"))
end
check("file:lines formats", function()
  local file = io.tmpfile()
  file:write("42 3.5\n0x10 x\n7")
  file:seek("set")
  local results = {}
  for first, second in file:lines("n", "n") do
    results[#results + 1] = first
    results[#results + 1] = second == nil and "nil" or second
  end
  local rest = file:read("a")
  local open = io.type(file)
  file:close()
  return #results, results[1], results[2], results[3], results[4], escaped(rest), open
end)
check("file:lines counts", function()
  local file = io.tmpfile()
  file:write("ab\ncd\n")
  file:seek("set")
  local parts = {}
  for first, rest in file:lines(1, "L") do
    parts[#parts + 1] = first .. "+" .. escaped(rest)
  end
  file:seek("set")
  local empty = file:lines(0)()
  file:close()
  return table.concat(parts, ","), empty == "" and "empty" or "missing"
end)
check("file:lines invalid format", function()
  local file = io.tmpfile()
  file:write("line\n")
  file:seek("set")
  for _ in file:lines("x") do end
end)
check("file:lines nil format", function()
  local file = io.tmpfile()
  file:write("line\n")
  file:seek("set")
  for _ in file:lines("l", nil) do end
end)
check("io.lines name formats", function()
  local name = os.tmpname()
  local file = io.open(name, "w")
  file:write("1 2\n3 4\nend\n")
  file:close()
  local count, iterator, state, control, closing = select("#", io.lines(name, "L")),
    io.lines(name, "n", "n")
  local before = io.type(closing)
  local sums = {}
  for first, second in iterator, state, control, closing do
    sums[#sums + 1] = first + second
  end
  local after = io.type(closing)
  local lines = {}
  for line in io.lines(name, "L") do lines[#lines + 1] = escaped(line) end
  local chunks = {}
  for chunk in io.lines(name, 5) do chunks[#chunks + 1] = escaped(chunk) end
  os.remove(name)
  return count, before, table.concat(sums, ","), after, table.concat(lines, ","),
    table.concat(chunks, ",")
end)
check("io.lines default input formats", function()
  local name = os.tmpname()
  local file = io.open(name, "w")
  file:write("5 6\n")
  file:close()
  local previous = io.input()
  io.input(name)
  local count = select("#", io.lines(nil, "n"))
  local values = {}
  for value in io.lines(nil, "n") do values[#values + 1] = value end
  local input = io.input()
  local open = io.type(input)
  io.input(previous)
  input:close()
  os.remove(name)
  return count, table.concat(values, ","), open
end)
check("io.lines format limit", function()
  local name = os.tmpname()
  local file = io.open(name, "w")
  file:write("a\nb\n")
  file:close()
  local formats = {}
  for index = 1, 250 do formats[index] = "l" end
  local iterator = io.lines(name, table.unpack(formats))
  local count = select("#", iterator())
  formats[251] = "l"
  local ok = pcall(io.lines, name, table.unpack(formats))
  local handle = io.open(name)
  local method_ok = pcall(handle.lines, handle, table.unpack(formats))
  handle:close()
  os.remove(name)
  return count, ok, method_ok
end)

-- debug
check("debug.getinfo", function()
  local upvalue = 1
  local function probe() return upvalue end
  local info = debug.getinfo(probe, "Su")
  return math.type(info.linedefined), math.type(info.lastlinedefined), info.nups
end)
check("debug line hook", function()
  local lines = {}
  debug.sethook(function(_, line) lines[#lines + 1] = line end, "l")
  local _ = 1
  debug.sethook()
  return math.type(lines[1])
end)
check("debug.getinfo C", function()
  local info = debug.getinfo(print, "S")
  return info.linedefined, info.lastlinedefined
end)
check("debug.getinfo level", function()
  local info = debug.getinfo(1, "l")
  return math.type(info.currentline), info.currentline > 0
end)
check("debug.getlocal level", function()
  local first = "value"
  return debug.getlocal(1, 1)
end)
check("debug.getinfo parameters", function()
  local function lua_function(a, b, ...) return a, b end
  local object = {}
  function object:method(a) return a end
  local lua = debug.getinfo(lua_function, "u")
  local method = debug.getinfo(object.method, "u")
  local native = debug.getinfo(print, "u")
  return lua.nparams, lua.isvararg, method.nparams, method.isvararg, native.nparams,
    native.isvararg
end)
check("debug count hook", function()
  local calls, arity, line = 0, nil, "unset"
  debug.sethook(function(...)
    calls = calls + 1
    arity = select("#", ...)
    line = select(2, ...)
  end, "", 1000)
  local _, mask, count = debug.gethook()
  local sum = 0
  for index = 1, 10000 do sum = sum + index end
  debug.sethook()
  return mask, count, sum, calls > 0, arity, line
end)
check("debug count hook builtin", function()
  debug.sethook(collectgarbage, "", 1)
  local sum = 0
  for index = 1, 100 do sum = sum + index end
  debug.sethook()
  return sum
end)
