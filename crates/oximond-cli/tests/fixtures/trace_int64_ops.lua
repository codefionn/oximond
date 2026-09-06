-- 64-bit integer cdata operations inside hot loops, so the trace JIT
-- compiles them. Every expression is evaluated by a loop that runs long
-- enough to trace, over operands held in upvalues (so nothing folds), and
-- the result of the last iteration prints with its type. The script returns
-- its output as a string: the test runs it under the interpreter, under
-- tracing execution with synchronous compilation, and under a live LuaJIT
-- when one is installed, and compares the three.
local out = {}
local function emit(...)
  local parts = { ... }
  for i = 1, #parts do parts[i] = tostring(parts[i]) end
  out[#out + 1] = table.concat(parts, " ")
end

local vals = {
  0ull, 1ull, 0xffffffffffffffffull, 0x8000000000000000ull,
  0ll, -1ll, 0x7fffffffffffffffll, 5ll,
  0, 3, -2.7, 4294967296, 2^53, 0/0,
}
local names = {}
for i = 1, #vals do names[i] = "v" .. i end
local header = "local " .. table.concat(names, ", ") .. " = ...\n"
  .. "local band, bor, bxor, lshift, rshift, arshift, rol, ror, tonumber, bnot, tobit = "
  .. "bit.band, bit.bor, bit.bxor, bit.lshift, bit.rshift, bit.arshift, bit.rol, bit.ror, "
  .. "tonumber, bit.bnot, bit.tobit\n"

local function is_cdata(v) return type(v) == "cdata" end

-- Runs `exprs` (a list of Lua expressions over v1..v14) in one hot loop and
-- prints each result of the last iteration.
local function run(label, exprs)
  local body = {}
  local results = {}
  for i, e in ipairs(exprs) do
    body[#body + 1] = "r" .. i .. " = " .. e
    results[#results + 1] = "r" .. i
  end
  local src = header .. "return function(n)\n  local " .. table.concat(results, ", ") .. "\n"
    .. "  for i = 1, n do\n    " .. table.concat(body, "\n    ") .. "\n  end\n"
    .. "  return " .. table.concat(results, ", ") .. "\nend"
  local chunk = assert(loadstring(src, "=" .. label))
  local f = chunk(unpack(vals))
  local r = { f(200) }
  for i, e in ipairs(exprs) do
    emit(label, e, type(r[i]), tostring(r[i]))
  end
end

local function pairs_with_cdata()
  local list = {}
  for i = 1, #vals do
    for j = 1, #vals do
      if is_cdata(vals[i]) or is_cdata(vals[j]) then
        list[#list + 1] = { i, j }
      end
    end
  end
  return list
end

local all_pairs = pairs_with_cdata()
local CHUNK = 14

local function sweep(label, template)
  local exprs = {}
  local function flush()
    if #exprs > 0 then run(label, exprs) end
    exprs = {}
  end
  for _, p in ipairs(all_pairs) do
    exprs[#exprs + 1] = (template:gsub("A", "v" .. p[1]):gsub("B", "v" .. p[2]))
    if #exprs == CHUNK then flush() end
  end
  flush()
end

sweep("add", "A + B")
sweep("sub", "A - B")
sweep("mul", "A * B")
sweep("div", "A / B")
sweep("mod", "A % B")
sweep("eq", "A == B")
sweep("lt", "A < B")
sweep("le", "A <= B")
sweep("gt", "A > B")
sweep("band", "band(A, B)")
sweep("bor", "bor(A, B)")
sweep("bxor", "bxor(A, B)")
sweep("lshift", "lshift(A, B)")
sweep("rshift", "rshift(A, B)")
sweep("arshift", "arshift(A, B)")
sweep("rol", "rol(A, B)")
sweep("ror", "ror(A, B)")

-- Unary operations over every cdata operand.
do
  local exprs = {}
  for i = 1, #vals do
    if is_cdata(vals[i]) then
      exprs[#exprs + 1] = "-v" .. i
      exprs[#exprs + 1] = "bnot(v" .. i .. ")"
      exprs[#exprs + 1] = "tonumber(v" .. i .. ")"
      exprs[#exprs + 1] = "tobit(v" .. i .. ")"
    end
  end
  for i = 1, #exprs, CHUNK do
    local part = {}
    for k = i, math.min(i + CHUNK - 1, #exprs) do part[#part + 1] = exprs[k] end
    run("unary", part)
  end
end

-- Literal operands, as the kernels write them.
run("literals", {
  "v1 * 0x100000ull + 12345",
  "0x100000000ull * v10 + v9",
  "(0xffffffffull - v1) - v9",
  "band(v3, 0xffffffffull)",
  "rshift(v3, 32)",
  "lshift(v8, 32)",
  "arshift(v6, 63)",
  "bor(lshift(rshift(v3, 32), 32), band(v3, 0xffffffffull))",
  "v1 % 65536ull",
  "v8 / 0ll",
  "v1 % 0ull",
  "0x7fffffffffffffffll + 1",
  "0ull - 1",
  "v7 * 2ll",
})

-- A value that changes between number and cdata across runs of one loop:
-- each round enters the trace with another operand type or finds another
-- element type in the table, so entry checks and array load guards fail
-- and the interpreter takes over.
do
  local data = { 1.5, 2 }
  local source = [[
    local data, band, rshift = ...
    return function(n, acc, step)
      for i = 1, n do
        acc = acc + step * 3 + band(data[1], 0xffull)
        if acc > 0x7ffffffffffffff0ll then acc = acc - 0x7ffffffffffffff0ll end
        acc = acc + rshift(data[2], 1)
      end
      return acc
    end
  ]]
  local f = assert(loadstring(source))(data, bit.band, bit.rshift)
  local rounds = {
    { 100, 0ull, 7 },
    { 100, 1ull, 7ull },
    { 100, 0ll, 7ll },
    { 100, 0, 7 },
    { 100, 5ll, 2^40 },
    { 100, 0ull, -3 },
  }
  for round = 1, 6 do
    if round == 3 then data[1] = 300ull end
    if round == 4 then data[2] = 6ll end
    if round == 5 then data[1] = 17; data[2] = 12 end
    local args = rounds[round]
    local ok, r = pcall(f, args[1], args[2], args[3])
    emit("mixed", round, ok, type(r), tostring(r))
  end
end

-- Overflow inside the loop: the accumulator wraps, changes sign, and a
-- comparison of it with another type leaves the trace.
do
  local f = assert(loadstring([[
    return function(n)
      local acc, count = 0x7ffffffffffffff0ll, 0
      for i = 1, n do
        acc = acc + 3ll
        if acc < 0ll then count = count + 1 end
      end
      return acc, count
    end
  ]]))()
  local a, c = f(200)
  emit("wrap", type(a), tostring(a), c)
  local g = assert(loadstring([[
    return function(n)
      local acc = 1ull
      for i = 1, n do acc = acc * 3 end
      return acc
    end
  ]]))()
  emit("umul", tostring(g(100)))
end

return table.concat(out, "\n")
