-- Scalar natives that fail mid-call, in the call shapes the interpreter
-- runs in place: a fixed result count, a single-result call, and an open
-- result count feeding another call. Each failure keeps its message,
-- position and traceback, and the registers around it stay intact.
local floor, tonumber, sqrt, abs = math.floor, tonumber, math.sqrt, math.abs

-- Both .expected files are the stock output (lua5.1 and luajit). The comment
-- keeps the line numbers the messages carry.
--
local function report(ok, err)
    print(ok, type(err), (tostring(err):gsub("^.-native_results_errors%.lua:", "")))
end

local function fixed(x)
    local keep = 7
    local y = floor(x)
    return y + keep
end

local function single(x)
    return (abs(x))
end

local function open(x)
    return tostring(tonumber(floor(x)))
end

local function nested(x)
    local a = sqrt(x)
    return a, abs(x)
end

for round = 1, 3 do
    report(pcall(fixed, 3.5))
    report(pcall(fixed, {}))
    report(pcall(fixed, "x"))
    report(pcall(single, nil))
    report(pcall(single, -4))
    report(pcall(open, {}))
    report(pcall(open, 2.5))
    report(pcall(nested, "y"))
    report(pcall(nested, 16))
end

-- The error object is the one the native raised, and the traceback names
-- the failing call site.
local function raise()
    local value = floor(setmetatable({}, { __index = function() return 1 end }))
    return value
end
local ok, err = xpcall(raise, function(message)
    return message .. "\n" .. debug.traceback("tb", 1)
end)
err = err:gsub("\n[^\n]*in function <[^\n]*", "\n<fn>")
err = err:gsub("\n\t%[C%][^\n]*", "")
err = err:gsub("\n<fn>\n<fn>", "\n<fn>")
print(ok, (err:gsub("%S*native_results_errors%.lua", "script")))

-- A string result replaces a string in the destination register, and a
-- failing call leaves the old register value alone.
local s = "before"
for i = 1, 5 do
    s = tostring(i)
    s = string.rep(s, 2)
end
print(s)
local keep = "kept"
local ok2 = pcall(function() keep = floor({}) end)
print(ok2, keep)

-- Strings, numbers and booleans as results of one call site.
local results = {}
for i = 1, 6 do
    local f = ({ tostring, floor, type, tonumber, abs, string.len })[i]
    results[#results + 1] = tostring(f(i + 0.5))
end
print(table.concat(results, ","))
