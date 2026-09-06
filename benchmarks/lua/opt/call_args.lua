-- Fixed calls with 0, 1, 4 and 8 number arguments, then calls with two
-- string arguments. Each loop passes caller-dead temporaries in one call and
-- locals the caller reads after the call in another. call_args_hook runs the
-- same code with a debug hook set.
local scale = tonumber(arg and arg[1]) or 1

local function f0()
    return 1
end
local function f1(a)
    return a + 1
end
local function f4(a, b, c, d)
    return a + b - c + d
end
local function f8(a, b, c, d, e, f, g, h)
    return a + b + c + d + e + f + g + h
end
local function fs(s, t)
    return #s + #t
end

local total = 0
for i = 1, 400000 * scale do
    -- Caller-dead arguments: temporaries the caller never reads again.
    total = total + f0() + f1(i % 7) + f4(i % 3, i % 5, i % 7, i % 11)
        + f8(i % 2, i % 3, i % 4, i % 5, i % 6, i % 7, i % 8, i % 9)
    -- Caller-live arguments: locals the caller reads after the call.
    local a, b, c, d = i % 13, i % 17, i % 19, i % 23
    total = total + f1(a) + f4(a, b, c, d) + f8(a, b, c, d, d, c, b, a)
    total = (total + a + b + c + d) % 1000003
end

-- String arguments get their own loop: traces do not take string values
-- yet, and keeping them out of the first loop lets that loop compile.
local words = {"alpha", "beta", "gamma", "delta"}
for i = 1, 100000 * scale do
    local s, t = words[i % 4 + 1], words[(i + 1) % 4 + 1]
    total = total + fs(words[(i + 2) % 4 + 1], words[(i + 3) % 4 + 1]) + fs(s, t)
    total = (total + #s + #t) % 1000003
end
print(total)
