local scale = tonumber(arg and arg[1]) or 1

-- Operator metamethods on a small vector class: arithmetic with the object on
-- either side of a number, comparison, equality, concatenation and tostring.
-- The arithmetic handlers update their left operand and give it back, as
-- accumulating code does, and the other handlers answer with numbers,
-- booleans and strings.
local Vec = {}
Vec.__index = Vec

local function vec(x, y)
    return setmetatable({x = x, y = y}, Vec)
end

Vec.__add = function(a, b)
    if type(b) == "number" then
        a.x, a.y = a.x + b, a.y + b
    else
        a.x, a.y = a.x + b.x, a.y + b.y
    end
    return a
end

Vec.__sub = function(a, b)
    a.x, a.y = a.x - b.x, a.y - b.y
    return a
end

-- Dot product of two vectors, or the sum of a vector's parts times a number.
Vec.__mul = function(a, b)
    if type(a) == "number" then
        return (b.x + b.y) * a
    elseif type(b) == "number" then
        return (a.x + a.y) * b
    end
    return a.x * b.x + a.y * b.y
end

Vec.__unm = function(a)
    return -a.x - a.y
end

Vec.__eq = function(a, b)
    return a.x == b.x and a.y == b.y
end

Vec.__lt = function(a, b)
    return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y
end

Vec.__le = function(a, b)
    return a.x * a.x + a.y * a.y <= b.x * b.x + b.y * b.y
end

Vec.__concat = function(a, b)
    if type(a) == "table" then
        return a.x .. "," .. a.y .. b
    end
    return a .. b.x .. "," .. b.y
end

Vec.__tostring = function(a)
    return "(" .. a.x .. "," .. a.y .. ")"
end

local pool, twins = {}, {}
for i = 1, 64 do
    pool[i] = vec(i % 7, i % 5)
    -- Equal to its pool entry without being it.
    twins[i] = vec(i % 7, i % 5)
end
-- Longer than any pool entry.
local big = vec(9, 9)

local acc = vec(0, 0)
local total, less, equal, length = 0, 0, 0, 0
for i = 1, 90000 * scale do
    local a = pool[i % 64 + 1]
    local b = pool[(i * 7) % 64 + 1]
    local r = acc + a
    r = r - b
    r = r + 1
    total = total + r.x - r.y + a * b + a * 2 + 3 * b - -a
    if a < big then less = less + 1 end
    if big <= a then less = less + 2 end
    if a <= big then less = less + 4 end
    if a == twins[i % 64 + 1] then equal = equal + 1 end
    if a == big then equal = equal + 5 end
    length = length + #("v=" .. a) + #(b .. ";") + #tostring(a)
end
print(total, acc.x, acc.y, less, equal, length)
