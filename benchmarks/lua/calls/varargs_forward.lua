local scale = tonumber(arg and arg[1]) or 1
local function sum(a, b, c, d, e)
    return a + b + c + d + e
end
local function forward(...)
    return sum(...)
end
local total = 0
for i = 1, 250000 * scale do
    total = total + forward(i, 2, 3, 4, 5)
end
print(total)
