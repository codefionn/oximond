local scale = tonumber(arg and arg[1]) or 1
local function sum(...)
    local a, b, c, d, e = ...
    return a + b + c + d + e
end
local total = 0
for i = 1, 250000 * scale do
    total = total + sum(i, 2, 3, 4, 5)
end
print(total)
