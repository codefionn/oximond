-- Float % float with positive and negative operands.
local scale = tonumber(arg and arg[1]) or 1
local acc = 0.0
local divisors = { 7.5, -7.5, 0.25, -0.25, 1e3, -3.0 }
for i = 1, 3000000 * scale do
    local x = (i % 1013 - 506) * 0.75
    local d = divisors[i % 6 + 1]
    acc = acc + x % d
end
print(string.format("%.4f", acc))
