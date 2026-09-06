-- Modulo sites that see integer, float and mixed operand pairs.
local scale = tonumber(arg and arg[1]) or 1
local values = {}
for i = 1, 32 do
    if i % 3 == 0 then
        values[i] = (i - 16) + 0.5
    else
        values[i] = i - 16
    end
end
local acc = 0.0
for i = 1, 2400000 * scale do
    local v = values[i % 32 + 1]
    local w = values[(i * 5) % 32 + 1]
    if w == 0 then w = 3 end
    acc = acc + v % w
end
print(string.format("%.2f", acc))
