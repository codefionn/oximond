local scale = tonumber(arg and arg[1]) or 1
local total = 0
for i = 1, 250000 * scale do
    local value = i % 1000 / 1000
    total = total + math.sin(value) * math.cos(value) + math.sqrt(value + 1)
end
print(string.format("%.6f", total))
