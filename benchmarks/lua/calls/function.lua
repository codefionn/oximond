local scale = tonumber(arg and arg[1]) or 1
local function step(value)
    return value * 1.000001 + 0.25
end
local value = 1
for _ = 1, 400000 * scale do
    value = step(value)
end
print(string.format("%.6f", value))
