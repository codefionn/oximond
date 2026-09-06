local scale = tonumber(arg and arg[1]) or 1
local i, value = 0, 1
local limit = 1200000 * scale
while i < limit do
    value = (value * 1.0000001 + i % 97) % 1000003
    i = i + 1
end
print(string.format("%.6f", value))
