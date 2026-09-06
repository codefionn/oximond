local scale = tonumber(arg and arg[1]) or 1
local sum = 0
for i = 1, 2000000 * scale do
    sum = sum + i
end
print(string.format("%.0f", sum))
