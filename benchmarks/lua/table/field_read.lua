local scale = tonumber(arg and arg[1]) or 1
local point = {x = 3, y = 5, z = 7, weight = 11}
local total = 0
for i = 1, 500000 * scale do
    total = total + point.x + point.y + point.z + point.weight + i % 3
end
print(total)
