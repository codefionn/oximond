-- Fields-only table constructors that die inside the iteration.
local scale = tonumber(arg and arg[1]) or 1
local total = 0
for i = 1, 150000 * scale do
    local v = {x = i, y = i + 1, z = i % 7}
    total = total + v.x + v.z
end
print(total)
