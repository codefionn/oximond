-- Array-only table constructors that die inside the iteration.
local scale = tonumber(arg and arg[1]) or 1
local total = 0
for i = 1, 150000 * scale do
    local v = {i, i + 1, i + 2}
    total = total + v[2] + #v
end
print(total)
