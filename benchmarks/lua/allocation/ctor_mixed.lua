-- Constructors with an array part and named fields that die inside the iteration.
local scale = tonumber(arg and arg[1]) or 1
local total = 0
for i = 1, 120000 * scale do
    local v = {i, i + 1, tag = i % 7, n = 2}
    total = total + v[1] + v.tag + v.n
end
print(total)
