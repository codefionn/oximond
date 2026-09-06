local scale = tonumber(arg and arg[1]) or 1
local function sweep(n, m)
    local acc = 0
    for i = 1, n do
        for j = i, m do
            acc = acc + (i * j) % 7
        end
    end
    return acc
end
local total = 0
for k = 1, 12000 * scale do
    total = (total + sweep(4 + k % 3, 12)) % 1000003
end
print(total)
