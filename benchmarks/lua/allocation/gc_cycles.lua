local scale = tonumber(arg and arg[1]) or 1
local total = 0
for round = 1, 20 * scale do
    local roots = {}
    for i = 1, 2000 do
        local left, right = {}, {}
        left.other = right
        right.other = left
        roots[i] = left
        total = total + i % 17
    end
    roots = nil
    collectgarbage("collect")
end
print(total)
