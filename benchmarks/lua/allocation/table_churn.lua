local scale = tonumber(arg and arg[1]) or 1
local total = 0
for i = 1, 80000 * scale do
    local value = {i, i + 1, i + 2, label = "node"}
    total = total + value[2]
end
print(total)
