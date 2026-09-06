local scale = tonumber(arg and arg[1]) or 1
local values = {}
for i = 1, 10000 do
    values[i] = i % 113
end
local total = 0
for _ = 1, 80 * scale do
    for _, value in ipairs(values) do
        total = total + value
    end
end
print(total)
