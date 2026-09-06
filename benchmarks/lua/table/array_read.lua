local scale = tonumber(arg and arg[1]) or 1
local values = {}
for i = 1, 20000 do
    values[i] = i % 101
end
local total = 0
for _ = 1, 80 * scale do
    for i = 1, 20000 do
        total = total + values[i]
    end
end
print(total)
