local scale = tonumber(arg and arg[1]) or 1
local values = {}
local limit = 800000 * scale
for i = 1, limit do
    values[(i - 1) % 20000 + 1] = i % 103
end
local total = 0
for i = 1, 20000 do
    total = total + values[i]
end
print(total)
