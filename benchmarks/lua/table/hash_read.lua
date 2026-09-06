local scale = tonumber(arg and arg[1]) or 1
local values, keys = {}, {}
for i = 1, 2000 do
    local key = "key_" .. i
    keys[i] = key
    values[key] = i % 107
end
local total = 0
for _ = 1, 300 * scale do
    for i = 1, 2000 do
        total = total + values[keys[i]]
    end
end
print(total)
