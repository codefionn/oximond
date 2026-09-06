-- Traverse mixed string and non-array integer keys with pairs.
local scale = tonumber(arg and arg[1]) or 1
local values = {}
for i = 1, 128 do
    values["item_" .. i] = i
    values[10000 + i] = i + 128
end

local total, count = 0, 0
for _ = 1, 1000 * scale do
    for _, value in pairs(values) do
        total = total + value
        count = count + 1
    end
end
print(total, count)
