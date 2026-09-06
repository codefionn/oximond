-- Delete and reinsert existing string keys while keeping table size bounded.
local scale = tonumber(arg and arg[1]) or 1
local size = 2048
local keys, values = {}, {}
for i = 1, size do
    local key = "item_" .. i
    keys[i] = key
    values[key] = 0
end

for i = 1, 300000 * scale do
    local key = keys[(i - 1) % size + 1]
    values[key] = nil
    values[key] = i % 101
end

local total = 0
for i = 1, size do
    total = total + values[keys[i]]
end
print(total)
