-- Rotate insert and remove positions in a fixed-size array.
local scale = tonumber(arg and arg[1]) or 1
local size = 128
local values = {}
for i = 1, size do
    values[i] = i
end

local removed_total = 0
for i = 1, 20000 * scale do
    table.insert(values, (i - 1) % size + 1, i % 97)
    removed_total = removed_total + table.remove(values, (i * 7) % (size + 1) + 1)
end

local remaining_total = 0
for i = 1, size do
    remaining_total = remaining_total + values[i]
end
print(removed_total, remaining_total)
