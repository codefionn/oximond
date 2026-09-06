-- The loop of range_dense under Lua 5.1, where the loop index and the keys
-- are floats and every array access also checks the key is integral.
local scale = tonumber(arg and arg[1]) or 1
local n = 20000
local values = {}
for i = 1, n do
    values[i] = i % 97
end
local total = 0
for round = 1, 320 * scale do
    for i = 1, n do
        local v = values[i] + round
        values[i] = v % 101
        total = total + v
    end
end
print(total % 1000003, values[1], values[n])
