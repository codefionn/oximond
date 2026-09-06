-- Dense array reads and writes in a stable counted loop. Runs under
-- --compat=lua54, so the loop index and the keys are integers.
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
