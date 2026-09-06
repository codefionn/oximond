-- Negative control for range proofs. Each round reads values[i + offset]
-- with an offset that changes per round, so up to 3000 of 20000 keys per
-- round fall past the end of the array and read nil.
local scale = tonumber(arg and arg[1]) or 1
local n = 20000
local values = {}
for i = 1, n do
    values[i] = i % 97
end
local total, missing = 0, 0
for round = 1, 320 * scale do
    local offset = (round % 4) * 1000
    for i = 1, n do
        local v = values[i + offset]
        if v then
            total = total + v
        else
            missing = missing + 1
        end
    end
end
print(total % 1000003, missing)
