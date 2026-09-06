local scale = tonumber(arg and arg[1]) or 1
local total = 0
for round = 1, 12 * scale do
    local values = {}
    local state = round
    for i = 1, 8000 do
        state = (state * 16807) % 2147483647
        values[i] = state
    end
    table.sort(values)
    total = (total + values[1] + values[#values]) % 1000000007
end
print(total)
