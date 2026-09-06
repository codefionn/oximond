local scale = tonumber(arg and arg[1]) or 1
local cells = {}
for i = 1, 6 do cells[i] = i * 3 + 1 end
local total = 0
for outer = 1, 60000 * scale do
    local n = 3 + outer % 4
    local sum = 0
    for i = 1, n do
        sum = sum + cells[i] * outer
    end
    total = (total + sum) % 1000003
end
print(total)
