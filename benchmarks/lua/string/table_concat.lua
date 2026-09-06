local scale = tonumber(arg and arg[1]) or 1
local total = 0
for _ = 1, 80 * scale do
    local parts = {}
    for i = 1, 2000 do
        parts[i] = tostring(i % 101)
    end
    total = total + #table.concat(parts, ",")
end
print(total)
