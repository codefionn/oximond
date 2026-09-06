local scale = tonumber(arg and arg[1]) or 1
local total = 0
for outer = 1, 800 * scale do
    for inner = 1, 800 do
        total = total + (outer + inner) % 31
    end
end
print(total)
