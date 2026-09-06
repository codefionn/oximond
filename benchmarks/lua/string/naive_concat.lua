local scale = tonumber(arg and arg[1]) or 1
local total = 0
for round = 1, 80 * scale do
    local value = ""
    for i = 1, 800 do
        value = value .. string.char(64 + i % 26)
    end
    total = total + #value
end
print(total)
