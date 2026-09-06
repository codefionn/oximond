local scale = tonumber(arg and arg[1]) or 1
local total = 0
for i = 1, 80000 * scale do
    local value = string.format("item=%06d value=%.3f hex=%x", i, i / 7, i % 65536)
    total = total + #value
end
print(total)
