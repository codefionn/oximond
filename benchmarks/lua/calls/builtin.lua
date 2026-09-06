local scale = tonumber(arg and arg[1]) or 1
local convert = tonumber
local total = 0
for i = 1, 200000 * scale do
    total = total + convert(i)
end
print(total)
