local scale = tonumber(arg and arg[1]) or 1
local parts = { "left", "right" }
local total = 0
for i = 1, 50000 * scale do
    total = total + math.max(i, 10)
    total = total + (string.find("haystack", "stack", 1, true) or 0)
    total = total + #table.concat(parts, ":")
end
print(total)
