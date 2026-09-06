local scale = tonumber(arg and arg[1]) or 1
local source = string.rep("alpha 123 beta 456 ", 40)
local total = 0
for _ = 1, 1000 * scale do
    local replaced, count = string.gsub(source, "%d+", "n")
    total = total + #replaced + count
end
print(total)
