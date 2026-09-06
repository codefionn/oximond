local scale = tonumber(arg and arg[1]) or 1
local source = string.rep("abcdefghij", 200) .. "needle"
local total = 0
for _ = 1, 20000 * scale do
    total = total + (string.find(source, "needle", 1, true) or 0)
end
print(total)
