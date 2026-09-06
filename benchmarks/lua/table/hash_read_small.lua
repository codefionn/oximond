-- Dynamic string-key reads on a small table (under 16 keys, so it stays in the
-- ordered representation). The keys come from an array, so the key operand of
-- every read is a runtime value, not a constant.
local scale = tonumber(arg and arg[1]) or 1
local names, values = {}, {}
for i = 1, 12 do
    local key = "field_" .. i
    names[i] = key
    values[key] = i * 3 + 1
end
local total = 0
for _ = 1, 80000 * scale do
    for i = 1, 12 do
        total = total + values[names[i]]
    end
end
print(total)
