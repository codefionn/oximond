-- Negative control for adaptive arithmetic. Runs under --compat=lua54: the
-- same add, multiply and subtract sites see integer pairs, float pairs and
-- mixed pairs, in a pattern that changes every iteration.
local scale = tonumber(arg and arg[1]) or 1
local values = {}
for i = 1, 64 do
    if i % 3 == 0 then
        values[i] = i + 0.5
    else
        values[i] = i
    end
end
local acc, count = 0, 0
for i = 1, 2400000 * scale do
    local v = values[i % 64 + 1]
    local w = values[(i * 7) % 64 + 1]
    local s = v + w
    local p = v * 3 - w
    acc = acc + s + p
    count = (count + i * 3) % 1000003
end
print(string.format("%.1f %d", acc, count))
