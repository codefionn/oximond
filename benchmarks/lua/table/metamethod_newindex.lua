-- Writes through a function __newindex; the proxy never acquires the keys.
local scale = tonumber(arg and arg[1]) or 1
local size = 256
local keys, backing = {}, {}
for i = 1, size do
    keys[i] = "item_" .. i
end
local proxy = setmetatable({}, {
    __newindex = function(_, key, value)
        backing[key] = value
    end
})

for i = 1, 600000 * scale do
    proxy[keys[(i - 1) % size + 1]] = i % 101
end

local total = 0
for i = 1, size do
    total = total + (backing[keys[i]] or 0)
end
print(total)
