local scale = tonumber(arg and arg[1]) or 1
local fallback = {x = 3, y = 5}
local object = setmetatable({}, {__index = fallback})
local total = 0
for i = 1, 200000 * scale do
    total = total + object.x + object.y + i % 11
end
print(total)
