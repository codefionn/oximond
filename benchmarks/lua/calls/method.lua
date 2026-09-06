local scale = tonumber(arg and arg[1]) or 1
local object = {value = 0}
function object:add(delta)
    self.value = self.value + delta
    return self.value
end
local total = 0
for i = 1, 250000 * scale do
    total = total + object:add(i % 7)
end
print(total, object.value)
