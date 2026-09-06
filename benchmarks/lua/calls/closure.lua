local scale = tonumber(arg and arg[1]) or 1
local function counter(start)
    local value = start
    return function(delta)
        value = value + delta
        return value
    end
end
local add = counter(0)
local result = 0
for i = 1, 300000 * scale do
    result = add(i % 5)
end
print(result)
