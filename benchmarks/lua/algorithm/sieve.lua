local scale = tonumber(arg and arg[1]) or 1
local result = 0
for _ = 1, 5 * scale do
    local limit = 50000
    local composite = {}
    for i = 2, limit do
        if not composite[i] then
            result = result + 1
            local multiple = i * i
            while multiple <= limit do
                composite[multiple] = true
                multiple = multiple + i
            end
        end
    end
end
print(result)
