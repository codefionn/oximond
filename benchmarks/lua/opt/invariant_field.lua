local scale = tonumber(arg and arg[1]) or 1
local config = {base = 7, step = 3, limit = 1000}
local total = 0
for i = 1, 500000 * scale do
    local v = (i * config.step + config.base) % config.limit
    total = (total + v + config.base) % 1000003
end
print(total)
