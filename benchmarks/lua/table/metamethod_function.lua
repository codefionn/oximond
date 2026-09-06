-- Dynamic reads through a function __index, including keys that stay absent.
local scale = tonumber(arg and arg[1]) or 1
local fallback = {alpha = 3, beta = 5, gamma = 7, delta = 11,
                  epsilon = 13, zeta = 17}
local keys = {"alpha", "beta", "missing_a", "gamma",
              "delta", "missing_b", "epsilon", "zeta"}
local object = setmetatable({}, {
    __index = function(_, key)
        return fallback[key]
    end
})

local total, missing = 0, 0
for i = 1, 600000 * scale do
    local value = object[keys[(i - 1) % 8 + 1]]
    if value then
        total = total + value
    else
        missing = missing + 1
    end
end
print(total, missing)
