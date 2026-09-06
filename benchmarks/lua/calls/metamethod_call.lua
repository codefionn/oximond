local scale = tonumber(arg and arg[1]) or 1

-- Call a table through __call with self and changing numeric arguments.
local callable = setmetatable({ bias = 11 }, {
    __call = function(self, a, b, c)
        return self.bias + a * 3 + b * 5 + c
    end,
})

local total = 0
for i = 1, 180000 * scale do
    total = total + callable(i % 97, i % 31, i % 13)
end
print(total)
