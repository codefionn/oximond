local scale = tonumber(arg and arg[1]) or 1

-- The branch keeps this call on the generic VM path rather than the
-- straight-line numeric inline path.
local function bump(x)
    if x % 2 == 0 then
        return x + 1
    else
        return x + 2
    end
end

local total = 0
for outer = 1, 1000000 * scale do
    local value = bump(outer)
    for inner = 1, 8 do
        total = total + (value + inner) % 31
    end
end
print(total)
