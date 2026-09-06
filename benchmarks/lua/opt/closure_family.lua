-- Eight closures made by one factory: one prototype, eight upvalue cells.
-- The locals rotate every iteration, so one call site sees all eight
-- closures without a table read. closure_protos is the control with eight
-- different prototypes.
local scale = tonumber(arg and arg[1]) or 1

local function make(k)
    local count = 0
    return function(x)
        count = count + 1
        return (x * k + count) % 1009
    end
end

local g1, g2, g3, g4 = make(1), make(2), make(3), make(4)
local g5, g6, g7, g8 = make(5), make(6), make(7), make(8)
local total = 0
for i = 1, 1600000 * scale do
    total = (total + g1(i)) % 1000003
    g1, g2, g3, g4, g5, g6, g7, g8 = g2, g3, g4, g5, g6, g7, g8, g1
end
print(total)
