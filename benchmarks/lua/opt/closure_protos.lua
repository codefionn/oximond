-- Control for closure_family: eight different prototypes, each with its
-- own upvalue, rotated through one call site the same way. Each prototype
-- computes what the matching closure_family closure computes, so the
-- checksums are equal.
local scale = tonumber(arg and arg[1]) or 1

local count1 = 0
local function p1(x)
    count1 = count1 + 1
    return (x * 1 + count1) % 1009
end
local count2 = 0
local function p2(x)
    count2 = count2 + 1
    return (x * 2 + count2) % 1009
end
local count3 = 0
local function p3(x)
    count3 = count3 + 1
    return (x * 3 + count3) % 1009
end
local count4 = 0
local function p4(x)
    count4 = count4 + 1
    return (x * 4 + count4) % 1009
end
local count5 = 0
local function p5(x)
    count5 = count5 + 1
    return (x * 5 + count5) % 1009
end
local count6 = 0
local function p6(x)
    count6 = count6 + 1
    return (x * 6 + count6) % 1009
end
local count7 = 0
local function p7(x)
    count7 = count7 + 1
    return (x * 7 + count7) % 1009
end
local count8 = 0
local function p8(x)
    count8 = count8 + 1
    return (x * 8 + count8) % 1009
end

local g1, g2, g3, g4, g5, g6, g7, g8 = p1, p2, p3, p4, p5, p6, p7, p8
local total = 0
for i = 1, 1600000 * scale do
    total = (total + g1(i)) % 1000003
    g1, g2, g3, g4, g5, g6, g7, g8 = g2, g3, g4, g5, g6, g7, g8, g1
end
print(total)
