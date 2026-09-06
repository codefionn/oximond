-- Functions with many registers that build temporary tables in scoped
-- blocks. The tables die at the end of each block, so their slots are dead
-- (and cleared) while later blocks run.
local scale = tonumber(arg and arg[1]) or 1

local function work(i)
    local a1, a2, a3, a4, a5, a6, a7, a8 = i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7
    local s = 0
    do
        local t1 = {a1, a2}
        local t2 = {x = a3, y = a4}
        s = s + t1[1] + t1[2] + t2.x % 7 * (t2.y % 5)
    end
    do
        local t3 = {a5, a6, a7, a8}
        local t4 = {first = t3[1], last = t3[4]}
        s = s + #t3 + t4.last - t4.first
    end
    do
        local b1, b2, b3, b4 = a1 % 3, a2 % 5, a3 % 7, a4 % 11
        local t5 = {b1, b2, b3, b4, {b1 + b2}}
        s = s + t5[5][1] + t5[3] * t5[4]
    end
    return s % 1009
end

local total = 0
for i = 1, 400000 * scale do
    total = (total + work(i)) % 1000003
end
print(total)
