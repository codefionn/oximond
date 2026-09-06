-- Control for dead_slots: the objects stay reachable. Locals hold tables
-- across the whole loop, closures capture tables, and each new table is kept
-- in a 256-entry ring.
local scale = tonumber(arg and arg[1]) or 1

local function make(i)
    local object = {v = i}
    return function()
        object.v = (object.v + 1) % 1009
        return object.v
    end
end

local closures = {}
for i = 1, 64 do
    closures[i] = make(i)
end
local ring = {}
local first, second = {1}, {v = 2}
local total = 0
for i = 1, 800000 * scale do
    local pair = {i % 17, i % 23}
    ring[i % 256 + 1] = pair
    total = (total + closures[i % 64 + 1]() + first[1] + second.v + pair[1] + pair[2]) % 1000003
    first[1] = (first[1] + 1) % 7
    second.v = (second.v + pair[2]) % 11
end
print(total, #ring)
