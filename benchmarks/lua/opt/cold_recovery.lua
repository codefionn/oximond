-- Guard-heavy loop whose temporaries are needed only on a cold branch. The
-- hot path reads config fields (each a guard) and keeps one value; one
-- iteration in 1000 reads every temporary.
local scale = tonumber(arg and arg[1]) or 1
local config = {k1 = 3, k2 = 7, k3 = 2, k4 = 13, k5 = 5}
local total, rare = 0, 0
for i = 1, 2400000 * scale do
    local a = i * config.k1
    local b = a + config.k2
    local c = b * config.k3 - i
    local d = c % config.k4
    local e = a + d
    local f = e * config.k5
    local g = f - b
    local h = g + c
    total = (total + d) % 1000003
    if i % 1000 == 0 then
        rare = (rare + a + b + c + e + f + g + h) % 1000003
    end
end
print(total, rare)
