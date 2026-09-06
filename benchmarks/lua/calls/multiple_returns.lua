local scale = tonumber(arg and arg[1]) or 1

-- Exercise fixed and open call results, including a nil between live values.
local function values(i)
    return i % 97, nil, i % 101
end

local function consume(a, b, c, d)
    return a + (b or 5) * 3 + (c or 7) * 5 + (d or 11) * 7
end

local total = 0
for i = 1, 150000 * scale do
    local a, b, c = values(i)
    total = total + a + (b or 13) + c
    total = total + consume(i % 17, values(i))
    total = total + consume(i % 17, values(i), 19)
end
print(total)
