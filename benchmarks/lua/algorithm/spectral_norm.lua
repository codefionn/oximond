local scale = tonumber(arg and arg[1]) or 1
local n = 120
local function a(i, j)
    local ij = i + j
    return 1 / (ij * (ij + 1) / 2 + i + 1)
end
local function multiply_av(input, output)
    for i = 0, n - 1 do
        local total = 0
        for j = 0, n - 1 do
            total = total + a(i, j) * input[j]
        end
        output[i] = total
    end
end
local function multiply_atv(input, output)
    for i = 0, n - 1 do
        local total = 0
        for j = 0, n - 1 do
            total = total + a(j, i) * input[j]
        end
        output[i] = total
    end
end
local function multiply_at_av(input, output)
    local temporary = {}
    multiply_av(input, temporary)
    multiply_atv(temporary, output)
end
local result = 0
for _ = 1, scale do
    local u, v = {}, {}
    for i = 0, n - 1 do
        u[i] = 1
    end
    for _ = 1, 10 do
        multiply_at_av(u, v)
        multiply_at_av(v, u)
    end
    local vbv, vv = 0, 0
    for i = 0, n - 1 do
        vbv = vbv + u[i] * v[i]
        vv = vv + v[i] * v[i]
    end
    result = result + math.sqrt(vbv / vv)
end
print(string.format("%.9f", result))
