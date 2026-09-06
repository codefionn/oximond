-- A three-point stencil: reads a[i-1], a[i] and a[i+1] and writes b[i],
-- then swaps the arrays. Every key is the loop index plus a constant.
local scale = tonumber(arg and arg[1]) or 1
local n = 20000
local a, b = {}, {}
for i = 1, n do
    a[i] = i % 89
    b[i] = 0
end
local total = 0
for _ = 1, 200 * scale do
    for i = 2, n - 1 do
        local v = (a[i - 1] + a[i] * 2 + a[i + 1]) % 1009
        b[i] = v
        total = total + v
    end
    a, b = b, a
end
print(total % 1000003, a[2], a[n - 1])
