local scale = tonumber(arg and arg[1]) or 1
local function f(i)
    return (i * 3 + 1) % 101
end
local t = {x = 0, y = 0}
local a = {0, 0}
local total = 0
for i = 1, 300000 * scale do
    t.x = f(i)
    total = total + t.x
    a[1] = i % 13
    a[2] = i % 17
    total = (total + a[1] + a[2] + t.x) % 1000003
end
print(total)
