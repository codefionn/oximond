local scale = tonumber(arg and arg[1]) or 1
local t = {a = 1, b = 2, c = 3, counter = 0}
local total = 0
for i = 1, 400000 * scale do
    local x = t.a
    t.counter = i % 7
    local y = t.a
    t.counter = t.counter + 1
    total = (total + x + y + t.a + t.b + t.counter) % 1000003
end
print(total, t.counter)
