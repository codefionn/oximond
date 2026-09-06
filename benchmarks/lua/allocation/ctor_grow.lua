-- One-key constructors that gain one or two fields right after.
local scale = tonumber(arg and arg[1]) or 1
local mt = {__index = {}}
local total = 0
for i = 1, 100000 * scale do
    local a = {n = i}
    a.x = i
    local b = {n = i}
    b.x, b.y = i, i + 1
    local c = setmetatable({name = i}, mt)
    c.x = i
    c.y = i + 1
    total = total + a.n + a.x + b.y + c.y
end
print(total)
