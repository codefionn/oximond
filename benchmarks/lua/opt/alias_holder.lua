local scale = tonumber(arg and arg[1]) or 1
local holder = {shared = 5, other = 9}
local meta = {__index = holder}
local p = setmetatable({own = 1}, meta)
local q = setmetatable({own = 2}, meta)
local total = 0
for i = 1, 300000 * scale do
    p.own = i % 11
    total = (total + p.shared + q.shared + p.own + q.own + p.other) % 1000003
end
print(total)
