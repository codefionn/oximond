-- Repeated reads of the same fields of one table, with a store to a different
-- table between them. The stored table is never the one being read.
local scale = tonumber(arg and arg[1]) or 1
local source = {x = 3, y = 5, z = 7, w = 11}
local sink = {count = 0, last = 0}
local total = 0
for i = 1, 150000 * scale do
    local a = source.x * source.y + source.x * source.z
    sink.count = sink.count + 1
    local b = source.y * source.z + source.x * source.w
    sink.last = a + b + i
    total = (total + a + b + source.x + source.y + sink.last % 13) % 1000003
end
print(total, sink.count)
