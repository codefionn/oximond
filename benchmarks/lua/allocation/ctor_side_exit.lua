-- A constructor loop whose rare branch keeps the table alive. The common path
-- never lets the table escape, so a compiled loop can drop it. Every 50th
-- iteration the table is stored, which forces the table to exist at that
-- point and then to be read back.
local scale = tonumber(arg and arg[1]) or 1
local kept, count = {}, 0
local total = 0
for i = 1, 120000 * scale do
    local v = {i, i + 1, tag = i % 5}
    if i % 50 == 0 then
        count = count + 1
        kept[count] = v
    end
    total = total + v[1] + v.tag
end
local kept_sum = 0
for j = 1, count do
    kept_sum = kept_sum + kept[j][2] + kept[j].tag
end
print(total, count, kept_sum)
