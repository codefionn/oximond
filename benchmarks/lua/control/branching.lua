local scale = tonumber(arg and arg[1]) or 1
local yes, no = 0, 0
for i = 1, 1200000 * scale do
    if i % 7 < 3 then
        yes = yes + i % 101
    else
        no = no + i % 89
    end
end
print(yes, no)
