local scale = tonumber(arg and arg[1]) or 1
local pcall = pcall

-- Most protected calls return values; sparse failures exercise caught errors.
local function work(i)
    if i % 4096 == 0 then
        error("expected", 0)
    end
    return i % 89, i % 23
end

local total = 0
for i = 1, 150000 * scale do
    local ok, a, b = pcall(work, i)
    if ok then
        total = total + a * 3 + b
    else
        total = total + 17
    end
end
print(total)
