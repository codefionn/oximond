local scale = tonumber(arg and arg[1]) or 1
local function fib(n)
    if n < 2 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end
local total = 0
for _ = 1, 4 * scale do
    total = total + fib(22)
end
print(total)
