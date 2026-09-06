local scale = tonumber(arg and arg[1]) or 1
local function even(n, total)
    if n == 0 then
        return total
    end
    return odd(n - 1, total + n % 13)
end
function odd(n, total)
    if n == 0 then
        return total
    end
    return even(n - 1, total + n % 17)
end
print(even(250000 * scale, 0))
