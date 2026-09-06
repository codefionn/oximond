-- Dynamic string-key reads on a table built by a constructor with constant
-- keys (a shaped table), keyed through an array of strings.
local scale = tonumber(arg and arg[1]) or 1
local names = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}
local record = {alpha = 11, beta = 7, gamma = 5, delta = 3, epsilon = 2, zeta = 13, eta = 17, theta = 19}
local total = 0
for _ = 1, 120000 * scale do
    for i = 1, 8 do
        total = total + record[names[i]]
    end
end
print(total)
