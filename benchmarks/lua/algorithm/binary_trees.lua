local scale = tonumber(arg and arg[1]) or 1
local function make(depth, seed)
    if depth == 0 then
        return {value = seed}
    end
    return {
        value = seed,
        left = make(depth - 1, seed * 2 - 1),
        right = make(depth - 1, seed * 2),
    }
end
local function check(tree)
    if not tree.left then
        return tree.value
    end
    return tree.value + check(tree.left) - check(tree.right)
end
local total = 0
for round = 1, 20 * scale do
    total = total + check(make(11, round))
end
print(total)
