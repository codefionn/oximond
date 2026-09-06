-- Integer % integer with positive and negative operands. Runs under
-- --compat=lua54, where both operands stay integers. The hash advances by
-- addition so every value stays below 2^53 and Lua 5.1 and LuaJIT print the
-- same checksum. MIN % -1 and the other edge cases are covered by
-- crates/oximond/src/vm/tests/fast_arith.rs.
local scale = tonumber(arg and arg[1]) or 1
local acc = 0
local h = 0
local divisors = { 7, -7, 13, -13, 1, -1, 1000003, -64 }
for i = 1, 3000000 * scale do
    h = (h + 2654435761) % 4294967296
    local n = h - 2147483648
    local d = divisors[i % 8 + 1]
    acc = (acc + n % d + i % 97) % 1000000007
end
print(acc)
