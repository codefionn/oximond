-- LuaJIT converts string indices to 32-bit integers, and a value outside
-- that range becomes INT_MIN.
local s = "abcdef"
local values = {0, 1, -1, 2^31 - 1, 2^31, 2^31 + 2, -2^31, -2^31 - 1, -2^31 - 3,
  2^32, 2^32 + 2, 2^32 + 3, 2^33 + 4, -2^32 - 2, 2^53, 2^53 + 2, -2^53, 2^62,
  -2^62, 2^63, -2^63, 2^64, 1e300, -1e300, 1/0, -1/0, 0/0, 2.5, -2.5, 3.9, -3.9, "3"}
for _, a in ipairs(values) do
  print(tostring(a), (s:sub(a)), (s:sub(1, a)), (s:byte(a)), select("#", s:byte(1, a)),
    s:find("c", a), s:find("", a), (#string.rep("ab", math.min(a, 3))))
end
print(#string.sub("\0G", 2^53, 229), #string.sub("abc", 2^31, 5), #string.sub("abc", -2^53, 2))
