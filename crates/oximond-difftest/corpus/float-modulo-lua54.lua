-- oximond-difftest repro profile=lua54 from=lua54 seed 15, lua55 seed 22
-- Lua 5.1 computes a % b as a - floor(a/b)*b; 5.3 and later use fmod and
-- add b when the signs differ. Printed with 17 digits.
local a, b, c = -583, 3.001, 1e10
print("o1#", string.format("%.17g", a % b))
print("o2#", string.format("%.17g", c % (4 / -2.25)))
print("o3#", string.format("%.17g", (2^53 * 3 + 5) % 1000003))
