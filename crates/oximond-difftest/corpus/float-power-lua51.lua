-- oximond-difftest repro profile=lua51 from=lua54 seed 3212
-- x ^ 2 must round like C pow(), which is correctly rounded here.
local x = 483 ^ 3
print("o1#", string.format("%.17g", x ^ 2))
print("o2#", string.format("%.17g", 1.1 ^ 3))
print("o3#", string.format("%.17g", 3.14159 ^ 3))
