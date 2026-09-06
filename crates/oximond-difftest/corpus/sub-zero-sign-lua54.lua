-- oximond-difftest repro profile=lua54 from=lua55 seed 1180 (generator 2, candidate c9f8733)
-- 5.4 compiles x - 0 as x + (-0) (OP_ADDI), so -0.0 - 0 is +0.0.
local z = -0.0
print("o1#", string.format("%.17g", z - 0))
print("o2#", string.format("%.17g", (-0.0) - 0))
print("o3#", string.format("%.17g", z - 0.0))
