-- oximond-difftest repro profile=luajit21 from=luajit21 seed 324 (baseline, candidate c9f8733)
-- LuaJIT converts string.sub's indexes to 32-bit integers.
print("o1#", #string.sub("\0G", 2^53, 229))
print("o2#", #string.sub("abc", 2^31, 5))
print("o3#", #string.sub("abc", -2^53, 2))
