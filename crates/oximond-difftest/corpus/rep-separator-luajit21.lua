-- oximond-difftest repro profile=luajit21 from=luajit21 seeds 107, 141, 163, 176
-- LuaJIT 2.1's string.rep takes a separator.
print("o1#", string.rep("ab", 3, ","))
print("o2#", string.rep("x", 1, ","))
