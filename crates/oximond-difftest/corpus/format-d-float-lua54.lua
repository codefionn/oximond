-- oximond-difftest repro profile=lua54 from=lua54 seed 7
print("o1#", pcall(string.format, "%d", 1.5))
print("o2#", pcall(string.format, "%d", 2.0))
