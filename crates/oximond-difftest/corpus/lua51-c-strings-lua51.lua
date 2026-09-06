-- oximond-difftest repro profile=lua51 from=lua51 seeds 146, 154, 187
-- Stock 5.1 builds %c through a C string, so %c of 0 adds nothing, and
-- tonumber with base 16 accepts a 0x prefix (strtoul).
print("o1#", #string.format("%c%c", 66, 0))
print("o2#", tonumber("0x1F", 16))
