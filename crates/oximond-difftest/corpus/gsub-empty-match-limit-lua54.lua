-- oximond-difftest repro profile=lua54 from=lua54 seed 481, lua55 seed 2025
-- After matching "x", 5.4 does not allow an empty match at the same end.
print("o1#", string.gsub("x,y,,z", "x*", "-", 2))
print("o2#", string.gsub("x,y,,z", "x*", "-"))
