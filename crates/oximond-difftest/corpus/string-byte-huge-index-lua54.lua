-- oximond-difftest repro profile=lua54 from=lua54 seeds 738, 898, 1963, 2061; lua55 seeds 287, 400, 508, 1707
-- string.byte with an index far outside the string returns nothing; oximond
-- raises "string index is out of range".
print("o1#", (string.byte("abc", math.maxinteger) or -1))
print("o2#", (string.byte("abc", math.mininteger) or -1))
print("o3#", select("#", string.byte("abc", math.maxinteger - 1, math.maxinteger)))
