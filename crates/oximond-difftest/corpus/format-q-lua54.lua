-- oximond-difftest repro profile=lua54
-- %q escapes: 5.4 writes \9 (three digits only before a digit), 5.1
-- leaves control characters other than \r, \0 and newline as they are.
print("o1#", string.format("%q", "a\tb\1c\0012\r\n"))
