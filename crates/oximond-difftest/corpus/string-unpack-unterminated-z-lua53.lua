-- oximond-difftest repro profile=lua53 from=generator 15 packing probes
-- Lua 5.3 uses the buffer's trailing NUL for an unterminated z string.
local ok, value, position = pcall(string.unpack, "z", "end")
print("o1#", ok, type(value), value == "end", position)
