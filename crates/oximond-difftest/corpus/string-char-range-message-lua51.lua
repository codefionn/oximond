-- oximond-difftest repro profile=lua51 from=lua51 baseline cases on candidate c9f8733
-- Stock 5.1 and LuaJIT say "invalid value"; 5.4 says "value out of range".
print("o1#", pcall(function() local w = string.char(256) end))
print("o2#", pcall(function() local w = string.char(-1) end))
