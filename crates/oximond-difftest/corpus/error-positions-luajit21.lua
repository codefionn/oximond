-- oximond-difftest repro profile=luajit21
-- error(msg, 2) names the caller's line (none when the caller is pcall);
-- 5.1 and LuaJIT also add a position to a number message.
print("o1#", pcall(function() error("caller", 2) end))
print("o2#", select(2, pcall(error, "caller", 2)))
print("o3#", pcall(function() error(42) end))
