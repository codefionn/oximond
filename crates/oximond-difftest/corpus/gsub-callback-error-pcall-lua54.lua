-- oximond-difftest repro profile=lua54 from=lua54 hand-written
-- pcall calls string.gsub directly and the replacement callback raises.
print(pcall(string.gsub, "a", "a", function() error("x", 0) end))
print(pcall(string.gsub, "a", "a", string.rep))
print(pcall(string.gsub, "a", "a", setmetatable({}, {__index = function() error("y", 0) end})))
print(pcall(string.gsub, "aa", "a", function() return {} end))
print(xpcall(string.gsub, function(m) return "H:" .. m end, "a", "a", function() error("z", 0) end))
print(pcall(string.gsub, "abc", "b", function(c) return c:upper() end))
print(pcall(string.gsub, "a", "a", setmetatable({}, {__index = string.rep})))
print(xpcall(string.gsub, function(m) return "H:" .. m end, "a", "a", setmetatable({}, {__index = string.rep})))
