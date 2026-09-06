-- oximond-difftest repro profile=lua54
-- Malformed patterns read the same through find, match, gsub and gmatch.
local pats = {"[a", "%", "%b", "%bx", "%f", "%fa", "%1", "(a)%2", "[%"}
for _, p in ipairs(pats) do
  print(p, pcall(string.find, "abc", p))
  print(p, pcall(string.match, "abc", p))
  print(p, pcall(string.gsub, "abc", p, "x"))
  print(p, pcall(function() for x in string.gmatch("abc", p) do end end))
end
print(pcall(string.find, "abc", "(a"))
print(pcall(string.match, "abc", "(a"))
-- gmatch compiles its pattern on the first step, not when it is created.
print(select("#", pcall(string.gmatch, "abc", "[a")), type((select(2, pcall(string.gmatch, "abc", "[a")))))
print(pcall(function() local it = string.gmatch("abc", "[a") return "created" end))
print(pcall(function() return string.gmatch("abc", "[a")() end))
print(pcall(function() local it = string.gmatch("abc", "[a") local r = it() return r end))
