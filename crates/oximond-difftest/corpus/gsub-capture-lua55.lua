-- oximond-difftest repro profile=lua55
-- Replacement strings with a bad capture index or a stray %.
print(pcall(string.gsub, "abc", "(a)", "%2"))
print(pcall(string.gsub, "abc", "a", "%1"))
print(pcall(string.gsub, "abc", "a", "%2"))
print(pcall(string.gsub, "abc", "(a)", "%9"))
print(pcall(string.gsub, "abc", "(a)(b)", "%3"))
print(pcall(string.gsub, "abc", "(a)", "%x"))
local ok, result, count = pcall(string.gsub, "abc", "a", "%")
print(ok, type(result) == "string" and #result or result, count)
print(pcall(string.gsub, "abc", "(a)", "%0%1"))
print(pcall(string.gsub, "abc", "(a)", "%%"))
