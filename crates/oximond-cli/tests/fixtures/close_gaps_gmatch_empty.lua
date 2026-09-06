-- Lua 5.3+ gmatch skips an empty match that ends where the previous match
-- ended; Lua 5.1 and LuaJIT accept it.
local function list(s, p)
  local out = {}
  for a, b in string.gmatch(s, p) do out[#out + 1] = tostring(a) .. (b and "," .. tostring(b) or "") end
  print(p, #out, table.concat(out, "|"))
end
list("ab c", "%a*")
list("key_1=Value42 next", "%a*()")
list("a=1 b=22", "(%a*)()")
list("", "x*")
list("xx", "x*")
list("abc", "")
list("a,,b,", "[^,]*")
