-- oximond-difftest repro profile=lua54 from=run-jit campaign (seeds lua51 43,
-- lua52 223, lua55 291)
-- A chain `a .. 5 .. b` calls the __concat handler twice. The trace inlines the
-- first call and then guards the second handler's identity. That guard used the
-- pc of the first handler's return instead of the concat's, so when the
-- metatable swap failed it, the interpreter resumed in the main chunk at the
-- wrong pc and called print("start") on a table.
print("start")
local A = {}
A.__concat = function(a, b) return 1 end
local B = {}
B.__concat = function(a, b) return 2 end
local objs = {}
for k = 1, 8 do objs[k] = setmetatable({v = k}, A) end
local ck = 0
for i = 1, 6274 do
  local o1 = objs[i % 8 + 1]
  local o2 = objs[(i * 3 + 1) % 8 + 1]
  ck = ck + (o1 .. 5 .. o2)
  if i == 3774 then setmetatable(objs[2], B) setmetatable(objs[3], A) end
end
print("done", ck)
