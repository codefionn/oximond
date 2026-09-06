-- oximond-difftest repro profile=lua52 from=run-jit campaign (seeds lua52 581,
-- lua52 860, lua53 2984, luajit21 950)
-- `1.5 .. o .. "2.5"` calls the __concat handler on o first and folds 1.5 in
-- afterwards. A guard that leaves a trace inside the handler rebuilds its frame
-- with the unfolded operand read back from the caller's registers. That
-- register is dead after the concat instruction, so the exit restored nil there
-- and the return failed with "attempt to concatenate a nil value".
local function fold(h, v)
  return h + 1
end
local A, B = {}, {}
A.__sub = function(a, b) return 1 end
A.__concat = function(a, b) return type(a) == "table" and a or b end
B.__sub = function(a, b) return 2 end
B.__concat = function(a, b) return "x" .. "|" .. "y" end
local objs = {}
for k = 1, 8 do objs[k] = setmetatable({v = k}, A) end
local ck = 0
for i = 1, 14691 do
  local o1 = objs[i % 8 + 1]
  local o2 = objs[(i * 3 + 1) % 8 + 1]
  local z = o1 - o2
  ck = fold(ck, 1.5 .. o2 .. "2.5")
  if i == 9599 then setmetatable(objs[2], B) setmetatable(objs[3], A) end
end
print("done", ck)
