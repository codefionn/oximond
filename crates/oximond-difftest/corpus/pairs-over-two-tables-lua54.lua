-- oximond-difftest repro profile=lua54 from=stage 8 numbers, found with a pairs loop over two tables in turn.
-- Open bug, not MIR specific (Cranelift gives the same sums). A hot loop
-- traverses a table with an array part and a hash part, and every other run
-- traverses a second one that is longer. Under tracing the sums differ from
-- the interpreter's: 335991 and 2328010 instead of 333000 and 2331000.
local ts = {}
for j = 1, 2 do
  local t = {}
  for i = 1, j * 13 do t[i] = i * j end
  for i = 1, j * 5 do t["z" .. i] = j + i end
  ts[j] = t
end
local s1, s2 = 0, 0
for r = 1, 6000 do
  local t = ts[r % 2 + 1]
  local s = 0
  for _, v in pairs(t) do s = s + v end
  if r % 2 == 0 then s1 = s1 + s else s2 = s2 + s end
end
print("o1#", s1, s2)
