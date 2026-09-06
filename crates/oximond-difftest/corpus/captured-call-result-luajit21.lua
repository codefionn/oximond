-- oximond-difftest repro profile=luajit21 from=lua51 seed 16 (reduced by hand)
-- f writes captured local a and reads captured local b; the caller then
-- stores f's result in b. Oximond keeps b's old value.
local a, b = 5, 1
local function f(p)
  a = 4
  return b + 3
end
b = f(nil)
print("o1#", b, a)
