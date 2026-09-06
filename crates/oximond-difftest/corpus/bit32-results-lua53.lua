-- oximond-difftest repro profile=lua53
local word = bit32.bnot(297)
print(word, 228 % word, math.type(word), math.type(bit32.bxor(10, 1)))
print(bit32.band(), bit32.bor(), bit32.bxor(), bit32.btest())
print(bit32.lshift(1, 32), bit32.rshift(-1, 32), bit32.arshift(-1, 32))
print(bit32.lshift(16, -4), bit32.rshift(1, -4))
print(bit32.extract(-1, 0, 32), bit32.replace(0, 15, 28, 4))
