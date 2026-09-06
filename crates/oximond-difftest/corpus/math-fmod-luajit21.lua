-- oximond-difftest repro profile=luajit21 from=luajit21 seed 2160
print("o1#", string.format("%.17g", math.fmod(5.3, 2)))
print("o2#", string.format("%.17g", math.fmod(-5.3, 2)))
print("o3#", string.format("%.17g", math.fmod(1e17 + 0.5, 3.1)))
print("o4#", string.format("%.17g", math.fmod(37.21, -6)))
