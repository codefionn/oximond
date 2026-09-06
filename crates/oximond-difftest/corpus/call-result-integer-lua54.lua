-- oximond-difftest repro profile=lua54 from=op-measure call-result-int-becomes-float.lua
local function f() return 9 end
local b = f()
print("o1#", b, math.type(b))
