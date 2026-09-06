-- Arithmetic sites that only ever see two non-integral floats: multiply-add
-- chains whose operand types never change.
local scale = tonumber(arg and arg[1]) or 1
local x, y, z = 0.5, 1.25, 0.75
local acc = 0.5
for _ = 1, 2400000 * scale do
    x = x * 0.999 + 0.0015
    y = y * 0.9999 + x * 0.0001
    z = (z + x * y) * 0.5
    acc = acc + z * 0.25 + x * 0.125
end
print(string.format("%.6f %.9f %.9f", acc, y, z))
