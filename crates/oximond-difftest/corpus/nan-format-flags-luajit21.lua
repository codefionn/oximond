-- oximond-difftest repro profile=luajit21
-- LuaJIT ignores the NaN sign and the plus flag, but keeps the space flag.
local nan = 0 / 0
print("print", nan, -nan)
print("tostring", tostring(nan), tostring(-nan))
print("concat", "" .. nan, "" .. (-nan))
io.write("write ", nan, " ", -nan, "\n")
-- Compute both zeros at runtime so constant folding cannot merge their signs.
local positive_zero = 1 / math.huge
local negative_zero = -1 / math.huge
print("zero", positive_zero, negative_zero)
print("zero tostring", tostring(positive_zero), tostring(negative_zero))
print("zero concat", "" .. positive_zero, "" .. negative_zero)
io.write("zero write ", positive_zero, " ", negative_zero, "\n")
print("zero reciprocal", 1 / positive_zero, 1 / negative_zero)
for _, value in ipairs({nan, -nan, math.huge, -math.huge, 1.5, positive_zero, negative_zero}) do
  for _, directive in ipairs({
    "%g", "%G", "%f", "%e", "%E",
    "%+g", "%+G", "%+f", "%+e", "%+E",
    "% g", "%+ g", "%+010g", "%+ 010g", "%-+10g", "%#g",
  }) do
    print(directive, "[" .. string.format(directive, value) .. "]")
  end
end
