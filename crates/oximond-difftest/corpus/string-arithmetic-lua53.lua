-- oximond-difftest repro profile=lua53 seed=105
local text, integer = "10", 1
local function observe(value)
  print(math.type(value), string.format("%.17g", value))
end
observe(text + integer)
observe(text - integer)
observe(text * integer)
observe(text // 3)
observe(text % 3)
observe(-text)
local total = 0
for i = 1, 40000 do
  total = total + text
end
observe(total)
