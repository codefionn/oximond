-- oximond-difftest repro profile=lua53
local sum = 0
for index = 1, 3.5 do
  sum = sum + index
  print(math.type(index), index)
end
print(math.type(sum), sum)
for index = 3, 0.5, -1 do
  print(math.type(index), index)
end
