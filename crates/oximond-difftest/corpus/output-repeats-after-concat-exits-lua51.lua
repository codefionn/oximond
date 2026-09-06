-- oximond-difftest repro profile=lua51 from=run-jit seed 43
-- Fixed: the guard on the second __concat handler of a chain left the trace at
-- the pc of the first handler's return, so the interpreter resumed in the main
-- chunk and printed earlier lines again. See concat-chain-guard-after-handler-lua54.
local function out(tag, ...)
  print(tag)
end
print("H8")
local A = {}
A.__concat = function(a, b) return 1 end
A.__add = function(a, b) return 1 end
local B = {}
B.__concat = function(a, b) return 2 end
local objs = {}
for k = 1, 8 do objs[k] = setmetatable({v = k}, A) end
local ck = 0
for i = 1, 6274 do
  local o1 = objs[i % 8 + 1]
  local o2 = objs[(i * 3 + 1) % 8 + 1]
  ck = ck + (o1 .. "tt" .. o2 .. "tt")
  if i == 3774 then setmetatable(objs[2], B) setmetatable(objs[3], A) end
end
print("OP9", ck)
