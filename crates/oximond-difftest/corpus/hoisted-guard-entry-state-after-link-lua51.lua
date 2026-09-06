-- oximond-difftest repro profile=lua51 from=run-jit campaign (seed lua51 1109)
-- An inner counting loop with an invariant branch guard (i % 6 == 0) gets the
-- guard hoisted into the loop's preheader. The root trace exits natively into
-- that loop with the loop's registers in the exit state, not stored. When the
-- hoisted guard failed, its entry-state snapshot left those registers
-- Unchanged, so the interpreter resumed the inner loop with stale limit and
-- step registers and failed with "attempt to perform numeric for a function
-- value". The snapshot now names every entry input.
local function show(v)
  return tostring(v)
end
local function out(tag, ...)
  local n = select("#", ...)
  local parts = {}
  for k = 1, n do parts[k] = show((select(k, ...))) end
  return #parts
end
local total = 0
for i = 1, 3000 do
  for j = 1, 76 do
    if i % 6 == 0 then
    end
  end
  for w in string.gmatch("", "%a+") do end
  total = total + out("g", i)
end
print("done", total)
