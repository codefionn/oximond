-- Constructors that name the same keys build tables of one layout, which one
-- read site and one write site see all of.
local function get(t) return t.mass end
local function bump(t, by) t.mass = t.mass + by end
local bodies = {
  {x = 0, y = 0, mass = 1},
  {x = 1, y = 2, mass = 2},
  {x = 3, y = 4, mass = 3},
  {x = 5, y = 6, mass = 4},
  {x = 7, y = 8, mass = 5},
}
local sum = 0
for round = 1, 200 do
  for i = 1, #bodies do
    sum = sum + get(bodies[i])
    bump(bodies[i], 1)
  end
end
assert(sum == 200 * 15 + 5 * 19900, sum)
for i = 1, #bodies do assert(get(bodies[i]) == i + 200) end
-- A layout of other keys, then an extra key on one of them.
local other = {mass = 10, label = "other"}
assert(get(other) == 10)
bodies[2].extra = true
assert(get(bodies[2]) == 202 and get(bodies[3]) == 203)
bodies[2].mass = nil
assert(get(bodies[2]) == nil and get(bodies[1]) == 201)
