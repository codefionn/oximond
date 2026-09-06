local function get(t) return t.v end
local function put(t, v) t.v = v end
local t = {v = 1, w = 2}
for i = 1, 30 do assert(get(t) == 1) end
for k in pairs(t) do t[k] = nil end
assert(get(t) == nil and next(t) == nil)
put(t, 2); assert(get(t) == 2)
for i = 1, 30 do put(t, i) end
assert(get(t) == 30)
put(t, nil); assert(get(t) == nil and rawget(t, "v") == nil)
put(t, 3); assert(get(t) == 3)

local big = {v = "kept"}
for i = 1, 60 do big["p" .. i] = i end
for i = 1, 30 do assert(get(big) == "kept") end
for k in pairs(big) do if k ~= "v" then big[k] = nil end end
assert(get(big) == "kept")
big.v = nil
for i = 1, 30 do assert(get(big) == nil) end
big.v = "again"
assert(get(big) == "again")
local count = 0
for _ in pairs(big) do count = count + 1 end
assert(count == 1)
