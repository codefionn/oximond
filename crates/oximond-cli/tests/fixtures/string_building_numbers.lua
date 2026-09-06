local vals = {0, 1, -1, 7, 255, 256, 12345, -12345, 0.5, -0.5, 2.5, 1/3, -1/3, 100, 1e2, 1e13, 99999999999999, 1e14, 1e15, 1e16, -1e14, 1e100, -1e100, 2^53, 2^63, -2^63, 2^31, 123456789012345, 3.14159265358979, 1e-5, 1.5e-10, 5e-324, 1.7976931348623157e308, (function() local z = 0.0; return -z end)(), 0.0, 1e14+0.5, 123456.789e3}
if math.type then
  vals[#vals+1] = math.tointeger(3); vals[#vals+1] = math.mininteger; vals[#vals+1] = math.maxinteger
  
end
vals[#vals+1] = 1/0; vals[#vals+1] = -1/0
local nan = 0/0
vals[#vals+1] = nan
local out = {}
local function add(...) out[#out+1] = table.concat({...}, " ") end
local pad = ("x"):rep(30)
local long = ("y"):rep(70)
for i, v in ipairs(vals) do
  local s = tostring(v)
  add(i, "cat", "a" .. v, v .. "b", v .. v, v .. "", pad .. v, long .. v .. long, v .. pad)
  add(i, "tcat", table.concat({v}), table.concat({v, v}, ","), table.concat({v, "z", v}, ""), table.concat({v, v, v, v}, pad), table.concat({v, v, v, v, v, v, v, v, v, v}, ", "))
  add(i, "fmt", string.format("%s", v), string.format("[%10s]", v), string.format("[%-10s]", v), string.format("%.3s", v), string.format("%s%s", v, v))
end
print(table.concat(out, "\n"))
