-- Table finalizers exist in Lua 5.2 and later only.
local version = tonumber((_VERSION:match("%d+%.%d+")))
if version >= 5.2 and not jit then
  local ran = 0
  local function setgc(mt, f) mt.__gc = f end
  for i = 1, 20 do
    local mt = {__gc = true}
    setmetatable({}, mt)
    setgc(mt, function() ran = ran + 1 end)
  end
  collectgarbage()
  collectgarbage()
  assert(ran == 20, ran)
end
