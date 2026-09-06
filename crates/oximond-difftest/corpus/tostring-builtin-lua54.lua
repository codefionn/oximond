-- oximond-difftest repro profile=lua54
-- A builtin (load, next, pcall, string.gsub, ...) as __tostring, under
-- tostring and string.format: its own result or argument error, and a
-- catchable C stack overflow when the handler is tostring itself.
local function T(f) return setmetatable({}, {__tostring = f}) end
local function show(ok, v, ...)
  if type(v) == "function" then v = "function" end
  if type(v) == "table" then v = "table" end
  if type(v) == "string" and v:find("^function: ") then v = "function" end
  print(ok, v)
end
local names = {"next", "pcall", "ipairs", "pairs", "load", "coroutine.wrap",
  "coroutine.running", "table.sort", "string.gsub", "rawget", "select", "type"}
for _, n in ipairs(names) do
  local f = _G
  for part in n:gmatch("[^.]+") do f = f[part] end
  io.write(n, " tostring: ") show(pcall(tostring, T(f)))
  io.write(n, " format: ") show(pcall(string.format, "%s", T(f)))
end
-- Stock LuaJIT recurses without end here.
if not jit then
  io.write("tostring tostring: ") show(pcall(tostring, T(tostring)))
  io.write("tostring format: ") show(pcall(string.format, "%s", T(tostring)))
end
