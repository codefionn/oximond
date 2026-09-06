-- oximond-difftest repro profile=lua55
-- xpcall runs its message handler before unwinding the failed function.
local log = {}
local ok, result = xpcall(function()
  local item <close> = setmetatable({}, {__close = function(_, err)
    log[#log + 1] = "close:" .. tostring(err)
  end})
  error("first", 0)
end, function(err)
  log[#log + 1] = "handler:" .. tostring(err)
  return "handled"
end)
print('X1#1 "' .. table.concat(log, ",") .. '"')
print('X2#2 ' .. tostring(ok) .. ' "' .. result .. '"')
