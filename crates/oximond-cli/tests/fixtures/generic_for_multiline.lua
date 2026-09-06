-- Generic-for headers that span several lines: errors raised by the iterator
-- report the line where the expression list starts.
local out = {}
local function P(...)
  local t = {}
  for i = 1, select('#', ...) do t[#t+1] = tostring((select(i, ...))) end
  out[#out+1] = table.concat(t, " ")
end
local t = {10, 20}
local function failing()
  for k, v
    in
    function() error("iter") end,
    t,
    0
  do
  end
end
local ok, e = pcall(failing)
P("err", ok, (tostring(e):gsub("^.-:(%d+):", "L%1:")))
local ok2, e2 = pcall(function()
  for x
    in
    5,
    6
  do
  end
end)
P("err2", ok2, (tostring(e2):gsub("^.-:(%d+):", "L%1:")))
print(table.concat(out, "\n"))
