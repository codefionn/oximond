local function run(name, f)
  local co = coroutine.create(f)
  local out = {}
  for i = 1, 4 do
    local r = {coroutine.resume(co, "r"..i)}
    local t = {}
    for j = 1, #r do t[#t+1] = tostring(r[j]) end
    out[#out+1] = coroutine.status(co) .. ":" .. table.concat(t, ",")
    if coroutine.status(co) == "dead" then break end
  end
  print(name, (table.concat(out, " ; "):gsub("(%a+): [%da-fx]+","%1: ADDR")))
end
run("close_yield", function() do local x <close> = setmetatable({}, {__close=coroutine.yield}) end return "closed" end)
run("close_tostring", function() do local x <close> = setmetatable({}, {__close=tostring}) end return "closed" end)
run("close_error_yield", function() local ok, e = pcall(function() local x <close> = setmetatable({}, {__close=coroutine.yield}); error("boom") end) return tostring(ok)..tostring(e) end)
