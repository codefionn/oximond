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
run("index_yield", function() local t = setmetatable({}, {__index=coroutine.yield}); return t.k end)
run("newindex_yield", function() local t = setmetatable({}, {__newindex=coroutine.yield}); t.k = 1; return "done" end)
run("index_tostring", function() local t = setmetatable({}, {__index=tostring}); return t.k end)
run("index_type", function() local t = setmetatable({}, {__index=type}); return t.k end)
run("index_rawget", function() local t = setmetatable({}, {__index=rawget}); return t.k end)
run("newindex_rawset", function() local t = setmetatable({}, {__newindex=rawset}); t.k = 5; return rawget(t,"k") end)
run("index_select", function() local t = setmetatable({}, {__index=select}); return t[1] end)
run("index_pcall", function() local t = setmetatable({}, {__index=pcall}); return t.x end)
