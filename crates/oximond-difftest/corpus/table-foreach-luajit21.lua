-- oximond-difftest repro profile=luajit21
-- table.foreach and foreachi return nothing unless the callback returns a
-- non-nil value.
local function count(...) return select("#", ...) end
print(count(table.foreach({}, print)))
print(count(table.foreach({10}, function() end)))
print(count(table.foreachi({10}, function() end)))
print(count(table.foreachi({}, function() end)))
print(count(table.foreach({a = 1}, function(k) return k end)))
print(table.foreach({a = 1}, function(k, v) return k end))
print(table.foreachi({5, 6}, function(k, v) return v end))
print(table.foreach({a = 1}, function() return false end))
print(count(pcall(table.foreach, {1}, function() end)))
print(count(pcall(table.foreachi, {1}, function() return 1 end)))
local t = {table.foreach({}, print)}
print(#t)
