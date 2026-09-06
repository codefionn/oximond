-- oximond-difftest repro profile=lua51
-- Exhausting a generic-for must not report a body line that never ran.
for count = 0, 2 do
  local values = count == 0 and "" or count == 1 and "'a'" or "'a','b'"
  local source = "for i,v in pairs{" .. values .. "} do\n  local item=i..v\nend\n"
  local f = assert(loadstring(source, "@hook-probe"))
  local lines = {}
  debug.sethook(function(_, line)
    if debug.getinfo(2, "S").source == "@hook-probe" then
      lines[#lines + 1] = line
    end
  end, "l")
  f(); debug.sethook()
  print(count, table.concat(lines, ","))
end
