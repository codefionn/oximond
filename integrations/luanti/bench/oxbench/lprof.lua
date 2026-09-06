-- Count-hook sampler: every N VM instructions record source:line of the running function.
if not core.settings:get_bool("oxbench_lprof", false) then return end
local ie = core.request_insecure_environment and core.request_insecure_environment()
local dbg = ie and ie.debug or debug
local hist, n = {}, 0
local getinfo = dbg.getinfo
dbg.sethook(function()
	local info = getinfo(2, "Sl")
	if info then
		local k = info.short_src .. ":" .. (info.currentline or 0)
		hist[k] = (hist[k] or 0) + 1
		n = n + 1
	end
end, "", 1000)
local tag = core.register_mapgen_script and "main" or "mapgen"
local function dump()
	local keys = {}
	for k in pairs(hist) do keys[#keys + 1] = k end
	table.sort(keys, function(a, b) return hist[a] > hist[b] end)
	local lines = { "[lprof " .. tag .. "] total " .. n }
	for i = 1, math.min(#keys, 60) do lines[#lines + 1] = "[lprof " .. tag .. "] " .. hist[keys[i]] .. " " .. keys[i] end
	core.log("action", table.concat(lines, "\n"))
end
if tag == "main" then
	core.register_on_shutdown(dump)
else
	local count = 0
	core.register_on_generated(function()
		count = count + 1
		dump()
	end)
end
