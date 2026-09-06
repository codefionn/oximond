dofile(core.get_modpath("oxbench") .. "/lprof.lua")
if core.register_mapgen_script then core.register_mapgen_script(core.get_modpath("oxbench") .. "/lprof.lua") end
dofile(core.get_modpath("oxbench") .. "/probe.lua")
if core.register_mapgen_script then core.register_mapgen_script(core.get_modpath("oxbench") .. "/probe.lua") end
local mode = core.settings:get("oxbench_mode") or "startup"
local out = core.get_worldpath() .. "/oxbench.txt"
local t_load = core.get_us_time()

local function finish(fields)
	local f = assert(io.open(out, "w"))
	for k, v in pairs(fields) do
		f:write(k, "=", tostring(v), "\n")
	end
	f:close()
	core.log("action", "[oxbench] done " .. mode)
	core.request_shutdown("oxbench done", false, 0)
end

core.register_on_mods_loaded(function()
	local t_mods = core.get_us_time()
	core.after(0, function()
		local t_first_step = core.get_us_time()
		if mode == "startup" then
			finish({ mode = mode, first_step_us = t_first_step - t_mods })
			return
		end
		local r = tonumber(core.settings:get("oxbench_radius") or "80")
		local ylo = tonumber(core.settings:get("oxbench_ylo") or "-32")
		local yhi = tonumber(core.settings:get("oxbench_yhi") or "47")
		local p1 = { x = -r, y = ylo, z = -r }
		local p2 = { x = r - 1, y = yhi, z = r - 1 }
		local t0 = core.get_us_time()
		local blocks = 0
		core.emerge_area(p1, p2, function(_, action, remaining)
			blocks = blocks + 1
			if remaining == 0 then
				finish({ mode = mode, emerge_us = core.get_us_time() - t0, blocks = blocks })
			end
		end)
	end)
end)
