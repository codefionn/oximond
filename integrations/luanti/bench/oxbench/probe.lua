local env = core.get_mapgen_object and "mapgen" or "main"
local fn = loadstring("local x = 0x3ull\nx = bit.band (x, 0x1ull)\nreturn x == 0x1ull")
local ok, ull = pcall(function() return fn and fn() end)
local okr, ffi = pcall(function() return require and require("ffi") end)
core.log("action", ("[oxbench-probe] env=%s jit=%s jit.version=%s ffi_global=%s require_ffi=%s ull=%s/%s"):format(
	core.get_current_modname and tostring(core.get_current_modname()) or env,
	tostring(core.global_exists("jit")), core.global_exists("jit") and tostring(jit.version) or "-",
	tostring(core.global_exists("ffi")), tostring(okr and type(ffi)), tostring(ok), tostring(ull)))
