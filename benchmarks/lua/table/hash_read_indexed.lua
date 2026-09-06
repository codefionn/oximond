-- Dynamic string-key reads on a large table (hundreds of keys, so the hash
-- index is used). One probe in eight names a key the table does not have.
local scale = tonumber(arg and arg[1]) or 1
local table_size, probe_count = 3000, 4096
local values = {}
for i = 1, table_size do
    values["item_" .. i] = i % 101
end
local probes = {}
for i = 1, probe_count do
    if i % 8 == 0 then
        probes[i] = "absent_" .. i
    else
        probes[i] = "item_" .. ((i * 7) % table_size + 1)
    end
end
local total, missing = 0, 0
for _ = 1, 160 * scale do
    for i = 1, probe_count do
        local value = values[probes[i]]
        if value then
            total = total + value
        else
            missing = missing + 1
        end
    end
end
print(total, missing)
