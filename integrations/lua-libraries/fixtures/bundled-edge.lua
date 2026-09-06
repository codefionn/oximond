local json = require('cjson')
local safe = require('cjson.safe')
local struct = require('struct')
local mp = require('cmsgpack')
local unpack = unpack or table.unpack
local function hex(s) return (s:gsub('.', function(c) return string.format('%02x', c:byte()) end)) end
local function canonical(v)
 if v == json.null then return 'null' end
 if type(v) == 'string' then
  -- Caller names belong to the VM profile. Compare the argument position and
  -- module error detail against Redis's Lua 5.1 module implementations.
  v = v:gsub("^(bad argument #%d+) to '[^']*' ", "%1 to '?' ")
  return 's:'..hex(v)
 end
 if type(v) == 'number' then return string.format('n:%.17g', v) end
 if type(v) ~= 'table' then return tostring(v) end
 local pairs = {}
 for k, value in next, v do pairs[#pairs+1] = canonical(k)..'='..canonical(value) end
 table.sort(pairs)
 return '{'..table.concat(pairs, ',')..'}'
end
local function probe(label, f, ...)
 local function results(...)
 local result = {...}
 local strings = {}
 for i = 2, select('#', ...) do strings[#strings+1] = canonical(result[i]) end
 print(label, tostring(result[1]), table.concat(strings, '|'))
 end
 results(pcall(f, ...))
end
for _, n in ipairs({0, -1/math.huge, 1.25, .1, 1e-5, 1e-4, 1e13, 1e14, 99999999999999.5, 1e300, -1e-300, 1/0, -1/0, 0/0}) do
 probe('json-number', json.encode, n)
end
for _, s in ipairs({'', '[', '[1,]', '{"x":}', '{"x" 1}', '{"x":1,}', '{"x":1', 'truex', 'null', '[]', '{}', '[1,null,false]', '"a\\u0000b"', '"\\ud834\\udd1e"', '"\\ud800"', '"\\udc00"', '"\\uZZZZ"', '"\\q"', '"unterminated', '"a\1b"', '01', '+2', '-.5', '1.', '1e+', '0x1.8p2', 'NaN', '-Infinity', 'nan(payload)'}) do
 probe('json-decode-'..hex(s), json.decode, s)
end
for _, n in ipairs({1, 3, 6, 14}) do
 json.encode_number_precision(n)
 probe('precision-'..n, json.encode, 1.23456789)
 probe('precision-'..n, json.encode, 9.99999e-5)
end
json.encode_number_precision(14)
json.decode_invalid_numbers(false)
for _, s in ipairs({'01', '+2', '-.5', '1.', '0x1p2', 'NaN', '1e999'}) do probe('strict-'..s, json.decode, s) end
for _, n in ipairs({false,true,'null'}) do
 json.encode_invalid_numbers(n)
 probe('invalid-mode-'..tostring(n), json.encode, 1/0)
 probe('invalid-mode-'..tostring(n), json.encode, 0/0)
end
for _, key in ipairs({1, 10, 11, 100}) do probe('sparse-'..key, json.encode, {[key]='x'}) end
json.encode_sparse_array(true)
for _, key in ipairs({1, 10, 11, 100}) do probe('sparse-convert-'..key, json.encode, {[key]='x'}) end
probe('json-key', json.encode, {[true]='x'})
probe('json-function', json.encode, function() end)
for _, format in ipairs({'c0','s','>I2i2','<!4BI4','!4BI3','I0','I32','I33','!3','!0','z'}) do
 probe('struct-size-'..format, struct.size, format)
end
for _, spec in ipairs({{'c3','xy'}, {'c2','xyz'}, {'c0','a\0b'}, {'s','a\0b'}, {'>I2',65537}, {'<i4',-1}, {'>I16',-1}, {'<!4BI4',1,0x12345678}, {'!4BI3',1,0x12345678}, {'>f',1.25}, {'<d',.1}}) do
 probe('struct-pack-'..spec[1], struct.pack, unpack(spec))
end
for _, spec in ipairs({{'Bc0', '\2ab'}, {'c0', 'abc'}, {'s','a\0b'}, {'I4','x'}, {'s','x'}, {'x','x',0}, {'<I2','\1\2'}, {'>i2','\255\254'}, {'i0',''}, {'i16',string.rep('\255',16)}}) do
 probe('struct-unpack-'..spec[1], struct.unpack, unpack(spec))
end
for _, n in ipairs({127,128,255,256,65535,65536,4294967295,4294967296,-32,-33,-128,-129,-32768,-32769,-2147483648,-2147483649,1.5,.1,1/0,-1/0,0/0}) do probe('mp-number', mp.pack, n) end
probe('mp-nil', mp.pack, nil)
probe('mp-function', mp.pack, function() end)
probe('mp-empty', mp.pack, {})
probe('mp-sparse', mp.pack, {[1]='x',[3]='z'})
for _, s in ipairs({'', '\192', '\193', '\217', '\146\1', '\129\192\1', '\207'..string.rep('\255',8), '\211'..string.rep('\255',8), '\217\0', '\196\1x'}) do probe('mp-unpack-'..hex(s), mp.unpack, s) end
probe('mp-one', mp.unpack_one, '\1\192\2', 1)
probe('mp-limit', mp.unpack_limit, '\1\192\2', 2, 1)
probe('mp-limit0', mp.unpack_limit, '\1\2', 0)
probe('mp-offset', mp.unpack_one, '\1', 2)
for _, option in ipairs({1, 'on', 'off', 'bad', true, false}) do probe('json-buffer-'..tostring(option), json.encode_keep_buffer, option) end
for _, option in ipairs({0, 15, '3.8', false, 'bad'}) do probe('json-precision-'..tostring(option), json.encode_number_precision, option) end
probe('json-sparse-badtype', json.encode_sparse_array, true, false)
probe('json-sparse-range', json.encode_sparse_array, true, -1)
probe('json-depth-extra', json.encode_max_depth, 4, 5)
probe('json-sparse-extra', json.encode_sparse_array, true, 2, 10, 1)
probe('safe-encode-noarg', safe.encode)
probe('safe-encode-extra', safe.encode, 1, 2)
probe('safe-decode-type', safe.decode, {})
probe('mp-noarg', mp.pack)
probe('mp-offset-negative', mp.unpack_one, '\1', -1)
probe('mp-limit-negative', mp.unpack_limit, '\1', -1)
probe('mp-limit-type', mp.unpack_limit, '\1', false)
