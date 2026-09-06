-- A small editor workload that runs after kickstart.nvim has loaded.
local api = vim.api
local buffer = api.nvim_create_buf(false, true)
local lines = {}
for i = 1, 400 do
  lines[i] = string.format('local item_%03d = %d', i, i * 7)
end
api.nvim_buf_set_lines(buffer, 0, -1, false, lines)

local sum = 0
for pass = 1, 40 do
  local current = api.nvim_buf_get_lines(buffer, 0, -1, false)
  for _, line in ipairs(current) do
    local number = tonumber(line:match('= (%d+)$'))
    sum = sum + number
  end
  api.nvim_buf_set_lines(buffer, pass - 1, pass, false, {
    string.format('local item_%03d = %d', pass, pass * 7),
  })
end

if sum ~= 22456000 or api.nvim_buf_line_count(buffer) ~= 400 then
  error(string.format('workload mismatch: sum=%d lines=%d', sum, api.nvim_buf_line_count(buffer)))
end
print('OXIMOND_NVIM_WORKLOAD_OK')
