-- string.find and string.match of a single-run pattern over short strings.
local scale = tonumber(arg and arg[1]) or 1
local subjects = { "id 4821", "x9", "order 77 shipped", "a1b22c333", "no 0", "ticket 123456" }
local count = #subjects
local total = 0
for i = 1, 400000 * scale do
    local s = subjects[i % count + 1]
    local first, last = string.find(s, "%d+")
    total = total + first + last
    total = total + #string.match(s, "[%w_]+")
end
print(total)
