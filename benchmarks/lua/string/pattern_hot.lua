-- Four fixed patterns run many times over short log lines: an anchored
-- capture parse, a bracketed field lookup, a word iterator and a
-- substitution. Per-call setup is paid on every call for the same few
-- patterns, so any compiled-pattern reuse shows here.
local scale = tonumber(arg and arg[1]) or 1

local lines = {
    "2024-01-17 INFO [net] connected to 10.0.0.1 after 3 tries",
    "2024-01-17 WARN [disk] usage at 91 percent on sda1",
    "2024-01-18 ERROR [auth] user alice failed login 5 times",
    "2024-01-18 INFO [net] sent 4096 bytes in 12 ms",
    "2024-01-19 DEBUG [cache] hit ratio 0.87 over 1000 lookups",
    "malformed line without a date",
}

local find, match, gmatch, gsub = string.find, string.match, string.gmatch, string.gsub
local total = 0
for round = 1, 3000 * scale do
    for l = 1, #lines do
        local line = lines[l]
        local year, month, day, level = match(line, "^(%d+)%-(%d+)%-(%d+) (%u+) ")
        if year then
            total = total + year + month * 31 + day + #level
        end
        local first, last, tag = find(line, "%[(%w+)%]")
        if first then
            total = total + first + last + #tag
        end
        if round % 4 == 0 then
            for word in gmatch(line, "%a+") do
                total = total + #word
            end
        end
        local _, count = gsub(line, "%d+", "#")
        total = (total + count) % 1000000007
    end
end
print(total)
