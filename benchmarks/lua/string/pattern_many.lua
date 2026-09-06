-- Many distinct patterns, each run a few times per round on short subjects.
-- Per-call pattern setup (validation, first-byte analysis, any cache lookup)
-- dominates over the match itself.
local scale = tonumber(arg and arg[1]) or 1

local prefixes = { "", "^", "x", "%s*" }
local classes = { "%a", "%d", "%l", "%u", "%w", "%p", "[aeiou]", "[%d_]" }
local quantifiers = { "", "+", "*", "-" }
local suffixes = { "", "%d", "(%a+)", "()", "=", "%f[%W]", "[^%s]?", "%s" }

local patterns = {}
for _, prefix in ipairs(prefixes) do
    for _, class in ipairs(classes) do
        for _, quantifier in ipairs(quantifiers) do
            for _, suffix in ipairs(suffixes) do
                patterns[#patterns + 1] = prefix .. class .. quantifier .. suffix
            end
        end
    end
end

local subjects = {
    "key_1=Value42 next",
    "  xHello World 2024!",
    "MIXED case_text, 7 items",
    "a=1 b=22 c=333",
}

local find, match, gmatch, gsub = string.find, string.match, string.gmatch, string.gsub
local total = 0
for _ = 1, 10 * scale do
    for p = 1, #patterns do
        local pattern = patterns[p]
        for s = 1, #subjects do
            local subject = subjects[s]
            local first, last = find(subject, pattern)
            if first then
                total = total + first * 3 + last
            end
            local captured = match(subject, pattern)
            if type(captured) == "string" then
                total = total + #captured
            elseif captured then
                total = total + captured
            end
        end
        for _ in gmatch(subjects[p % #subjects + 1], pattern) do
            total = total + 1
        end
        local _, count = gsub(subjects[(p + 1) % #subjects + 1], pattern, "")
        total = (total + count) % 1000000007
    end
end
print(#patterns, total)
