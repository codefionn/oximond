local scale = tonumber(arg and arg[1]) or 1
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield

-- Resume a fixed set of coroutines and consume each yielded response.
local function worker(first)
    local value = first
    while true do
        value = yield((value * 3 + 7) % 997)
    end
end

local threads = {}
local total = 0
for slot = 1, 8 do
    local thread = create(worker)
    threads[slot] = thread
    local ok, value = resume(thread, slot)
    assert(ok)
    total = total + value
end

for i = 1, 160000 * scale do
    local thread = threads[(i - 1) % 8 + 1]
    local ok, value = resume(thread, i % 997)
    assert(ok)
    total = total + value
end
print(total)
