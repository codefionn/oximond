-- A loop nest that warms a tree up: the inner loop becomes a stone, the
-- outer loop's continuation a side, and both compile and run. Traced by
-- scripts/startup-order.sh to list the functions of a trace's life, which
-- crates/oximond-cli/build.rs places after the ones every start runs.
local bodies = {}
for i = 1, 5 do
    bodies[i] = {x = i, y = i * 0.5, z = 0, vx = 0.1 * i, vy = 0, vz = 0.01, mass = 1 + i}
end
local function advance(dt)
    for i = 1, #bodies do
        local first = bodies[i]
        for j = i + 1, #bodies do
            local second = bodies[j]
            local dx, dy, dz = first.x - second.x, first.y - second.y, first.z - second.z
            local distance2 = dx * dx + dy * dy + dz * dz
            local magnitude = dt / (distance2 * math.sqrt(distance2))
            first.vx, first.vy, first.vz = first.vx - dx * second.mass * magnitude, first.vy - dy * second.mass * magnitude, first.vz - dz * second.mass * magnitude
            second.vx, second.vy, second.vz = second.vx + dx * first.mass * magnitude, second.vy + dy * first.mass * magnitude, second.vz + dz * first.mass * magnitude
        end
    end
    for i = 1, #bodies do
        local body = bodies[i]
        body.x, body.y, body.z = body.x + dt * body.vx, body.y + dt * body.vy, body.z + dt * body.vz
    end
end
for _ = 1, 120 do
    advance(0.01)
end
print(string.format("%.6f", bodies[1].x + bodies[5].vz))

-- Loops of other kinds, so that the compiler's paths for integers, arrays,
-- strings, library calls, closures and `pairs` are listed too.
local sum = 0
local numbers = {}
for i = 1, 100 do
    numbers[i] = i
end
for round = 1, 40 do
    for i = 1, #numbers do
        sum = sum + numbers[i] % 7
        numbers[i] = numbers[i] + round
    end
end
local parts = {}
for i = 1, 120 do
    parts[#parts + 1] = string.format("%d:%s", i, string.rep("x", i % 5))
    sum = sum + string.byte("abc", 1 + i % 3) + #parts[i]
end
local joined = table.concat(parts, ",")
local record = {a = 1, b = 2, c = 3}
local function add(x, y)
    return x + y
end
for _ = 1, 80 do
    for _, value in pairs(record) do
        sum = add(sum, value)
    end
    for index, value in ipairs(numbers) do
        if index > 10 then
            break
        end
        sum = add(sum, value)
    end
end
print(sum, #joined)
