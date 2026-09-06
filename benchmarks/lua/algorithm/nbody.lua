local scale = tonumber(arg and arg[1]) or 1
local bodies = {
    {x = 0, y = 0, z = 0, vx = 0, vy = 0, vz = 0, mass = 39.47841760435743},
    {x = 4.841431442, y = -1.160320044, z = -0.103622044, vx = 0.606326393, vy = 2.811986844, vz = -0.025218362, mass = 0.037693675},
    {x = 8.343366719, y = 4.124798564, z = -0.403523417, vx = -1.010774346, vy = 1.825662371, vz = 0.008415762, mass = 0.011286326},
    {x = 12.894369562, y = -15.111151402, z = -0.223307578, vx = 1.082791006, vy = 0.868713018, vz = -0.010832637, mass = 0.001723724},
    {x = 15.379697115, y = -25.919314609, z = 0.179258772, vx = 0.979090732, vy = 0.594698999, vz = -0.034755956, mass = 0.002033687},
}
local function advance(dt)
    for i = 1, #bodies do
        local first = bodies[i]
        for j = i + 1, #bodies do
            local second = bodies[j]
            local dx, dy, dz = first.x - second.x, first.y - second.y, first.z - second.z
            local distance2 = dx * dx + dy * dy + dz * dz
            local magnitude = dt / (distance2 * math.sqrt(distance2))
            local first_scale, second_scale = second.mass * magnitude, first.mass * magnitude
            first.vx, first.vy, first.vz = first.vx - dx * first_scale, first.vy - dy * first_scale, first.vz - dz * first_scale
            second.vx, second.vy, second.vz = second.vx + dx * second_scale, second.vy + dy * second_scale, second.vz + dz * second_scale
        end
    end
    for i = 1, #bodies do
        local body = bodies[i]
        body.x, body.y, body.z = body.x + dt * body.vx, body.y + dt * body.vy, body.z + dt * body.vz
    end
end
for _ = 1, 1000 * scale do
    advance(0.01)
end
local total = 0
for i = 1, #bodies do
    local body = bodies[i]
    total = total + body.x + body.y + body.z + body.vx + body.vy + body.vz
end
print(string.format("%.9f", total))
