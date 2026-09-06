local scale = tonumber(arg and arg[1]) or 1
local size = 180
local checksum = 0
for repeat_index = 1, scale do
    for y = 0, size - 1 do
        local ci = 2 * y / size - 1
        for x = 0, size - 1 do
            local cr = 2 * x / size - 1.5
            local zr, zi, iterations = 0, 0, 0
            while zr * zr + zi * zi <= 4 and iterations < 50 do
                zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
                iterations = iterations + 1
            end
            checksum = checksum + iterations
        end
    end
end
print(checksum)
