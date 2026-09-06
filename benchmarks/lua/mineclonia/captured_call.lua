-- A loop in a chunk whose locals are captured by closures, as in the
-- self-tests at the end of mcl_levelgen/random.lua: the chunk defines
-- dozens of helper locals that later functions use as upvalues, then
-- calls `rng:next_integer ()` and a library function in a hot loop.
local scale = tonumber(arg and arg[1]) or 1

local floor = math.floor
local fmod = math.fmod
local UINT_MAX = 4294967295
local MODULUS = UINT_MAX + 1
local MULTIPLIER = 69069
local INCREMENT = 12345

local function normalize(l) return (l + MODULUS) % MODULUS end
local function lo16(x) return x % 65536 end
local function hi16(x) return floor(x / 65536) end
local function ull(hi, lo) return { lo, hi } end
local function zeroull(x) return x[1] == 0 and x[2] == 0 end
local function equalull(a, b) return a[1] == b[1] and a[2] == b[2] end
local function copyull(a, b) a[1], a[2] = b[1], b[2] end
local function addkull(a, k)
    local lo = a[1] + k
    if lo > UINT_MAX then
        a[1] = lo - MODULUS
        a[2] = (a[2] + 1) % MODULUS
    else
        a[1] = lo
    end
end
local function step(x) return (x * MULTIPLIER + INCREMENT) % MODULUS end
local function mix(a, b) return (a * 31 + b) % MODULUS end
local function clamp(x, lo, hi)
    if x < lo then return lo elseif x > hi then return hi end
    return x
end
local function sign(x) if x < 0 then return -1 end return 1 end
local function rtz(n) if n < 0 then return -floor(-n) end return floor(n) end
local function lerp(t, a, b) return a + t * (b - a) end
local function square(x) return x * x end
local function within(x, bound) return x % bound end
local function scale16(x) return lo16(x) * hi16(x) end
local function rem(a, b) return fmod(a, b) end
local K1, K2, K3, K4 = 0x9e3779b9, 0x7f4a7c15, 1779033703, 4089235721
local SEED_A, SEED_B = ull(K1, K2), ull(K3, K4)
local names = { "lo16", "hi16", "mix", "step" }
local cache = {}
local counter = 0
local last_value = 0
local tmp = ull(0, 0)

local function jvm_lcg(seed)
    local state = { normalize(seed), 0 }
    return function(bits)
        state[1] = step(state[1])
        state[2] = mix(state[2], state[1])
        return floor(state[1] / 2 ^ (32 - bits))
    end
end

local function jvm_random(seed)
    local fn = jvm_lcg(seed)
    return {
        next_integer = function(self)
            return fn(32)
        end,
        next_within = function(self, bound)
            return within(fn(31), bound)
        end,
    }
end

-- The closures below exist only to capture the helpers above, as the
-- rest of random.lua does.
local function exports()
    return normalize, lo16, hi16, ull, zeroull, equalull, copyull, addkull,
        step, mix, clamp, sign, rtz, lerp, square, within, scale16, rem,
        SEED_A, SEED_B, names, cache, tmp, K1, K2, K3, K4
end
local function bump()
    counter = counter + 1
    last_value = counter
end
cache.exports = exports
cache.bump = bump

local rng = jvm_random(304853)
local other = jvm_random(304853)
local sum = 0
for i = 1, 5000 * scale do
    local a, b = rng:next_integer(), other:next_within(5000)
    sum = fmod(sum * 3 + a + b, 1000000007)
end
print(sum, counter, last_value)
