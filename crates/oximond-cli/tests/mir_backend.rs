//! The MIR backend against Cranelift (`OXIMOND_JIT_BACKEND=cranelift`) on
//! Lua programs.
//!
//! Each snippet runs with the default backend, with `cranelift` and with
//! `mir`, on tracing, with the compile on the VM thread so every tree is
//! compiled before the loop goes on. The output must be identical. Under mir
//! the log must say the MIR backend compiled trees and say why it handed any
//! other to Cranelift. The default is MIR, so its log says the same; under
//! `cranelift` it must not mention the MIR backend.
//!
//! The snippets together exercise every operation the MIR backend lowers:
//! the optimized IR (`OXIMOND_JIT_TRACE_IR_OPT`) of each must hold the
//! operations it lists, and the tree that holds them must have been compiled
//! by MIR. `GuardType` is the one operation no Lua program reaches (the
//! recorder does not emit it); `jit::trace::mir::tests` builds it by hand.
//!
//! The string-index probe is on by default (`OXIMOND_JIT_MIR_PROBE`), so the
//! snippets that probe compile their trees through MIR's probe. One test
//! checks that with the variable set to `0` MIR declines those trees.

use std::process::Command;

struct Snippet {
    name: &'static str,
    compat: &'static str,
    source: &'static str,
    /// Operations the optimized IR must hold, as `op: Name`.
    ops: &'static [&'static str],
    /// Reasons the backend may give for the trees it hands to Cranelift
    /// (setup loops, sides that rejoin). Any other reason fails the test.
    fallbacks: &'static [&'static str],
    /// Whether MIR must have compiled a tree.
    compiled: bool,
}

/// `sum` walks a table of tables with `pairs` and reads `box[5]` of each; the
/// loop after it fills the same lanes with the rows of `m`'s tables. Rows are
/// shared by every trace of the thread and not cleared between runs, so a row
/// the step did not fill holds `m`'s array address: wrong sums. 400 runs of
/// `sum` reuse the lanes.
const PAIRS_SOURCE: &str = "local m = {} for i = 1, 10 do m[i] = {0, 0, 0, 0, 1000 + i, 0} end local total = 0 local function sum(fixed) local t = 0 for _, box in pairs(fixed) do t = t + box[5] end return t end local bad = 0 for d = 1, 400 do local fixed = {} for b = 1, 4 do fixed[b] = {0, 0, 0, 0, b, 0} end if sum(fixed) ~= 10 then bad = bad + 1 end if d % 50 == 0 then for k = 1, 3000 do total = total + m[k % 10 + 1][5] end end end print(bad, total)";

const SNIPPETS: &[Snippet] = &[
    Snippet {
        name: "float arithmetic",
        compat: "lua51",
        source: "local x, y = 0, 1 for i = 1, 300000 do x = x + i * 0.5 - y / 3 y = y * 1.0000001 end print(x, y)",
        ops: &[
            "op: Add",
            "op: Subtract",
            "op: Multiply",
            "op: Divide",
            "op: Constant",
            "op: GuardBranch",
            "op: Compare",
        ],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "float modulo, floor, ceil, abs and sqrt",
        compat: "lua51",
        source: "local s = 0 for i = 1, 300000 do s = s + (i % 7) + math.floor(i / 3) + math.abs(i - 500) + math.ceil(i * 0.3) + math.sqrt(i) end print(s)",
        ops: &[
            "op: Modulo",
            "op: Floor",
            "op: Ceil",
            "op: Abs",
            "op: NativeUnary",
        ],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "negation and conversions",
        compat: "lua54",
        source: "local s, n = 0.0, 0 for i = 1, 300000 do s = s - i / 2 + (-i) * 0.25 n = n + (-i) end print(s, n)",
        ops: &["op: Negate", "op: IntToFloat"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "integer arithmetic, bit operations and constant shifts",
        compat: "lua54",
        source: "local a, b, c = 0, 1, 7 for i = 1, 300000 do a = a + i b = (b * 3 + i) & 0xffff c = c ~ (i << 3) a = a + (i % 4) + (i // 2) c = c | 1 c = c >> 2 end print(a, b, c)",
        ops: &[
            "op: BitXor",
            "op: BitOr",
            "op: BitAnd",
            "op: ShiftLeft",
            "op: ShiftRight",
            "op: Modulo",
            "op: FloorDivide",
            "op: Multiply",
        ],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "comparisons, branches, not and select",
        compat: "lua51",
        source: "local nan = 0/0 local a, b = 0, 0 for i = 1, 300000 do if i > 0 then a = a + 1 end if not (i < nan) then b = b + 1 end if i ~= nan then a = a + 2 end if i <= 300001 and i >= 1 then b = b - 1 end if i == i then a = a + 1 end end print(a, b)",
        ops: &["op: Compare", "op: GuardBranch", "op: Select"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a select between floats",
        compat: "lua51",
        source: "local s = 0 for i = 1, 300000 do local m = (i > 5) and i or 5 s = s + m end print(s)",
        ops: &["op: Select"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "array loads and stores with float keys",
        compat: "lua51",
        source: "local t = {} for i = 1, 1000 do t[i] = i * 1.5 end local s = 0 for r = 1, 300 do for i = 1, 1000 do t[i] = t[i] + 1 s = s + t[i] end end print(s)",
        // `t[i] = t[i] + 1` stores into the slot the load just reached, so
        // the store is the unguarded overwrite.
        ops: &["op: ArrayLoad", "op: ArrayOverwrite"],
        fallbacks: &["root that links to sides compiled apart"],
        compiled: true,
    },
    Snippet {
        name: "array loads and stores with integer keys",
        compat: "lua54",
        source: "local s = 0 local t = {} for i = 1, 64 do t[i] = i end for r = 1, 5000 do for i = 1, 64 do s = s + t[i] * 2 t[i] = t[i] + 1 end end print(s)",
        // As above: the store follows the load of the same slot.
        ops: &["op: ArrayLoad", "op: ArrayOverwrite"],
        fallbacks: &["root that links to sides compiled apart"],
        compiled: true,
    },
    Snippet {
        // A store to a slot nothing in the iteration read first keeps its
        // guard.
        name: "guarded array stores",
        compat: "lua54",
        source: "local t = {} for i = 1, 64 do t[i] = i end local s = 0 for r = 1, 5000 do for i = 1, 63 do t[i] = t[i + 1] + r s = s + t[i] end end print(s)",
        ops: &["op: ArrayLoad", "op: ArrayStore"],
        fallbacks: &["root that links to sides compiled apart"],
        compiled: true,
    },
    Snippet {
        name: "boolean array elements",
        compat: "lua51",
        source: "local f = {} for i = 1, 100 do f[i] = false end local c = 0 for r = 1, 3000 do for i = 1, 100 do f[i] = not f[i] if f[i] then c = c + 1 end end end print(c)",
        ops: &["op: Not", "op: ArrayLoad", "op: ArrayOverwrite"],
        fallbacks: &["root that links to sides compiled apart"],
        compiled: true,
    },
    Snippet {
        name: "fields of a table",
        compat: "lua51",
        source: "local o = {x = 1, y = 2} for i = 1, 300000 do o.x = o.x + o.y end print(o.x)",
        ops: &["op: FieldLoad", "op: FieldStore"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "fields of a table with a metatable",
        compat: "lua54",
        source: "local o = setmetatable({x = 1}, {__index = {}}) local n = 0 for i = 1, 300000 do n = n + o.x o.x = o.x + 1 end print(n)",
        ops: &["op: FieldLoad", "op: FieldStore"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "table length",
        compat: "lua54",
        source: "local t = {} for i = 1, 64 do t[i] = i end local s = 0 for r = 1, 5000 do for i = 1, 64 do s = s + #t + t[i] end end print(s)",
        ops: &["op: TableLength", "op: ArrayLoad"],
        fallbacks: &["root that links to sides compiled apart"],
        compiled: true,
    },
    Snippet {
        name: "string bytes and string length",
        compat: "lua54",
        source: "local s = \"hello world, text\" local n = 0 for i = 1, 300000 do n = n + string.byte(s, (i & 7) + 1) + #s end print(n)",
        ops: &["op: StringByte", "op: StringLength"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "table constructors that never allocate",
        compat: "lua54",
        source: "local n = 0 for i = 1, 300000 do local p = {i, i + 1} n = n + p[2] end print(n)",
        ops: &["op: PhantomTable"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a value that changes type inside the loop",
        compat: "lua54",
        source: "local x = 1 local s = 0 for i = 1, 300000 do if i % 100000 == 0 then x = x + 0.5 end s = s + x end print(s)",
        ops: &["op: GuardBranch"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a loop that leaves through its guard and starts again",
        compat: "lua51",
        source: "local total = 0 for r = 1, 20000 do local x = 0 for i = 1, 40 do x = x + i if x > 700 then break end end total = total + x end print(total)",
        ops: &["op: GuardBranch"],
        fallbacks: &["root that links to sides compiled apart"],
        compiled: false,
    },
    Snippet {
        name: "power and math library calls",
        compat: "lua54",
        source: "local s, x = 0.0, 0.5 for i = 1, 300000 do x = x + 0.001 s = s + (x ^ 1.5) + math.sin(x) * math.exp(x * 0.00001) end print(s)",
        ops: &["op: PowerSquare", "op: NativeUnary"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "float modulo that calls the compat function",
        compat: "lua54",
        source: "local s = 0.0 for i = 1, 300000 do s = s + ((i * 1.25) % 7.5) + (i * 0.5) % 3 end print(s)",
        ops: &["op: Modulo"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // Divisors of both signs, zero (NaN), infinities and a dividend whose
        // quotient is too large for the in-line form, which calls the compat
        // function.
        name: "float modulo answered in line, and by the compat function off the fast path",
        compat: "lua54",
        source: "local d = {7.5, -7.5, 0.25, -0.25, 0.0, 1 / 0, -1 / 0, 3.0, -3.0} local s, calls = 0.0, 0 for i = 1, 300000 do local x = (i % 1013 - 506) * 0.75 if i % 9973 == 0 then x = 1e300 end local m = x % d[i % 9 + 1] if m == m then s = s + m else calls = calls + 1 end end print(s, calls)",
        ops: &["op: ModuloWith"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "numbers that are integers or floats at run time take the modulo of their type",
        compat: "lua54",
        source: "local values = {} for i = 1, 32 do if i % 3 == 0 then values[i] = (i - 16) + 0.5 else values[i] = i - 16 end end local acc = 0.0 for i = 1, 300000 do local v = values[i % 32 + 1] local w = values[(i * 5) % 32 + 1] if w == 0 then w = 3 end acc = acc + v % w end print(string.format('%.2f', acc))",
        ops: &["op: NumberModulo"],
        // The tree with the side the loop's `if` makes passes a number that
        // is an integer or a float at run time to it, and the side rejoins
        // the loop: MIR compiles it (see the test of the rejoining side).
        fallbacks: &["root that links to sides compiled apart"],
        compiled: true,
    },
    Snippet {
        // Only the exit that leaves the loop one iteration in a thousand
        // reads the derived values.
        name: "values only an exit reads are computed in its stub",
        compat: "lua51",
        source: "local c = {k1 = 3, k2 = 7, k3 = 2, k4 = 13, k5 = 5} local total, rare = 0, 0 for i = 1, 300000 do local a = i * c.k1 local b = a + c.k2 local e = b * c.k3 - i local f = e % c.k4 local g = (f + a) * c.k5 total = (total + f) % 1000003 if i % 1000 == 0 then rare = (rare + a + b + e + g) % 1000003 end end print(total, rare)",
        ops: &["op: Multiply"],
        fallbacks: &["root that links to sides compiled apart"],
        compiled: true,
    },
    Snippet {
        name: "a branch that merges back into the loop (a side that rejoins)",
        compat: "lua54",
        source: "local a, b, c = 0, 0, 0 for i = 1, 600000 do if i % 3 == 0 then a = a + i elseif i % 7 == 0 then b = b + 2 else c = c + 1 end end print(a, b, c)",
        ops: &["op: Modulo", "op: GuardBranch"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "table stores that grow an array",
        compat: "lua54",
        source: "local t = {} for i = 1, 300000 do t[#t + 1] = i end local s = 0 for i = 1, #t do s = s + t[i] end print(s)",
        ops: &["op: ArraySet", "op: ArrayLoad", "op: TableLength"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // A strict header: the append at the end of a dense list is the only
        // set the trace takes, and a table of strings goes to the VM's store
        // entry from the same code.
        name: "appends under a strict header, and a pop",
        compat: "lua54",
        source: "local s = 0 for r = 1, 300 do local t = {} for i = 1, 200 do t[#t + 1] = i end for i = 1, 100 do t[#t] = nil end s = s + #t end print(s)",
        ops: &["op: ArraySet", "op: TableLength"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // The sieve: a boolean over a slot inside the array, and one a few
        // slots past its end, which grows it with nil holes.
        name: "a sieve that fills holes past the end of its array",
        compat: "lua54",
        source: "local found = 0 for _ = 1, 3 do local limit = 20000 local composite = {} for i = 2, limit do if not composite[i] then found = found + 1 local multiple = i * i while multiple <= limit do composite[multiple] = true multiple = multiple + i end end end end print(found)",
        ops: &["op: ArraySet"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // Strings stored into a dense list go through the VM's store entry
        // (the table takes a counted reference), a number overwrites in place.
        name: "strings and numbers stored into a list under a header",
        compat: "lua54",
        source: "local n = 0 for r = 1, 400 do local t = {} for i = 1, 50 do t[i] = 'k' .. (i % 5) end for i = 1, 50 do t[i] = i end n = n + #t end print(n)",
        ops: &["op: ArraySet"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // `row[j]` of a row loaded inside the loop: the words of the row's
        // array come through the table's address in the lane's row.
        name: "reads of the array of a table met by lane",
        compat: "lua54",
        source: "local m = {} for i = 1, 8 do local row = {} for j = 1, 6 do row[j] = i * 10 + j end m[i] = row end local total = 0 for k = 1, 30000 do local row = m[k % 8 + 1] for j = 1, 6 do total = total + row[j] end end print(total)",
        ops: &["op: LaneArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "float rows read at a varying float index",
        compat: "lua51",
        source: "local m = {} for i = 1, 8 do local row = {} for j = 1, 6 do row[j] = i * 0.5 + j end m[i] = row end local total = 0 for k = 1, 60000 do local row = m[k % 8 + 1] total = total + row[k % 6 + 1] + row[1] end print(total)",
        ops: &["op: LaneArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // The store of `region.solids[idx] = region.solids[idx] + bit`, as
        // in Mineclonia's `mcl_util/shape.lua`: a lane read and a lane store
        // of the same slot.
        name: "counters stored into the array of a table met by lane",
        compat: "lua54",
        source: "local regions = {} for i = 1, 4 do local solids = {} for j = 1, 16 do solids[j] = 0 end regions[i] = { solids = solids, size = 16 } end local function mark(region, idx, bit) region.solids[idx] = region.solids[idx] + bit end for k = 1, 40000 do local region = regions[k % 4 + 1] mark(region, k % region.size + 1, 1) end local total = 0 for i = 1, 4 do for j = 1, 16 do total = total + regions[i].solids[j] end end print(total)",
        ops: &["op: LaneArrayLoad", "op: LaneArrayStore"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // A row whose slot changes type and a nil hole leave the lane store
        // and take the raw set.
        name: "a lane store whose slot changes type and a nil hole in the row",
        compat: "lua54",
        source: "local regions = {} for i = 1, 4 do local solids = {} for j = 1, 16 do solids[j] = 0 end regions[i] = { solids = solids, size = 16 } end local function mark(region, idx, bit) region.solids[idx] = region.solids[idx] + bit end for k = 1, 40000 do if k == 20000 then regions[2].solids[5] = 0.25 end if k == 30000 then regions[3].solids[7] = nil end local region = regions[k % 4 + 1] if region.solids[k % 16 + 1] then mark(region, k % region.size + 1, 1) end end local total = 0 for i = 1, 4 do for j = 1, 16 do total = total + (regions[i].solids[j] or 0) end end print(total)",
        ops: &["op: LaneArrayLoad", "op: LaneArrayStore"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // The value of a `pairs` step is the second result of the `next`
        // helper: a table that gets a lane, read through the lane's row
        // (`box[5]`, as in Mineclonia's `gwp_compute_fixed_ground_height`).
        name: "a table the next step gives, read through its row",
        compat: "lua54",
        source: PAIRS_SOURCE,
        ops: &["op: LaneArrayLoad", "op: CallHelper"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "helper calls, host calls and the release of strings",
        compat: "lua54",
        source: "local s = 0 for i = 1, 200000 do local x = tostring(i) s = s + #x end print(s)",
        ops: &["op: CallHelper", "op: HostCall", "op: ReleaseStrings"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a string key is looked up in the index in line",
        compat: "lua54",
        source: "local keys = {\"alpha\", \"beta\", \"gamma\"} local t = {alpha = 1, beta = 2, gamma = 3} local s = 0 for i = 1, 300000 do s = s + t[keys[(i % 3) + 1]] end print(s)",
        ops: &["op: HashLoad", "op: ArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a string key the table lacks reads as nil in line",
        compat: "lua54",
        // Every key is missing, so each probe ends at an empty bucket.
        source: "local keys = {\"x1\", \"x2\", \"x3\", \"x4\"} local t = {alpha = 1, beta = 2, gamma = 3} local n = 0 for i = 1, 300000 do if t[keys[(i % 4) + 1]] == nil then n = n + 1 end end print(n)",
        ops: &["op: HashLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a probe after the hash part is rebuilt reads the new index",
        compat: "lua54",
        // `sum` is compiled on its first calls over a small table. The
        // table then grows past its index several times, so the later runs of
        // the same tree get the words of a new index each time. Chains of
        // colliding keys come with the growth.
        source: "local keys, t = {}, {} for i = 1, 8 do keys[i] = 'k' .. i t[keys[i]] = i end local function sum() local s = 0 for i = 1, 8 do s = s + t[keys[i]] end return s end local total = 0 for r = 1, 60000 do total = total + sum() if r % 10000 == 0 then for j = 1, 300 do t['grow' .. r .. '_' .. j] = j end end end print(total)",
        ops: &["op: HashLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a number stored under a present string key is stored in line",
        compat: "lua54",
        source: "local keys = {\"alpha\", \"beta\", \"gamma\"} local t = {alpha = 1, beta = 2, gamma = 3} for i = 1, 300000 do local k = keys[(i % 3) + 1] t[k] = t[k] + 1 end print(t.alpha, t.beta, t.gamma)",
        ops: &["op: HashLoad", "op: CallHelper"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a key deleted and stored again is deleted and revived in line",
        compat: "lua54",
        // The hash_churn shape: a delete and a store of the same key, which
        // both backends fuse into one probe (`fused_delete_stores`).
        source: "local keys = {} local t = {} for i = 1, 101 do keys[i] = 'k' .. i t[keys[i]] = i end for i = 1, 300000 do local k = keys[(i - 1) % 101 + 1] t[k] = nil t[k] = i % 101 end print(t.k1, t.k50, t.k101)",
        ops: &["op: CallHelper"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a delete and a store a guard apart keep both stores",
        compat: "lua54",
        source: "local t = {} local keys = {} for i = 1, 32 do keys[i] = 'g' .. i t[keys[i]] = 1.5 end local s = 0 for i = 1, 300000 do local k = keys[i % 32 + 1] t[k] = nil if i % 3 == 0 then s = s + 1 end t[k] = i * 0.25 end print(s, t.g1, t.g32)",
        ops: &["op: CallHelper", "op: GuardBranch"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a key that may name a metamethod is stored by the helper",
        compat: "lua54",
        source: "local t = {} local keys = {'__x', '__gc', 'plain', '__index'} for i = 1, 300000 do local k = keys[i % 4 + 1] t[k] = nil t[k] = i * 1.0 end print(t.__x, t.__gc, t.plain, t.__index)",
        ops: &["op: CallHelper"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a dense index is walked in line past its first buckets",
        compat: "lua54",
        // 3000 keys in the index of a 4096-bucket table (load 0.73), so
        // probe sequences are long: lookups of the same strings in other
        // allocations, of absent keys, stores over present keys and a delete
        // with a store of one key. The walk passes the steps made in line, so
        // the native walk behind them answers some of the lookups.
        source: "local n = 3000 local t, probes, keys = {}, {}, {} for i = 1, n do keys[i] = 'item_' .. i t[keys[i]] = i % 101 end for i = 1, 4096 do if i % 8 == 0 then probes[i] = 'absent_' .. i else probes[i] = 'item_' .. ((i * 7) % n + 1) end end local total, missing = 0, 0 for r = 1, 60 do for i = 1, 4096 do local v = t[probes[i]] if v then total = total + v else missing = missing + 1 end end end for i = 1, 200000 do local k = keys[i % n + 1] t[k] = i % 101 end for i = 1, 100000 do local k = keys[(i * 13) % n + 1] t[k] = nil t[k] = i % 7 end print(total, missing, t.item_1, t.item_2999, t.item_3000)",
        ops: &["op: HashLoad", "op: CallHelper"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a tree that moves hash storage reads its keys through the VM",
        compat: "lua54",
        // The delete and the store go through a helper that may rebuild the
        // table's hash part, so the index of the run cannot be trusted and no
        // probe is made: the lookup is the VM's, as it is for Cranelift.
        source: "local keys = {\"alpha\", \"beta\", \"gamma\"} local t = {alpha = 1, beta = 2, gamma = 3} local s = 0 for i = 1, 300000 do local k = keys[(i % 3) + 1] s = s + t[k] t[k] = nil t[k] = i end print(s, t.alpha, t.beta, t.gamma)",
        ops: &["op: HashLoad", "op: CallHelper"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "closure cells and prototype checks are read and written in line",
        compat: "lua54",
        // Four closures of one prototype and an integer cell each: the call
        // site checks the prototype and reads and writes the cell through the
        // closure's row. `other` is called in its own loop, so no check runs on it.
        source: "local function make(k) local n = 0 return function(x) n = n + 1 return (x * k + n) % 1009 end end local function other(x) return x + 1 end local fs = {make(1), make(2), make(3), make(4)} local t = 0 for i = 1, 300000 do t = t + fs[(i % 4) + 1](i) end local o = 0 for i = 1, 300000 do o = o + other(i) end print(t, o)",
        ops: &["op: CallHelper"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a float cell written by one closure is read by the other that shares it",
        compat: "lua54",
        // Each pair shares one float cell. The call sites check the prototype
        // (CLOSURE_SAME_PROTO) and write the cell through one closure
        // (CELL_SET) and read it through the other (CELL_GET_*).
        source: "local function pair(k) local n = 0.5 * k return function() n = n + 1.5 return n end, function() return n end end local fs = {} for k = 1, 4 do local inc, get = pair(k) fs[k] = {inc = inc, get = get} end local t = 0 for i = 1, 300000 do local f = fs[(i % 4) + 1] f.inc() t = t + f.get() end print(t)",
        ops: &["op: CallHelper", "322,", "324,", "327,"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a call site meets a native function with no closure row after it compiled",
        compat: "lua54",
        // The tree guards on the prototype. From iteration 200000 two slots
        // hold math.abs and math.floor, which have no closure row, so the
        // guard leaves the trace and a side takes the call.
        source: "local function make(k) local n = k return function(x) n = n + x return n end end local fs = {make(1), make(2), make(3), make(4)} local t = 0 for i = 1, 300000 do if i == 200000 then fs[2] = math.abs fs[4] = math.floor end t = t + fs[(i % 4) + 1](1) end print(t)",
        ops: &["op: CallHelper", "322,", "323,", "327,"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a call site meets a closure of another prototype after it compiled",
        compat: "lua54",
        source: "local function make(k) local n = 0 return function(x) n = n + 1 return (x * k + n) % 1009 end end local function other(x) return x + 1 end local fs = {make(1), make(2), make(3), make(4)} local t = 0 for i = 1, 300000 do if i == 200000 then fs[3] = other end t = t + fs[(i % 4) + 1](i) end print(t)",
        ops: &["op: CallHelper", "322,"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a table held by a closure cell is read in line",
        compat: "lua54",
        // Each closure keeps a table in an upvalue, so the tree reads it
        // through CELL_GET_TABLE and the closure's row names its lane.
        source: "local function mk(k) local t = {v = k} local n = 0 return function() n = n + t.v return n end end local fs = {mk(1), mk(2), mk(3), mk(4)} local s = 0 for i = 1, 300000 do s = s + fs[(i % 4) + 1]() end print(s)",
        ops: &["op: CallHelper", "326,"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "nested loops over a table of tables",
        compat: "lua54",
        source: "local bodies = {{x = 1.0, v = 0.5}, {x = 2.0, v = -0.5}, {x = 3.0, v = 0.25}} local function step(dt) for i = 1, #bodies do local a = bodies[i] for j = i + 1, #bodies do local b = bodies[j] local d = a.x - b.x a.v = a.v - d * dt b.v = b.v + d * dt end end for i = 1, #bodies do local b = bodies[i] b.x = b.x + dt * b.v end end for _ = 1, 20000 do step(0.01) end local s = 0 for i = 1, #bodies do s = s + bodies[i].x + bodies[i].v end print(string.format(\"%.9f\", s))",
        ops: &["op: ArrayLoad", "op: TableFieldLoad", "op: TableFieldStore"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // `starts[key]` of a sparse number key is a lookup call
        // (`TABLE_GET_TABLE`), and the fields read after it go through the
        // row the call's fill makes this run's, as Cranelift fills it.
        name: "a table a lookup gives, read through its row",
        compat: "lua54",
        source: "local starts = {} for k = 1, 64 do starts[k * 8191 + 4096] = {n = k, m = k * 2} end local s = 0 for i = 1, 300000 do local record = starts[(i % 64 + 1) * 8191 + 4096] s = (s + record.n + record.m) % 1000003 end print(s)",
        ops: &["op: CallHelper", "op: TableFieldLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a lookup of a lookup, read through its row",
        compat: "lua54",
        source: "local grid = {} for a = 1, 16 do local row = {} for b = 1, 16 do row[b * 4099 + 17] = {n = a + b, m = a * b} end grid[a * 8191 + 4096] = row end local s = 0 for i = 1, 300000 do local row = grid[(i % 16 + 1) * 8191 + 4096] local cell = row[((i // 16) % 16 + 1) * 4099 + 17] s = (s + cell.n + cell.m) % 1000003 end print(s)",
        ops: &["op: CallHelper", "op: TableFieldLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "strings from a table of strings",
        compat: "lua54",
        source: "local words = {} for i = 1, 50 do words[i] = \"w\" .. i end local n = 0 for r = 1, 3000 do for i = 1, #words do local w = words[i] n = n + #w + (w:find(\"1\", 1, true) or 0) end end print(n)",
        ops: &["op: ArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // `#s` of a string the loop made is a host call, which MIR answers
        // from the run's length table, as Cranelift does.
        name: "the length of a string a helper made is read in line",
        compat: "lua54",
        source: "local total = 0 for i = 1, 200000 do local s = string.format('item=%06d value=%.3f', i, i / 7) total = total + #s end print(total)",
        ops: &["op: HostCall", "op: CallHelper"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "find and match of a pattern over short strings",
        compat: "lua51",
        source: "local subjects = { 'id 4821', 'x9', 'order 77 shipped', 'a1b22c333', 'no 0', 'ticket 123456' } local total = 0 for i = 1, 200000 do local s = subjects[i % 6 + 1] local first, last = string.find(s, '%d+') total = total + first + last total = total + #string.match(s, '[%w_]+') end print(total)",
        ops: &["op: HostCall", "op: CallHelper", "op: HelperResult"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "pack, unpack and the utf8 library",
        compat: "lua54",
        source: "local words = { 'na\\u{EF}ve', 'caf\\u{E9}', 'ascii', '\\u{1F600}ok' } local total = 0 for round = 1, 60000 do local name = words[round % 4 + 1] local record = string.pack('<I4s1', round, name) local id, label = string.unpack('<I4s1', record) local first = utf8.codepoint(label, 1) local shifted = utf8.char(first + 1, 0x20AC) total = (total + id + utf8.len(label) + first + #shifted + #record) % 1000000007 end print(total)",
        ops: &["op: HostCall", "op: CallHelper", "op: HelperResult"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a hoisted call into the VM under a side that rejoins",
        compat: "lua52",
        source: "local function f(i) error('boom') end local function h(e) if type(e) == 'string' then return 'h:' .. e end return type(e) end local _, ref = xpcall(f, h, 0) local n, bad = 0, 0 for i = 1, 100000 do local ok, e = xpcall(f, h, i) if ok then bad = bad + 1000000 elseif rawequal(e, ref) then n = n + 1 else bad = bad + 1 end end print(n, bad)",
        ops: &["op: CallHelper", "op: ReleaseStrings"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a side that rejoins after moving the array the root reads",
        compat: "lua54",
        // The side inserts into `t`, which may move its array, and stores
        // into the slot the root reads next: the root has to read the array's
        // words again after the link.
        source: "local t={} for i=1,64 do t[i]=i end local pads={} for j=1,200 do local p={} for i=1,64 do p[i]=i end pads[j]=p end local s=0 local insert=table.insert for i=1,300000 do local k=(i%64)+1 s=s+t[k] if i%50==0 then insert(t,'s') local n=(k%64)+1 t[n]=t[n]+1000 end end print(s,#t)",
        ops: &["op: ArrayLoad", "op: CallHelper"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "an integer compared with a float falls back to Cranelift",
        compat: "lua54",
        source: "local s, x = 0, 150000.5 for i = 1, 300000 do if i < x then s = s + 1 end end print(s)",
        ops: &["op: Compare"],
        fallbacks: &["comparison of an integer with a float"],
        compiled: false,
    },
];

/// The snippets that exercise the string-index probe, which MIR declines with
/// `OXIMOND_JIT_MIR_PROBE=0`.
const PROBE_SNIPPETS: &[&str] = &[
    "a string key is looked up in the index in line",
    "a string key the table lacks reads as nil in line",
    "a probe after the hash part is rebuilt reads the new index",
    "a number stored under a present string key is stored in line",
    "a key deleted and stored again is deleted and revived in line",
    "a delete and a store a guard apart keep both stores",
    "a key that may name a metamethod is stored by the helper",
    "a dense index is walked in line past its first buckets",
    "a tree that moves hash storage reads its keys through the VM",
];

struct Run {
    stdout: String,
    stderr: String,
}

/// A run of `snippet` under `backend`, with the MIR probe as the default
/// has it.
fn run(snippet: &Snippet, backend: Option<&str>) -> Run {
    run_with_probe(snippet, backend, None)
}

/// A run of `snippet` under `backend` with `OXIMOND_JIT_MIR_PROBE` set to
/// `probe`, or not set.
fn run_with_probe(snippet: &Snippet, backend: Option<&str>, probe: Option<&str>) -> Run {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
    command
        .args([
            &format!("--compat={}", snippet.compat),
            "--execution=tracing",
            "-e",
            snippet.source,
        ])
        .env("OXIMOND_JIT_SYNC", "1")
        .env("OXIMOND_JIT_TRACE", "1")
        .env("OXIMOND_JIT_TRACE_IR_OPT", "1")
        .env_remove("OXIMOND_JIT_BACKEND")
        .env_remove("OXIMOND_JIT_MIR_PROBE");
    if let Some(backend) = backend {
        command.env("OXIMOND_JIT_BACKEND", backend);
    }
    if let Some(probe) = probe {
        command.env("OXIMOND_JIT_MIR_PROBE", probe);
    }
    let output = command.output().expect("run oximond");
    assert!(output.status.success(), "{}: {output:?}", snippet.name);
    Run {
        stdout: String::from_utf8(output.stdout).unwrap(),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

#[test]
fn every_snippet_prints_the_same_under_both_backends() {
    for snippet in SNIPPETS {
        let default = run(snippet, None);
        let cranelift = run(snippet, Some("cranelift"));
        let mir = run(snippet, Some("mir"));
        assert!(!default.stdout.is_empty(), "{}: it prints", snippet.name);
        assert_eq!(default.stdout, cranelift.stdout, "{}", snippet.name);
        assert_eq!(default.stdout, mir.stdout, "{}", snippet.name);
        assert_eq!(
            default.stderr.contains("mir backend"),
            mir.stderr.contains("mir backend"),
            "{}: MIR is the default backend",
            snippet.name
        );
        assert!(
            !cranelift.stderr.contains("mir backend"),
            "{}: no MIR code runs under cranelift",
            snippet.name
        );
    }
}

/// The remainders and the values only exits read print what the interpreter
/// prints, under every profile whose modulo differs (Lua 5.1 and the floor
/// form, Lua 5.3 and 5.4 with `fmod`).
#[test]
fn modulo_and_exit_only_snippets_print_what_the_interpreter_prints() {
    let names = [
        "float modulo that calls the compat function",
        "float modulo answered in line, and by the compat function off the fast path",
        "numbers that are integers or floats at run time take the modulo of their type",
        "values only an exit reads are computed in its stub",
    ];
    for snippet in SNIPPETS.iter().filter(|s| names.contains(&s.name)) {
        for compat in ["lua51", "lua53", "lua54", "lua55"] {
            let run_in = |execution: &str, backend: Option<&str>| {
                let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
                command
                    .args([
                        &format!("--compat={compat}"),
                        &format!("--execution={execution}"),
                        "-e",
                        snippet.source,
                    ])
                    .env("OXIMOND_JIT_SYNC", "1")
                    .env_remove("OXIMOND_JIT_BACKEND");
                if let Some(backend) = backend {
                    command.env("OXIMOND_JIT_BACKEND", backend);
                }
                let output = command.output().expect("run oximond");
                assert!(
                    output.status.success(),
                    "{} under {compat}: {output:?}",
                    snippet.name
                );
                String::from_utf8(output.stdout).unwrap()
            };
            let interpreted = run_in("interpreter", None);
            assert_eq!(
                interpreted,
                run_in("tracing", Some("mir")),
                "{} under {compat}, mir",
                snippet.name
            );
            assert_eq!(
                interpreted,
                run_in("tracing", Some("cranelift")),
                "{} under {compat}, cranelift",
                snippet.name
            );
        }
    }
}

#[test]
fn the_mir_backend_compiles_what_each_snippet_exercises_and_says_why_it_declines_the_rest() {
    for snippet in SNIPPETS {
        let mir = run(snippet, Some("mir"));
        for op in snippet.ops {
            assert!(
                mir.stderr.contains(op),
                "{}: the optimized IR holds {op}",
                snippet.name
            );
        }
        let compiled = mir
            .stderr
            .lines()
            .filter(|line| line.starts_with("jit-trace: mir backend compiled tree"))
            .count();
        if snippet.compiled {
            assert!(
                compiled > 0,
                "{}: MIR compiled a tree\n{}",
                snippet.name,
                mir.stderr
            );
        }
        if snippet.name.contains("a side that rejoins") {
            // The tree with the rejoining side is a second compile, and MIR
            // made it.
            assert!(
                mir.stderr.contains("terminator: Link {"),
                "{}: a side links back",
                snippet.name
            );
            assert!(
                compiled >= 2,
                "{}: MIR compiled the tree with the side",
                snippet.name
            );
        }
        if snippet.name.contains("only an exit reads") {
            // The guard that leaves in the loop's rare iteration computes
            // the values it keeps in its stub.
            assert!(
                mir.stderr.contains("operations in exit stubs"),
                "{}: an exit stub computes values\n{}",
                snippet.name,
                mir.stderr
            );
        }
        let fallbacks: Vec<&str> = mir
            .stderr
            .lines()
            .filter(|line| line.starts_with("jit-trace: mir backend fallback to cranelift"))
            .collect();
        for line in &fallbacks {
            assert!(
                snippet.fallbacks.iter().any(|reason| line.contains(reason)),
                "{}: an unexpected fallback: {line}",
                snippet.name
            );
        }
        if !snippet.compiled {
            assert!(
                !fallbacks.is_empty() || compiled > 0,
                "{}: the MIR backend saw a tree",
                snippet.name
            );
        }
    }
}

/// The count `name` has in the statistics line of a run of `source` under
/// `backend`.
fn stat(source: &str, backend: &str, name: &str) -> u64 {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat=lua54", "--execution=tracing", "-e", source])
        .env("OXIMOND_JIT_SYNC", "1")
        .env("OXIMOND_JIT_STATS", "1")
        .env("OXIMOND_JIT_BACKEND", backend)
        .output()
        .expect("run oximond");
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    let at = stderr
        .find(&format!("{name}: "))
        .unwrap_or_else(|| panic!("no {name} in {stderr}"));
    stderr[at + name.len() + 2..]
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

/// The statistics of a run of `source` under `backend`: the exits taken, the
/// side exits and the iterations, as the line prints them.
fn exits_of(source: &str, backend: &str) -> (String, String, String, String) {
    exits_in("lua54", source, backend)
}

/// [`exits_of`] under the profile `compat`.
fn exits_in(compat: &str, source: &str, backend: &str) -> (String, String, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args([
            &format!("--compat={compat}"),
            "--execution=tracing",
            "-e",
            source,
        ])
        .env("OXIMOND_JIT_SYNC", "1")
        .env("OXIMOND_JIT_STATS", "1")
        .env("OXIMOND_JIT_BACKEND", backend)
        .output()
        .expect("run oximond");
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    let after = |name: &str, end: char| {
        let at = stderr
            .find(name)
            .unwrap_or_else(|| panic!("no {name} in {stderr}"));
        stderr[at + name.len()..]
            .split(end)
            .next()
            .unwrap()
            .to_string()
    };
    (
        String::from_utf8(output.stdout).unwrap(),
        after(", exits: [", ']'),
        after("side_exits: [", ']'),
        after("iterations: ", ','),
    )
}

/// Sides that link back to the root's loop run the same exits, side exits and
/// iterations under both backends, with a call into the VM hoisted above the
/// loop or an array a side moves.
#[test]
fn a_side_that_rejoins_the_loop_leaves_where_cranelift_leaves() {
    let hoisted = SNIPPETS
        .iter()
        .find(|snippet| snippet.name == "a hoisted call into the VM under a side that rejoins")
        .expect("the snippet is there")
        .source;
    // The root reads `t[k]`, and a side inserts into `t` (which may move its
    // array), overwrites a slot the root reads next and links back. The
    // second store is what the old array words would read wrongly.
    let moved = SNIPPETS
        .iter()
        .find(|snippet| snippet.name == "a side that rejoins after moving the array the root reads")
        .expect("the snippet is there")
        .source;
    let moved_once = "local t={} for i=1,64 do t[i]=i end local pads={} for j=1,200 do local p={} for i=1,64 do p[i]=i end pads[j]=p end local s=0 local insert=table.insert for i=1,300000 do local k=(i%64)+1 s=s+t[k] if i%50==0 then insert(t,'s') end end print(s,#t)";
    let grown = "local t={} for i=1,8 do t[i]=i end local s=0 local insert=table.insert for i=1,300000 do local k=(i%8)+1 s=s+t[k] if i%97==0 then insert(t,i) t[k]=t[k]+1 end end print(s,#t)";
    for source in [hoisted, moved, moved_once, grown] {
        let mir = exits_of(source, "mir");
        let cranelift = exits_of(source, "cranelift");
        assert_eq!(mir, cranelift, "{source}");
        let interpreted = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat=lua54", "--execution=interpreter", "-e", source])
            .output()
            .expect("run oximond");
        assert_eq!(
            String::from_utf8(interpreted.stdout).unwrap(),
            mir.0,
            "{source}"
        );
    }
}

/// The row of a table a lookup call gives is filled after the call under
/// both backends, so the loop reads the fields of this run's table: it runs
/// the same exits, side exits and iterations as under Cranelift, in a hit, a
/// miss, a record that is replaced and a lookup of a lookup.
#[test]
fn a_lookup_row_is_filled_where_cranelift_fills_it() {
    let hit = SNIPPETS
        .iter()
        .find(|snippet| snippet.name == "a table a lookup gives, read through its row")
        .expect("the snippet is there")
        .source;
    let nested = SNIPPETS
        .iter()
        .find(|snippet| snippet.name == "a lookup of a lookup, read through its row")
        .expect("the snippet is there")
        .source;
    let miss = "local starts = {} for k = 1, 64 do starts[k * 8191 + 4096] = {n = k, m = k * 2} end local s = 0 for i = 1, 300000 do local record = starts[(i % 70 + 1) * 8191 + 4096] if record then s = (s + record.n + record.m) % 1000003 else s = s + 1 end end print(s)";
    let replaced = "local starts = {} for k = 1, 64 do starts[k * 8191 + 4096] = {n = k, m = k * 2} end local s = 0 for i = 1, 300000 do if i % 20000 == 0 then starts[(i % 64 + 1) * 8191 + 4096] = {m = i, n = -i, extra = 1} end local record = starts[(i % 64 + 1) * 8191 + 4096] s = (s + record.n + record.m) % 1000003 end print(s)";
    for source in [hit, nested, miss, replaced] {
        let mir = exits_of(source, "mir");
        let cranelift = exits_of(source, "cranelift");
        assert_eq!(mir, cranelift, "{source}");
        let interpreted = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat=lua54", "--execution=interpreter", "-e", source])
            .output()
            .expect("run oximond");
        assert_eq!(
            String::from_utf8(interpreted.stdout).unwrap(),
            mir.0,
            "{source}"
        );
        // A row left unfilled leaves the loop at the first field read of
        // every entry.
        let iterations: u64 = mir.3.trim().parse().unwrap();
        assert!(iterations >= 250_000, "{iterations} iterations: {source}");
    }
}

/// A table the `next` step gives has its row filled after the helper call
/// under both backends (every table result of a helper, not only the first),
/// across many runs that reuse lanes: the sums are the interpreter's, the
/// exits and iterations equal Cranelift's, and the keys and values of a
/// `pairs` over arrays of different lengths and element types agree.
#[test]
fn the_row_of_a_table_the_next_step_gives_is_this_runs_under_both_backends() {
    let sums_of_floats = "local m = {} for i = 1, 10 do m[i] = {0.5, 0.5, 0.5, 0.5, 1000.5 + i, 0.5} end local total = 0 local function sum(fixed) local t = 0 for k, box in pairs(fixed) do t = t + box[5] * k end return t end local bad = 0 for d = 1, 400 do local fixed = {} for b = 1, 5 do fixed[b] = {0.5, 0.5, 0.5, 0.5, b + 0.25, 0.5} end if sum(fixed) ~= 56.25 then bad = bad + 1 end if d % 40 == 0 then for k = 1, 3000 do total = total + m[k % 10 + 1][5] end end end print(bad, total)";
    let several_tables = "local a, b = {}, {} for i = 1, 8 do a[i] = {i, i * 2, i * 3} b[i] = {-i, -i * 2, -i * 3} end local function sum(t) local s = 0 for _, box in pairs(t) do s = s + box[1] + box[3] end return s end local out = 0 for r = 1, 600 do out = out + sum(a) + sum(b) * 2 + sum(a) end print(out)";
    for source in [PAIRS_SOURCE, sums_of_floats, several_tables] {
        let mir = exits_of(source, "mir");
        let cranelift = exits_of(source, "cranelift");
        assert_eq!(mir, cranelift, "{source}");
        let interpreted = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat=lua54", "--execution=interpreter", "-e", source])
            .output()
            .expect("run oximond");
        assert_eq!(
            String::from_utf8(interpreted.stdout).unwrap(),
            mir.0,
            "{source}"
        );
    }
    let printed = exits_of(PAIRS_SOURCE, "mir").0;
    assert!(printed.starts_with("0\t"), "no wrong sum: {printed}");
}

#[test]
fn the_probe_leaves_the_trace_where_cranelift_leaves_it() {
    // Each program changes the table under a compiled loop: the type of a
    // value, a key gone, a table swapped for an empty one, a store of another
    // type. The loop must take the same exits the same number of times.
    for source in [
        "local keys = {'a', 'b', 'c'} local t = {a = 1, b = 2, c = 3} local s = 0 for i = 1, 300000 do if i == 200000 then t.b = 2.5 end local v = t[keys[i % 3 + 1]] s = s + v end print(s)",
        "local keys = {'a', 'b', 'c'} local t = {a = 1, b = 2, c = 3} local s = 0 for i = 1, 300000 do if i == 200000 then t.b = nil end local v = t[keys[i % 3 + 1]] if v then s = s + v end end print(s)",
        "local keys = {'a', 'b', 'c'} local t = {a = 1, b = 2, c = 3} for i = 1, 300000 do if i == 200000 then t.b = 2.5 end local k = keys[i % 3 + 1] t[k] = t[k] + 1 end print(t.a, t.b, t.c)",
        "local keys = {'a', 'b', 'c'} local t = {a = 1, b = 2, c = 3} for i = 1, 300000 do local k = keys[i % 3 + 1] t[k] = t[k] + 1 if i == 200000 then t.c = 'x' end if i > 200000 and i % 3 == 0 then t.c = 3 end end print(t.a, t.b, t.c)",
        "local t = {} local keys = {} for i = 1, 64 do keys[i] = 'm' .. i end local s = 0 for i = 1, 300000 do if i % 1000 == 0 then t = {} end local k = keys[i % 64 + 1] t[k] = nil t[k] = i % 2 == 0 if t[k] then s = s + 1 end end print(s, t.m1, t.m64)",
        "local t = {} local keys = {'__x', 'plain', '__index', 'other'} for i = 1, 300000 do local k = keys[i % 4 + 1] t[k] = nil t[k] = i * 1.0 end print(t.__x, t.plain, t.__index, t.other)",
        "local t = {} local keys = {} for i = 1, 40 do keys[i] = 's' .. i t[keys[i]] = 1 end local a = 0.5 for i = 1, 300000 do local k = keys[i % 40 + 1] t[k] = nil t[k] = i * 0.5 + a t[keys[(i + 17) % 40 + 1]] = 'w' a = a * 0.75 + 1 end print(a, t.s1, t.s40)",
    ] {
        let mir = exits_of(source, "mir");
        let cranelift = exits_of(source, "cranelift");
        assert_eq!(mir, cranelift, "{source}");
    }
}

#[test]
fn the_probe_answers_what_it_can_without_the_vm() {
    let snippet = |name: &str| {
        SNIPPETS
            .iter()
            .find(|snippet| snippet.name == name)
            .unwrap_or_else(|| panic!("no snippet {name}"))
            .source
    };
    // Lookups and stores through the index make no call to the VM, as they
    // make none under Cranelift.
    for (name, counter) in [
        (
            "a string key is looked up in the index in line",
            "hash_helper_calls",
        ),
        (
            "a string key the table lacks reads as nil in line",
            "hash_helper_calls",
        ),
        (
            "a number stored under a present string key is stored in line",
            "hash_helper_calls",
        ),
        (
            "a key deleted and stored again is deleted and revived in line",
            "delete_string_helper_calls",
        ),
        (
            "a key deleted and stored again is deleted and revived in line",
            "set_revive_helper_calls",
        ),
        (
            "a number stored under a present string key is stored in line",
            "set_present_helper_calls",
        ),
    ] {
        let source = snippet(name);
        let mir = stat(source, "mir", counter);
        let cranelift = stat(source, "cranelift", counter);
        assert!(mir <= 2000, "{name}: {counter} is {mir} under mir");
        assert_eq!(mir, cranelift, "{name}: {counter}");
    }
    // A dense index, walked past the steps made in line: the VM is called as
    // often as under Cranelift, which walks the whole sequence in line.
    let source = snippet("a dense index is walked in line past its first buckets");
    for counter in [
        "hash_helper_calls",
        "set_present_helper_calls",
        "delete_string_helper_calls",
        "set_revive_helper_calls",
    ] {
        let mir = stat(source, "mir", counter);
        assert_eq!(mir, stat(source, "cranelift", counter), "{counter}");
    }
    // A tree that moves hash storage reads through the VM under both.
    let source = snippet("a tree that moves hash storage reads its keys through the VM");
    let mir = stat(source, "mir", "hash_helper_calls");
    assert!(mir > 100_000, "{mir} lookups reached the VM");
    assert_eq!(mir, stat(source, "cranelift", "hash_helper_calls"));
}

#[test]
fn with_the_probe_off_mir_declines_the_trees_that_probe() {
    let find = |name: &str| {
        SNIPPETS
            .iter()
            .find(|snippet| snippet.name == name)
            .unwrap_or_else(|| panic!("no snippet {name}"))
    };
    for name in PROBE_SNIPPETS {
        find(name);
    }
    for (name, reason) in [
        (
            "a string key is looked up in the index in line",
            "HashLoad without the inline probe",
        ),
        (
            "a key deleted and stored again is deleted and revived in line",
            "hash store or delete without the inline probe",
        ),
    ] {
        let snippet = find(name);
        let off = run_with_probe(snippet, Some("mir"), Some("0"));
        assert!(
            off.stderr.lines().any(|line| {
                line.starts_with("jit-trace: mir backend fallback to cranelift")
                    && line.contains(reason)
            }),
            "{name}: MIR declines with {reason:?}\n{}",
            off.stderr
        );
        assert_eq!(off.stdout, run(snippet, Some("mir")).stdout, "{name}");
    }
}

#[test]
fn the_default_and_mir_runs_compile_the_same_trees() {
    // Selecting `mir` is the default, not a variant of it.
    let snippet = &SNIPPETS[0];
    let count = |run: &Run| {
        run.stderr
            .lines()
            .filter(|line| {
                line.starts_with("jit-trace: compiled trace")
                    || line.starts_with("jit-trace: mir backend compiled tree")
            })
            .count()
    };
    let default = run(snippet, None);
    assert!(default.stderr.contains("mir backend compiled tree"));
    assert_eq!(count(&default), count(&run(snippet, Some("mir"))));
    // Another value is not a backend: Cranelift compiles.
    assert!(
        !run(snippet, Some("nonsense"))
            .stderr
            .contains("mir backend")
    );
}

#[test]
fn the_icount_lines_name_the_backend_of_each_tree() {
    let snippet = &SNIPPETS[0];
    for (backend, name) in [("mir", "mir"), ("cranelift", "cranelift")] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--execution=tracing", "-e", snippet.source])
            .env("OXIMOND_JIT_SYNC", "1")
            .env("OXIMOND_JIT_ICOUNT", "1")
            .env("OXIMOND_JIT_BACKEND", backend)
            .output()
            .expect("run oximond");
        assert!(output.status.success());
        let text = String::from_utf8(output.stderr).unwrap();
        assert!(
            text.lines()
                .any(|line| line.starts_with(&format!("jit-icount: tree backend={name} "))),
            "{backend}: {text}"
        );
    }
}

/// Loops whose array keys a range plan proves. Under MIR the optimized IR
/// must hold the proven accesses and the log must say the tree's loop runs in
/// the range-proven version, which Cranelift makes too.
const RANGE_SNIPPETS: &[Snippet] = &[
    Snippet {
        name: "a counted loop over an array with integer keys",
        compat: "lua54",
        source: "local t = {} for i = 1, 100 do t[i] = i end local s = 0 for r = 1, 5000 do for i = 1, 100 do s = s + t[i] end end print(s)",
        ops: &["op: ArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a counted loop with offset and negative offset keys",
        compat: "lua54",
        source: "local t = {} for i = 1, 100 do t[i] = i end local s = 0 for r = 1, 5000 do for i = 2, 99 do s = s + t[i - 1] + t[i + 1] end end print(s)",
        ops: &["op: ArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a counted loop over an array with float keys",
        compat: "lua51",
        source: "local t = {} for i = 1, 100 do t[i] = i * 1.5 end local s = 0 for r = 1, 5000 do for i = 1, 100 do s = s + t[i] end end print(s)",
        ops: &["op: ArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // The last pass runs the proof into a failure: its limit is past the
        // array, so the loop leaves through a guard in whichever version it
        // runs.
        name: "a counted loop whose range ends past the array",
        compat: "lua54",
        source: "local t = {} for i = 1, 100 do t[i] = i end local s = 0 for r = 1, 5000 do local n = 100 if r % 1000 == 0 then n = 150 end for i = 1, n do s = s + (t[i] or 1000) end end print(s)",
        ops: &["op: ArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        name: "a counted loop that stores into an array",
        compat: "lua54",
        source: "local t = {} for i = 1, 100 do t[i] = i end local s = 0 for r = 1, 5000 do for i = 1, 100 do t[i] = t[i] + 1 s = s + t[i] end end print(s)",
        ops: &["op: ArrayLoad", "op: ArrayOverwrite"],
        fallbacks: &[],
        compiled: true,
    },
];

#[test]
fn counted_loops_with_range_proofs_run_versioned_through_mir() {
    for snippet in RANGE_SNIPPETS {
        let default = run(snippet, None);
        let cranelift = run(snippet, Some("cranelift"));
        let mir = run(snippet, Some("mir"));
        assert!(!default.stdout.is_empty(), "{}: it prints", snippet.name);
        assert_eq!(default.stdout, cranelift.stdout, "{}", snippet.name);
        assert_eq!(default.stdout, mir.stdout, "{}", snippet.name);
        for op in snippet.ops {
            assert!(mir.stderr.contains(op), "{}: {op}", snippet.name);
        }
        assert!(
            mir.stderr.contains("range-proven loop version"),
            "{}: MIR versioned the loop\n{}",
            snippet.name,
            mir.stderr
        );
        assert!(
            !mir.stderr.contains("mir backend panicked"),
            "{}",
            snippet.name
        );
        for line in mir
            .stderr
            .lines()
            .filter(|line| line.starts_with("jit-trace: mir backend fallback to cranelift"))
        {
            assert!(
                ["root that links to sides compiled apart"]
                    .iter()
                    .chain(snippet.fallbacks)
                    .any(|reason| line.contains(reason)),
                "{}: an unexpected fallback: {line}",
                snippet.name
            );
        }
        assert!(
            !cranelift.stderr.contains("range-proven loop version"),
            "{}",
            snippet.name
        );
    }
}

/// Loops over arrays whose fast version closes on one compare, and loops
/// that must not take that form. The bound lies past the array, between two
/// keys, below the start; the array ends on a hole, grows while the loop
/// runs, or is another table on each run; the loop runs downward, by two, by
/// a step in a variable; a `while` loop stores after its bound test, so its
/// guard stays in the body. Each prints the interpreter's output under every
/// profile, where the induction is a float (5.1, `LuaJIT`) or an integer.
const FOLD_SNIPPETS: &[(&str, &str)] = &[
    (
        "a bound past the array reads nil where the array ends",
        "local t = {} for i = 1, 100 do t[i] = i end local s, n = 0, 0 for r = 1, 400 do for i = 1, 150 do local v = t[i] if v then s = s + v else n = n + 1 end end end print(s, n)",
    ),
    (
        "a bound between two keys",
        "local t = {} for i = 1, 100 do t[i] = i * 0.5 end local s = 0 for r = 1, 400 do for i = 1, 60.5 do s = s + t[i] end end print(s)",
    ),
    (
        "a downward loop that ends above the first key",
        "local t = {} for i = 1, 100 do t[i] = i end local s = 0 for r = 1, 400 do for i = 100, 2.5, -1 do s = s + t[i] end end print(s)",
    ),
    (
        "a downward loop that walks below the first key",
        "local t = {} for i = 1, 100 do t[i] = i end local s, n = 0, 0 for r = 1, 400 do for i = 100, -20, -1 do local v = t[i] if v then s = s + v else n = n + 1 end end end print(s, n)",
    ),
    (
        "two keys an iteration, by two, to an odd end",
        "local t = {} for i = 1, 101 do t[i] = i end local s, n = 0, 0 for r = 1, 400 do for i = 1, 120, 2 do local a, b = t[i], t[i + 1] if a then s = s + a end if b then s = s - b else n = n + 1 end end end print(s, n)",
    ),
    (
        "a step in a variable, and arrays of other lengths",
        "local ts = {} for j = 1, 5 do local t = {} for i = 1, 20 * j do t[i] = i + j end ts[j] = t end local s, n = 0, 0 for r = 1, 3000 do local t, step = ts[r % 5 + 1], r % 3 + 1 for i = 1, 70, step do local v = t[i] if v then s = s + v else n = n + 1 end end end print(s, n)",
    ),
    (
        "a loop that grows the array it reads",
        "local s = 0 for r = 1, 300 do local t = {1} for i = 1, 120 do t[i + 1] = t[i] + i s = s + t[i] end end print(s)",
    ),
    (
        "an array with a hole in the middle",
        "local t = {} for i = 1, 100 do t[i] = i end t[50] = nil local s, n = 0, 0 for r = 1, 400 do for i = 1, 100 do local v = t[i] if v then s = s + v else n = n + 1 end end end print(s, n)",
    ),
    (
        "a loop that never runs, and one of a single iteration",
        "local t = {} for i = 1, 100 do t[i] = i end local s = 0 for r = 1, 3000 do local last = r % 100 == 0 and 0 or (r % 7 == 0 and 1 or 30) for i = 1, last do s = s + t[i] end end print(s)",
    ),
    (
        "a while loop that stores after its bound test",
        "local t = {} for i = 1, 100 do t[i] = i end local s = 0 for r = 1, 400 do local i = 1 while i <= 120 do local v = t[i] if v then t[i] = v + 1 s = s + v end i = i + 1 end end print(s)",
    ),
    (
        "a bound that moves in the loop",
        "local t = {} for i = 1, 200 do t[i] = i end local s = 0 for r = 1, 400 do local i, n = 1, 50 while i <= n do s = s + t[i] if i % 25 == 0 and n < 150 then n = n + 10 end i = i + 1 end end print(s)",
    ),
    (
        // Each entry of the inner loop comes from the side that runs the
        // outer one, with a bound that gives every trip count in turn.
        "inner loops of every trip count from none to twelve",
        "local t = {} for i = 1, 12 do t[i] = i * 3 + 1 end local counts = {} local total = 0 for outer = 1, 6000 do local n = outer % 13 local s = 0 for i = 1, n do s = s + t[i] * outer end counts[n + 1] = (counts[n + 1] or 0) + s % 7 total = (total + s) % 1000003 end print(total, table.concat(counts, ' '))",
    ),
    (
        "downward and stepped inner loops of every trip count",
        "local t = {} for i = 1, 30 do t[i] = i * 5 - 2 end local total = 0 for outer = 1, 5000 do local n = outer % 13 local s = 0 for i = n, 1, -1 do s = s + t[i] end for i = 2, 2 * n, 2 do s = s + t[i] * 3 end for i = 25, 26 - 2 * n, -2 do s = s - t[i] end total = (total + s * outer) % 1000003 end print(total)",
    ),
    (
        // The bound lies past the array for some entries: those leave the
        // fast loop at the end of the array and read nil in the guarded one.
        "short inner loops whose bound runs past the array",
        "local t = {} for i = 1, 6 do t[i] = i end local total, holes = 0, 0 for outer = 1, 6000 do local n = outer % 9 for i = 1, n do local v = t[i] if v then total = total + v * outer % 11 else holes = holes + 1 end end end print(total, holes)",
    ),
    (
        // Eight prototypes through one call site: a chain of sides, each
        // taking what the root keeps for its exits, and a ninth callee that
        // leaves the tree now and then.
        "closures of eight prototypes rotated through one call site",
        "local c1, c2, c3, c4, c5, c6, c7, c8 = 0, 0, 0, 0, 0, 0, 0, 0 local function p1(x) c1 = c1 + 1 return (x + c1) % 1009 end local function p2(x) c2 = c2 + 1 return (x * 2 + c2) % 1009 end local function p3(x) c3 = c3 + 1 return (x * 3 + c3) % 1009 end local function p4(x) c4 = c4 + 1 return (x * 4 + c4) % 1009 end local function p5(x) c5 = c5 + 1 return (x * 5 + c5) % 1009 end local function p6(x) c6 = c6 + 1 return (x * 6 + c6) % 1009 end local function p7(x) c7 = c7 + 1 return (x * 7 + c7) % 1009 end local function p8(x) c8 = c8 + 1 return (x * 8 + c8) % 1009 end local odd = setmetatable({}, { __call = function(_, x) return x % 5 end }) local g1, g2, g3, g4, g5, g6, g7, g8 = p1, p2, p3, p4, p5, p6, p7, p8 local total = 0 for i = 1, 40000 do total = (total + g1(i)) % 1000003 g1, g2, g3, g4, g5, g6, g7, g8 = g2, g3, g4, g5, g6, g7, g8, g1 if i % 9973 == 0 then g3 = odd elseif i % 9973 == 20 then g1 = p3 end end print(total, c1, c2, c3, c4, c5, c6, c7, c8)",
    ),
];

#[test]
fn loops_that_close_on_one_compare_print_what_the_interpreter_prints() {
    for &(name, source) in FOLD_SNIPPETS {
        for compat in ["lua51", "lua53", "lua54", "lua55", "luajit21"] {
            let run_in = |execution: &str, backend: Option<&str>| {
                let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
                command
                    .args([
                        &format!("--compat={compat}"),
                        &format!("--execution={execution}"),
                        "-e",
                        source,
                    ])
                    .env("OXIMOND_JIT_SYNC", "1")
                    .env_remove("OXIMOND_JIT_BACKEND");
                if let Some(backend) = backend {
                    command.env("OXIMOND_JIT_BACKEND", backend);
                }
                let output = command.output().expect("run oximond");
                assert!(output.status.success(), "{name} under {compat}: {output:?}");
                String::from_utf8(output.stdout).unwrap()
            };
            let interpreted = run_in("interpreter", None);
            assert!(!interpreted.is_empty(), "{name}: it prints");
            for backend in [None, Some("mir"), Some("cranelift")] {
                assert_eq!(
                    interpreted,
                    run_in("tracing", backend),
                    "{name} under {compat}, {backend:?}"
                );
            }
        }
    }
}

/// `modulo_matrix.lua` takes the remainder of every pair of signs and
/// subtypes, with zero divisors, infinities, NaN and the ends of the integer
/// range, in loops compiled on ordinary operands first. Each profile has its
/// own remainder, and each backend must print the interpreter's.
#[test]
fn every_remainder_prints_what_the_interpreter_prints_under_each_profile() {
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/modulo_matrix.lua");
    for compat in ["lua51", "lua53", "lua54", "lua55", "luajit21"] {
        let run_in = |execution: &str, backend: Option<&str>| {
            let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
            command
                .args([
                    &format!("--compat={compat}"),
                    &format!("--execution={execution}"),
                    script,
                ])
                .env("OXIMOND_JIT_SYNC", "1")
                .env("OXIMOND_JIT_TRACE", "1")
                .env_remove("OXIMOND_JIT_BACKEND");
            if let Some(backend) = backend {
                command.env("OXIMOND_JIT_BACKEND", backend);
            }
            let output = command.output().expect("run oximond");
            assert!(output.status.success(), "under {compat}: {output:?}");
            Run {
                stdout: String::from_utf8(output.stdout).unwrap(),
                stderr: String::from_utf8(output.stderr).unwrap(),
            }
        };
        let interpreted = run_in("interpreter", None).stdout;
        assert!(
            interpreted.lines().count() > 70 && interpreted.contains("nan"),
            "under {compat}: every loop prints"
        );
        for backend in [None, Some("mir"), Some("cranelift")] {
            let traced = run_in("tracing", backend);
            let (mut wanted, mut got) = (interpreted.lines(), traced.stdout.lines());
            loop {
                let (wanted, got) = (wanted.next(), got.next());
                assert_eq!(wanted, got, "under {compat}, {backend:?}");
                if wanted.is_none() {
                    break;
                }
            }
            if backend != Some("cranelift") {
                let compiled = traced.stderr.matches("mir backend compiled tree").count();
                assert!(
                    compiled > 50,
                    "under {compat}, {backend:?}: {compiled} trees went through MIR"
                );
            }
        }
    }
}

// ---- Stage 8, arrays -----------------------------------------------------------------------

/// Array steps and the loops around them: a `pairs` loop over a list steps
/// along its array part (`ArrayNext`), as Mineclonia's `_mcl_autogroup`
/// digtime loop and the lists of tables its pathfinding walks do. MIR must
/// compile each tree itself, with no reason to hand it back.
const ARRAY_SNIPPETS: &[Snippet] = &[
    Snippet {
        // The step finds the next slot with a value, and the key's slot is the
        // slot the load reads.
        name: "pairs over a list of numbers steps along its array part",
        compat: "lua54",
        source: "local t = {} for i = 1, 100 do t[i] = i * 0.5 end local total = 0 for r = 1, 3000 do for _, v in pairs(t) do total = total + v end end print(total)",
        ops: &["op: ArrayNext", "op: ArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // `_mcl_autogroup`: the values of a list are read in a loop that
        // calls `table.insert` on another list.
        name: "pairs over a list in a loop that inserts into another list",
        compat: "lua51",
        source: "local hardness = {} for i = 1, 40 do hardness[i] = i * 0.25 end local function digtimes(speed) local out = {} for _, h in pairs(hardness) do local d = (h or 0) / speed if d <= 0.05 then d = 0 else d = math.ceil(d * 20) / 20 end table.insert(out, d) end return out end local total = 0 for r = 1, 2000 do local out = digtimes(1 + r % 5) total = total + out[1] + out[#out] end print(total)",
        ops: &["op: ArrayNext"],
        fallbacks: &["numeric operation that needs a conversion"],
        compiled: true,
    },
    Snippet {
        // Lists of other lengths in one loop: the array words of every run
        // are the run's own.
        name: "lists of other lengths and a list of holes walked by one loop",
        compat: "lua54",
        source: "local function sum(list) local t = 0 for _, v in pairs(list) do t = t + v end return t end local total = 0 for r = 1, 600 do local dense, holey = {}, {} local n = r % 37 + 1 for i = 1, n do dense[i] = i * 0.5 if i % 3 ~= 0 then holey[i] = i * 2 end end total = total + sum(dense) + sum(holey) end print(total)",
        ops: &["op: ArrayNext"],
        fallbacks: &[],
        compiled: true,
    },
    Snippet {
        // `gwp_compute_fixed_ground_height` over lists of other lengths: each
        // run fills the lane rows of its own tables.
        name: "a list of tables walked with pairs, each read through its row",
        compat: "lua54",
        source: "local m = {} for i = 1, 10 do m[i] = {1, 2, 3, 4, 1000 + i, 6} end local function sum(list) local t = 0 for _, box in pairs(list) do t = t + box[5] + box[1] end return t end local bad, total = 0, 0 for d = 1, 400 do local list = {} for b = 1, d % 6 + 1 do list[b] = {0, 0, 0, 0, b, 0} end local want = 0 for b = 1, d % 6 + 1 do want = want + b end if sum(list) ~= want then bad = bad + 1 end if d % 40 == 0 then for k = 1, 3000 do total = total + m[k % 10 + 1][5] end end end print(bad, total)",
        ops: &["op: LaneArrayLoad"],
        fallbacks: &[],
        compiled: true,
    },
];

#[test]
fn array_steps_run_through_mir_as_they_run_through_cranelift() {
    for snippet in ARRAY_SNIPPETS {
        let default = run(snippet, None);
        let cranelift = run(snippet, Some("cranelift"));
        let mir = run(snippet, Some("mir"));
        assert!(!default.stdout.is_empty(), "{}: it prints", snippet.name);
        assert_eq!(default.stdout, cranelift.stdout, "{}", snippet.name);
        assert_eq!(default.stdout, mir.stdout, "{}", snippet.name);
        for op in snippet.ops {
            assert!(mir.stderr.contains(op), "{}: {op}", snippet.name);
        }
        assert!(
            mir.stderr
                .lines()
                .any(|line| line.starts_with("jit-trace: mir backend compiled tree")),
            "{}: MIR compiled a tree\n{}",
            snippet.name,
            mir.stderr
        );
        for line in mir
            .stderr
            .lines()
            .filter(|line| line.starts_with("jit-trace: mir backend fallback to cranelift"))
        {
            assert!(
                !line.contains("ArrayNext"),
                "{}: the array step is lowered: {line}",
                snippet.name
            );
            assert!(
                ["root that links to sides compiled apart"]
                    .iter()
                    .chain(snippet.fallbacks)
                    .any(|reason| line.contains(reason)),
                "{}: an unexpected fallback: {line}",
                snippet.name
            );
        }
        assert!(
            !cranelift.stderr.contains("mir backend"),
            "{}",
            snippet.name
        );
    }
}

/// The native step of a boxed `next` loop (the attempt along the array part
/// and along the hash part's snapshot before the helper's call) gives up
/// where Cranelift's gives up: the same exits, side exits and iterations in a
/// list with holes, integer keys of the hash part, a table with a string key
/// among the numbers, a value of another type, a slot cleared and a key added
/// during the traversal, and loops that nest.
#[test]
fn a_native_next_step_gives_up_where_cranelift_gives_up() {
    let programs = [
        // The keys and values of a list with holes.
        "local function walk(t) local s = 0 for k, v in pairs(t) do s = s + k * v end return s end local out = 0 for r = 1, 3000 do local t = {} for i = 1, r % 19 + 3 do if i % 4 ~= 0 then t[i] = i + 0.5 end end out = out + walk(t) end print(out)",
        // The key alone.
        "local function walk(t) local s = 0 for k in pairs(t) do s = s + k end return s end local out = 0 for r = 1, 3000 do local t = {} for i = 1, r % 19 + 3 do if i % 4 ~= 0 then t[i] = true end end out = out + walk(t) end print(out)",
        // Integer keys of the hash part.
        "local t = {} for i = 1, 60 do t[i * 100] = i end local s = 0 for r = 1, 4000 do for k, v in pairs(t) do s = s + k + v end end print(s)",
        // A string key among the numbers: the step gives up there.
        "local t = {} for i = 1, 30 do t[i] = i end t.name = 'x' t.other = 5 local s = 0 for r = 1, 4000 do for k, v in pairs(t) do if type(k) == 'number' then s = s + k + v end end end print(s)",
        // A value of another type than the loop saw, now and then.
        "local function walk(t) local s = 0 for k, v in pairs(t) do if v ~= 'z' then s = s + k + v else s = s - 1 end end return s end local out = 0 for r = 1, 3000 do local t = {} for i = 1, 20 do t[i] = i end if r % 100 == 0 then t[7] = 'z' end out = out + walk(t) end print(out)",
        // A slot cleared during the traversal.
        "local function walk(t) local s = 0 for k, v in pairs(t) do s = s + v t[k] = nil end return s end local out = 0 for r = 1, 3000 do local t = {} for i = 1, 15 do t[i] = i end out = out + walk(t) end print(out)",
        // Nested loops over lists that have holes.
        "local a, b = {}, {} for i = 1, 12 do if i % 3 ~= 0 then a[i] = i end b[i * 2] = i end local s = 0 for r = 1, 600 do for ka, va in pairs(a) do for kb, vb in pairs(b) do s = s + ka * vb + va end end end print(s)",
        // The value is declared and never read: the step keeps it in a lane
        // of the bank, and gives up where the value is not a number or a
        // boolean.
        "local t = {} for i = 1, 40 do if i % 5 ~= 0 then t[i] = i * 0.25 end end t[12] = 'x' local u = {} for i = 1, 50 do u[i * 64] = i end local s = 0 for r = 1, 4000 do for k, v in pairs(t) do s = s + k end for k, v in pairs(u) do s = s + k end end print(s)",
        // Floats as the numbers of the array part (Lua 5.1 keys).
        "local t = {} for i = 1, 40 do if i % 5 ~= 0 then t[i] = i * 0.25 end end local s = 0 for r = 1, 4000 do for k, v in pairs(t) do s = s + k * v end end print(s)",
    ];
    for source in programs {
        for compat in ["lua51", "lua54"] {
            let run_under = |backend: &str| {
                let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
                    .args([
                        &format!("--compat={compat}"),
                        "--execution=tracing",
                        "-e",
                        source,
                    ])
                    .env("OXIMOND_JIT_SYNC", "1")
                    .env("OXIMOND_JIT_STATS", "1")
                    .env("OXIMOND_JIT_BACKEND", backend)
                    .output()
                    .expect("run oximond");
                assert!(output.status.success(), "{output:?}");
                let stderr = String::from_utf8(output.stderr).unwrap();
                let after = |name: &str, end: char| {
                    let at = stderr
                        .find(name)
                        .unwrap_or_else(|| panic!("no {name} in {stderr}"));
                    stderr[at + name.len()..]
                        .split(end)
                        .next()
                        .unwrap()
                        .to_string()
                };
                (
                    String::from_utf8(output.stdout).unwrap(),
                    after(", exits: [", ']'),
                    after("side_exits: [", ']'),
                    after("iterations: ", ','),
                )
            };
            let mir = run_under("mir");
            let cranelift = run_under("cranelift");
            assert_eq!(mir, cranelift, "{compat}: {source}");
            let interpreted = Command::new(env!("CARGO_BIN_EXE_oximond"))
                .args([
                    &format!("--compat={compat}"),
                    "--execution=interpreter",
                    "-e",
                    source,
                ])
                .output()
                .expect("run oximond");
            assert_eq!(
                String::from_utf8(interpreted.stdout).unwrap(),
                mir.0,
                "{compat}: {source}"
            );
        }
    }
}

// ---- Stage 8, links: direct links to the loop header -----------------------------------

/// Counted loops over an array whose body takes a branch that merges back into
/// the loop (a side that rejoins it), many times, with the range proof
/// holding and failing. A link that steps like the loop goes straight to the
/// fast loop; the output must equal the other backends'.
const LINK_SNIPPETS: &[(&str, &str, &str)] = &[
    (
        "a branch every few iterations of a proven loop",
        "lua54",
        "local t = {} for i = 1, 200 do t[i] = i % 5 == 0 and i + 0.5 or i end local s = 0 for r = 1, 4000 do for i = 1, 200 do local v = t[i] if v == math.floor(v) then s = s + v else s = s + 2 * v end end end print(s)",
    ),
    (
        "a branch every other iteration with float keys",
        "lua51",
        "local t = {} for i = 1, 100 do t[i] = i * 1.5 end local s = 0 for r = 1, 4000 do for i = 1, 100 do local v = t[i] if i % 2 == 0 then s = s + v else s = s - v end end end print(s)",
    ),
    (
        "a branch with a proof that fails on the last pass",
        "lua54",
        "local t = {} for i = 1, 100 do t[i] = i end local s = 0 for r = 1, 4000 do local n = 100 if r % 1000 == 0 then n = 130 end for i = 1, n do local v = t[i] or 7 if v % 3 == 0 then s = s + 2 * v else s = s + v end end end print(s)",
    ),
];

#[test]
fn links_that_step_like_the_loop_print_what_the_other_backends_print() {
    for &(name, compat, source) in LINK_SNIPPETS {
        let snippet = Snippet {
            name,
            compat,
            source,
            ops: &[],
            fallbacks: &[],
            compiled: true,
        };
        let default = run(&snippet, None);
        let cranelift = run(&snippet, Some("cranelift"));
        let mir = run(&snippet, Some("mir"));
        assert!(!default.stdout.is_empty(), "{name}: it prints");
        assert_eq!(default.stdout, cranelift.stdout, "{name}");
        assert_eq!(default.stdout, mir.stdout, "{name}");
        assert!(!mir.stderr.contains("mir backend panicked"), "{name}");
        assert!(
            !mir.stderr.contains("produced an invalid function"),
            "{name}\n{}",
            mir.stderr
        );
    }
}

// ---- stage 8, numbers --------------------------------------------------------------------

/// Snippets of the numeric operations that convert, and of numbers that are
/// an integer or a float at run time. Each must compile through MIR with no
/// fallback but the root that links to sides.
const NUMBER_SNIPPETS: &[(Snippet, &[&str])] = &[
    (
        Snippet {
            // The hash of a chunk position in Mineclonia (`struct_hash`) mixes
            // two coordinates with the `bit` library, which reads a float as its
            // low word by the rounding trick.
            name: "the bit library reads floats by the rounding trick",
            compat: "luajit21",
            source: "local bit = require('bit') local acc = 0 for i = 1, 200000 do local h = bit.bxor(bit.band(i * 73856093, 0xffff), bit.band(i * 19349663, 0xfff)) acc = acc + bit.rshift(h, 2) end print(acc)",
            ops: &["TrickBits"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["luajit21"],
    ),
    (
        Snippet {
            name: "the absolute value of an integer and a floor that gives an integer",
            compat: "lua54",
            source: "local acc = 0 for i = 1, 200000 do local v = (i % 97) - 48 acc = acc + math.abs(v) + math.floor(i / 7) end print(acc)",
            ops: &["Abs", "FloatToInteger", "Floor"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
    (
        Snippet {
            // `math.abs` of the integer minimum wraps to itself, and of a float
            // clears the sign.
            name: "the absolute value of the integer minimum and of negative zero",
            compat: "lua54",
            source: "local lo, acc = math.mininteger, 0 for i = 1, 100000 do local a = math.abs(lo + (i % 3)) local f = math.abs(-0.0 * (i % 2)) acc = acc + a % 7 + f end print(acc, math.abs(lo))",
            ops: &["Abs"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua53", "lua54", "lua55"],
    ),
    (
        Snippet {
            // Mineclonia shifts by a distance it computes (the hash of a
            // position); the distances include negative ones and 64 and more.
            name: "shifts by a distance known at run time",
            compat: "lua54",
            source: "local acc = 0 for i = 1, 200000 do local d = (i % 131) - 66 acc = acc ~ ((i * 2654435761) << d) ~ ((i * 40503) >> (d + 3)) end print(acc)",
            ops: &["ShiftLeft", "ShiftRight"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua53", "lua54", "lua55"],
    ),
    (
        Snippet {
            // `TABLE_NEW` leaves the address of the table's slots and each field
            // is stored where it is; the rows of tables kept in `keep` are read
            // after 100000 tables were made.
            name: "the fields of a table the loop makes are stored in line",
            compat: "luajit21",
            source: "local keep = {} local acc = 0 for i = 1, 100000 do local p = {x = i, y = i * 0.5, ok = (i % 2 == 0)} keep[i % 16 + 1] = p acc = acc + p.x + p.y if p.ok then acc = acc + 1 end end local s = 0 for i = 1, 16 do s = s + keep[i].x end print(acc, s)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["luajit21"],
    ),
    (
        Snippet {
            // A short list of numbers and a boolean, which the helper makes with
            // its slots and the loop fills.
            name: "a short list the loop makes is filled in line",
            compat: "luajit21",
            source: "local keep = {} local acc = 0 for i = 1, 100000 do local q = {i, i * 0.5, i % 2 == 0} keep[i % 16 + 1] = q acc = acc + q[1] + q[2] if q[3] then acc = acc + 1 end end local s = 0 for i = 1, 16 do s = s + keep[i][1] end print(acc, s)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["luajit21"],
    ),
    (
        Snippet {
            // `Words::Bit32Floats`: a float operand goes through a guard and the
            // rounding trick, and a float count through the conversion.
            name: "bit32 reads a float operand and a float count by the rounding trick",
            compat: "lua52",
            source: "local acc = 0 for i = 1, 200000 do  local x = i * 1.5  local n = (i % 5) + 0.0  acc = acc + bit32.band(x * 7, 0xffff) + bit32.rshift(i * 13, n) + bit32.lshift(x, n) + bit32.bxor(x, i) end print(acc)",
            ops: &["TrickBits", "FloatToInteger"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua52"],
    ),
    (
        Snippet {
            // `Words::Bit32Integers` in Lua 5.3: a float operand or count must be
            // integral, so the recorder checks it with a floor and converts it.
            name: "bit32 reads an integral float operand and count through a conversion",
            compat: "lua53",
            source: "local acc = 0 for i = 1, 200000 do  local x = i * 2.0  local n = (i % 5) + 0.0  acc = acc + bit32.band(x * 7, 0xffff) + bit32.rshift(i * 13, n) + bit32.lshift(x, n) + bit32.bxor(x, i) end print(acc)",
            ops: &["FloatToInteger", "Floor"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua52", "lua53"],
    ),
    (
        Snippet {
            name: "floor, ceiling and absolute value in Lua 5.1 and LuaJIT",
            compat: "lua51",
            source: "local acc = 0 for i = 1, 200000 do  local v = (i % 97) - 48  acc = acc + math.abs(v) + math.floor(i / 7) + math.abs(i * -0.5) + math.ceil(i / 3) end print(acc)",
            ops: &["Abs", "Floor", "Ceil"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
    (
        Snippet {
            name: "a pairs loop over an array part and a hash part steps in line",
            compat: "lua54",
            source: "local n, c = 0, 0 local t = {} for i = 1, 60 do t[i] = i end for i = 1, 60 do t['k' .. i] = i + 100 end for r = 1, 4000 do for k, v in pairs(t) do n = n + v c = c + 1 end end print(n, c)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
    (
        Snippet {
            name: "a pairs loop over an array with holes steps in line",
            compat: "lua54",
            source: "local n,c=0,0 local t={} for i=1,90 do t[i]=i*2 end for i=1,90,4 do t[i]=nil end t.x=7 t.y=9 for r=1,4000 do for k,v in pairs(t) do n=n+v c=c+1 end end print(n,c)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
    (
        Snippet {
            name: "a pairs loop over float values steps in line",
            compat: "lua51",
            source: "local n,c=0,0 local t={} for i=1,50 do t[i]=i+0.25 end t.a=1.5 t.b=2.5 for r=1,4000 do for k,v in pairs(t) do n=n+v c=c+1 end end print(n,c)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
    (
        Snippet {
            name: "a pairs loop over boolean values steps in line",
            compat: "lua54",
            source: "local n,c=0,0 local t={} for i=1,50 do t[i]=(i%3==0) end t.a=true t.b=false for r=1,4000 do for k,v in pairs(t) do if v then n=n+1 end c=c+1 end end print(n,c)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
    (
        Snippet {
            name: "a pairs loop that reads only the key steps in line",
            compat: "lua54",
            source: "local n,c=0,0 local t={} for i=1,40 do t[i]=i end for i=1,40 do t[\"s\"..i]=i end for r=1,4000 do for k in pairs(t) do if type(k)==\"number\" then n=n+k end c=c+1 end end print(n,c)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
    (
        Snippet {
            name: "a pairs loop that never reads the value steps in line",
            compat: "lua53",
            source: "local n,c=0,0 local t={} for i=1,50 do t[i]=i end t.a=1 t.b=2 for r=1,4000 do for k,v in pairs(t) do c=c+1 end end print(n,c)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
    (
        Snippet {
            name: "a next loop steps in line",
            compat: "lua55",
            source: "local n,c=0,0 local t={} for i=1,50 do t[i]=i end t.a=1 t.b=2 for r=1,4000 do for k,v in next,t do n=n+v c=c+1 end end print(n,c)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
    (
        Snippet {
            // The lanes and the snapshot of the first run are reused by the next, which
            // has another array and hash part, so a slot the step leaves must not stay.
            name: "a pairs loop that takes several tables of different sizes in turn",
            compat: "lua54",
            source: "local ts={} for j=1,4 do local t={} for i=1,j*13 do t[i]=(i%2==0) and i+0.5 or i end for i=1,j*5 do t[\"z\"..i]=j+i end ts[j]=t end local n,c=0,0 for r=1,6000 do local t=ts[r%4+1] for k,v in pairs(t) do n=n+v c=c+1 end end print(n,c)",
            ops: &["CallHelper"],
            fallbacks: &["root that links to sides compiled apart"],
            compiled: true,
        },
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    ),
];

#[test]
fn numeric_conversions_and_numbers_of_either_subtype_run_through_mir() {
    for (snippet, _) in NUMBER_SNIPPETS {
        let default = run(snippet, None);
        let cranelift = run(snippet, Some("cranelift"));
        let mir = run(snippet, Some("mir"));
        assert!(!default.stdout.is_empty(), "{}: it prints", snippet.name);
        assert_eq!(default.stdout, cranelift.stdout, "{}", snippet.name);
        assert_eq!(default.stdout, mir.stdout, "{}", snippet.name);
        for op in snippet.ops {
            assert!(mir.stderr.contains(op), "{}: {op}", snippet.name);
        }
        let compiled = mir
            .stderr
            .lines()
            .filter(|line| line.starts_with("jit-trace: mir backend compiled tree"))
            .count();
        assert!(compiled > 0, "{}: MIR compiled a tree", snippet.name);
        for line in mir
            .stderr
            .lines()
            .filter(|line| line.starts_with("jit-trace: mir backend fallback to cranelift"))
        {
            assert!(
                snippet.fallbacks.iter().any(|reason| line.contains(reason)),
                "{}: an unexpected fallback: {line}",
                snippet.name
            );
        }
    }
}

/// The conversions and the steps print what the interpreter prints, under
/// MIR and under Cranelift, in every profile the snippet lists: those that
/// have the library function it calls, and the integer or float form of the
/// numbers they record.
#[test]
fn numeric_conversions_print_what_the_interpreter_prints() {
    for (snippet, profiles) in NUMBER_SNIPPETS {
        assert!(
            !profiles.is_empty(),
            "{}: it lists its profiles",
            snippet.name
        );
        for compat in *profiles {
            let run_in = |execution: &str, backend: Option<&str>| {
                let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
                command
                    .args([
                        &format!("--compat={compat}"),
                        &format!("--execution={execution}"),
                        "-e",
                        snippet.source,
                    ])
                    .env("OXIMOND_JIT_SYNC", "1")
                    .env_remove("OXIMOND_JIT_BACKEND");
                if let Some(backend) = backend {
                    command.env("OXIMOND_JIT_BACKEND", backend);
                }
                let output = command.output().expect("run oximond");
                assert!(
                    output.status.success(),
                    "{} under {compat}: {output:?}",
                    snippet.name
                );
                String::from_utf8(output.stdout).unwrap()
            };
            let interpreted = run_in("interpreter", None);
            for backend in ["mir", "cranelift"] {
                assert_eq!(
                    interpreted,
                    run_in("tracing", Some(backend)),
                    "{} under {compat}, {backend}",
                    snippet.name
                );
            }
        }
    }
}

/// A side entered with a number of run-time subtype that rejoins the loop is
/// compiled by MIR with the rest of its tree (the second tree of
/// `opt/mod_mixed`), and the output is Cranelift's.
#[test]
fn a_rejoining_side_that_takes_a_number_of_either_subtype_runs_through_mir() {
    let snippet = SNIPPETS
        .iter()
        .find(|s| {
            s.name
                .starts_with("numbers that are integers or floats at run time")
        })
        .expect("the snippet is there");
    let mir = run(snippet, Some("mir"));
    let cranelift = run(snippet, Some("cranelift"));
    assert_eq!(mir.stdout, cranelift.stdout);
    assert!(
        mir.stderr.contains("terminator: Link {"),
        "a side links back\n{}",
        mir.stderr
    );
    assert!(
        !mir.stderr
            .contains("register that is a float or an integer at run time"),
        "{}",
        mir.stderr
    );
    let compiled = mir
        .stderr
        .lines()
        .filter(|line| line.starts_with("jit-trace: mir backend compiled tree"))
        .count();
    assert!(
        compiled >= 2,
        "MIR compiled the tree with the side\n{}",
        mir.stderr
    );
}

/// Loops under a count hook (counted trees, which carry the hook's budget
/// beside the fuel): each records the value of `acc` every time the hook
/// fires, so a hook that fires one instruction early or late prints another
/// list. The second has a side that rejoins with a cost of its own, the
/// third leaves through a break, the fourth is a range-proven loop over an
/// array, the fifth inlines calls, the last fires the hook every few
/// iterations.
const HOOK_SNIPPETS: &[(&str, &str)] = &[
    (
        "a counted loop",
        "local n, acc, seen = 0, 0, {} debug.sethook(function() n = n + 1 if #seen < 60 then seen[#seen + 1] = acc end end, '', 997) for i = 1, 100000 do acc = acc + i % 7 end debug.sethook() print(n, acc, table.concat(seen, ','))",
    ),
    (
        "a counted loop with a side that rejoins",
        "local n, acc, seen = 0, 0, {} debug.sethook(function() n = n + 1 if #seen < 60 then seen[#seen + 1] = acc end end, '', 997) for i = 1, 100000 do if i % 3 == 0 then acc = acc + 2 + i % 5 else acc = acc - 1 end end debug.sethook() print(n, acc, table.concat(seen, ','))",
    ),
    (
        "a counted loop that breaks",
        "local n, acc, seen = 0, 0, {} debug.sethook(function() n = n + 1 if #seen < 60 then seen[#seen + 1] = acc end end, '', 1013) for i = 1, 100000 do acc = acc + i % 5 if acc > 150000 then break end end debug.sethook() print(n, acc, table.concat(seen, ','))",
    ),
    (
        "a counted range-proven loop",
        "local t = {} for i = 1, 200 do t[i] = i * 0.5 end local n, acc, seen = 0, 0, {} debug.sethook(function() n = n + 1 if #seen < 60 then seen[#seen + 1] = acc end end, '', 991) for r = 1, 400 do for i = 1, #t do acc = acc + t[i] end end debug.sethook() print(n, acc, table.concat(seen, ','))",
    ),
    (
        "a counted loop with inlined calls",
        "local function f1(a) return a + 1 end local function f4(a, b, c, d) return a + b - c + d end local n, acc, seen = 0, 0, {} debug.sethook(function() n = n + 1 if #seen < 60 then seen[#seen + 1] = acc end end, '', 1009) for i = 1, 60000 do acc = (acc + f1(i % 7) + f4(i % 3, i % 5, i % 7, i % 11)) % 1000003 end debug.sethook() print(n, acc, table.concat(seen, ','))",
    ),
    (
        "a counted loop whose hook fires every few iterations",
        "local n, acc, seen = 0, 0, {} debug.sethook(function() n = n + 1 if #seen < 60 then seen[#seen + 1] = acc end end, '', 23) for i = 1, 100000 do if i % 4 == 0 then acc = acc + 3 else acc = acc + 1 end end debug.sethook() print(n, acc, table.concat(seen, ','))",
    ),
];

/// Counted trees compile through MIR (no `counted trace` fallback) and fire
/// the hook at the instructions the interpreter fires it at, as Cranelift's
/// code does, in two profiles.
#[test]
fn counted_loops_run_through_mir_and_fire_the_hook_where_the_interpreter_does() {
    for (name, source) in HOOK_SNIPPETS {
        for compat in ["lua51", "lua54"] {
            let run_in = |execution: &str, backend: Option<&str>| {
                let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
                command
                    .args([
                        &format!("--compat={compat}"),
                        &format!("--execution={execution}"),
                        "-e",
                        source,
                    ])
                    .env("OXIMOND_JIT_SYNC", "1")
                    .env("OXIMOND_JIT_TRACE", "1")
                    .env("OXIMOND_JIT_TRACE_IR_OPT", "1")
                    .env_remove("OXIMOND_JIT_BACKEND")
                    .env_remove("OXIMOND_JIT_MIR_PROBE");
                if let Some(backend) = backend {
                    command.env("OXIMOND_JIT_BACKEND", backend);
                }
                let output = command.output().expect("run oximond");
                assert!(output.status.success(), "{name} under {compat}: {output:?}");
                Run {
                    stdout: String::from_utf8(output.stdout).unwrap(),
                    stderr: String::from_utf8(output.stderr).unwrap(),
                }
            };
            let interpreted = run_in("interpreter", None).stdout;
            let mir = run_in("tracing", Some("mir"));
            let cranelift = run_in("tracing", Some("cranelift"));
            assert_ne!(interpreted, "");
            assert_eq!(interpreted, mir.stdout, "{name} under {compat}, mir");
            assert_eq!(
                interpreted, cranelift.stdout,
                "{name} under {compat}, cranelift"
            );
            assert!(
                mir.stderr.contains("counted: true"),
                "{name} under {compat}: a counted tree was recorded\n{}",
                mir.stderr
            );
            assert!(
                mir.stderr
                    .lines()
                    .any(|line| line.starts_with("jit-trace: mir backend compiled tree")),
                "{name} under {compat}: MIR compiled a tree\n{}",
                mir.stderr
            );
            assert!(
                !mir.stderr.contains("mir backend fallback"),
                "{name} under {compat}: MIR declined a tree\n{}",
                mir.stderr
            );
        }
    }
}

/// Outer roots (see "Outer roots" and "Loops in calls" in `vm::jit::trace`):
/// two `while` loops side by side in a `for` loop, and the `dx` loop of
/// Mineclonia's `locate_biome_in_area` with its predicate's `ipairs` scan
/// (`trace_callee_loops.rs`), whose list changes on the way. The root ends in
/// front of the first inner loop, the inner loops are loop sides (the scan's
/// in the frames of the calls, its parameters without registers), and the
/// side of the last closing guard links back to the root's header (under
/// the `luajit21` profile, Mineclonia's; Lua 5.4 makes other trees of the
/// scan). MIR compiles every tree of each, with the same exits, side exits
/// and iterations as Cranelift and the interpreter's output.
#[test]
fn outer_roots_and_loops_in_calls_run_through_mir_like_cranelift() {
    let siblings = "local function run(n) local acc = 0 for i = 1, n do local j = 0 while j < 40 do acc = acc + j j = j + 1 end local k = 0 while k < 30 do acc = acc + k * 2 k = k + 1 end end return acc end print(run(3000))";
    let scan = |extra: &str| {
        format!(
            "local function indexof(list, val) for i, v in ipairs(list) do if v == val then return i end end return -1 end \
             local list = {{}} for i = 1, 40 do list[i] = 'biome_' .. i end \
             local names = {{}} for i = 1, 64 do names[i] = 'biome_' .. (i * 7 % 90) end \
             local function pred(name) return indexof(list, name) ~= -1 end \
             local function locate(n, predicate) local hits, acc = 0, 0 \
             for k = 1, n do {extra} for dx = 1, 64 do local name = names[dx] acc = (acc * 31 + dx) % 1000003 \
             if predicate(name) then hits = hits + 1 end end end return hits, acc end \
             print(locate(300, pred))"
        )
    };
    let changing = scan(
        "if k == 100 then list[10] = nil end if k == 130 then list[10] = 10 end \
         if k == 200 then list[10] = 'x' list[41] = 'biome_41' end",
    );
    for source in [siblings, &scan(""), &changing] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat=luajit21", "--execution=tracing", "-e", source])
            .env("OXIMOND_JIT_SYNC", "1")
            .env("OXIMOND_JIT_TRACE", "1")
            .env("OXIMOND_JIT_TRACE_IR_OPT", "1")
            .env("OXIMOND_JIT_BACKEND", "mir")
            .output()
            .expect("run oximond");
        assert!(output.status.success(), "{output:?}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains("terminator: Link {"),
            "a side links back to the outer root: {source}"
        );
        assert!(
            !stderr.contains("mir backend fallback to cranelift"),
            "MIR compiles every tree: {source}\n{stderr}"
        );
        let compiled = |prefix: &str| stderr.lines().filter(|l| l.starts_with(prefix)).count();
        assert!(
            compiled("jit-trace: mir backend compiled tree") > 0,
            "{source}"
        );
        let mir = exits_in("luajit21", source, "mir");
        assert_eq!(mir, exits_in("luajit21", source, "cranelift"), "{source}");
        let interpreted = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat=luajit21", "--execution=interpreter", "-e", source])
            .output()
            .expect("run oximond");
        assert_eq!(
            String::from_utf8(interpreted.stdout).unwrap(),
            mir.0,
            "{source}"
        );
    }
}
