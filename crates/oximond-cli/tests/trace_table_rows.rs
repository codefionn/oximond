//! Native traces over tables a loop meets while it runs: records read out of
//! an array, and tables held in registers the loops around a loop write.
//!
//! Every program runs under the interpreter and under tracing execution, in
//! every profile it makes sense for. The results must match exactly, errors
//! included, and counters must show native trace code ran. The programs hold the engine to:
//!
//! - A field of a table known only by lane is read and written in place, in
//!   the slot the interpreter would use, and a store is committed once.
//! - A record that lacks the field, holds another type there, answers through
//!   `__index`, is not a table at all, or is the same table as another
//!   element is the interpreter's business: the trace leaves before the
//!   access.
//! - More tables than a run has lanes for leave the trace at the load, and
//!   the loop goes on in the interpreter with the same answers.
//! - Nested loops over records (the n-body shape) run through outer side
//!   traces with the same results.

use oximond::vm::{TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

const INTEGER_PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` under the interpreter and under tracing execution, checks
/// the results agree, and returns the result and the tracing statistics.
fn agree(profile: CompatibilityProfile, source: &str) -> (String, TraceStatistics) {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        tracing, expected,
        "{profile:?} tracing\n{source}\n{stats:?}"
    );
    (expected, stats)
}

fn agree_natively(
    profiles: &[CompatibilityProfile],
    source: &str,
) -> Vec<(String, TraceStatistics)> {
    profiles
        .iter()
        .map(|profile| {
            let (result, stats) = agree(*profile, source);
            assert!(
                stats.iterations > 0,
                "{profile:?}: no native iteration ran\n{source}\n{stats:?}"
            );
            (result, stats)
        })
        .collect()
}

/// A loop that reads and writes the fields of the records of an array.
const RECORDS: &str = "
    local function records(n)
      local rows = {}
      for i = 1, n do
        rows[i] = {x = i * 0.5, y = (i % 7) + 0.25, hits = 0}
      end
      return rows
    end";

#[test]
fn fields_of_records_read_out_of_an_array_run_natively() {
    let source = format!(
        "{RECORDS}
        local function step(rows, n, dt)
          local sum = 0
          for i = 1, n do
            local r = rows[i]
            r.x = r.x + r.y * dt
            r.y = r.y - r.x * 0.001
            sum = sum + r.x
          end
          return sum
        end
        local rows = records(300)
        local total = 0
        for round = 1, 30 do total = total + step(rows, 300, 0.125) end
        return total, rows[1].x, rows[1].y, rows[300].x, rows[300].y"
    );
    for (_, stats) in agree_natively(&ALL_PROFILES, &source) {
        assert!(stats.iterations > 3_000, "{stats:?}");
    }
}

#[test]
fn integer_fields_keep_their_type_and_wrap_like_the_interpreter() {
    let source = format!(
        "{RECORDS}
        local function bump(rows, n)
          local total = 0
          for i = 1, n do
            local r = rows[i]
            r.hits = r.hits + 1
            total = total + r.hits
          end
          return total
        end
        local rows = records(200)
        for i = 1, 200 do rows[i].hits = i end
        local total = 0
        for round = 1, 30 do total = total + bump(rows, 200) end
        return total, math.type(rows[1].hits), rows[200].hits"
    );
    agree_natively(&INTEGER_PROFILES, &source);
}

#[test]
fn the_same_table_twice_in_an_array_is_one_table() {
    let source = format!(
        "{RECORDS}
        local function bump(rows, n)
          local sum = 0
          for i = 1, n do
            local a = rows[i]
            local b = rows[n + 1 - i]
            a.x = a.x + 1
            sum = sum + b.x
          end
          return sum
        end
        local rows = records(120)
        -- slots 5 and 80 name the same record, and so do the mirrored ones
        rows[80] = rows[5]
        rows[41] = rows[80]
        local total = 0
        for round = 1, 30 do total = total + bump(rows, 120) end
        return total, rows[5].x, rows[41].x"
    );
    agree_natively(&ALL_PROFILES, &source);
}

#[test]
fn a_record_without_the_field_or_with_another_type_runs_in_the_interpreter() {
    let source = format!(
        "{RECORDS}
        local function sum(rows, n)
          local s = 0
          for i = 1, n do
            local r = rows[i]
            r.x = r.x + 1
            s = s + r.x
          end
          return s
        end
        local rows = records(200)
        local total = 0
        for round = 1, 12 do total = total + sum(rows, 200) end
        -- from now on a few records are not what the trace was recorded on
        rows[50].x = nil
        rows[51] = {{x = 3, y = 1}}
        rows[52] = setmetatable({{}}, {{__index = function(t, k) return 10 end,
                                      __newindex = function(t, k, v) rawset(t, k, v * 2) end}})
        local first, second = pcall(sum, rows, 49)
        total = total + second
        local ok, err = pcall(sum, rows, 200)
        return total, ok, err, rows[49].x, rows[51].x, rows[52].x, rawget(rows[52], 'x')"
    );
    for profile in ALL_PROFILES {
        agree(profile, &source);
    }
}

#[test]
fn an_element_that_is_not_a_table_leaves_before_the_load() {
    let source = format!(
        "{RECORDS}
        local function sum(rows, n)
          local s = 0
          for i = 1, n do
            local r = rows[i]
            s = s + r.x
          end
          return s
        end
        local rows = records(100)
        local total = 0
        for round = 1, 20 do total = total + sum(rows, 100) end
        rows[70] = 'text'
        local ok, err = pcall(sum, rows, 100)
        rows[70] = 4
        local ok2, err2 = pcall(sum, rows, 100)
        rows[70] = rows[1]
        return total, ok, err, ok2, err2, sum(rows, 100)"
    );
    for profile in ALL_PROFILES {
        agree(profile, &source);
    }
}

#[test]
fn a_run_with_more_tables_than_lanes_leaves_at_the_load_and_goes_on() {
    // 1000 distinct records are more than a run has lanes for. The trace
    // leaves at the first table past them, the interpreter does that
    // iteration, and the trace is entered again at the next backedge.
    let source = format!(
        "{RECORDS}
        local function step(rows, n)
          local sum = 0
          for i = 1, n do
            local r = rows[i]
            r.x = r.x + 1
            sum = sum + r.x - r.y
          end
          return sum
        end
        local rows = records(1000)
        local total = 0
        for round = 1, 6 do total = total + step(rows, 1000) end
        return total, rows[1].x, rows[255].x, rows[256].x, rows[257].x, rows[1000].x"
    );
    for (_, stats) in agree_natively(&ALL_PROFILES, &source) {
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > 0,
            "the lanes ran out: {stats:?}"
        );
    }
}

#[test]
fn records_that_change_between_runs_are_found_again_at_every_entry() {
    // Another table at every slot, the same slots a different shape, a field
    // that grows the table's storage: the rows are rebuilt at each entry.
    let source = format!(
        "{RECORDS}
        local function step(rows, n)
          local sum = 0
          for i = 1, n do
            local r = rows[i]
            r.x = r.x + 1
            sum = sum + r.x + r.y
          end
          return sum
        end
        local rows = records(150)
        local total = 0
        for round = 1, 24 do
          total = total + step(rows, 150)
          if round % 3 == 0 then
            for i = 1, 150 do
              rows[i] = {{y = i, x = round, hits = 0}}
            end
          elseif round % 3 == 1 then
            for i = 1, 150, 2 do
              rows[i].extra = i
              rows[i].more = round
              rows[i].zzz = true
            end
          end
        end
        return total, rows[1].x, rows[2].x, rows[150].x"
    );
    agree_natively(&ALL_PROFILES, &source);
}

#[test]
fn a_metatable_that_appears_on_a_record_changes_nothing_for_present_fields() {
    let source = format!(
        "{RECORDS}
        local log = {{}}
        local meta = {{
          __index = function(t, k) log[#log + 1] = 'index ' .. tostring(k); return 0 end,
          __newindex = function(t, k, v) log[#log + 1] = 'new ' .. tostring(k); rawset(t, k, v) end,
        }}
        local function step(rows, n)
          local sum = 0
          for i = 1, n do
            local r = rows[i]
            r.x = r.x + r.y
            sum = sum + r.x
          end
          return sum
        end
        local rows = records(100)
        for i = 1, 100, 3 do setmetatable(rows[i], meta) end
        local total = 0
        for round = 1, 30 do total = total + step(rows, 100) end
        rows[10].y = nil
        local ok, err = pcall(step, rows, 100)
        return total, #log, log[1], ok, err, rows[10].x"
    );
    agree_natively(&ALL_PROFILES, &source);
}

#[test]
fn nested_loops_over_records_run_through_outer_side_traces() {
    // The n-body shape: an outer loop reads a record, an inner loop pairs it
    // with the later ones and updates both.
    let source = "
        local bodies = {}
        for i = 1, 5 do
          bodies[i] = {x = i + 0.5, y = i * 2.5, vx = 0.1 * i, vy = -0.05 * i, mass = 1 + i * 0.25}
        end
        local function advance(dt)
          for i = 1, #bodies do
            local first = bodies[i]
            for j = i + 1, #bodies do
              local second = bodies[j]
              local dx, dy = first.x - second.x, first.y - second.y
              local d2 = dx * dx + dy * dy
              local mag = dt / (d2 * (d2 ^ 0.5))
              first.vx = first.vx - dx * second.mass * mag
              first.vy = first.vy - dy * second.mass * mag
              second.vx = second.vx + dx * first.mass * mag
              second.vy = second.vy + dy * first.mass * mag
            end
          end
          for i = 1, #bodies do
            local body = bodies[i]
            body.x = body.x + dt * body.vx
            body.y = body.y + dt * body.vy
          end
        end
        for _ = 1, 3000 do advance(0.01) end
        local out = {}
        for i = 1, #bodies do
          out[#out + 1] = string.format('%.9f %.9f %.9f %.9f', bodies[i].x, bodies[i].y,
                                        bodies[i].vx, bodies[i].vy)
        end
        return table.concat(out, ';')";
    for (_, stats) in agree_natively(&ALL_PROFILES, source) {
        assert!(stats.iterations > 5_000, "{stats:?}");
        assert!(stats.outer_sides > 0, "no outer side ran: {stats:?}");
    }
}

#[test]
fn a_table_held_in_a_register_the_outer_loop_writes_is_not_bound_at_entry() {
    // `row` is another table at every outer iteration, so the inner loop
    // cannot bind its storage once. The first outer iteration's table must not
    // leak into the later ones.
    let source = "
        local rows = {}
        for i = 1, 40 do rows[i] = {a = i * 1.5, b = 0.5} end
        local function run(rounds)
          local total = 0
          for i = 1, #rows do
            local row = rows[i]
            for k = 1, rounds do
              row.b = row.b + row.a * 0.01
              total = total + row.b
            end
          end
          return total
        end
        local out = 0
        for r = 1, 40 do out = out + run(30) end
        return out, rows[1].b, rows[40].b";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn indexing_a_loaded_table_with_a_number_or_taking_its_length_stays_correct() {
    let source = "
        local grid = {}
        for i = 1, 30 do
          grid[i] = {}
          for j = 1, 30 do grid[i][j] = (i * j) % 7 + 0.5 end
        end
        local function sum(g, n)
          local s = 0
          for i = 1, n do
            local row = g[i]
            for j = 1, #row do s = s + row[j] end
          end
          return s
        end
        local total = 0
        for r = 1, 40 do total = total + sum(grid, 30) end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn records_with_mixed_number_types_keep_the_interpreters_arithmetic() {
    let source = "
        local rows = {}
        for i = 1, 90 do
          rows[i] = {n = i % 2 == 0 and i or i + 0.5}
        end
        local function run(rounds)
          local s = 0
          for r = 1, rounds do
            for i = 1, #rows do
              local row = rows[i]
              row.n = row.n + 1
              s = s + row.n
            end
          end
          return s
        end
        local total = 0
        for k = 1, 8 do total = total + run(12) end
        return total, rows[1].n, rows[2].n, rows[3].n";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

// --------------------------------------------------------------- generated

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % bound as u64).unwrap()
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// Programs that loop over an array of records, reading and writing a few
/// fields, with records corrupted in the ways a trace has to leave for.
fn program(rng: &mut Rng) -> String {
    let fields = ["a", "b", "c"];
    let mut body = Vec::new();
    for _ in 0..(2 + rng.below(4)) {
        let target = rng.pick(&fields);
        let left = rng.pick(&fields);
        let right = rng.pick(&fields);
        let operator = rng.pick(&["+", "-", "*"]);
        body.push(match rng.below(5) {
            0 | 1 => format!("r.{target} = (r.{left} {operator} r.{right}) % 997"),
            2 => format!("s = (s + r.{left}) % 100003"),
            3 => format!(
                "if r.{left} > r.{right} then s = s + 1 else r.{target} = r.{target} + 1 end"
            ),
            _ => format!(
                "if i % {} == 0 then other.{target} = other.{target} + r.{left} end",
                2 + rng.below(5)
            ),
        });
    }
    let corruption = rng.below(7);
    let hole = 1 + rng.below(60);
    let alias = rng.below(3);
    format!(
        "
        local function build(round)
          local rows = {{}}
          for i = 1, 60 do
            rows[i] = {{a = (i * 3 + round) % 11 + 0.5, b = i % 5 + 1.25, c = (i + round) % 7}}
          end
          local other = {{a = 1.5, b = 2.5, c = 3}}
          if round % 3 == {corruption} % 3 then
            local at = (round * {hole}) % 60 + 1
            local kind = round % 4
            if kind == 0 then rows[at].a = nil
            elseif kind == 1 then rows[at].b = 'text'
            elseif kind == 2 then rows[at] = setmetatable({{c = 1}}, {{__index = function(t, k) return 2 end}})
            else rows[at] = true end
          end
          if {alias} == 1 then rows[17] = rows[3] elseif {alias} == 2 then rows[40] = rows[39] end
          return rows, other
        end
        local function run(rows, other, n)
          local s = 0
          for i = 1, n do
            local r = rows[i]
            {body}
          end
          return s
        end
        local out = {{}}
        for round = 1, 8 do
          local rows, other = build(round)
          local ok, s = pcall(run, rows, other, 60)
          local sum = 0
          for i = 1, 60 do
            local r = rows[i]
            if type(r) == 'table' then
              for _, k in ipairs({{'a', 'b', 'c'}}) do
                local v = rawget(r, k)
                if type(v) == 'number' then sum = sum + v end
              end
            end
          end
          out[#out + 1] = tostring(ok) .. ':' .. tostring(s) .. ':' .. tostring(sum) .. ':' .. tostring(other.a) .. tostring(other.b) .. tostring(other.c)
        end
        return table.concat(out, ',')",
        body = body.join("\n            ")
    )
}

#[test]
fn generated_record_loops_match_the_interpreter_and_run_natively() {
    let mut runs = 0;
    let mut native_runs = 0;
    let mut iterations = 0;
    for seed in 1..=60_u64 {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ seed.wrapping_mul(0x1000_0000_01b3) | 1);
        let source = program(&mut rng);
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::LuaJit21,
        ] {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (tracing, stats) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(
                tracing, expected,
                "seed {seed} {profile:?} tracing\n{source}\n{stats:?}"
            );
            runs += 1;
            native_runs += usize::from(stats.iterations > 0);
            iterations += stats.iterations;
        }
    }
    assert!(
        native_runs * 3 > runs,
        "{native_runs} of {runs} runs ran natively"
    );
    assert!(iterations > 2_000, "{iterations} native iterations");
}
