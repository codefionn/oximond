//! Reproducers from differential fuzzing of the interpreter against tracing
//! execution.
//!
//! Each program runs under the interpreter and under tracing execution. Both
//! must print what stock Lua prints, which the tests spell out.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn run(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> String {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    format!("{results:?}")
}

/// Checks `source` against `expected` under the interpreter and tracing
/// execution.
fn agree(profile: CompatibilityProfile, source: &str, expected: &str) {
    let interpreter = run(profile, ExecutionMode::Interpreter, source);
    assert_eq!(interpreter, expected, "{profile:?} interpreter\n{source}");
    let tracing = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(tracing, expected, "{profile:?} tracing\n{source}");
}

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

#[test]
fn tostring_call_keeps_a_captured_local_written_since_its_last_flush() {
    // The interpreter's fast loop kept `z + 1` in its register and left the
    // cell behind. The `tostring` fast path then synced the cell back over
    // the register, so `z` went from 3 back to 1 in the first function and
    // lost every doubling in the second. It needs a closure that captured
    // `z`, a write to `z`, and a `tostring` call with `z` as the argument.
    for profile in PROFILES {
        agree(
            profile,
            r"
            local function a()
              local z = 1
              local f = function() return z end
              z = z + 1
              local n = tostring(z)
              z = z + 1
              local m = #tostring(z)
              return z + f() * 0, n, m
            end
            local function b()
              local z = 1
              for i = 1, 4 do
                local n = #tostring(z)
                local f = function() return z end
                z = z + f()
              end
              return z
            end
            local x, n, m = a()
            return string.format('%d,%s,%d,%d', x, n, m, b())
            ",
            "[String(b\"3,2,1,16\")]",
        );
    }
}

#[test]
fn upvaluejoin_during_coroutine_close_keeps_the_callers_captures_alive() {
    // `__close` runs `debug.upvaluejoin` while the caller of `coroutine.close`
    // sits parked on the Rust stack. Its frame borrows `f`'s capture cells,
    // and the join replaces that allocation. Freeing it left the frame with
    // a dangling borrow and `invalid cell handle` once the slots were reused.
    // The first call still returns the stale `a` (stock Lua prints 100), as
    // it did before frames borrowed cells; only soundness is checked here.
    agree(
        CompatibilityProfile::Lua54,
        r"
local a, b, c = 1, 2, 3
local function f()
  local s = a + b + c
  local x = 100
  local function g() return x end
  local co = coroutine.create(function()
    local t <close> = setmetatable({}, {__close = function() debug.upvaluejoin(f, 1, g, 1) end})
    coroutine.yield()
  end)
  coroutine.resume(co)
  coroutine.close(co)
  co = nil
  collectgarbage() collectgarbage()
  local junk = {}
  for i = 1, 20000 do
    junk[i] = {i, i, i, i}
    junk[-i] = function() return i, junk end
  end
  return a, b, c
end
local out = {}
for i = 1, 3 do out[#out + 1] = table.concat({f()}, ',') end
return table.concat(out, ' ')
",
        "[String(b\"1,2,3 100,2,3 100,2,3\")]",
    );
}
