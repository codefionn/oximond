//! Integer and float operands of one operation, in the constant-operand and
//! register forms, on both execution modes. The script asserts the values
//! stock Lua 5.4 gives, including integers beyond 2^53.

use std::process::Command;

const SCRIPT: &str = r#"
local big = 9007199254740993
local function check(name, got, want)
  if got ~= want and not (got ~= got and want ~= want) then
    error(name .. ": " .. tostring(got) .. " vs " .. tostring(want))
  end
  if math.type(got) ~= math.type(want) then error(name .. ": subtype") end
end
local n = 7
local f = 2.5
for _ = 1, 300 do
  check("add_k", n + 0.5, 7.5)
  check("add_k2", f + 3, 5.5)
  check("sub_k", n - 0.5, 6.5)
  check("mul_k", f * 4, 10.0)
  check("div_k", n / 2.0, 3.5)
  check("mod_k", n % 2.5, 2.0)
  check("mod_k2", -n % 2.5, 0.5)
  check("add", n + f, 9.5)
  check("sub", f - n, -4.5)
  check("mul", n * f, 17.5)
  check("div", n / f, 2.8)
  check("mod", n % f, 2.0)
  check("neg_mod", -n % f, 0.5)
  check("idiv", n // f, 2.0)
  check("pow", n ^ 2, 49.0)
  check("lt_exact", big < 9007199254740992.0, false)
  check("gt_exact", big > 9007199254740992.0, true)
  check("le_exact", big <= 9007199254740992.0, false)
  check("eq_exact", big == 9007199254740992.0, false)
  check("eq_same", 3 == 3.0, true)
  check("eq_k", n == 7.0, true)
  check("lt_k", f < 3, true)
  check("le_k", 3 <= f, false)
  check("max_lt", math.maxinteger < 9223372036854775808.0, true)
  check("max_eq", math.maxinteger == 9223372036854775808.0, false)
  check("min_eq", math.mininteger == -9223372036854775808.0, true)
  local nan = 0.0 / 0.0
  check("nan_lt", n < nan, false)
  check("nan_gt", nan < n, false)
  check("nan_eq", n == nan, false)
  check("nan_add", (n + nan) ~= (n + nan), true)
  check("inf_lt", math.maxinteger < math.huge, true)
  check("ninf_lt", -math.huge < math.mininteger, true)
  check("mod_inf", n % math.huge, 7.0)
  check("mod_ninf", n % -math.huge, -math.huge)
  check("neg_zero", 1 / (-0.0 * 3), -math.huge)
  check("zero_mod", 1 / (0 * -1.0 % 5), -math.huge)
  check("div_zero", n / 0.0, math.huge)
  check("idiv_zero", n // 0.0, math.huge)
  local ok, err = pcall(function() return n % 0 end)
  if ok or not err:find("attempt to perform 'n%%0'") then error("mod zero: " .. tostring(err)) end
  ok, err = pcall(function() return n // 0 end)
  if ok or not err:find("attempt to divide by zero") then error("idiv zero: " .. tostring(err)) end
end
print("ok")
"#;

#[test]
fn mixed_integer_and_float_operands_follow_lua54() {
    for mode in ["interpreter", "tracing"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .arg("--compat=lua54")
            .arg(format!("--execution={mode}"))
            .arg("-e")
            .arg(SCRIPT)
            .env("OXIMOND_JIT_SYNC", "1")
            .output()
            .expect("run oximond");
        assert!(
            output.status.success() && output.stdout == b"ok\n",
            "{mode}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
