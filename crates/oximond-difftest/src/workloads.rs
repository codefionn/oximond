//! Isolated hot workloads. Native activity belongs to the workload, not to
//! every feature that happened to occur elsewhere in a generated program.

use crate::profile::Profile;

/// Each source has one hot loop, a bounded checksum and no output in the
/// loop. Native entries establish that this workload compiled; they do not
/// establish that every operation in its body was lowered to native code.
#[must_use]
pub fn workloads(_profile: Profile) -> Vec<(&'static str, String)> {
    [
        ("math_floor", "", "total = total + math.floor(i / 3)"),
        ("math_sqrt", "", "total = total + math.sqrt(i % 16)"),
        ("string_byte", "local s = 'abcdefgh'", "total = total + string.byte(s, i % 8 + 1)"),
        ("string_sub", "local s = 'abcdefgh'", "total = total + #string.sub(s, i % 8 + 1)"),
        ("table_read_write", "local t = {1, 2, 3, 4}", "local k = i % 4 + 1; t[k] = t[k] + 1; total = total + t[k]"),
        ("closure_call", "local bias = 3; local function add(x) return x + bias end", "total = total + add(i % 17)"),
        ("metatable_index", "local t = setmetatable({}, {__index = function(_, k) return k + 2 end})", "total = total + t[i % 17]"),
    ]
    .into_iter()
    .map(|(name, setup, body)| {
        (name, format!("{setup}\nlocal total = 0\nfor i = 1, 40000 do\n  {body}\nend\nprint(string.format('%.17g', total))\n"))
    })
    .collect()
}
