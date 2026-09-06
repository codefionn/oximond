//! The start position of `string.gmatch`, which Lua 5.4 added. The expected
//! lines are what stock Lua 5.4 and 5.5 print for the same script; the
//! earlier profiles ignore a third argument.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, Lua};

const SCRIPT: &str = "
local function run(s, p, init)
  local out = {}
  for a, b in string.gmatch(s, p, init) do out[#out + 1] = tostring(a) .. (b and ('/' .. tostring(b)) or '') end
  return table.concat(out, ',')
end
return run('abc def ghi', '%a+', 5), run('abc def ghi', '%a+', -3),
  run('abc def ghi', '%a+', -100), run('abc def ghi', '%a+', 0),
  run('abc def ghi', '%a+', 12), run('abc def ghi', '%a+', 13),
  run('abc', '%a*', 2), run('abc', '()', 4), run('abc', '()', 5),
  run('a b', '(%a)', 2), run('abc', '^a', 1)";

fn lines(profile: CompatibilityProfile) -> Vec<String> {
    let mut lua = Lua::builder().profile(profile).build().unwrap();
    lua.exec(SCRIPT)
        .unwrap()
        .iter()
        .map(|value| match value {
            Value::String(text) => String::from_utf8_lossy(text.as_bytes()).into_owned(),
            other => panic!("{other:?}"),
        })
        .collect()
}

#[test]
fn the_start_position_is_honoured_from_lua_54() {
    let expected = [
        "def,ghi",
        "ghi",
        "abc,def,ghi",
        "abc,def,ghi",
        "",
        "",
        "bc",
        "4",
        "",
        "b",
        "",
    ];
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        assert_eq!(lines(profile), expected, "{profile:?}");
    }
}

#[test]
fn earlier_profiles_ignore_it() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::LuaJit21,
    ] {
        let actual = lines(profile);
        assert_eq!(actual[0], "abc,def,ghi", "{profile:?}");
        assert_eq!(actual[1], "abc,def,ghi", "{profile:?}");
    }
}
