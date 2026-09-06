//! Bounded library cases whose file effects stay in the runner's case directory.

use crate::profile::Profile;

/// Sources for host-facing library behavior. The runner must execute each case
/// in a fresh directory, since the sources create files with fixed names.
#[must_use]
pub fn fixtures(_profile: Profile) -> Vec<(&'static str, String)> {
    vec![
        (
            "io_binary",
            r#"
local path = "fixture-binary.dat"
local file = assert(io.open(path, "w+b"))
assert(file:write("A\000B\255\nC"))
assert(file:flush())
print(file:seek("cur", 0) == 6)
assert(file:seek("set", 0))
local bytes = assert(file:read("*a"))
print(#bytes, string.byte(bytes, 1, #bytes))
assert(file:seek("set", 1))
local part = file:read(2)
print(part == "\000B", file:seek("cur", 0) == 3)
assert(file:close())
local ok, err = pcall(function() return file:read(1) end)
print(ok, type(err))
local missing, open_err = io.open("fixture-does-not-exist.dat", "rb")
print(missing == nil, type(open_err))
assert(os.remove(path))
"#
            .to_owned(),
        ),
        (
            "io_lines",
            r#"
local path = "fixture-lines.txt"
local file = assert(io.open(path, "wb"))
assert(file:write("first\nsecond\nthird\n"))
assert(file:close())
local lines = {}
for line in io.lines(path) do lines[#lines + 1] = line end
print(#lines, table.concat(lines, "|"))
file = assert(io.open(path, "rb"))
local count = 0
for line in file:lines() do count = count + #line end
print(count)
assert(file:close())
assert(os.remove(path))
"#
            .to_owned(),
        ),
        (
            "loadfile_dofile",
            r#"
local path = "fixture-source.lua"
local file = assert(io.open(path, "wb"))
assert(file:write("local x = ...; if x == nil then x = 11 end; return x * 2, 'loaded'\n"))
assert(file:close())
local chunk = assert(loadfile(path))
local doubled, label = chunk(21)
print(doubled, label)
local result, word = dofile(path)
print(result, word)
local missing, err = loadfile("fixture-no-source.lua")
print(missing == nil, type(err))
assert(os.remove(path))
"#
            .to_owned(),
        ),
        (
            "require_preload",
            r#"
local name = "oximond_difftest_fixture_module"
local calls = 0
package.loaded[name] = nil
package.preload[name] = function(module_name)
  calls = calls + 1
  return {name = module_name, count = calls}
end
local first = require(name)
local second = require(name)
print(first == second, first.name == name, first.count, calls)
print(package.loaded[name] == first)
package.loaded[name] = nil
local third = require(name)
print(third ~= first, third.count, calls)
package.preload[name] = function() error("fixture preload failure") end
package.loaded[name] = nil
local ok, err = pcall(require, name)
print(ok, type(err))
package.preload[name] = nil
package.loaded[name] = nil
"#
            .to_owned(),
        ),
        (
            "os_utc",
            r#"
print(os.difftime(72, 12) == 60)
print(os.date("!%Y-%m-%d %H:%M:%S", 946684800))
local utc = os.date("!*t", 946684800)
print(utc.year, utc.month, utc.day, utc.hour, utc.min, utc.sec)
"#
            .to_owned(),
        ),
    ]
}
