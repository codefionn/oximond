//! Base-library behaviour that differs per profile: `tonumber` with a base,
//! `math.max` and `math.min` with NaN, and `coroutine.isyieldable`. The
//! expected output of every case was captured from the stock interpreter for
//! the profile (`lua5.1`, `lua5.4`, `lua5.5`, `luajit`).

use std::env;
use std::process::Command;

fn run(profile: &str, source: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "-e", source])
        .output()
        .expect("run oximond");
    assert_eq!(output.status.code(), Some(0), "{profile}: {output:?}");
    String::from_utf8(output.stdout).expect("utf-8 output")
}

const TONUMBER: &str = r#"local t={} for _,a in ipairs{{"0x1F",16},{"0X1f",16},{" 0x1F ",16},{"-0x1F",16},{"-1F",16},{"+0x1",16},{"0x",16},{"0x-1",16},{"0b11",2},{"-0b11",2},{"0b11",16},{"ffffffffffffffffff",16},{"-ffffffffffffffffff",16},{"1F\v",16},{"1\0002",16},{"-0",16},{"0x1F",10},{"0x1F",36},{"-0x",36},{"z",36}} do t[#t+1]=tostring(tonumber(a[1],a[2])) end print(table.concat(t," "))"#;

#[test]
fn tonumber_base_follows_strtoul_in_lua51() {
    // strtoul wraps a negative numeral to 2^64 - n and saturates on overflow.
    assert_eq!(
        run("lua51", TONUMBER),
        "31 31 31 1.844674407371e+19 1.844674407371e+19 1 nil nil 3 1.844674407371e+19 2833 1.844674407371e+19 1.844674407371e+19 31 1 0 31 42819 1.844674407371e+19 35\n"
    );
}

#[test]
fn tonumber_base_follows_strtoul_in_luajit() {
    // LuaJIT accepts the same prefixes but negates the converted double.
    assert_eq!(
        run("luajit21", TONUMBER),
        "31 31 31 -31 -31 1 nil nil 3 -3 2833 1.844674407371e+19 -1.844674407371e+19 31 1 -0 31 42819 -33 35\n"
    );
    assert_eq!(run("luajit21", r#"print(1/tonumber("-0",16))"#), "-inf\n");
}

#[test]
fn tonumber_base_rejects_prefixes_in_lua54_and_lua55() {
    for profile in ["lua54", "lua55"] {
        assert_eq!(
            run(profile, TONUMBER),
            "nil nil nil nil -31 nil nil nil nil nil 2833 -1 1 31 nil 0 nil 42819 -33 35\n",
            "{profile}"
        );
        assert_eq!(
            run(
                profile,
                r#"print(tonumber("0x1F",16), tonumber("-1F",16), tonumber("\v1F\v",16), pcall(tonumber,"1",16.5)) print(tonumber("7",8.0))"#
            ),
            "nil\t-31\t31\tfalse\tbad argument #2 to 'tonumber' (number has no integer representation)\n7\n",
            "{profile}"
        );
    }
}

#[test]
fn tonumber_base_truncates_a_fractional_base_in_lua51_and_luajit() {
    for profile in ["lua51", "luajit21"] {
        assert_eq!(
            run(profile, r#"print(pcall(tonumber,"1",16.5))"#),
            "true\t1\n",
            "{profile}"
        );
    }
}

const EXTREMES: &str = r#"local u=function(a,b,c) if c==nil and not (#{a,b}>1) then return a end if c==nil then return a,b end return a,b,c end local nan=0/0 local function f(x) return x~=x and "nan" or tostring(x) end
local t={} for _,a in ipairs{{nan,1},{1,nan},{nan,nan},{2,3},{3,2},{nan,1,2},{1,nan,2},{1,2,nan},{nan,nan,3}} do
 t[#t+1]=f(math.max(u(a[1],a[2],a[3])))..","..f(math.min(u(a[1],a[2],a[3]))) end print(table.concat(t," "))"#;

#[test]
fn math_max_min_with_nan_follow_the_stock_comparison() {
    // Lua 5.1 and later keep the running result unless the new value
    // compares greater (or less).
    for profile in ["lua51", "lua54", "lua55"] {
        assert_eq!(
            run(profile, EXTREMES),
            "nan,nan 1,1 nan,nan 3,2 3,2 nan,nan 2,1 2,1 nan,nan\n",
            "{profile}"
        );
        assert_eq!(
            run(profile, "print(1/math.max(0,-0), 1/math.min(0,-0))"),
            "inf\tinf\n",
            "{profile}"
        );
    }
    // LuaJIT replaces the result unless the new value compares less, so a
    // NaN argument replaces and is replaced, and equal zeros take the later.
    assert_eq!(
        run("luajit21", EXTREMES),
        "1,1 nan,nan nan,nan 3,2 3,2 2,1 2,2 nan,nan 3,3\n"
    );
    assert_eq!(
        run("luajit21", "print(1/math.max(0,-0), 1/math.min(0,-0))"),
        "-inf\t-inf\n"
    );
}

#[test]
fn math_max_min_with_nan_agree_in_compiled_loops() {
    // A hot loop reaches the numeric fast path instead of the native call.
    let source = r"local nan=0/0 local r={} for i=1,300 do r[1]=math.max(nan,i) r[2]=math.max(i,nan) r[3]=math.min(nan,i) r[4]=math.min(i,nan) end
print(r[1]~=r[1], r[2]~=r[2], r[3]~=r[3], r[4]~=r[4])";
    assert_eq!(run("luajit21", source), "false\ttrue\tfalse\ttrue\n");
    assert_eq!(run("lua51", source), "true\tfalse\ttrue\tfalse\n");
}

const ISYIELDABLE: &str = r##"print(type(coroutine.isyieldable)) if coroutine.isyieldable then print(coroutine.isyieldable(), select("#",coroutine.isyieldable()), coroutine.wrap(function() return coroutine.isyieldable() end)(), coroutine.wrap(function() return pcall(coroutine.isyieldable) end)()) end"##;

#[test]
fn isyieldable_exists_from_lua53_and_in_luajit() {
    assert_eq!(run("lua51", ISYIELDABLE), "nil\n");
    for profile in ["lua53", "lua54", "lua55", "luajit21"] {
        assert_eq!(
            run(profile, ISYIELDABLE),
            "function\nfalse\t1\ttrue\ttrue\ttrue\n",
            "{profile}"
        );
    }
}

#[test]
fn isyieldable_takes_a_thread_in_lua54_and_lua55() {
    let source = r"local co=coroutine.create(function() end) local main=coroutine.running()
print(coroutine.isyieldable(main), coroutine.isyieldable(co), coroutine.wrap(function() return coroutine.isyieldable(main), coroutine.isyieldable(coroutine.running()) end)())
coroutine.resume(co) print(coroutine.isyieldable(co)) print(pcall(coroutine.isyieldable, 1))";
    for profile in ["lua54", "lua55"] {
        let output = run(profile, source);
        let mut lines = output.lines();
        assert_eq!(lines.next(), Some("false\ttrue\tfalse\ttrue"), "{profile}");
        assert_eq!(lines.next(), Some("true"), "{profile}");
        let error = lines.next().expect("error line");
        assert!(
            error.starts_with("false\tbad argument #1 to ")
                && error.ends_with("(thread expected, got number)"),
            "{profile}: {error}"
        );
    }
}

#[test]
fn luajit_isyieldable_ignores_its_argument() {
    let source = r"local co=coroutine.create(function() end) local main=coroutine.running()
print(coroutine.isyieldable(main), coroutine.isyieldable(co), coroutine.wrap(function() return coroutine.isyieldable(main), coroutine.isyieldable(coroutine.running()) end)())
coroutine.resume(co) print(coroutine.isyieldable(co)) print(pcall(coroutine.isyieldable, 1))";
    assert_eq!(
        run("luajit21", source),
        "false\tfalse\ttrue\ttrue\nfalse\ntrue\tfalse\n"
    );
}

#[test]
fn isyieldable_is_false_across_host_call_boundaries() {
    // Values from stock lua5.4, lua5.5 and luajit. Stock also reports true
    // inside __index (all) and __tostring (luajit), where it can resume a
    // yield; oximond rejects a yield there, so it reports false instead.
    let source = r#"local out = {}
coroutine.wrap(function()
  string.gsub("a", "a", function() out[#out+1] = tostring(coroutine.isyieldable()) end)
  table.sort({2, 1}, function(a, b) out[#out+1] = tostring(coroutine.isyieldable()) return a < b end)
  out[#out+1] = tostring(coroutine.isyieldable())
end)()
print(table.concat(out, " "))"#;
    for profile in ["lua54", "lua55", "luajit21"] {
        assert_eq!(run(profile, source), "false false true\n", "{profile}");
    }
    let tostring_case = r#"local r
coroutine.wrap(function()
  tostring(setmetatable({}, {__tostring = function() r = coroutine.isyieldable() return "" end}))
end)()
print(r)"#;
    assert_eq!(run("lua54", tostring_case), "false\n");
    assert_eq!(run("lua55", tostring_case), "false\n");
}
