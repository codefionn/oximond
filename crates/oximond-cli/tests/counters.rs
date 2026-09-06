//! `OXIMOND_COUNTERS` prints one `key=value` line to stderr at a clean exit,
//! and prints nothing without it. `benchmarks/measure.py` parses the line.

use std::io::Write;
use std::process::Command;

// Tests run in parallel, so each call gets its own script file.
static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn run(counters: bool, execution: &str, source: &str) -> (String, String) {
    run_with(counters, execution, &[], source)
}

fn run_with(counters: bool, execution: &str, flags: &[&str], source: &str) -> (String, String) {
    let mut file = tempfile_path();
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    file.push(format!("counters_test_{id}.lua"));
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::File::create(&file)
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
    command
        .arg(format!("--execution={execution}"))
        .args(flags)
        .arg(&file)
        .env_remove("OXIMOND_COUNTERS");
    if counters {
        command.env("OXIMOND_COUNTERS", "1");
    }
    let output = command.output().unwrap();
    assert!(output.status.success(), "{output:?}");
    (
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

fn tempfile_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("counters-{}", std::process::id()))
}

fn value(line: &str, key: &str) -> u64 {
    line.split_whitespace()
        .find_map(|item| item.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("{key} missing from {line}"))
        .parse()
        .unwrap()
}

const SOURCE: &str = "local keep = {} for i = 1, 3000 do keep[#keep + 1] = {i} end print(#keep)";

#[test]
fn counters_are_off_by_default() {
    let (stdout, stderr) = run(false, "interpreter", SOURCE);
    assert_eq!(stdout, "3000\n");
    assert_eq!(stderr, "");
}

#[test]
fn counters_report_real_tables_and_trace_activity() {
    let (stdout, stderr) = run(true, "interpreter", SOURCE);
    assert_eq!(stdout, "3000\n");
    let line = stderr
        .lines()
        .find(|l| l.starts_with("oximond-counters:"))
        .unwrap();
    assert!(value(line, "tables_created") >= 3000);
    assert_eq!(value(line, "traces_compiled"), 0);
    let (_, stderr) = run(
        true,
        "tracing",
        "local s = 0 for i = 1, 200000 do s = s + i % 7 end print(s)",
    );
    let line = stderr
        .lines()
        .find(|l| l.starts_with("oximond-counters:"))
        .unwrap();
    assert!(value(line, "native_entries") >= 1, "{line}");
    assert!(value(line, "iterations") > 0, "{line}");
    assert_eq!(value(line, "elided_tables"), 0);
}

fn tracing_line(source: &str) -> String {
    let (_, stderr) = run(true, "tracing", source);
    stderr
        .lines()
        .find(|l| l.starts_with("oximond-counters:"))
        .unwrap()
        .to_owned()
}

/// [`tracing_line`] on Lua 5.4, which has the `utf8` library.
fn modern_line(source: &str) -> String {
    let (_, stderr) = run_with(true, "tracing", &["--compat=lua54"], source);
    stderr
        .lines()
        .find(|l| l.starts_with("oximond-counters:"))
        .unwrap()
        .to_owned()
}

#[test]
fn optimizer_keys_are_always_present() {
    let line = tracing_line("local s = 0 for i = 1, 200000 do s = s + i % 7 end print(s)");
    for key in ["loads_forwarded", "loads_merged", "loads_hoisted"] {
        value(&line, key);
    }
}

#[cfg(feature = "alloc-stats")]
#[test]
fn alloc_keys_are_present_and_positive() {
    let line = tracing_line(SOURCE);
    for key in ["alloc_count", "alloc_bytes", "alloc_peak_bytes"] {
        assert!(value(&line, key) > 0, "{key} in {line}");
    }
    assert!(value(&line, "alloc_peak_bytes") <= value(&line, "alloc_bytes"));
}

#[cfg(not(feature = "alloc-stats"))]
#[test]
fn alloc_keys_are_absent_without_the_feature() {
    let line = tracing_line(SOURCE);
    assert!(!line.contains("alloc_"), "{line}");
}

/// Runs `source` under `compat` in the interpreter and returns `alloc_count`.
#[cfg(feature = "alloc-stats")]
fn alloc_count(compat: &str, source: &str) -> u64 {
    let mut file = tempfile_path();
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    file.push(format!("counters_alloc_{id}.lua"));
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args([
            format!("--compat={compat}").as_str(),
            "--execution=interpreter",
        ])
        .arg(&file)
        .env("OXIMOND_COUNTERS", "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    value(&stderr, "alloc_count")
}

/// Startup and per-function compilation allocate nothing beyond the b0f63c2
/// baseline. A compiler field that held a `Vec` per function scope once added
/// one allocation per compiled function, which shows in both numbers below.
/// The bounds are the b0f63c2 release counts (lua51 491/1127, lua54
/// 538/1178); a debug build may count one fewer, so they are ceilings. The
/// closure-minus-empty difference does not depend on the build profile and
/// catches a per-function allocation on its own.
#[cfg(feature = "alloc-stats")]
fn assert_baseline(compat: &str, empty_bound: u64, closure_bound: u64) {
    let empty = alloc_count(compat, "\n");
    let closure = alloc_count(compat, "local function f(x) return x end print(f(1))");
    assert!(
        empty <= empty_bound,
        "{compat}: empty script allocated {empty} times"
    );
    assert!(
        closure <= closure_bound,
        "{compat}: closure script allocated {closure} times"
    );
    assert!(
        closure - empty <= closure_bound - empty_bound,
        "{compat}: one more function costs {} allocations",
        closure - empty
    );
}

#[cfg(feature = "alloc-stats")]
#[test]
fn lua51_startup_and_closure_allocations_stay_at_baseline() {
    assert_baseline("lua51", 491, 1127);
}

#[cfg(feature = "alloc-stats")]
#[test]
fn lua54_startup_and_closure_allocations_stay_at_baseline() {
    assert_baseline("lua54", 538, 1178);
}

/// Runs `source` under `compat` in the interpreter and returns
/// `alloc_peak_bytes`. The peak counts the bytes of `argv[0]` and the script
/// path, so the run fixes both: `argv[0]` is `oximond` and the script is
/// `s.lua` in its own directory, which is the working directory.
#[cfg(feature = "alloc-stats")]
fn alloc_peak_bytes(compat: &str, source: &str) -> u64 {
    use std::os::unix::process::CommandExt;
    let mut directory = tempfile_path();
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    directory.push(format!("counters_peak_{id}"));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("s.lua"), source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg0("oximond")
        .args([
            format!("--compat={compat}").as_str(),
            "--execution=interpreter",
            "s.lua",
        ])
        .current_dir(&directory)
        .env("OXIMOND_COUNTERS", "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    value(&stderr, "alloc_peak_bytes")
}

/// Startup's live heap stays at the 9236910 baseline (lua51 51,512 bytes,
/// lua54 55,256, the same in release and debug builds). The `Vm` is boxed,
/// so every field added to it or to its `Runtime` raises the peak of every
/// script: seven more `Option<NativeFunctionHandle>` slots and a `usize`
/// depth counter once added 64 bytes.
#[cfg(feature = "alloc-stats")]
#[test]
fn startup_peak_bytes_stay_at_baseline() {
    for (compat, bound) in [("lua51", 51_512), ("lua54", 55_256)] {
        let peak = alloc_peak_bytes(compat, "\n");
        assert!(
            peak <= bound,
            "{compat}: an empty script peaked at {peak} bytes, above {bound}"
        );
    }
}

#[test]
fn counters_name_the_library_function_that_aborted_a_recording() {
    let line = tracing_line("local n = 0 for i = 1, 5000 do n = n + math.atan2(i, 2) end print(n)");
    assert!(value(&line, "abort_native.math.atan2") > 0, "{line}");
    // `string.match` is a helper now: the loop traces.
    let line = tracing_line(
        "local function f(...) local s = 0 for i = 1, 50000 do s = s + #string.match('abc', 'b') end return s end \
         print(f(1, 2, 3))",
    );
    assert!(!line.contains("abort_native"), "{line}");
}

#[test]
fn counters_name_no_function_for_the_inline_builtins() {
    let line = tracing_line(
        "local function f(...) return select('#', ...) end local s = 0 \
         for i = 1, 5000 do s = s + f(i, 2, 3) + math.floor(i / 3) + math.ceil(i / 3) end \
         print(s)",
    );
    assert!(!line.contains("abort_native"), "{line}");
    // The root frame's own varargs trace too.
    let line = tracing_line(
        "local function f(...) local s = 0 for i = 1, 5000 do s = s + select('#', ...) end return s end \
         print(f(1, 2, 3))",
    );
    assert!(!line.contains("abort_native"), "{line}");
}

#[test]
fn counters_name_no_function_when_the_loop_traces() {
    let line = tracing_line("local s = 0 for i = 1, 200000 do s = s + i % 7 end print(s)");
    assert!(!line.contains("abort_native"), "{line}");
}

#[test]
fn counters_name_a_builtin_called_through_a_local_alias() {
    // `load` is a VM builtin; the alias moves it into a register first.
    let line = tracing_line(
        "local load = load local t = 0 \
         for i = 1, 20000 do local f = load('return ' .. i) t = t + f() end print(t)",
    );
    assert!(value(&line, "abort_native.load") > 0, "{line}");
    // `pcall` of a Lua function is a protected call the trace records.
    let line = tracing_line(
        "local pcall = pcall local function work(i) return i % 89 end local t = 0 \
         for i = 1, 20000 do local ok, a = pcall(work, i) t = t + a end print(t)",
    );
    assert!(!line.contains("abort_native.pcall"), "{line}");
    // A native `pcall` cannot hold is named, not `pcall`.
    let line = tracing_line(
        "local pcall, error = pcall, error local t = 0 \
         for i = 1, 20000 do local ok = pcall(error, 'x') if not ok then t = t + 1 end end print(t)",
    );
    assert!(value(&line, "abort_native.error") > 0, "{line}");
    assert!(!line.contains("abort_native.pcall"), "{line}");
    // `next` is the iterator `pairs` hands the generic for. A traversal of
    // numbers traces; one over values the trace cannot hold names it.
    let line = tracing_line(
        "local v = {} for i = 1, 64 do v['k' .. i] = i end local t = 0 \
         for _ = 1, 2000 do for _, x in pairs(v) do t = t + x end end print(t)",
    );
    assert!(!line.contains("abort_native.next"), "{line}");
    assert!(!line.contains("abort_native.pairs"), "{line}");
    let line = tracing_line(
        "local v = {} for i = 1, 64 do v['k' .. i] = print end local t = 0 \
         for _ = 1, 2000 do for _, x in pairs(v) do t = t + 1 end end print(t)",
    );
    assert!(value(&line, "abort_native.next") > 0, "{line}");
}

#[test]
fn counters_name_a_native_called_through_a_local_alias() {
    // `string.gsub` is read into a local before the loop; the name is the
    // one the installer gave it, not whatever `package.loaded` holds now.
    let line = tracing_line(
        "local gsub = string.gsub local s = string.rep('ab12 ', 40) local n = 0 \
         local f = function(d) return d end \
         for _ = 1, 3000 do local r, c = gsub(s, '%d+', f) n = n + c end print(n)",
    );
    assert!(value(&line, "abort_native.string.gsub") > 0, "{line}");
    // A string replacement is a helper: the loop traces.
    let line = tracing_line(
        "local gsub = string.gsub local s = string.rep('ab12 ', 40) local n = 0 \
         for _ = 1, 3000 do local r, c = gsub(s, '%d+', 'n') n = n + c end print(n)",
    );
    assert!(!line.contains("abort_native"), "{line}");
    // The method form names it too.
    let line = tracing_line(
        "local s = string.rep('ab12 ', 40) local n = 0 \
         local f = function(d) return d end \
         for _ = 1, 3000 do local r, c = s:gsub('%d+', f) n = n + c end print(n)",
    );
    assert!(value(&line, "abort_native.string.gsub") > 0, "{line}");
    let line = tracing_line(
        "local s = string.rep('ab12 ', 40) local n = 0 \
         for _ = 1, 3000 do local r, c = s:gsub('%d+', 'n') n = n + c end print(n)",
    );
    assert!(!line.contains("abort_native"), "{line}");
    // An argument that aborts is not blamed on the method it is passed to.
    let line = tracing_line(
        "local obj = setmetatable({}, {__index = function() return 2 end}) \
         local s = 'hello world' local n = 0 \
         for i = 1, 5000 do n = n + #s:sub(1, obj.x) end print(n)",
    );
    assert!(!line.contains("abort_native.string.sub"), "{line}");
    let line = tracing_line(
        "local clock = os.clock package.loaded.os = nil local n = 0 \
         for i = 1, 5000 do if clock() < 0 then n = n + 1 end n = n + i end print(n)",
    );
    assert!(value(&line, "abort_native.os.clock") > 0, "{line}");
}

#[test]
fn counters_name_the_utf8_and_pack_function_that_aborted_a_recording() {
    // More code points than a helper holds (four).
    let line = modern_line(
        "local s = ('abcdefgh'):rep(2) local n = 0 \
         for i = 1, 20000 do local a, b, c, d, e = utf8.codepoint(s, 1, 5) n = n + a + e end print(n)",
    );
    assert!(value(&line, "abort_native.utf8.codepoint") > 0, "{line}");
    // More values than a helper holds (three, and the position).
    let line = modern_line(
        "local n = 0 for i = 1, 20000 do \
         local a, b, c, d, p = string.unpack('<i1i1i1i1', '\\1\\2\\3\\4') n = n + a + d + p end print(n)",
    );
    assert!(value(&line, "abort_native.string.unpack") > 0, "{line}");
    // More values to pack than a helper takes (five).
    let line = modern_line(
        "local n = 0 for i = 1, 20000 do \
         n = n + #string.pack('<i1i1i1i1i1i1', 1, 2, 3, 4, 5, i % 100) end print(n)",
    );
    assert!(value(&line, "abort_native.string.pack") > 0, "{line}");
    // The calls the helpers take leave no blame.
    let line = modern_line(
        "local s = ('h\\u{E9}llo '):rep(4) local n = 0 \
         for i = 1, 20000 do local a, b = utf8.len(s) n = n + a + utf8.codepoint(s, 1) + (utf8.offset(s, 3) or 0) \
         local x = string.pack('<i4', i) n = n + (string.unpack('<i4', x)) + #utf8.char(i % 200 + 60) end print(n)",
    );
    assert!(!line.contains("abort_native"), "{line}");
    // A codes step that meets an invalid string while recording names
    // utf8.codes, though no library table holds the step.
    let line = modern_line(
        "local function walk(s) local n = 0 for _, c in utf8.codes(s) do n = n + c end return n end \
         local t = 0 for i = 1, 20000 do local ok, v = pcall(walk, 'ab\\226\\130') t = t + (ok and v or 1) end print(t)",
    );
    assert!(value(&line, "abort_native.utf8.codes") > 0, "{line}");
    // The loop over utf8.codes traces: its step is a helper.
    let line = modern_line(
        "local s = ('h\\u{E9}llo '):rep(2000) local n = 0 \
         for r = 1, 4 do for p, c in utf8.codes(s) do n = n + c end end print(n)",
    );
    assert!(!line.contains("abort_native"), "{line}");
}
