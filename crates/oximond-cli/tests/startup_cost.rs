//! What a bare start of the executable pays for. `startup/empty` is a 0.6 ms
//! run, so a library mapped at start, a page of code faulted in, or a
//! teardown nobody sees shows in it.

use std::path::PathBuf;
use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_oximond")
}

fn scratch(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("oximond-startup-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn needed_libraries() -> String {
    let output = Command::new("readelf")
        .args(["-d", binary()])
        .output()
        .expect("run readelf");
    assert!(output.status.success(), "{output:?}");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// libffi.so is opened on the first foreign call, not mapped and bound at
/// every start (about 30 us and 8 page faults).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn binary_does_not_link_libffi() {
    let needed = needed_libraries();
    assert!(!needed.contains("libffi"), "{needed}");
}

/// The unwinder is linked into the executable when the toolchain has it, so
/// `libgcc_s.so` is not loaded at start (about 30 us and 8 page faults).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn binary_does_not_link_libgcc_s() {
    let cc = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
    let archive = Command::new(cc)
        .arg("-print-file-name=libgcc_eh.a")
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default();
    if !std::path::Path::new(&archive).is_absolute() {
        return;
    }
    let needed = needed_libraries();
    assert!(!needed.contains("libgcc_s"), "{needed}");
}

/// libm.so is opened on the first math call, not mapped and bound at every
/// start (about 30 us, 37,000 instructions and 8 page faults).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn binary_does_not_link_libm() {
    let needed = needed_libraries();
    assert!(!needed.contains("libm.so"), "{needed}");
}

/// The loader copies the thread-local image into every thread when it starts
/// and zeroes the rest of the block, used or not. A 64 KiB pattern cache in a
/// thread local once cost each start 16 page faults and 12,000 instructions.
/// A large per-thread table belongs on the heap, made on first use.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn thread_local_block_stays_small() {
    let output = Command::new("readelf")
        .args(["-lW", binary()])
        .output()
        .expect("run readelf");
    assert!(output.status.success(), "{output:?}");
    let headers = String::from_utf8_lossy(&output.stdout);
    let Some(tls) = headers
        .lines()
        .find(|line| line.trim_start().starts_with("TLS "))
    else {
        return;
    };
    // Type, Offset, VirtAddr, PhysAddr, FileSiz, MemSiz, ...
    let memory_size = tls
        .split_whitespace()
        .nth(5)
        .and_then(|field| u64::from_str_radix(field.trim_start_matches("0x"), 16).ok())
        .expect("TLS header has a MemSiz");
    assert!(memory_size <= 16 * 1024, "{tls}");
}

/// The functions the executable defines in libm's place give libm's results,
/// and libm is not mapped until the first call. `ffi.C` still finds libm
/// functions the executable does not define, and the executable exports
/// its Lua API and not its Rust symbols.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn libm_opens_on_first_call() {
    let script = r#"
local function mapped()
  local file = assert(io.open("/proc/self/maps"))
  local text = file:read("*a")
  file:close()
  return text:find("libm.so", 1, true) ~= nil
end
local before = mapped()
local ok = math.sin(1) == 0.8414709848082 or math.abs(math.sin(1) - 0.8414709848082) < 1e-12
local all = math.cos(0) == 1 and math.exp(0) == 1 and math.log(1) == 0 and 2 ^ 3 == 8
all = all and math.abs(math.atan2(1, 1) - math.pi / 4) < 1e-15
all = all and math.abs(math.tan(1) - 1.5574077246549) < 1e-12 and math.floor(2.5) == 2
print(before, ok, all, mapped())
"#;
    let output = Command::new(binary())
        .args(["--compat=lua51", "-e", script])
        .env_remove("LUA_INIT")
        .output()
        .expect("run oximond");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"false\ttrue\ttrue\ttrue\n");
}

/// `ffi.C` reaches libm functions the executable does not define.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn ffi_finds_libm_functions() {
    let script = r#"
local ffi = require("ffi")
ffi.cdef[[double cbrt(double); double sin(double);]]
print(math.abs(ffi.C.cbrt(27) - 3) < 1e-12, math.abs(ffi.C.sin(1) - math.sin(1)) < 1e-15)
"#;
    let output = Command::new(binary())
        .args(["--compat=luajit21", "-e", script])
        .env_remove("LUA_INIT")
        .output()
        .expect("run oximond");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"true\ttrue\n");
}

/// Only the C names are in the dynamic symbol table. `--export-dynamic` put
/// 3,100 Rust symbols there, a 410 KB string table that ld.so searched for
/// every symbol it binds.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn dynamic_symbols_are_the_c_api() {
    let output = Command::new("readelf")
        .args(["--dyn-syms", "-W", binary()])
        .output()
        .expect("run readelf");
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8_lossy(&output.stdout);
    let defined: Vec<&str> = text
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            (fields.len() >= 8 && fields[6] != "UND").then(|| fields[7])
        })
        .collect();
    assert!(defined.contains(&"lua_pushnumber"), "{defined:?}");
    assert!(defined.contains(&"sin"), "{defined:?}");
    assert!(
        defined.len() < 400,
        "{} symbols are exported",
        defined.len()
    );
}

/// The FFI still works with libffi loaded lazily, and libffi is not mapped
/// until the first call.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn ffi_opens_libffi_on_first_call() {
    let script = r#"
local function mapped()
  local file = assert(io.open("/proc/self/maps"))
  local text = file:read("*a")
  file:close()
  return text:find("libffi", 1, true) ~= nil
end
local ffi = require("ffi")
ffi.cdef[[int abs(int);]]
local before = mapped()
local value = ffi.C.abs(-7)
print(before, value, mapped())
"#;
    let output = Command::new(binary())
        .args(["--compat=luajit21", "-e", script])
        .env_remove("LUA_INIT")
        .output()
        .expect("run oximond");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"false\t7\ttrue\n");
}

/// A start leaves without freeing the heap, so the exit must still flush a
/// file the script never closed.
#[test]
fn unclosed_file_is_flushed_at_exit() {
    let directory = scratch("flush");
    let path = directory.join("out.txt");
    let script = format!(
        "local file = assert(io.open({:?}, 'w')); file:setvbuf('full'); file:write('kept ', 'text')",
        path.to_str().unwrap()
    );
    let output = Command::new(binary())
        .args(["--compat=lua51", "-e", &script])
        .output()
        .expect("run oximond");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"kept text");
    std::fs::remove_dir_all(directory).unwrap();
}

/// A pipe's child is waited for before the process leaves.
#[cfg(unix)]
#[test]
fn unclosed_pipe_is_waited_for_at_exit() {
    let directory = scratch("pipe");
    let path = directory.join("pipe.txt");
    let script = format!(
        "local pipe = assert(io.popen('cat > ' .. {:?}, 'w')); pipe:write('through the pipe')",
        path.to_str().unwrap()
    );
    let output = Command::new(binary())
        .args(["--compat=lua51", "-e", &script])
        .output()
        .expect("run oximond");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"through the pipe");
    std::fs::remove_dir_all(directory).unwrap();
}

/// The budget of a bare tracing start, from `perf stat` pinned to nothing in
/// particular. Skipped without perf, and in a debug build, which is several
/// times larger. `startup/empty` was 1.05M instructions and 250 page faults
/// before the start was trimmed; lua5.1 takes 111 faults and 1.5M
/// instructions.
#[cfg(all(target_os = "linux", not(debug_assertions)))]
#[test]
fn bare_start_stays_within_its_budget() {
    let directory = scratch("budget");
    let empty = directory.join("empty.lua");
    std::fs::write(&empty, "").unwrap();
    let output = match Command::new("perf")
        .args(["stat", "-x,", "-e", "instructions:u,page-faults", "--"])
        .arg(binary())
        .args(["--compat=lua51", "--execution=tracing", "-E"])
        .arg(&empty)
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => {
            std::fs::remove_dir_all(directory).unwrap();
            return;
        }
    };
    let report = String::from_utf8_lossy(&output.stderr);
    let count = |event: &str| -> Option<u64> {
        report
            .lines()
            .find(|line| line.contains(event))
            .and_then(|line| line.split(',').next())
            .and_then(|field| field.trim().parse().ok())
    };
    if let (Some(instructions), Some(faults)) = (count("instructions"), count("page-faults")) {
        assert!(instructions <= 1_100_000, "{instructions} instructions");
        assert!(faults <= 180, "{faults} page faults");
    }
    std::fs::remove_dir_all(directory).unwrap();
}
