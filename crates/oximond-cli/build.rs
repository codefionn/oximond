fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=startup-order.txt");
    println!("cargo:rerun-if-env-changed=OXIMOND_NO_SYMBOL_ORDER");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os == "linux" && target_env == "gnu" {
        export_c_symbols();
        startup_layout();
    } else {
        // Native modules resolve the Lua API against the executable itself.
        #[cfg(unix)]
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
    }
}

/// Exports the C names of the executable, and no others. Native modules and
/// the C++ bridge resolve `lua_*`, `luaL_*`, `oxi_*` and `oximond_*` against
/// the executable, and `src/libm.rs` defines the libm functions that native
/// code may import. `--export-dynamic` would add every Rust symbol as well:
/// 3,100 entries, a 410 KB string table and a 24 KB hash table that ld.so
/// probes for each of the 143 symbols it binds at start.
///
/// The unwinder that `startup_layout` links in from `libgcc_eh` is not
/// exported either, so the C++ bridge, once loaded, brings `libgcc_s` and the
/// process holds two unwinders. That is safe as long as no exception crosses
/// from one to the other: C++ exceptions are thrown, resumed and caught in
/// `libgcc_s`, inside the bridge and the modules it calls, and Rust panics stay
/// in Rust frames. `cxx_exceptions_are_caught_inside_cxx_modules` in
/// `tests/native_modules.rs` checks it.
fn export_c_symbols() {
    let patterns = [
        "lua*", "oxi*", "acos", "asin", "atan", "atan2", "cos", "cosh", "exp", "log", "log10",
        "log2", "pow", "sin", "sinh", "tan", "tanh",
    ];
    for pattern in patterns {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic-symbol={pattern}");
    }
}

/// Link steps that shorten process start on linux-gnu. `startup/empty` is a
/// 0.6 ms run, and each step below removes a share of it.
fn startup_layout() {
    // Put the functions that every start runs next to each other. A start
    // touches about 440 KB of code spread over 3.5 MB, which cost 55 page
    // faults; packed, it costs 9. The list goes on with the functions that
    // warming up a trace runs after them (the compiler's, mostly): packed,
    // a tracing run of `algorithm/nbody` takes 70 fewer page faults and
    // 0.7M fewer cycles. The list comes from `scripts/startup-order.sh`. It
    // goes to the linker as a script that collects the functions' sections
    // ahead of `.text`, not as a `--symbol-ordering-file`: LTO renames a
    // function it promotes to `name.llvm.<hash>`, and the hash changes with
    // the build directory, so a list of exact names matched about 30 of its
    // 285 entries in a build from another directory (19 more page faults). A
    // section pattern ends in `*` there. A name that no longer exists matches
    // nothing, so a stale list only loses part of the gain. gold and mold do
    // not take the script: see `linker_takes_script`.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let order = std::path::Path::new(&manifest).join("startup-order.txt");
    if linker_takes_script()
        && let Ok(names) = std::fs::read_to_string(&order)
    {
        let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap_or_default());
        let script = out.join("startup-order.ld");
        if std::fs::write(&script, ordering_script(&names)).is_ok() {
            println!("cargo:rustc-link-arg=-Wl,-T,{}", script.display());
        }
    }
    // Link the unwinder into the executable instead of loading libgcc_s.so
    // at start, which cost about 30 us and 8 page faults. The archive is
    // linked whole through `rustc-link-lib`, which places it ahead of rustc's
    // `-lgcc_s`, so its definitions replace the ones libgcc_s would provide
    // and the linker drops libgcc_s as unneeded. A trailing link-arg would
    // not do: GNU ld binds a symbol to the first definition it sees, so with
    // the archive last it kept libgcc_s NEEDED (and failed on `__tls_get_addr`
    // with libc already passed). `binary_does_not_link_libgcc_s` guards the
    // default build; a `-C link-arg=-fuse-ld=bfd` build also lists only libc
    // and ld.so. The C++ bridge still loads libgcc_s for itself: see
    // `export_c_symbols`.
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    let eh = std::process::Command::new(cc)
        .arg("-print-file-name=libgcc_eh.a")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_owned())
        .filter(|path| {
            std::path::Path::new(path).is_absolute() && std::path::Path::new(path).exists()
        });
    // A copy in OUT_DIR keeps the search path from offering the rest of the
    // GCC library directory to the link.
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap_or_default());
    if let Some(path) = eh
        && std::fs::copy(&path, out.join("libgcc_eh.a")).is_ok()
    {
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static:+whole-archive=gcc_eh");
    }
}

/// Whether the linker takes the ordering script. lld and GNU ld accept a
/// `SECTIONS` block with `INSERT BEFORE .text`. gold rejects `INSERT`, and
/// mold takes only scripts without `SECTIONS`, so flags that name either
/// skip the ordering. Set `OXIMOND_NO_SYMBOL_ORDER` to skip it with any linker.
fn linker_takes_script() -> bool {
    if std::env::var_os("OXIMOND_NO_SYMBOL_ORDER").is_some() {
        return false;
    }
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    !flags
        .split('\x1f')
        .any(|f| f.contains("gold") || f.contains("mold"))
}

/// A linker script that places the sections of the listed functions, in list
/// order, in an output section ahead of `.text`.
fn ordering_script(names: &str) -> String {
    use std::fmt::Write;
    let mut script = String::from("SECTIONS\n{\n  .text.startup_order :\n  {\n");
    for name in names.lines().map(str::trim).filter(|name| !name.is_empty()) {
        // `name.llvm.<hash>`: the hash is not stable, so match any.
        let pattern = match name.find(".llvm.") {
            Some(at) => format!("{}.llvm.*", &name[..at]),
            None => name.to_owned(),
        };
        // A cold function (`grow_one`, `finish_grow`) sits in `.text.unlikely.`.
        let _ = writeln!(script, "    *(.text.{pattern} .text.unlikely.{pattern})");
    }
    script.push_str("  }\n}\nINSERT BEFORE .text;\n");
    script
}
