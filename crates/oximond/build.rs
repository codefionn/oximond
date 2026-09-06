use std::path::Path;
use std::process::Command;

fn main() {
    // wasm32-unknown-unknown has no native loader, C toolchain, or libffi.
    // Its unsupported backend is implemented in Rust instead.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        return;
    }
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" | "android" | "freebsd" | "openbsd" | "netbsd" | "dragonfly" | "macos" => {
            let mut native = cc::Build::new();
            native.file("src/ffi_native/native.c");
            if matches!(target_os.as_str(), "linux" | "android") {
                // Open libffi on the first foreign call. Linking it makes
                // every start map and bind libffi.so, about 30 us.
                native.define("OXIMOND_LAZY_LIBFFI", None);
                let soname = libffi_soname(&native);
                native.define("OXIMOND_LIBFFI_SONAME", Some(&*format!("\"{soname}\"")));
                println!("cargo:rustc-env=OXIMOND_LIBFFI_SONAME={soname}");
                println!("cargo:rustc-link-lib=dl");
            } else {
                println!("cargo:rustc-link-lib=ffi");
            }
            native.compile("oximond_ffi_native");
        }
        "windows" if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") => {
            cc::Build::new()
                .file("src/ffi_native/native_windows.c")
                .compile("oximond_ffi_native");
        }
        _ => {
            cc::Build::new()
                .file("src/ffi_native/unsupported.c")
                .compile("oximond_ffi_unsupported");
        }
    }
    println!("cargo:rerun-if-changed=src/ffi_native/native.c");
    println!("cargo:rerun-if-changed=src/ffi_native/native_windows.c");
    println!("cargo:rerun-if-changed=src/ffi_native/unsupported.c");
}

/// The soname of the libffi that `-lffi` would link. native.c is compiled
/// against the installed ffi.h, so it may only open the library of that
/// ABI: libffi.so.7 and libffi.so.8 lay out `ffi_cif` differently.
fn libffi_soname(build: &cc::Build) -> String {
    let compiler = build.get_compiler();
    let output = Command::new(compiler.path())
        .args(compiler.args())
        .arg("-print-file-name=libffi.so")
        .output();
    let path = output
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    if let Some(path) = path.as_deref().filter(|path| Path::new(path).is_absolute()) {
        println!("cargo:rerun-if-changed={path}");
        if let Some(soname) = std::fs::read(path).ok().and_then(|elf| elf_soname(&elf)) {
            return soname;
        }
    }
    panic!(
        "oximond: cannot find the soname of libffi.so (resolved to {path:?}); \
         install the libffi development files"
    );
}

/// `DT_SONAME` of an ELF shared object, read from its `SHT_DYNAMIC` section.
fn elf_soname(elf: &[u8]) -> Option<String> {
    if elf.get(..4)? != b"\x7fELF" {
        return None;
    }
    let wide = match elf.get(4)? {
        1 => false,
        2 => true,
        _ => return None,
    };
    let little = match elf.get(5)? {
        1 => true,
        2 => false,
        _ => return None,
    };
    let read = |at: usize, len: usize| -> Option<u64> {
        let bytes = elf.get(at..at.checked_add(len)?)?;
        let mut value = 0u64;
        for i in 0..len {
            let byte = if little { bytes[len - 1 - i] } else { bytes[i] };
            value = value << 8 | u64::from(byte);
        }
        Some(value)
    };
    let word = if wide { 8 } else { 4 };
    let usize_at = |at: usize, len: usize| read(at, len).and_then(|v| usize::try_from(v).ok());
    let (shoff, shentsize, shnum) = if wide {
        (usize_at(0x28, 8)?, usize_at(0x3a, 2)?, usize_at(0x3c, 2)?)
    } else {
        (usize_at(0x20, 4)?, usize_at(0x2e, 2)?, usize_at(0x30, 2)?)
    };
    // Offsets of sh_type, sh_offset, sh_size and sh_link in a section header.
    let (type_at, offset_at, size_at, link_at) = if wide {
        (4, 0x18, 0x20, 0x28)
    } else {
        (4, 0x10, 0x14, 0x18)
    };
    let header = |index: usize| shoff.checked_add(index.checked_mul(shentsize)?);
    for index in 0..shnum {
        let section = header(index)?;
        if read(section + type_at, 4)? != 6 {
            continue; // not SHT_DYNAMIC
        }
        let start = usize_at(section + offset_at, word)?;
        let size = usize_at(section + size_at, word)?;
        let strtab = header(usize_at(section + link_at, 4)?)?;
        let strings = usize_at(strtab + offset_at, word)?;
        for entry in (start..start.checked_add(size)?).step_by(2 * word) {
            match read(entry, word)? {
                0 => return None, // DT_NULL
                14 => {
                    // DT_SONAME
                    let name = strings.checked_add(usize_at(entry + word, word)?)?;
                    let tail = elf.get(name..)?;
                    let end = tail.iter().position(|&b| b == 0)?;
                    return String::from_utf8(tail[..end].to_vec()).ok();
                }
                _ => {}
            }
        }
    }
    None
}
