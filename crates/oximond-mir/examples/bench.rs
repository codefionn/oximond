//! Bench driver. One mode per process, so `perf stat` counts exactly what the
//! mode does; `bench.sh` differences two repeat counts to get a marginal cost.
//!
//!   bench mir-build K N      N x build kernel K into a reused Func
//!   bench mir-compile K N    N x compile with a reused Compiler
//!   bench mir-allocs K N     allocations per compile, steady state
//!   bench clif-translate K N N x MIR -> CLIF only
//!   bench clif-compile K N   N x the per-tree path of emit.rs (new JITModule,
//!                            translate, define, finalize)
//!   bench clif-allocs K N    allocations per compile
//!   bench mir-run K FUEL     compile once, run the loop FUEL iterations
//!   bench clif-run K FUEL
//!   bench mir-dump K FILE    machine code of the loop kernel
//!   bench clif-dump K FILE
//!
//! Output is one line of `key=value` pairs on stdout.
#![allow(unsafe_code)]

#[path = "../tests/common/clif.rs"]
mod clif;

#[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
mod real {
    use super::clif;

    use oximond_mir::exec::ExecMem;
    use oximond_mir::ir::Func;
    use oximond_mir::kernels::{self, Input, Kernel, Scenario};
    use oximond_mir::ra::Compiler;
    use oximond_mir::x64::X64;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::hint::black_box;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    struct Counting;
    static COUNT: AtomicU64 = AtomicU64::new(0);
    static BYTES: AtomicU64 = AtomicU64::new(0);

    // SAFETY: forwards to `System` unchanged and only counts around the calls.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size() as u64, Relaxed);
            unsafe { System.alloc(l) }
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size() as u64, Relaxed);
            unsafe { System.alloc_zeroed(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(n as u64, Relaxed);
            unsafe { System.realloc(p, l, n) }
        }
    }

    #[global_allocator]
    static A: Counting = Counting;

    fn allocs() -> (u64, u64) {
        (COUNT.load(Relaxed), BYTES.load(Relaxed))
    }

    fn kernel(name: &str) -> &'static Kernel {
        kernels::ALL
            .iter()
            .find(|k| k.name == name)
            .expect("kernel name")
    }

    /// A scenario that keeps the loop running for `fuel` iterations.
    fn run_scenario(k: &Kernel, f: &Func, fuel: i64) -> Scenario {
        let (_, mut data) = (k.scenarios)().remove(0);
        if k.name == "int_side" {
            data = (0..(fuel as usize + 80))
                .map(|i| (i * 37 % 900) as u64)
                .collect();
        }
        let mut sc = k.scenario(f, fuel, data);
        sc.fix_pointers(k.inputs);
        sc
    }

    type Entry = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64) -> i64;

    fn checksum(sc: &Scenario) -> u64 {
        sc.area
            .iter()
            .chain(sc.data.iter())
            .fold(0u64, |a, &w| a.rotate_left(5) ^ w)
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let (mode, kname, n) = (args[1].as_str(), args[2].as_str(), args[3].as_str());
        let k = kernel(kname);
        let mut f = Func::new();
        (k.build)(&mut f);
        let count = n.parse::<u64>().unwrap_or(0);
        match mode {
            "mir-build" => {
                for _ in 0..count {
                    (k.build)(black_box(&mut f));
                }
                println!("mir_insts={}", f.insts.len());
            }
            "mir-compile" => {
                let mut c = Compiler::new(X64::detect());
                let mut st = c.compile(&f);
                for _ in 0..count {
                    st = c.compile(black_box(&f));
                }
                println!(
                    "mir_insts={} code_bytes={} {st:?}",
                    f.insts.len(),
                    c.code_len()
                );
            }
            "mir-allocs" => {
                let mut c = Compiler::new(X64::detect());
                for _ in 0..3 {
                    c.compile(&f);
                }
                let before = allocs();
                for _ in 0..count {
                    c.compile(black_box(&f));
                }
                let after = allocs();
                println!(
                    "allocs_per_compile={:.3} bytes_per_compile={:.1}",
                    (after.0 - before.0) as f64 / count as f64,
                    (after.1 - before.1) as f64 / count as f64
                );
            }
            "clif-translate" => {
                let mut fbc = cranelift_frontend::FunctionBuilderContext::new();
                let mut insts = 0;
                for _ in 0..count {
                    let mut func = cranelift_codegen::ir::Function::new();
                    func.signature = clif::signature();
                    clif::translate(black_box(&f), &mut func, &mut fbc);
                    insts = func.dfg.num_insts();
                }
                println!("clif_insts={insts}");
            }
            "clif-compile" => {
                let mut insts = 0;
                for _ in 0..count {
                    let c = clif::compile(black_box(&f), false);
                    insts = c.clif_insts;
                    black_box(&c.entry);
                }
                println!("clif_insts={insts}");
            }
            "clif-allocs" => {
                drop(clif::compile(&f, false));
                let before = allocs();
                for _ in 0..count {
                    drop(black_box(clif::compile(&f, false)));
                }
                let after = allocs();
                println!(
                    "allocs_per_compile={:.1} bytes_per_compile={:.1}",
                    (after.0 - before.0) as f64 / count as f64,
                    (after.1 - before.1) as f64 / count as f64
                );
            }
            "mir-run" | "clif-run" => {
                let fuel = count as i64;
                let mut sc = run_scenario(k, &f, fuel);
                let (r, ck);
                // `BENCH_DUMP=FILE` writes the machine code, and the line
                // `code_at=ADDRESS` says where it ran, so a profile of the
                // run (`benchmarks/mir_profile.py`) can name each address.
                let dump = std::env::var("BENCH_DUMP").ok();
                if mode == "mir-run" {
                    let mut c = Compiler::new(X64::detect());
                    c.compile(&f);
                    let mut img = Vec::new();
                    c.copy_code(&mut img);
                    let code = ExecMem::new(&[&img]);
                    if let Some(path) = &dump {
                        std::fs::write(path, &img).unwrap();
                        println!("code_at={:#x} code_len={}", code.address(), img.len());
                    }
                    r = code.call(&mut sc.inputs, &mut sc.control, &mut sc.area);
                } else {
                    let code = clif::compile(&f, dump.is_some());
                    if let Some(path) = &dump {
                        std::fs::write(path, &code.code).unwrap();
                        println!(
                            "code_at={:#x} code_len={}",
                            code.entry as usize,
                            code.code.len()
                        );
                    }
                    // SAFETY: declared as (ptr, ptr, ptr) -> i64; buffers outlive the call.
                    let e: Entry = unsafe { std::mem::transmute::<*const u8, Entry>(code.entry) };
                    r = unsafe {
                        e(
                            sc.inputs.as_mut_ptr(),
                            sc.control.as_mut_ptr(),
                            sc.area.as_mut_ptr(),
                        )
                    };
                }
                ck = checksum(&sc);
                println!("exit={} checksum={ck:#x}", -1 - r);
            }
            "mir-dump" | "clif-dump" => {
                let mut img = Vec::new();
                if mode == "mir-dump" {
                    let mut c = Compiler::new(X64::detect());
                    c.compile(&f);
                    c.copy_code(&mut img);
                } else {
                    img = clif::compile(&f, true).code;
                }
                std::fs::write(n, &img).unwrap();
                println!("bytes={}", img.len());
            }
            _ => panic!("unknown mode"),
        }
        let _ = Input::DataPtr;
    }
}

fn main() {
    #[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
    real::main();
    #[cfg(not(all(any(target_os = "linux", windows), target_arch = "x86_64")))]
    eprintln!("this tool runs on x86-64 Linux and Windows only");
}
