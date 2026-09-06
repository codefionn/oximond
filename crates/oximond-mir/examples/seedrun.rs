//! Runs one random seed natively and in the evaluator, printing the MIR.
//!   cargo run -p oximond-mir --example seedrun -- 0x9d88a676e7ec747f [out.bin]
#![allow(unsafe_code)]

#[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
mod real {

    use oximond_mir::exec::ExecMem;
    use oximond_mir::ir::Func;
    use oximond_mir::kernels::check_native;
    use oximond_mir::ra::Compiler;
    use oximond_mir::x64::X64;

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let seed = u64::from_str_radix(args[1].trim_start_matches("0x"), 16).unwrap();
        let mut f = Func::new();
        let (mut sc, _) = oximond_mir::randgen::generate(seed, &mut f);
        println!("{}", f.dump());
        let mut c = Compiler::new(X64::detect());
        let st = c.compile(&f);
        println!("{st:?}");
        let mut img = Vec::new();
        c.copy_code(&mut img);
        if let Some(p) = args.get(2) {
            std::fs::write(p, &img).unwrap();
        }
        let code = ExecMem::new(&[&img]);
        println!("{:?}", check_native(&f, &code, &mut sc));
    }
}

fn main() {
    #[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
    real::main();
    #[cfg(not(all(any(target_os = "linux", windows), target_arch = "x86_64")))]
    eprintln!("this tool runs on x86-64 Linux and Windows only");
}
