//! Writes the machine code of a kernel (or a random program) to a file, for
//! `objdump -D -b binary -mi386:x86-64 -M intel`.
//!
//!   cargo run -p oximond-mir --example dump -- nbody out.bin
//!   cargo run -p oximond-mir --example dump -- seed:42 out.bin

use oximond_mir::ir::Func;
use oximond_mir::ra::Compiler;
use oximond_mir::x64::X64;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut f = Func::new();
    if let Some(seed) = args[1].strip_prefix("seed:") {
        oximond_mir::randgen::generate(seed.parse().unwrap(), &mut f);
    } else {
        let k = oximond_mir::kernels::ALL
            .iter()
            .find(|k| k.name == args[1])
            .expect("kernel name");
        (k.build)(&mut f);
    }
    let mut c = Compiler::new(X64::detect());
    let stats = c.compile(&f);
    let mut img = Vec::new();
    c.copy_code(&mut img);
    let (body, _) = c.code_parts();
    std::fs::write(&args[2], &img).unwrap();
    eprintln!(
        "{stats:?}\nbody {} bytes, image {} bytes",
        body.len(),
        img.len()
    );
}
