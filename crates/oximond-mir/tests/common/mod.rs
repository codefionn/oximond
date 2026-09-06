#![allow(dead_code)]
use oximond_mir::exec::ExecMem;
use oximond_mir::ir::Func;
use oximond_mir::ra::{Compiler, Stats};
use oximond_mir::x64::X64;

/// Both encodings: AVX when the CPU has it, and plain SSE2.
pub fn targets() -> Vec<X64> {
    let mut v = vec![X64::sse2()];
    if X64::detect().avx {
        v.push(X64::detect());
    }
    v
}

pub fn compile(f: &Func) -> (ExecMem, Stats) {
    compile_on(X64::detect(), f)
}

pub fn compile_on(t: X64, f: &Func) -> (ExecMem, Stats) {
    let mut c = Compiler::new(t);
    let st = c.compile(f);
    let mut img = Vec::new();
    c.copy_code(&mut img);
    (ExecMem::new(&[&img]), st)
}

pub fn compile_with(c: &mut Compiler<X64>, f: &Func) -> (ExecMem, Stats) {
    let st = c.compile(f);
    let mut img = Vec::new();
    c.copy_code(&mut img);
    (ExecMem::new(&[&img]), st)
}
