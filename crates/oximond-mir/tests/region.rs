//! The executable region: carved per compile, bytes go back when a block is
//! dropped, and what is written there runs.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
#![allow(unsafe_code)]

use oximond_mir::exec::CodeRegion;

#[test]
fn blocks_are_aligned_distinct_and_go_back_to_the_region() {
    let region = CodeRegion::leaked(1 << 20).expect("maps");
    assert_eq!(region.usage(), (0, 0));
    let a = region.carve(100, |b| b.fill(0x90)).unwrap();
    let b = region.carve(100, |b| b.fill(0x90)).unwrap();
    let c = region.carve(5000, |b| b.fill(0x90)).unwrap();
    for block in [&a, &b, &c] {
        assert_eq!(block.address() % 16, 0, "entry points are 16-byte aligned");
        assert!(region.contains(block.address()));
    }
    assert_eq!(a.len(), 112);
    assert_eq!(b.address(), a.address() + 112);
    assert_eq!(c.address(), b.address() + 112);
    let (live, top) = region.usage();
    assert_eq!((live, top), (112 + 112 + 5008, 112 + 112 + 5008));
    // A freed block is reused by one that fits.
    let (a_address, b_address) = (a.address(), b.address());
    drop(a);
    let d = region.carve(50, |b| b.fill(0)).unwrap();
    assert_eq!(d.address(), a_address, "first fit into the hole");
    // Neighbouring holes merge, and a hole at the top lowers the top.
    drop(d);
    drop(b);
    let e = region.carve(200, |b| b.fill(0)).unwrap();
    assert_eq!(
        e.address(),
        a_address,
        "the two holes became one of 224 bytes"
    );
    let _ = b_address;
    drop(c);
    drop(e);
    assert_eq!(region.usage(), (0, 0), "everything went back");
}

#[test]
fn code_written_into_a_block_runs() {
    let region = CodeRegion::leaked(1 << 16).expect("maps");
    // mov rax, 42 ; ret
    let code = [0x48, 0xC7, 0xC0, 42, 0, 0, 0, 0xC3];
    let block = region
        .carve(code.len(), |b| b[..code.len()].copy_from_slice(&code))
        .unwrap();
    // SAFETY: the block holds a complete function of this signature.
    let run: extern "C" fn() -> i64 = unsafe { std::mem::transmute(block.address()) };
    assert_eq!(run(), 42);
    // The region is full when the carve cannot fit.
    assert!(region.carve(1 << 20, |_| {}).is_none());
}
