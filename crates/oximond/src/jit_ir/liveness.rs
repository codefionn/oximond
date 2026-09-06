//! Bytecode register observability for exact native exits.
//!
//! A native exit resumes the interpreter at some bytecode pc. Only registers
//! the rest of the frame can observe need their native values written back:
//! registers the interpreter may read before writing them again, named locals
//! in scope (visible through `debug.getlocal`), and registers captured by a
//! closure. Every other register may keep its entry value, because no Lua
//! program can tell the difference.
//!
//! The first two come from the chunk's own liveness
//! ([`Chunk::live_registers`]), the analysis the collector roots frames by.
//! The compiler does not clear dead temporaries, so a register outside that
//! set may hold any stale value (or a reference the collector has since
//! freed): native code must neither read it nor guard its type. Captured
//! registers are added on top for the whole chunk.

use crate::bytecode::{Chunk, UpvalueSource};

/// A set of the 256 addressable frame registers.
pub type RegisterSet = [u64; 4];

fn insert(set: &mut RegisterSet, register: u8) {
    set[usize::from(register) / 64] |= 1 << (register % 64);
}

#[must_use]
pub fn register_set_contains(set: &RegisterSet, register: u8) -> bool {
    set[usize::from(register) / 64] >> (register % 64) & 1 == 1
}

/// For every pc in `chunk`, the registers native code must write back when
/// it resumes the interpreter at that pc.
#[must_use]
pub fn observable_registers(chunk: &Chunk) -> Vec<RegisterSet> {
    let mut live = (0..chunk.code.len())
        .map(|pc| {
            let registers = chunk.live_registers(pc);
            let mut set = [0u64; 4];
            // Every register in the set lies below the live count.
            for register in (0..chunk.live_register_count(pc)).filter_map(|r| u8::try_from(r).ok())
            {
                if registers.contains(register) {
                    insert(&mut set, register);
                }
            }
            set
        })
        .collect::<Vec<_>>();
    let mut captured = [0u64; 4];
    for prototype in &chunk.prototypes {
        for upvalue in &prototype.upvalues {
            if let UpvalueSource::ParentRegister(register) = upvalue.source {
                insert(&mut captured, register);
            }
        }
    }
    for set in &mut live {
        for word in 0..4 {
            set[word] |= captured[word];
        }
    }
    live
}
