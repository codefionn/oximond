//! Code buffers. The body grows down (backward emission) inside one reusable
//! byte vector; exit stubs and the epilogue grow up in a second one and follow
//! the body in the final layout, so a body-to-stub branch is `body_distance +
//! stub_offset` and needs no fixup.

use crate::target::{Sink, Tgt};

#[derive(Default)]
pub struct CodeBuf {
    buf: Vec<u8>,
    /// Index of the lowest written byte; the body is `buf[pos..]`.
    pos: usize,
    /// Branches to the loop header, waiting for its address.
    pending_label: Vec<u32>,
    /// Branches to the outer headers (`Op::Join`): the field to patch and the
    /// region whose header it is.
    pending_outer: Vec<(u32, u32)>,
    /// Branches to the epilogue, waiting for the stub region's end.
    pending_epi: Vec<u32>,
    /// Calls of code at an offset from the image's first byte, waiting for
    /// the body's length: the field and the offset.
    pending_start: Vec<(u32, i32)>,
    /// Constant pool of the body and the branches waiting for it.
    pub pool: Vec<u64>,
    pending_pool: Vec<(u32, u32)>,
}

const HEADROOM: usize = 512;

impl CodeBuf {
    /// Empties the buffer, keeping its allocation. `hint` is a size estimate.
    pub fn reset(&mut self, hint: usize) {
        let want = hint.max(2 * HEADROOM);
        if self.buf.len() < want {
            self.buf.resize(want, 0);
        }
        self.pos = self.buf.len();
        self.pending_label.clear();
        self.pending_outer.clear();
        self.pending_epi.clear();
        self.pending_start.clear();
        self.pool.clear();
        self.pending_pool.clear();
    }

    /// Resolves the calls of code at an offset from the image's first byte,
    /// once the body is complete (its top is that byte).
    pub fn patch_start(&mut self) {
        let len = self.len() as i64;
        let pending = std::mem::take(&mut self.pending_start);
        for &(e, at) in &pending {
            // The field ends `len - e` bytes into the image.
            let rel = i64::from(at) - (len - i64::from(e));
            self.patch_rel(e, i32::try_from(rel).expect("a call within 2 GiB"));
        }
        self.pending_start = pending;
        self.pending_start.clear();
    }

    /// Resolves pool references once the stub region has `stub_len` bytes.
    pub fn patch_pool(&mut self, stub_len: u32) {
        let pending = std::mem::take(&mut self.pending_pool);
        for &(e, k) in &pending {
            self.patch_rel(e, (stub_len + 8 * k + e) as i32);
        }
        self.pending_pool = pending;
        self.pending_pool.clear();
    }

    /// Makes sure the next instruction fits; grows by moving the tail up.
    #[inline]
    pub fn ensure(&mut self) {
        if self.pos < HEADROOM {
            let used = self.buf.len() - self.pos;
            let new_len = self.buf.len() * 2;
            self.buf.resize(new_len, 0);
            let len = self.buf.len();
            self.buf.copy_within(self.pos..self.pos + used, len - used);
            self.pos = len - used;
        }
    }

    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.buf[self.pos..]
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn patch_rel(&mut self, t_end: u32, rel: i32) {
        let end = self.buf.len() - t_end as usize;
        self.buf[end - 4..end].copy_from_slice(&rel.to_le_bytes());
    }

    /// The loop header is at the current position.
    pub fn bind_label(&mut self) {
        let t = self.mark();
        let pending = std::mem::take(&mut self.pending_label);
        for &e in &pending {
            self.patch_rel(e, e as i32 - t as i32);
        }
        self.pending_label = pending;
        self.pending_label.clear();
    }

    /// The outer header (join) with label `region` is at the current
    /// position.
    pub fn bind_outer(&mut self, region: u32) {
        let t = self.mark();
        let pending = std::mem::take(&mut self.pending_outer);
        for &(e, r) in &pending {
            if r == region {
                self.patch_rel(e, e as i32 - t as i32);
            }
        }
        self.pending_outer = pending;
        self.pending_outer.retain(|&(_, r)| r != region);
    }

    /// The epilogue starts at stub offset `s_epi`.
    pub fn patch_epilogue(&mut self, s_epi: u32) {
        let pending = std::mem::take(&mut self.pending_epi);
        for &e in &pending {
            self.patch_rel(e, (s_epi + e) as i32);
        }
        self.pending_epi = pending;
        self.pending_epi.clear();
    }
}

impl Sink for CodeBuf {
    const BACKWARD: bool = true;

    #[inline]
    fn mark(&self) -> u32 {
        (self.buf.len() - self.pos) as u32
    }

    #[inline]
    fn put(&mut self, bytes: &[u8]) {
        self.pos -= bytes.len();
        self.buf[self.pos..self.pos + bytes.len()].copy_from_slice(bytes);
    }

    fn rel(&mut self, tgt: Tgt, end: u32) -> Option<i32> {
        match tgt {
            Tgt::Body(t) => Some(end as i32 - t as i32),
            Tgt::Stub(s) => Some((s + end) as i32),
            Tgt::Epilogue => {
                self.pending_epi.push(end);
                None
            }
            Tgt::Label => {
                self.pending_label.push(end);
                None
            }
            Tgt::Outer(region) => {
                self.pending_outer.push((end, region));
                None
            }
            Tgt::Pool(k) => {
                self.pending_pool.push((end, k));
                None
            }
            Tgt::Start(at) => {
                self.pending_start.push((end, at));
                None
            }
        }
    }

    fn pool(&mut self, bits: u64) -> u32 {
        pool_index(&mut self.pool, bits)
    }
}

fn pool_index(pool: &mut Vec<u64>, bits: u64) -> u32 {
    if let Some(k) = pool.iter().position(|&b| b == bits) {
        return k as u32;
    }
    pool.push(bits);
    (pool.len() - 1) as u32
}

#[derive(Default)]
pub struct Stubs {
    pub bytes: Vec<u8>,
    pending_epi: Vec<u32>,
    pub pool: Vec<u64>,
    pending_pool: Vec<(u32, u32)>,
}

impl Stubs {
    pub fn reset(&mut self) {
        self.bytes.clear();
        self.pending_epi.clear();
        self.pool.clear();
        self.pending_pool.clear();
    }

    /// Resolves pool references; the stub pool follows the body pool of
    /// `body_pool_len` entries after the stubs.
    pub fn patch_pool(&mut self, body_pool_len: u32) {
        let len = self.bytes.len() as u32;
        let pending = std::mem::take(&mut self.pending_pool);
        for &(e, k) in &pending {
            let rel = (len + 8 * (body_pool_len + k)) as i32 - e as i32;
            self.bytes[e as usize - 4..e as usize].copy_from_slice(&rel.to_le_bytes());
        }
        self.pending_pool = pending;
        self.pending_pool.clear();
    }

    /// Resolves jumps to the epilogue, which starts at the current end.
    pub fn patch_epilogue(&mut self, s_epi: u32) {
        let pending = std::mem::take(&mut self.pending_epi);
        for &e in &pending {
            let rel = s_epi as i32 - e as i32;
            self.bytes[e as usize - 4..e as usize].copy_from_slice(&rel.to_le_bytes());
        }
        self.pending_epi = pending;
        self.pending_epi.clear();
    }
}

impl Sink for Stubs {
    const BACKWARD: bool = false;

    fn mark(&self) -> u32 {
        self.bytes.len() as u32
    }

    fn put(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn rel(&mut self, tgt: Tgt, end: u32) -> Option<i32> {
        match tgt {
            Tgt::Body(t) => Some(-(t as i32) - end as i32),
            Tgt::Stub(s) => Some(s as i32 - end as i32),
            Tgt::Epilogue => {
                self.pending_epi.push(end);
                None
            }
            Tgt::Label | Tgt::Outer(_) => panic!("stubs never jump to a loop header"),
            Tgt::Start(_) => panic!("stubs never call the image's code"),
            Tgt::Pool(k) => {
                self.pending_pool.push((end, k));
                None
            }
        }
    }

    fn pool(&mut self, bits: u64) -> u32 {
        pool_index(&mut self.pool, bits)
    }
}
