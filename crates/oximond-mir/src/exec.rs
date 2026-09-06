//! Executable memory: map read-write, copy the code in, flip to read-execute.
//! x86 needs no instruction-cache maintenance. This is the only module (with
//! the bench example) that uses unsafe.
#![allow(unsafe_code)]

/// The operating system's half: Linux maps with `mmap`, Windows reserves and
/// commits with `VirtualAlloc`.
#[cfg(target_os = "linux")]
mod os {
    use std::ffi::c_void;

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;
    const MAP_PRIVATE: i32 = 2;
    const MAP_ANONYMOUS: i32 = 0x20;
    const MAP_NORESERVE: i32 = 0x4000;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
    }

    fn map(len: usize, prot: i32, flags: i32) -> Option<*mut u8> {
        // SAFETY: a fresh anonymous private mapping; the arguments are valid.
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, prot, flags, -1, 0) };
        (ptr as isize != -1).then_some(ptr.cast())
    }

    /// `len` bytes of fresh read-write memory.
    pub(super) fn map_rw(len: usize) -> Option<*mut u8> {
        map(len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS)
    }

    /// Makes a mapping from [`map_rw`] read-execute.
    pub(super) fn protect_rx(ptr: *mut u8, len: usize) -> bool {
        // SAFETY: the caller passes a whole mapping from `map_rw`.
        unsafe { mprotect(ptr.cast(), len, PROT_READ | PROT_EXEC) == 0 }
    }

    pub(super) fn unmap(ptr: *mut u8, len: usize) {
        // SAFETY: the caller passes a whole mapping it owns, once.
        unsafe { munmap(ptr.cast(), len) };
    }

    /// Address space for `len` bytes of read-write-execute memory. Nothing is
    /// committed: pages are backed as code is written.
    pub(super) fn reserve_rwx(len: usize) -> Option<*mut u8> {
        map(
            len,
            PROT_READ | PROT_WRITE | PROT_EXEC,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
        )
    }

    /// Backs `ptr..ptr + len` of a reservation. The kernel does it on first
    /// touch.
    pub(super) fn commit_rwx(_ptr: *mut u8, _len: usize) -> bool {
        true
    }
}

#[cfg(windows)]
mod os {
    use std::ffi::c_void;

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;
    const PAGE_EXECUTE_READ: u32 = 0x20;
    const PAGE_EXECUTE_READWRITE: u32 = 0x40;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualProtect(addr: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
    }

    pub(super) fn map_rw(len: usize) -> Option<*mut u8> {
        // SAFETY: a fresh allocation; the arguments are valid.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        (!ptr.is_null()).then_some(ptr.cast())
    }

    pub(super) fn protect_rx(ptr: *mut u8, len: usize) -> bool {
        let mut old = 0;
        // SAFETY: the caller passes a whole allocation from `map_rw`.
        unsafe { VirtualProtect(ptr.cast(), len, PAGE_EXECUTE_READ, &raw mut old) != 0 }
    }

    pub(super) fn unmap(ptr: *mut u8, _len: usize) {
        // SAFETY: the caller passes a whole allocation it owns, once.
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }

    /// Reserves address space only. Windows charges committed pages against
    /// the commit limit whether or not they are touched, so the region commits
    /// as it grows (see [`commit_rwx`]).
    pub(super) fn reserve_rwx(len: usize) -> Option<*mut u8> {
        // SAFETY: a fresh reservation; the arguments are valid.
        let ptr = unsafe { VirtualAlloc(std::ptr::null_mut(), len, MEM_RESERVE, PAGE_NOACCESS) };
        (!ptr.is_null()).then_some(ptr.cast())
    }

    pub(super) fn commit_rwx(ptr: *mut u8, len: usize) -> bool {
        // SAFETY: the caller passes a range inside a reservation from
        // `reserve_rwx`; committing committed pages again is allowed.
        let got = unsafe { VirtualAlloc(ptr.cast(), len, MEM_COMMIT, PAGE_EXECUTE_READWRITE) };
        !got.is_null()
    }
}

/// `(inputs, control, exit_area) -> -1 - exit index`
pub type Entry = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64) -> i64;

// Calls `entry(inputs, control, area)` with distinct values in every
// callee-saved register and reports which ones changed, so a missing push in
// a generated prologue shows up as a test failure instead of corrupting the
// caller. rdi = entry, rsi = inputs, rdx = control, rcx = area, r8 = out mask.
#[cfg(not(windows))]
std::arch::global_asm!(
    ".globl oximond_mir_call_checked",
    "oximond_mir_call_checked:",
    "push rbx",
    "push rbp",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "push r8",
    "movabs rbx, 0x1111111111111111",
    "movabs rbp, 0x2222222222222222",
    "movabs r12, 0x3333333333333333",
    "movabs r13, 0x4444444444444444",
    "movabs r14, 0x5555555555555555",
    "movabs r15, 0x6666666666666666",
    "mov rax, rdi",
    "mov rdi, rsi",
    "mov rsi, rdx",
    "mov rdx, rcx",
    "call rax",
    "xor r9d, r9d",
    "movabs r10, 0x1111111111111111",
    "cmp rbx, r10",
    "jz 2f",
    "or r9d, 1",
    "2:",
    "movabs r10, 0x2222222222222222",
    "cmp rbp, r10",
    "jz 3f",
    "or r9d, 2",
    "3:",
    "movabs r10, 0x3333333333333333",
    "cmp r12, r10",
    "jz 4f",
    "or r9d, 4",
    "4:",
    "movabs r10, 0x4444444444444444",
    "cmp r13, r10",
    "jz 5f",
    "or r9d, 8",
    "5:",
    "movabs r10, 0x5555555555555555",
    "cmp r14, r10",
    "jz 6f",
    "or r9d, 16",
    "6:",
    "movabs r10, 0x6666666666666666",
    "cmp r15, r10",
    "jz 7f",
    "or r9d, 32",
    "7:",
    "pop r8",
    "mov [r8], r9",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbp",
    "pop rbx",
    "ret",
);

// The same for Win64, which also keeps rdi, rsi and xmm6 to xmm15. rcx =
// entry, rdx = inputs, r8 = control, r9 = area, the out mask on the stack. The
// mask bits are rbx rbp rdi rsi r12 r13 r14 r15, then xmm6 to xmm15 (all 128
// bits).
#[cfg(windows)]
std::arch::global_asm!(
    ".globl oximond_mir_call_checked",
    "oximond_mir_call_checked:",
    "push rbx",
    "push rbp",
    "push rdi",
    "push rsi",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "mov rax, [rsp + 104]",
    "sub rsp, 360",
    "mov [rsp + 352], rax",
    "movups [rsp + 32], xmm6",
    "movups [rsp + 48], xmm7",
    "movups [rsp + 64], xmm8",
    "movups [rsp + 80], xmm9",
    "movups [rsp + 96], xmm10",
    "movups [rsp + 112], xmm11",
    "movups [rsp + 128], xmm12",
    "movups [rsp + 144], xmm13",
    "movups [rsp + 160], xmm14",
    "movups [rsp + 176], xmm15",
    "mov r10, rcx",
    "mov rcx, rdx",
    "mov rdx, r8",
    "mov r8, r9",
    "movabs rbx, 0x1111111111111111",
    "movabs rbp, 0x2222222222222222",
    "movabs rdi, 0x3333333333333333",
    "movabs rsi, 0x4444444444444444",
    "movabs r12, 0x5555555555555555",
    "movabs r13, 0x6666666666666666",
    "movabs r14, 0x7777777777777777",
    "movabs r15, 0x8888888888888888",
    "movabs rax, 0x0606060606060606",
    "movq xmm6, rax",
    "movabs rax, 0x0707070707070707",
    "movq xmm7, rax",
    "movabs rax, 0x0808080808080808",
    "movq xmm8, rax",
    "movabs rax, 0x0909090909090909",
    "movq xmm9, rax",
    "movabs rax, 0x0a0a0a0a0a0a0a0a",
    "movq xmm10, rax",
    "movabs rax, 0x0b0b0b0b0b0b0b0b",
    "movq xmm11, rax",
    "movabs rax, 0x0c0c0c0c0c0c0c0c",
    "movq xmm12, rax",
    "movabs rax, 0x0d0d0d0d0d0d0d0d",
    "movq xmm13, rax",
    "movabs rax, 0x0e0e0e0e0e0e0e0e",
    "movq xmm14, rax",
    "movabs rax, 0x0f0f0f0f0f0f0f0f",
    "movq xmm15, rax",
    "call r10",
    "xor r9d, r9d",
    "movabs r10, 0x1111111111111111",
    "cmp rbx, r10",
    "jz 2f",
    "or r9d, 1",
    "2:",
    "movabs r10, 0x2222222222222222",
    "cmp rbp, r10",
    "jz 3f",
    "or r9d, 2",
    "3:",
    "movabs r10, 0x3333333333333333",
    "cmp rdi, r10",
    "jz 4f",
    "or r9d, 4",
    "4:",
    "movabs r10, 0x4444444444444444",
    "cmp rsi, r10",
    "jz 5f",
    "or r9d, 8",
    "5:",
    "movabs r10, 0x5555555555555555",
    "cmp r12, r10",
    "jz 6f",
    "or r9d, 16",
    "6:",
    "movabs r10, 0x6666666666666666",
    "cmp r13, r10",
    "jz 7f",
    "or r9d, 32",
    "7:",
    "movabs r10, 0x7777777777777777",
    "cmp r14, r10",
    "jz 8f",
    "or r9d, 64",
    "8:",
    "movabs r10, 0x8888888888888888",
    "cmp r15, r10",
    "jz 9f",
    "or r9d, 128",
    "9:",
    "movups [rsp + 192], xmm6",
    "movups [rsp + 208], xmm7",
    "movups [rsp + 224], xmm8",
    "movups [rsp + 240], xmm9",
    "movups [rsp + 256], xmm10",
    "movups [rsp + 272], xmm11",
    "movups [rsp + 288], xmm12",
    "movups [rsp + 304], xmm13",
    "movups [rsp + 320], xmm14",
    "movups [rsp + 336], xmm15",
    "movabs r10, 0x0606060606060606",
    "cmp [rsp + 192], r10",
    "jnz 12f",
    "cmp qword ptr [rsp + 200], 0",
    "jz 13f",
    "12:",
    "or r9d, 256",
    "13:",
    "movabs r10, 0x0707070707070707",
    "cmp [rsp + 208], r10",
    "jnz 14f",
    "cmp qword ptr [rsp + 216], 0",
    "jz 15f",
    "14:",
    "or r9d, 512",
    "15:",
    "movabs r10, 0x0808080808080808",
    "cmp [rsp + 224], r10",
    "jnz 16f",
    "cmp qword ptr [rsp + 232], 0",
    "jz 17f",
    "16:",
    "or r9d, 1024",
    "17:",
    "movabs r10, 0x0909090909090909",
    "cmp [rsp + 240], r10",
    "jnz 18f",
    "cmp qword ptr [rsp + 248], 0",
    "jz 19f",
    "18:",
    "or r9d, 2048",
    "19:",
    "movabs r10, 0x0a0a0a0a0a0a0a0a",
    "cmp [rsp + 256], r10",
    "jnz 20f",
    "cmp qword ptr [rsp + 264], 0",
    "jz 21f",
    "20:",
    "or r9d, 4096",
    "21:",
    "movabs r10, 0x0b0b0b0b0b0b0b0b",
    "cmp [rsp + 272], r10",
    "jnz 22f",
    "cmp qword ptr [rsp + 280], 0",
    "jz 23f",
    "22:",
    "or r9d, 8192",
    "23:",
    "movabs r10, 0x0c0c0c0c0c0c0c0c",
    "cmp [rsp + 288], r10",
    "jnz 24f",
    "cmp qword ptr [rsp + 296], 0",
    "jz 25f",
    "24:",
    "or r9d, 16384",
    "25:",
    "movabs r10, 0x0d0d0d0d0d0d0d0d",
    "cmp [rsp + 304], r10",
    "jnz 26f",
    "cmp qword ptr [rsp + 312], 0",
    "jz 27f",
    "26:",
    "or r9d, 32768",
    "27:",
    "movabs r10, 0x0e0e0e0e0e0e0e0e",
    "cmp [rsp + 320], r10",
    "jnz 28f",
    "cmp qword ptr [rsp + 328], 0",
    "jz 29f",
    "28:",
    "or r9d, 65536",
    "29:",
    "movabs r10, 0x0f0f0f0f0f0f0f0f",
    "cmp [rsp + 336], r10",
    "jnz 30f",
    "cmp qword ptr [rsp + 344], 0",
    "jz 31f",
    "30:",
    "or r9d, 131072",
    "31:",
    "mov r8, [rsp + 352]",
    "mov [r8], r9",
    "movups xmm6, [rsp + 32]",
    "movups xmm7, [rsp + 48]",
    "movups xmm8, [rsp + 64]",
    "movups xmm9, [rsp + 80]",
    "movups xmm10, [rsp + 96]",
    "movups xmm11, [rsp + 112]",
    "movups xmm12, [rsp + 128]",
    "movups xmm13, [rsp + 144]",
    "movups xmm14, [rsp + 160]",
    "movups xmm15, [rsp + 176]",
    "add rsp, 360",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rsi",
    "pop rdi",
    "pop rbp",
    "pop rbx",
    "ret",
);

/// Names the bits of the mask the checking shim reports.
#[cfg(not(windows))]
const SAVED_NAMES: &str = "rbx rbp r12 r13 r14 r15";
#[cfg(windows)]
const SAVED_NAMES: &str = "rbx rbp rdi rsi r12 r13 r14 r15 xmm6..xmm15";

unsafe extern "C" {
    fn oximond_mir_call_checked(
        entry: Entry,
        inputs: *mut u64,
        control: *mut u64,
        area: *mut u64,
        bad: *mut u64,
    ) -> i64;
}

pub struct ExecMem {
    ptr: *mut u8,
    len: usize,
}

impl ExecMem {
    /// Maps `parts` back to back and makes them executable.
    ///
    /// # Panics
    ///
    /// Panics if the mapping or the protection change fails.
    #[must_use]
    pub fn new(parts: &[&[u8]]) -> ExecMem {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let len = total.next_multiple_of(4096).max(4096);
        let ptr = os::map_rw(len).expect("mapping code memory failed");
        let mut off = 0;
        for p in parts {
            // SAFETY: `ptr..ptr+len` is writable and `off + p.len() <= total <= len`.
            unsafe { std::ptr::copy_nonoverlapping(p.as_ptr(), ptr.add(off), p.len()) };
            off += p.len();
        }
        assert!(
            os::protect_rx(ptr, len),
            "making code memory executable failed"
        );
        ExecMem { ptr, len }
    }

    /// Address of the first byte of the code.
    #[must_use]
    pub fn address(&self) -> usize {
        self.ptr as usize
    }

    #[must_use]
    pub fn entry(&self) -> Entry {
        // SAFETY: the code was produced for this signature and starts at `ptr`.
        unsafe { std::mem::transmute::<*mut u8, Entry>(self.ptr) }
    }

    /// Runs the code on the given buffers.
    ///
    /// The generated code reads and writes only through the pointers it is
    /// given and its own stack, as the MIR program says. The prototype treats
    /// its own output as trusted, so this is a safe wrapper.
    pub fn call(&self, inputs: &mut [u64], control: &mut [u64], area: &mut [u64]) -> i64 {
        let f = self.entry();
        let mut bad = 0u64;
        // SAFETY: see above; the pointers stay valid for the call, and the
        // assembly shim keeps the caller's callee-saved registers itself.
        let r = unsafe {
            oximond_mir_call_checked(
                f,
                inputs.as_mut_ptr(),
                control.as_mut_ptr(),
                area.as_mut_ptr(),
                &raw mut bad,
            )
        };
        assert_eq!(
            bad, 0,
            "generated code clobbered callee-saved registers (mask {bad:#b}: {SAVED_NAMES})"
        );
        r
    }
}

impl Drop for ExecMem {
    fn drop(&mut self) {
        os::unmap(self.ptr, self.len);
    }
}

/// One executable region for a process, carved per compile.
///
/// The region is mapped read-write-execute once, with the address space
/// reserved and no memory committed (pages are touched as code is written),
/// so a compile costs a copy and no system call. Windows commits the region a
/// megabyte at a time as it grows. Blocks are handed back to a
/// free list when their code is dropped.
pub struct CodeRegion {
    base: *mut u8,
    len: usize,
    alloc: std::sync::Mutex<Carver>,
}

// SAFETY: the base pointer names a mapping that lives for the whole process;
// all bookkeeping is behind the mutex and blocks never overlap.
unsafe impl Send for CodeRegion {}
unsafe impl Sync for CodeRegion {}

struct Carver {
    /// Offset of the first never-used byte.
    top: usize,
    /// Freed `(offset, len)` ranges below `top`, sorted by offset, never
    /// adjacent.
    free: Vec<(usize, usize)>,
    /// Bytes handed out and not yet returned.
    live: usize,
    /// Bytes from the base that [`os::commit_rwx`] has backed.
    committed: usize,
}

/// The step the region commits in, so most compiles make no system call.
const COMMIT_STEP: usize = 1 << 20;

const BLOCK_ALIGN: usize = 16;

/// A carved piece of the region. Dropping it returns the bytes.
pub struct CodeBlock {
    region: &'static CodeRegion,
    offset: usize,
    len: usize,
}

// SAFETY: a block names bytes only it owns; the code in them is read-only
// once written.
unsafe impl Send for CodeBlock {}
unsafe impl Sync for CodeBlock {}

impl CodeBlock {
    /// Address of the first byte, where the entry function starts.
    #[must_use]
    pub fn address(&self) -> usize {
        self.region.base as usize + self.offset
    }

    /// Bytes reserved for this block (the code, rounded up).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for CodeBlock {
    fn drop(&mut self) {
        self.region.release(self.offset, self.len);
    }
}

impl CodeRegion {
    /// The process's region, mapped on first use. `None` if the system
    /// refuses a read-write-execute mapping.
    #[must_use]
    pub fn global() -> Option<&'static CodeRegion> {
        static REGION: std::sync::OnceLock<Option<CodeRegion>> = std::sync::OnceLock::new();
        REGION.get_or_init(|| CodeRegion::map(128 << 20)).as_ref()
    }

    /// A region of its own (never unmapped), for tests of the carving.
    #[must_use]
    pub fn leaked(len: usize) -> Option<&'static CodeRegion> {
        Self::map(len).map(|region| &*Box::leak(Box::new(region)))
    }

    fn map(len: usize) -> Option<CodeRegion> {
        let ptr = os::reserve_rwx(len)?;
        Some(CodeRegion {
            base: ptr,
            len,
            alloc: std::sync::Mutex::new(Carver {
                top: 0,
                committed: 0,
                free: Vec::with_capacity(64),
                live: 0,
            }),
        })
    }

    /// Reserves `len` bytes and lets `fill` write the code into them.
    /// `None` when the region is full.
    pub fn carve(&'static self, len: usize, fill: impl FnOnce(&mut [u8])) -> Option<CodeBlock> {
        let len = len.next_multiple_of(BLOCK_ALIGN).max(BLOCK_ALIGN);
        let offset = {
            let mut c = self.alloc.lock().ok()?;
            let found = c.free.iter().position(|&(_, l)| l >= len);
            let offset = if let Some(k) = found {
                let (o, l) = c.free[k];
                if l == len {
                    c.free.remove(k);
                } else {
                    c.free[k] = (o + len, l - len);
                }
                o
            } else if c.top + len <= self.len {
                let o = c.top;
                let end = o + len;
                if end > c.committed {
                    let upto = end.next_multiple_of(COMMIT_STEP).min(self.len);
                    // SAFETY: `committed..upto` is inside the reservation.
                    let from = unsafe { self.base.add(c.committed) };
                    if !os::commit_rwx(from, upto - c.committed) {
                        return None;
                    }
                    c.committed = upto;
                }
                c.top = end;
                o
            } else {
                return None;
            };
            c.live += len;
            offset
        };
        // SAFETY: `offset..offset + len` is inside the mapping and owned by
        // this call until the block is dropped, so no one else touches it.
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.base.add(offset), len) };
        fill(bytes);
        Some(CodeBlock {
            region: self,
            offset,
            len,
        })
    }

    fn release(&self, offset: usize, len: usize) {
        let Ok(mut c) = self.alloc.lock() else { return };
        c.live -= len;
        let at = c.free.partition_point(|&(o, _)| o < offset);
        c.free.insert(at, (offset, len));
        // Merge with the next range, then the previous one.
        if at + 1 < c.free.len() && c.free[at].0 + c.free[at].1 == c.free[at + 1].0 {
            c.free[at].1 += c.free[at + 1].1;
            c.free.remove(at + 1);
        }
        if at > 0 && c.free[at - 1].0 + c.free[at - 1].1 == c.free[at].0 {
            c.free[at - 1].1 += c.free[at].1;
            c.free.remove(at);
            let at = at - 1;
            if c.free[at].0 + c.free[at].1 == c.top {
                c.top = c.free[at].0;
                c.free.remove(at);
            }
        } else if c.free[at].0 + c.free[at].1 == c.top {
            c.top = c.free[at].0;
            c.free.remove(at);
        }
    }

    /// Bytes of the region in use by live blocks, and the high-water mark.
    #[must_use]
    pub fn usage(&self) -> (usize, usize) {
        self.alloc.lock().map_or((0, 0), |c| (c.live, c.top))
    }

    /// Bytes mapped.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.len
    }

    /// Whether `address` is inside the region.
    #[must_use]
    pub fn contains(&self, address: usize) -> bool {
        let base = self.base as usize;
        (base..base + self.len).contains(&address)
    }
}
