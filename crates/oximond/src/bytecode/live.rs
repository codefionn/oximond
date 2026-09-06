//! Per-instruction live register counts.
//!
//! `live_register_count(pc)` is one more than the highest register that can
//! still matter when execution resumes at `pc`. Every register at or above it
//! is a dead temporary or unused: along every path from `pc` it is written
//! before any instruction reads it. The collector roots a Lua frame only up
//! to that count, which is why the compiler does not have to clear dead
//! temporaries.
//!
//! The same analysis keeps the exact set of live registers at every pc
//! (deduplicated, so a pc costs a small index). The count is the set's
//! highest register plus one; the collector roots only registers in the set,
//! which is a subset of the registers below the count. Registers below the
//! count that are dead, such as a stale condition value under the operands of
//! a later statement, are then not roots either.
//!
//! The counts come from a backward dataflow over the finished chunk. A
//! register matters when
//!
//! - some path reads it before writing it,
//! - a named local that is in scope at `pc` lives in it (`debug.getlocal`
//!   and to-be-closed locals can observe it),
//! - it holds a parameter or the `arg` table.
//!
//! A closure reads the registers it captures when it is created. After that
//! the cell holds the value, and the cell is a root of its own.
//!
//! Opcodes whose register effects are not modelled read every register, so a
//! gap in the model only costs precision.

use crate::bytecode::{Chunk, OpCode, RegisterSet, UpvalueSource};
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

type Set = [u64; 4];
const ALL: Set = [u64::MAX; 4];

fn insert(set: &mut Set, register: u8) {
    set[usize::from(register) / 64] |= 1 << (register % 64);
}

fn insert_range(set: &mut Set, start: u8, count: usize) {
    for register in usize::from(start)..(usize::from(start) + count).min(256) {
        insert(set, u8::try_from(register).expect("register below 256"));
    }
}

fn insert_from(set: &mut Set, start: u8) {
    insert_range(set, start, 256 - usize::from(start));
}

/// The count one past the highest register in `set`.
fn count_of(set: &Set) -> u16 {
    for word in (0..4).rev() {
        if set[word] != 0 {
            return u16::try_from(word * 64 + 64 - set[word].leading_zeros() as usize)
                .expect("at most 256");
        }
    }
    0
}

struct Effects {
    reads: Set,
    kills: Set,
    successors: [Option<usize>; 3],
}

/// The reads, kills and successors of the instruction at `pc`. The test
/// `every_opcode_keeps_live_what_the_interpreter_reads` holds them against a
/// table written from the interpreter's handlers.
#[allow(
    clippy::too_many_lines,
    reason = "one exhaustive opcode classification keeps the analysis auditable"
)]
// `ClearRange` and `Vararg` kill the same registers but are distinct opcodes
// documented separately, so the arms stay separate.
#[allow(clippy::match_same_arms)]
fn effects(chunk: &Chunk, pc: usize) -> Effects {
    let everything = Effects {
        reads: ALL,
        kills: [0; 4],
        successors: [None; 3],
    };
    let Some(&instruction) = chunk.code.get(pc) else {
        return everything;
    };
    let Ok(opcode) = instruction.opcode() else {
        return everything;
    };
    let (a, b, c) = (instruction.a(), instruction.b(), instruction.c());
    let next = Some(pc + 1);
    let relative = |offset: i32| {
        i64::try_from(pc)
            .ok()
            .and_then(|pc| usize::try_from(pc + 1 + i64::from(offset)).ok())
    };
    let short = i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes()));
    let mut reads = [0; 4];
    let mut kills = [0; 4];
    let mut successors = [next, None, None];
    match opcode {
        OpCode::LoadNil
        | OpCode::LoadConstant
        | OpCode::LoadConstantWide
        | OpCode::GetGlobal
        | OpCode::GetGlobalWide
        | OpCode::GetUpvalue
        | OpCode::GetTabUp
        | OpCode::NewTable => insert(&mut kills, a),
        // The new closure's cells take the values of the parent registers it
        // captures.
        OpCode::Closure => {
            if let Some(prototype) = chunk.prototypes.get(usize::from(instruction.d())) {
                for upvalue in &prototype.upvalues {
                    if let UpvalueSource::ParentRegister(register) = upvalue.source {
                        insert(&mut reads, register);
                    }
                }
            } else {
                reads = ALL;
            }
            insert(&mut kills, a);
        }
        OpCode::Move
        | OpCode::Negate
        | OpCode::Not
        | OpCode::Length
        | OpCode::BitNot
        | OpCode::AddK
        | OpCode::SubtractK
        | OpCode::MultiplyK
        | OpCode::DivideK
        | OpCode::ModuloK
        | OpCode::GetField => {
            insert(&mut reads, b);
            insert(&mut kills, a);
        }
        OpCode::Add
        | OpCode::Subtract
        | OpCode::Multiply
        | OpCode::Divide
        | OpCode::Modulo
        | OpCode::Power
        | OpCode::FloorDivide
        | OpCode::BitAnd
        | OpCode::BitOr
        | OpCode::BitXor
        | OpCode::ShiftLeft
        | OpCode::ShiftRight
        | OpCode::Equal
        | OpCode::LessThan
        | OpCode::LessEqual
        | OpCode::GetTable
        | OpCode::CallOne => {
            insert(&mut reads, b);
            insert(&mut reads, c);
            insert(&mut kills, a);
        }
        OpCode::Concat => {
            if b <= c {
                insert_range(&mut reads, b, usize::from(c - b) + 1);
            }
            insert(&mut kills, a);
        }
        OpCode::SelfField => {
            insert(&mut reads, b);
            insert(&mut kills, a);
            insert(&mut kills, a.wrapping_add(1));
        }
        OpCode::SetField => {
            insert(&mut reads, a);
            insert(&mut reads, c);
        }
        OpCode::SetTabUp => insert(&mut reads, c),
        OpCode::ClearRange => insert_range(&mut kills, a, usize::from(b)),
        OpCode::SetTable | OpCode::DeclareTable => {
            insert(&mut reads, a);
            insert(&mut reads, b);
            insert(&mut reads, c);
        }
        OpCode::SetGlobal
        | OpCode::SetGlobalWide
        | OpCode::DeclareGlobal
        | OpCode::DeclareGlobalWide
        | OpCode::SetUpvalue
        | OpCode::TableVararg
        | OpCode::TableOpen
        | OpCode::MarkClose
        | OpCode::CloseLocal
        | OpCode::CloseUpvalue => insert(&mut reads, a),
        // `Vararg` writes `b` registers from `a` (nil past the last vararg)
        // and reads none.
        OpCode::Vararg => insert_range(&mut kills, a, usize::from(b)),
        OpCode::ExtraArgument => {}
        OpCode::Jump => successors = [relative(instruction.jump_offset()), None, None],
        // The word after the test is its `Jump`; skipping it continues at
        // pc + 2. The constant forms read only `a`.
        OpCode::TestEqual | OpCode::TestLessThan | OpCode::TestLessEqual => {
            insert(&mut reads, a);
            insert(&mut reads, b);
            successors = [next, Some(pc + 2), None];
        }
        OpCode::TestEqualK | OpCode::TestLessThanK | OpCode::TestLessEqualK => {
            insert(&mut reads, a);
            successors = [next, Some(pc + 2), None];
        }
        OpCode::JumpIfFalse => {
            insert(&mut reads, a);
            successors = [next, relative(short), None];
        }
        // An integer loop with an empty range skips its latch too, to the
        // instruction after the jump target.
        OpCode::NumericForPrep => {
            insert_range(&mut reads, a, 4);
            successors = [next, relative(short), relative(short + 1)];
        }
        OpCode::NumericForLoop => {
            insert_range(&mut reads, a, 4);
            successors = [next, relative(short), None];
        }
        // The call reads the iterator, state and control, and writes the
        // call window `a+3 ..= a+5` and the results.
        OpCode::ForInCall => {
            insert_range(&mut reads, a, 3);
            insert_range(&mut kills, a.saturating_add(3), usize::from(b.max(3)));
        }
        // Copies the first variable into the control register when the
        // loop continues.
        OpCode::ForInLoop => {
            insert(&mut reads, a.saturating_add(3));
            successors = [next, relative(short), None];
        }
        // The callee and its `b` fixed arguments. The varargs and pending
        // open results of the vararg and open forms do not live in registers.
        // A tail call continues at the next instruction too: a callee that is
        // not a Lua closure runs as a call with open results, and the
        // `ReturnOpen` after the tail call returns them.
        // A call that wants `c` results writes `a .. a + c` when it returns
        // (nil past the callee's last result), so a result register is dead
        // before the call: the generic `for` that moves three results of
        // `pairs (t)` out of the way does not keep its third register live
        // around the loop that holds it. An open result count (`c == 0`)
        // writes no register.
        OpCode::Call | OpCode::CallVararg | OpCode::CallOpen => {
            insert_range(&mut reads, a, usize::from(b) + 1);
            insert_range(&mut kills, a, usize::from(c));
        }
        OpCode::TailCall | OpCode::TailCallVararg | OpCode::TailCallOpen => {
            insert_range(&mut reads, a, usize::from(b) + 1);
        }
        // The VM returns every register from `a` up for a zero count.
        OpCode::Return if b == 0 => {
            insert_from(&mut reads, a);
            successors = [None; 3];
        }
        OpCode::Return | OpCode::ReturnVararg | OpCode::ReturnOpen => {
            insert_range(&mut reads, a, usize::from(b));
            successors = [None; 3];
        }
        OpCode::Halt => successors = [None; 3],
    }
    Effects {
        reads,
        kills,
        successors,
    }
}

/// The result of the analysis for one chunk.
struct Analysis {
    /// One more than the highest live register, per pc.
    counts: Vec<u16>,
    /// Index into `sets`, per pc.
    index: Vec<u32>,
    sets: Vec<RegisterSet>,
}

impl Analysis {
    /// Every register live everywhere: the answer when nothing is proven.
    fn conservative(len: usize) -> Self {
        Self {
            counts: vec![256; len],
            index: vec![0; len],
            sets: vec![RegisterSet::ALL],
        }
    }
}

/// Adds the register of every named local to the sets of the pcs where it is
/// in scope. One sweep over the pcs with the scope starts and ends sorted, so
/// a main chunk with many long-lived locals does not pay locals times pcs.
fn add_named_locals(chunk: &Chunk, live: &mut [Set]) {
    let len = live.len();
    let locals = chunk.debug.local_variables();
    if locals.is_empty() {
        return;
    }
    // (pc, register, whether the scope starts there)
    let mut events = Vec::with_capacity(locals.len() * 2);
    for local in locals {
        let end = local.end_pc().min(len);
        let start = local.start_pc().min(end);
        if start < end {
            events.push((start, local.register(), true));
            events.push((end, local.register(), false));
        }
    }
    events.sort_unstable_by_key(|(pc, _, _)| *pc);
    let mut open = [0u32; 256];
    let mut current: Set = [0; 4];
    let mut next = 0;
    for (pc, set) in live.iter_mut().enumerate() {
        while let Some(&(at, register, starts)) = events.get(next)
            && at == pc
        {
            let slot = &mut open[usize::from(register)];
            if starts {
                *slot += 1;
                insert(&mut current, register);
            } else {
                *slot -= 1;
                if *slot == 0 {
                    current[usize::from(register) / 64] &= !(1 << (register % 64));
                }
            }
            next += 1;
        }
        for word in 0..4 {
            set[word] |= current[word];
        }
    }
}

/// Computes the live registers at every pc of `chunk`.
fn analyze(chunk: &Chunk) -> Analysis {
    let len = chunk.code.len();
    let effects = (0..len).map(|pc| effects(chunk, pc)).collect::<Vec<_>>();
    // The pass walks backwards, so a successor above `pc` is final by the
    // time `pc` reads it. Only a successor at or below `pc` (a loop's back
    // edge) can be stale, and only then does a second round find changes.
    let loops = effects.iter().enumerate().any(|(pc, effect)| {
        effect
            .successors
            .iter()
            .flatten()
            .any(|successor| *successor <= pc)
    });
    let mut live = vec![[0u64; 4]; len];
    let mut rounds = 0;
    let mut changed = true;
    while changed {
        changed = false;
        rounds += 1;
        if rounds > 64 {
            return Analysis::conservative(len);
        }
        if rounds == 2 && !loops {
            break;
        }
        for pc in (0..len).rev() {
            let effect = &effects[pc];
            let mut out = [0u64; 4];
            for successor in effect.successors.into_iter().flatten() {
                // Falling off the end or jumping outside the chunk is
                // malformed; keep everything live there.
                let successor_live = live.get(successor).copied().unwrap_or(ALL);
                for word in 0..4 {
                    out[word] |= successor_live[word];
                }
            }
            let mut live_in = [0u64; 4];
            for word in 0..4 {
                live_in[word] = effect.reads[word] | (out[word] & !effect.kills[word]);
            }
            if live_in != live[pc] {
                live[pc] = live_in;
                changed = true;
            }
        }
    }
    let mut always = [0u64; 4];
    insert_range(&mut always, 0, usize::from(chunk.parameter_count));
    if let Some(register) = chunk.arg_register {
        insert(&mut always, register);
    }
    for set in &mut live {
        for word in 0..4 {
            set[word] |= always[word];
        }
    }
    add_named_locals(chunk, &mut live);
    let mut palette = FxHashMap::default();
    let mut sets = Vec::new();
    let mut index = Vec::with_capacity(len);
    let mut counts = Vec::with_capacity(len);
    let mut previous: Option<(Set, u32, u16)> = None;
    for set in live {
        // Neighbouring pcs often share a set; reuse it without hashing.
        if let Some((last, id, count)) = previous
            && last == set
        {
            index.push(id);
            counts.push(count);
            continue;
        }
        let id = *palette.entry(set).or_insert_with(|| {
            sets.push(RegisterSet(set));
            u32::try_from(sets.len() - 1).expect("fewer distinct sets than instructions")
        });
        let count = count_of(&set);
        index.push(id);
        counts.push(count);
        previous = Some((set, id, count));
    }
    Analysis {
        counts,
        index,
        sets,
    }
}

/// The lazily computed per-pc counts stored in a [`Chunk`].
///
/// It is a cache of a pure function of the chunk. Two chunks compare equal
/// whatever they have computed. A clone of a chunk whose analysis exists
/// shares it (it sits behind an `Arc`), so cloning never repeats the work.
/// That is sound because a chunk's code and debug locals do not change after
/// they are first analysed; `Chunk::with_debug_metadata` resets the result,
/// and other code that edits a chunk must call [`LiveRegisters::reset`].
///
/// The compiler leaves the analysis to the first call of the function
/// ([`Chunk::entry_register_count`] needs it), since most loaded functions
/// never run. A frame exists only once its function was called, so a
/// collection that roots a frame finds the analysis already computed. The
/// binary loader still computes it for every chunk it decodes.
pub struct LiveRegisters {
    analysis: OnceLock<Arc<Analysis>>,
    /// Where the analysis is kept instead once [`Self::make_shareable`]
    /// ran: one slot for every copy [`Self::share`] makes.
    shared: Option<Arc<OnceLock<Arc<Analysis>>>>,
    /// [`Chunk::entry_register_count`], or `u16::MAX` before the first
    /// call. A word the call path reads without touching the analysis.
    entry: AtomicU16,
    /// [`Chunk::calls_leave_registers_dead`]: `CALLS_UNKNOWN` before the
    /// first query.
    calls: AtomicU8,
    /// [`Chunk::entry_info`]: `ENTRY_INFO_UNKNOWN` before the first query.
    entry_info: AtomicU64,
}

const ENTRY_INFO_UNKNOWN: u64 = u64::MAX;

/// Set in [`Chunk::entry_info`] when the chunk starts with a `Vararg`.
pub const ENTRY_VARARG_BIT: u64 = 1 << 16;
const CALLS_UNKNOWN: u8 = 0;
const CALLS_DEAD: u8 = 1;
const CALLS_MIXED: u8 = 2;

impl Default for LiveRegisters {
    fn default() -> Self {
        Self {
            analysis: OnceLock::new(),
            shared: None,
            entry: AtomicU16::new(u16::MAX),
            calls: AtomicU8::new(CALLS_UNKNOWN),
            entry_info: AtomicU64::new(ENTRY_INFO_UNKNOWN),
        }
    }
}

impl LiveRegisters {
    /// The slot that holds the analysis.
    fn slot(&self) -> &OnceLock<Arc<Analysis>> {
        self.shared.as_deref().unwrap_or(&self.analysis)
    }

    /// Moves the analysis, computed or not, to a slot that the copies
    /// [`Self::share`] makes will share.
    pub fn make_shareable(&mut self) {
        if self.shared.is_none() {
            self.shared = Some(Arc::new(core::mem::take(&mut self.analysis)));
        }
    }

    /// For an exact copy of the chunk: after [`Self::make_shareable`], the
    /// same analysis, computed or not, so whichever copy runs first computes
    /// it for all of them; otherwise a clone. A copy that changes the code
    /// or the debug locals must not share.
    #[must_use]
    pub fn share(&self) -> Self {
        match &self.shared {
            Some(shared) => Self {
                shared: Some(Arc::clone(shared)),
                ..Self::default()
            },
            None => self.clone(),
        }
    }

    /// Drops a computed analysis. Anything that changes a chunk's code or
    /// debug locals after it may have been analysed calls this, since clones
    /// share the result and nothing would recompute it.
    pub fn reset(&mut self) {
        self.analysis = OnceLock::new();
        self.shared = None;
        *self.entry.get_mut() = u16::MAX;
        *self.calls.get_mut() = CALLS_UNKNOWN;
        *self.entry_info.get_mut() = ENTRY_INFO_UNKNOWN;
    }
}

impl Clone for LiveRegisters {
    fn clone(&self) -> Self {
        let clone = Self::default();
        if let Some(analysis) = self.slot().get() {
            let _ = clone.analysis.set(Arc::clone(analysis));
        }
        clone
    }
}

impl PartialEq for LiveRegisters {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl core::fmt::Debug for LiveRegisters {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("LiveRegisters")
    }
}

impl Chunk {
    /// Computes the live-register analysis now if it has not been computed.
    /// Call it once a chunk is complete so later collections only read it.
    pub fn compute_live_registers(&self) {
        let _ = self.live_register_counts();
    }

    /// Whether the analysis has been computed (or shared from a clone).
    #[must_use]
    pub fn has_live_registers(&self) -> bool {
        self.live.slot().get().is_some()
    }

    /// The live register count at every pc; see the module documentation.
    #[must_use]
    pub fn live_register_counts(&self) -> &[u16] {
        &self
            .live
            .slot()
            .get_or_init(|| Arc::new(analyze(self)))
            .counts
    }

    /// The exact set of registers that can matter at `pc`. A pc outside the
    /// code reports every register.
    #[must_use]
    pub fn live_registers(&self, pc: usize) -> RegisterSet {
        let analysis = self.live.slot().get_or_init(|| Arc::new(analyze(self)));
        analysis
            .index
            .get(pc)
            .map_or(RegisterSet::ALL, |index| analysis.sets[*index as usize])
    }

    /// The registers the instruction at `pc` reads, by its register effects
    /// alone. An instruction whose effects are not modelled reads every
    /// register.
    #[must_use]
    pub fn instruction_reads(&self, pc: usize) -> RegisterSet {
        RegisterSet(effects(self, pc).reads)
    }

    /// Whether some path from `pc` reads `register` before it writes it,
    /// judged by the register effects of the instructions alone. Unlike
    /// [`Self::live_registers`] it does not count a named local in scope that
    /// nothing reads, so it tells a loop variable the body never looks at
    /// from one it does. An instruction whose effects are not modelled reads
    /// every register.
    #[must_use]
    pub fn reads_before_write(&self, pc: usize, register: u8) -> bool {
        let contains = |set: &Set| set[usize::from(register) / 64] >> (register % 64) & 1 == 1;
        let mut seen = vec![false; self.code.len()];
        let mut pending = vec![pc];
        while let Some(pc) = pending.pop() {
            if seen.get(pc).copied().unwrap_or(false) {
                continue;
            }
            if let Some(seen) = seen.get_mut(pc) {
                *seen = true;
            }
            let effects = effects(self, pc);
            if contains(&effects.reads) {
                return true;
            }
            if !contains(&effects.kills) {
                pending.extend(effects.successors.into_iter().flatten());
            }
        }
        false
    }

    /// Whether some instruction in `start..end` may write `register`, judged
    /// by the register effects of the instructions alone. An instruction
    /// whose effects are not modelled counts as writing.
    #[must_use]
    pub fn writes_register_between(&self, start: usize, end: usize, register: u8) -> bool {
        let contains = |set: &Set| set[usize::from(register) / 64] >> (register % 64) & 1 == 1;
        (start..end.min(self.code.len())).any(|pc| {
            let effects = effects(self, pc);
            contains(&effects.kills) || effects.reads == ALL
        })
    }

    /// How many registers of a fresh activation must start nil: the
    /// parameters, which a caller may leave out and `debug.getlocal` shows,
    /// and every register the chunk can read before writing it. Registers
    /// from this count up are written before any read on every path from
    /// entry (see [`Self::live_registers`]), so their old contents never
    /// matter.
    #[must_use]
    #[allow(clippy::inline_always, reason = "read on every call of a closure")]
    #[inline(always)]
    pub fn entry_register_count(&self) -> usize {
        let cached = self.live.entry.load(Ordering::Relaxed);
        if cached == u16::MAX {
            return self.entry_register_count_uncached();
        }
        usize::from(cached)
    }

    /// [`Self::entry_register_count`] when it is already computed, and
    /// `None` when it is not: a caller that must not call anything on its way
    /// (the fast loop's call) leaves that to a cold path.
    #[must_use]
    #[allow(clippy::inline_always, reason = "read on every call of a closure")]
    #[inline(always)]
    pub fn entry_register_count_cached(&self) -> Option<usize> {
        let cached = self.live.entry.load(Ordering::Relaxed);
        (cached != u16::MAX).then_some(usize::from(cached))
    }

    #[cold]
    #[inline(never)]
    fn entry_register_count_uncached(&self) -> usize {
        let count = self
            .live_register_count(0)
            .max(usize::from(self.parameter_count))
            .min(usize::from(self.register_count));
        self.live.entry.store(
            u16::try_from(count).expect("a register count fits u16"),
            Ordering::Relaxed,
        );
        count
    }

    /// [`Self::entry_register_count`] and the chunk's first instruction when
    /// it is a `Vararg` whose registers lie inside the chunk's window, in one
    /// word: the entry count in bits 0..16, then `ENTRY_VARARG_BIT` and the
    /// instruction's destination register `a` in bits 24..32 and its count
    /// `b` in bits 32..40. Without such a first instruction the bits above
    /// the count are 0. Computed once, then one load: the call path of a
    /// vararg closure reads both on every call and needs not fetch and
    /// decode the instruction.
    #[must_use]
    #[allow(clippy::inline_always, reason = "read on every call of a closure")]
    #[inline(always)]
    pub fn entry_info(&self) -> u64 {
        let cached = self.live.entry_info.load(Ordering::Relaxed);
        if cached == ENTRY_INFO_UNKNOWN {
            return self.entry_info_uncached();
        }
        cached
    }

    #[cold]
    #[inline(never)]
    fn entry_info_uncached(&self) -> u64 {
        let head = match self.code.first() {
            Some(first)
                if first.opcode() == Ok(OpCode::Vararg)
                    && usize::from(first.a()) + usize::from(first.b())
                        <= usize::from(self.register_count) =>
            {
                ENTRY_VARARG_BIT | (u64::from(first.a()) << 24) | (u64::from(first.b()) << 32)
            }
            _ => 0,
        };
        let packed = head | self.entry_register_count() as u64;
        self.live.entry_info.store(packed, Ordering::Relaxed);
        packed
    }

    /// Whether the `Call` at `pc`, which wants at least one result, leaves
    /// every register from its own up dead except the ones it writes: the
    /// code after it reads nothing there but the results. A callee's window
    /// can then take the place of the call's arguments and of everything
    /// above them. Any other instruction, and a call that keeps no result,
    /// reports `false`.
    #[must_use]
    pub fn call_leaves_registers_dead(&self, pc: usize) -> bool {
        let Some(instruction) = self.code.get(pc) else {
            return false;
        };
        instruction.opcode() == Ok(OpCode::Call)
            && instruction.c() != 0
            && self.live_register_count(pc + 1)
                <= usize::from(instruction.a()) + usize::from(instruction.c())
    }

    /// Whether [`Self::call_leaves_registers_dead`] holds at every `Call` of
    /// the chunk that wants a result. Computed once, then one load: the call
    /// path of the fast loop asks it on every call.
    #[must_use]
    #[allow(clippy::inline_always, reason = "read on every call of a closure")]
    #[inline(always)]
    pub fn calls_leave_registers_dead(&self) -> bool {
        match self.live.calls.load(Ordering::Relaxed) {
            CALLS_DEAD => true,
            CALLS_MIXED => false,
            _ => self.calls_leave_registers_dead_uncached(),
        }
    }

    #[cold]
    #[inline(never)]
    fn calls_leave_registers_dead_uncached(&self) -> bool {
        let all = self.code.iter().enumerate().all(|(pc, instruction)| {
            instruction.opcode() != Ok(OpCode::Call)
                || instruction.c() == 0
                || self.call_leaves_registers_dead(pc)
        });
        self.live.calls.store(
            if all { CALLS_DEAD } else { CALLS_MIXED },
            Ordering::Relaxed,
        );
        all
    }

    /// One more than the highest register that can matter at `pc`. A pc
    /// outside the code reports the whole 256-register window.
    #[must_use]
    pub fn live_register_count(&self, pc: usize) -> usize {
        self.live_register_counts()
            .get(pc)
            .map_or(256, |count| usize::from(*count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Instruction;
    use std::sync::Arc;

    fn counts(code: Vec<Instruction>) -> Vec<u16> {
        Chunk::new(8, Vec::new(), code)
            .live_register_counts()
            .to_vec()
    }

    #[test]
    fn straight_line_code_is_live_from_its_first_read() {
        let live = counts(vec![
            Instruction::load_nil(5),
            Instruction::move_(2, 5),
            Instruction::return_(2),
        ]);
        // r5 is written at pc 0 and read at pc 1; r2 is read by the return.
        assert_eq!(live, vec![0, 6, 3]);
    }

    #[test]
    fn calls_that_leave_the_registers_above_dead_are_told_apart() {
        // r1 = f(r2): nothing above r1 is read after the call.
        let clean = Chunk::new(
            8,
            Vec::new(),
            vec![
                Instruction::load_nil(1),
                Instruction::load_nil(2),
                Instruction::call(1, 1, 1),
                Instruction::return_(1),
            ],
        );
        assert!(clean.call_leaves_registers_dead(2));
        assert!(clean.calls_leave_registers_dead());
        // A value parked in r5 before the call and read after it is not.
        let parked = Chunk::new(
            8,
            Vec::new(),
            vec![
                Instruction::load_nil(1),
                Instruction::load_nil(2),
                Instruction::load_nil(5),
                Instruction::call(1, 1, 1),
                Instruction::return_values(1, 5),
            ],
        );
        assert!(!parked.call_leaves_registers_dead(3));
        assert!(!parked.calls_leave_registers_dead());
        // Instructions that are not calls, and calls that keep nothing, say no.
        assert!(!parked.call_leaves_registers_dead(0));
        assert!(!parked.call_leaves_registers_dead(99));
        let discarded = Chunk::new(
            8,
            Vec::new(),
            vec![Instruction::call(1, 0, 0), Instruction::halt()],
        );
        assert!(!discarded.call_leaves_registers_dead(0));
        assert!(discarded.calls_leave_registers_dead());
    }

    #[test]
    fn a_call_window_is_live_until_the_call() {
        let live = counts(vec![
            Instruction::load_nil(1),
            Instruction::load_nil(2),
            Instruction::call(1, 1, 1),
            Instruction::halt(),
        ]);
        assert_eq!(live, vec![0, 2, 3, 0]);
    }

    #[test]
    fn a_call_writes_its_result_registers() {
        // r1, r2, r3 = f(r2); return r3. The call writes r3, so r3 is dead
        // before it, though the code reads it after.
        let chunk = Chunk::new(
            8,
            Vec::new(),
            vec![
                Instruction::load_nil(1),
                Instruction::load_nil(2),
                Instruction::call(1, 1, 3),
                Instruction::return_(3),
            ],
        );
        assert!(chunk.live_registers(3).contains(3));
        assert!(!chunk.live_registers(2).contains(3));
        assert!(!chunk.live_registers(0).contains(3));
        // The callee and its argument stay live up to the call.
        assert!(chunk.live_registers(2).contains(1));
        assert!(chunk.live_registers(2).contains(2));
        // An open result count writes no register: r3 stays live through it.
        let open = Chunk::new(
            8,
            Vec::new(),
            vec![
                Instruction::load_nil(1),
                Instruction::call(1, 0, 0),
                Instruction::return_(3),
            ],
        );
        assert!(open.live_registers(1).contains(3));
        assert!(open.live_registers(0).contains(3));
    }

    #[test]
    fn a_generic_for_keeps_its_control_registers_live_around_the_loop() {
        // setup r0..r2; body (r3 is the variable); ForInCall; ForInLoop.
        let live = counts(vec![
            Instruction::load_nil(0),
            Instruction::load_nil(1),
            Instruction::load_nil(2),
            Instruction::aj(OpCode::Jump, 1).unwrap(),
            Instruction::move_(7, 3),
            Instruction::for_in_call(0, 1),
            Instruction::for_in_loop(0, -3),
            Instruction::halt(),
        ]);
        // The call reads r0..r2 and writes the window from r3 up, so r3 is
        // not live before it. The test reads r3 and the next call reads r2.
        assert_eq!(live[5], 3);
        assert_eq!(live[6], 4);
        assert!(live[4] >= 4);
    }

    #[test]
    fn a_loop_variable_is_read_only_when_the_body_reads_it() {
        let chunk = |source: u8| {
            Chunk::new(
                8,
                Vec::new(),
                vec![
                    Instruction::load_nil(0),
                    Instruction::load_nil(1),
                    Instruction::load_nil(2),
                    Instruction::aj(OpCode::Jump, 1).unwrap(),
                    Instruction::move_(7, source),
                    Instruction::for_in_call(0, 2),
                    Instruction::for_in_loop(0, -3),
                    Instruction::halt(),
                ],
            )
        };
        // The body at pc 4 copies r3: the variable is read before the call
        // writes it again. The second variable, r4, is not, and r3 is not
        // when the body copies r6, whatever the scope of a named local says.
        assert!(chunk(3).reads_before_write(4, 3));
        assert!(!chunk(3).reads_before_write(4, 4));
        assert!(!chunk(6).reads_before_write(4, 3));
        assert!(chunk(6).reads_before_write(4, 6));
        // The control registers are read by the call itself.
        assert!(chunk(6).reads_before_write(4, 2));
    }

    #[test]
    fn an_activation_starts_nil_only_up_to_the_parameters_and_the_first_reads() {
        // Two parameters, the second unused; r2 and r3 are written first.
        let chunk = Chunk::with_parts(
            6,
            2,
            Vec::new(),
            vec![
                Instruction::load_nil(2),
                Instruction::move_(3, 2),
                Instruction::move_(4, 0),
                Instruction::return_(4),
            ],
            Vec::new(),
            Vec::new(),
        );
        // r0 is read before it is written; r1 is a parameter and counts too.
        assert_eq!(chunk.entry_register_count(), 2);
        // The answer is cached, and a clone agrees.
        assert_eq!(chunk.entry_register_count(), 2);
        assert_eq!(chunk.clone().entry_register_count(), 2);
    }

    #[test]
    fn a_register_read_before_it_is_written_extends_the_nil_prefix() {
        // r4 is read at once, with no write: it must start nil.
        let chunk = Chunk::with_parts(
            6,
            1,
            Vec::new(),
            vec![Instruction::move_(1, 4), Instruction::return_(1)],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(chunk.entry_register_count(), 5);
    }

    #[test]
    fn the_nil_prefix_never_exceeds_the_register_count() {
        let chunk = Chunk::with_parts(
            2,
            2,
            Vec::new(),
            vec![Instruction::halt()],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(chunk.entry_register_count(), 2);
        let empty = Chunk::with_parts(0, 0, Vec::new(), Vec::new(), Vec::new(), Vec::new());
        assert_eq!(empty.entry_register_count(), 0);
    }

    #[test]
    fn clones_share_a_computed_analysis_and_start_empty_otherwise() {
        let chunk = Chunk::new(
            8,
            Vec::new(),
            vec![Instruction::load_nil(5), Instruction::return_(5)],
        );
        let fresh_clone = chunk.clone();
        assert!(!chunk.has_live_registers());
        assert!(!fresh_clone.has_live_registers());
        chunk.compute_live_registers();
        let clone = chunk.clone();
        assert!(clone.has_live_registers());
        assert!(Arc::ptr_eq(
            chunk.live.analysis.get().unwrap(),
            clone.live.analysis.get().unwrap()
        ));
        assert_eq!(clone.live_register_counts(), chunk.live_register_counts());
        // The earlier clone still computes by itself.
        assert!(!fresh_clone.has_live_registers());
    }

    #[test]
    fn binary_loaded_chunks_are_analysed_on_first_use() {
        let mut parent = Chunk::new(
            4,
            Vec::new(),
            vec![Instruction::load_nil(1), Instruction::return_(1)],
        );
        parent.prototypes.push(Arc::new(Chunk::new(
            2,
            Vec::new(),
            vec![Instruction::load_nil(0), Instruction::return_(0)],
        )));
        let bytes = crate::bytecode::binary::dump(&parent).unwrap();
        let loaded = crate::bytecode::binary::load(&bytes).unwrap();
        assert!(!loaded.has_live_registers());
        assert!(
            loaded
                .prototypes
                .iter()
                .all(|child| !child.has_live_registers())
        );
        assert_eq!(loaded.live_register_counts(), [0, 2]);
        assert!(loaded.has_live_registers());
    }

    /// What the interpreter does with the registers of the instruction at
    /// `pc`, written from its handlers in `oximond::vm` (`interp.rs`,
    /// `fast_path.rs`, `registers.rs`, `instructions.rs`, `calls.rs`,
    /// `upvalues.rs`), not from this module.
    struct InterpreterUse {
        /// Every register the instruction may read.
        reads: Vec<u8>,
        /// Registers every normal completion overwrites before anything
        /// after it can read them.
        writes: Vec<u8>,
        /// Every pc execution may continue at.
        successors: Vec<usize>,
    }

    /// `None` for an instruction validation rejects.
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per opcode, so a new opcode must be described here"
    )]
    fn interpreter_use(chunk: &Chunk, pc: usize) -> Option<InterpreterUse> {
        let instruction = chunk.code[pc];
        let (a, b, c) = (instruction.a(), instruction.b(), instruction.c());
        let range = |start: u8, count: usize| -> Vec<u8> {
            (usize::from(start)..(usize::from(start) + count).min(256))
                .map(|register| u8::try_from(register).expect("below 256"))
                .collect()
        };
        let target = |offset: i32| {
            usize::try_from(i64::try_from(pc).ok()? + 1 + i64::from(offset))
                .ok()
                .filter(|target| *target < chunk.code.len())
        };
        let short = i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes()));
        // Lua 5.5 reads the varargs from the named vararg table.
        let vararg_table: Vec<u8> = chunk.arg_register.into_iter().collect();
        let next = vec![pc + 1];
        let (reads, writes, successors) = match instruction.opcode().ok()? {
            OpCode::LoadNil
            | OpCode::LoadConstant
            | OpCode::GetGlobal
            | OpCode::GetUpvalue
            | OpCode::GetTabUp
            | OpCode::NewTable => (vec![], vec![a], next),
            // `consume_wide_index` steps over the `ExtraArgument`.
            OpCode::LoadConstantWide | OpCode::GetGlobalWide => (vec![], vec![a], vec![pc + 2]),
            OpCode::SetGlobalWide | OpCode::DeclareGlobalWide => (vec![a], vec![], vec![pc + 2]),
            OpCode::Move
            | OpCode::Not
            | OpCode::Negate
            | OpCode::BitNot
            | OpCode::Length
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK
            | OpCode::GetField => (vec![b], vec![a], next),
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight
            | OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::GetTable
            // `b` is the callee, `c` the argument.
            | OpCode::CallOne => (vec![b, c], vec![a], next),
            OpCode::Concat if b > c => return None,
            OpCode::Concat => (range(b, usize::from(c - b) + 1), vec![a], next),
            OpCode::ClearRange => (vec![], range(a, usize::from(b)), next),
            OpCode::SelfField => (vec![b], vec![a, a.wrapping_add(1)], next),
            OpCode::SetField => (vec![a, c], vec![], next),
            OpCode::SetTabUp => (vec![c], vec![], next),
            OpCode::SetTable | OpCode::DeclareTable => (vec![a, b, c], vec![], next),
            OpCode::SetGlobal
            | OpCode::DeclareGlobal
            | OpCode::SetUpvalue
            | OpCode::MarkClose
            | OpCode::CloseLocal
            | OpCode::TableOpen => (vec![a], vec![], next),
            // The cell holds the value; the register is not read.
            OpCode::CloseUpvalue => (vec![], vec![], next),
            OpCode::TableVararg => ([vec![a], vararg_table].concat(), vec![], next),
            OpCode::Vararg => (vararg_table, range(a, usize::from(b)), next),
            OpCode::Closure => {
                let prototype = chunk.prototypes.get(usize::from(instruction.d()))?;
                let captured = prototype
                    .upvalues
                    .iter()
                    .filter_map(|upvalue| match upvalue.source {
                        UpvalueSource::ParentRegister(register) => Some(register),
                        _ => None,
                    })
                    .collect();
                (captured, vec![a], next)
            }
            OpCode::Jump => (vec![], vec![], vec![target(instruction.jump_offset())?]),
            OpCode::JumpIfFalse => (vec![a], vec![], vec![pc + 1, target(short)?]),
            // The interpreter reads the jump word itself or skips it.
            OpCode::TestEqual | OpCode::TestLessThan | OpCode::TestLessEqual => {
                (vec![a, b], vec![], vec![pc + 1, pc + 2])
            }
            OpCode::TestEqualK | OpCode::TestLessThanK | OpCode::TestLessEqualK => {
                (vec![a], vec![], vec![pc + 1, pc + 2])
            }
            // The float path jumps to the latch, an empty integer range past
            // it, and a non-empty one into the body. Some paths leave the
            // control registers unwritten.
            OpCode::NumericForPrep => {
                let latch = target(short)?;
                (range(a, 3), vec![], vec![pc + 1, latch, latch + 1])
            }
            // An integer control that would overflow is left unwritten.
            OpCode::NumericForLoop => (range(a, 3), vec![], vec![pc + 1, target(short)?]),
            // The copies read the three control registers and rewrite the
            // call window; the results land from `a + 3`.
            OpCode::ForInCall => (
                range(a, 3),
                range(a.checked_add(3)?, usize::from(b.max(3))),
                next,
            ),
            // The loop variable moves into the control register when it
            // continues, so the control write is not certain.
            OpCode::ForInLoop => (
                range(a.checked_add(3)?, 1),
                vec![],
                vec![pc + 1, target(short)?],
            ),
            // The callee and `b` arguments; `c` results land from `a` once
            // it returns, and zero leaves them as open results.
            OpCode::Call | OpCode::CallOpen => {
                (range(a, usize::from(b) + 1), range(a, usize::from(c)), next)
            }
            OpCode::CallVararg => (
                [range(a, usize::from(b) + 1), vararg_table].concat(),
                range(a, usize::from(c)),
                next,
            ),
            // A callee that is not a Lua closure is called with open results,
            // and the `ReturnOpen` after the tail call returns them.
            OpCode::TailCall | OpCode::TailCallOpen => (range(a, usize::from(b) + 1), vec![], next),
            OpCode::TailCallVararg => (
                [range(a, usize::from(b) + 1), vararg_table].concat(),
                vec![],
                next,
            ),
            // A zero count returns the window from `a` up.
            OpCode::Return if b == 0 => (
                range(a, usize::from(chunk.register_count).saturating_sub(usize::from(a))),
                vec![],
                vec![],
            ),
            OpCode::Return | OpCode::ReturnOpen => (range(a, usize::from(b)), vec![], vec![]),
            OpCode::ReturnVararg => (
                [range(a, usize::from(b)), vararg_table].concat(),
                vec![],
                vec![],
            ),
            // Dispatching an `ExtraArgument` on its own is an error.
            OpCode::Halt | OpCode::ExtraArgument => (vec![], vec![], vec![]),
        };
        Some(InterpreterUse {
            reads,
            writes,
            successors,
        })
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the operand grid, the expected words and both checks read best together"
    )]
    fn every_opcode_keeps_live_what_the_interpreter_reads() {
        const LEN: usize = 16;
        const AT: usize = 8;
        const VARARG_TABLE: u8 = 12;
        let prototype = |sources: Vec<UpvalueSource>| {
            let mut chunk = Chunk::new(2, Vec::new(), vec![Instruction::halt()]);
            chunk.upvalues = sources
                .into_iter()
                .map(|source| crate::bytecode::UpvalueDescriptor {
                    name: "u".into(),
                    source,
                })
                .collect();
            Arc::new(chunk)
        };
        let prototypes = vec![
            prototype(vec![
                UpvalueSource::ParentRegister(5),
                UpvalueSource::ParentUpvalue(0),
                UpvalueSource::ParentRegister(201),
            ]),
            prototype(vec![UpvalueSource::Detached]),
        ];
        let chunk_with = |code: Vec<Instruction>| {
            let mut chunk = Chunk::new(255, Vec::new(), code);
            chunk.prototypes.clone_from(&prototypes);
            chunk.is_vararg = true;
            chunk.arg_register = Some(VARARG_TABLE);
            chunk
        };
        let mut covered = 0;
        for byte in 0..=u8::MAX {
            let Ok(opcode) = OpCode::try_from(byte) else {
                continue;
            };
            // Operands from a grid, read as `abc`, `ad` or `aj` words, plus
            // jumps that land inside the chunk.
            let mut samples = Vec::new();
            for a in [0, 2, 7, 200] {
                for b in [0, 1, 3, 9] {
                    for c in [0, 1, 4, 30] {
                        samples.push(Instruction::abc(opcode, a, b, c));
                    }
                }
            }
            for offset in -9_i16..=6 {
                for a in [0, 7, 200] {
                    samples.push(Instruction::ad(opcode, a, offset.cast_unsigned()));
                }
                if let Ok(jump) = Instruction::aj(opcode, i32::from(offset)) {
                    samples.push(jump);
                }
            }
            let mut checked = 0;
            for instruction in samples {
                let mut code = vec![Instruction::halt(); LEN];
                code[AT] = instruction;
                // The words the interpreter expects after these.
                match opcode {
                    OpCode::LoadConstantWide
                    | OpCode::GetGlobalWide
                    | OpCode::SetGlobalWide
                    | OpCode::DeclareGlobalWide => {
                        code[AT + 1] = Instruction::extra_argument(0).unwrap();
                    }
                    OpCode::TailCall | OpCode::TailCallVararg | OpCode::TailCallOpen => {
                        code[AT + 1] = Instruction::return_open(instruction.a(), 0);
                    }
                    opcode if opcode.is_test() => {
                        code[AT + 1] = Instruction::aj(OpCode::Jump, 3).unwrap();
                    }
                    _ => {}
                }
                let chunk = chunk_with(code.clone());
                let Some(used) = interpreter_use(&chunk, AT) else {
                    continue;
                };
                checked += 1;
                let what = format!("{opcode:?} {:#010x}", instruction.raw());
                let live = chunk.live_registers(AT);
                for register in &used.reads {
                    assert!(
                        live.contains(*register),
                        "{what}: the interpreter reads r{register}, which is not live"
                    );
                }
                // A register the instruction does not overwrite stays live
                // into every successor that reads it.
                let mut probes: Vec<u8> =
                    vec![0, 1, 3, 5, 6, 8, 9, 10, 31, 199, 201, 202, 230, 254];
                for base in [instruction.a(), instruction.b(), instruction.c()] {
                    probes.extend((0..5).filter_map(|offset| base.checked_add(offset)));
                }
                probes.retain(|register| !used.writes.contains(register));
                for &successor in &used.successors {
                    if successor == AT || successor >= LEN {
                        continue;
                    }
                    for &register in &probes {
                        let mut with_probe = code.clone();
                        with_probe[successor] = Instruction::return_(register);
                        let live = chunk_with(with_probe).live_registers(AT);
                        assert!(
                            live.contains(register),
                            "{what}: pc {successor} reads r{register}, which is not live before it"
                        );
                    }
                }
            }
            assert!(checked > 0, "{opcode:?}: no sample checked");
            covered += 1;
        }
        assert!(covered > 60, "{covered} opcodes");
    }

    #[test]
    fn shared_copies_analyse_once() {
        let chunk = Chunk::new(
            8,
            Vec::new(),
            vec![Instruction::load_nil(5), Instruction::return_(5)],
        );
        let mut chunk = chunk;
        chunk.live.make_shareable();
        let mut copy = chunk.clone();
        copy.live = chunk.live.share();
        assert!(!chunk.has_live_registers());
        assert_eq!(copy.entry_register_count(), 0);
        assert!(chunk.has_live_registers());
        assert!(Arc::ptr_eq(
            chunk.live.slot().get().unwrap(),
            copy.live.slot().get().unwrap()
        ));
        assert!(chunk.clone().live.shared.is_none());
        assert_eq!(chunk.live_register_counts(), copy.live_register_counts());
    }

    #[test]
    fn new_debug_metadata_drops_an_analysis_computed_before() {
        let chunk = Chunk::new(2, Vec::new(), vec![Instruction::halt()]);
        assert!(!chunk.live_registers(0).contains(1));
        let debug = crate::bytecode::DebugMetadata::default()
            .with_local_variables(vec![crate::bytecode::LocalVariable::new("x", 1, 0, 1)]);
        let chunk = chunk.with_debug_metadata(debug);
        assert!(!chunk.has_live_registers());
        assert!(chunk.live_registers(0).contains(1));
    }
}
