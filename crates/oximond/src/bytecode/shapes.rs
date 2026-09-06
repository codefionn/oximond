//! Which string keys a table constructor stores right after `NewTable`.
//!
//! `{value = v, left = l, right = r}` compiles to a `NewTable` followed by one
//! `SetField` per key. The keys are known from the bytecode alone, so the VM
//! can build the table with a key list shared by every table the site makes.
//! This pass finds the key list of each such site.

use crate::bytecode::captures::{Written, written_by};
use crate::bytecode::{Chunk, Instruction, OpCode};
use crate::string::LuaString;
use std::any::Any;
use std::sync::{Arc, OnceLock};

/// Fewest keys worth a shared list: one key costs a single clone anyway. A
/// one-key table is shaped only when the constructor is returned right away
/// (see [`analyze`]), so nothing is added to it afterwards.
const MIN_KEYS: usize = 2;
/// Most keys a shaped table holds; matches the runtime's small table limit.
const MAX_KEYS: usize = 8;

/// The keys one `NewTable` site stores, plus a slot where the VM keeps the
/// shared shape it builds from them.
pub struct ConstructorShape {
    keys: Box<[LuaString]>,
    shared: OnceLock<Box<dyn Any + Send + Sync>>,
}

impl ConstructorShape {
    /// The distinct string keys, in the order the constructor first stores
    /// them.
    #[must_use]
    pub fn keys(&self) -> &[LuaString] {
        &self.keys
    }

    /// The value the VM derived from [`Self::keys`], built by `init` on first
    /// use. The bytecode crate cannot name the runtime's shape type, so the
    /// slot is typed by the caller, who must always ask for the same `T`.
    ///
    /// # Panics
    ///
    /// Panics if an earlier call used a different `T`.
    #[must_use]
    #[inline]
    pub fn shared<T: Any + Send + Sync>(&self, init: impl FnOnce(&[LuaString]) -> T) -> &T {
        self.shared
            .get_or_init(|| Box::new(init(&self.keys)))
            .downcast_ref::<T>()
            .expect("a constructor shape slot holds one type")
    }
}

/// The shapes of one chunk, found by `NewTable` pc without a search: a
/// constructor site is looked up on every table the VM creates, and most
/// sites have no shape.
struct Sites {
    /// Per pc, one more than the index into `shapes`, or zero. Empty when
    /// the chunk has no shaped site, so that lookup ends at the bounds check.
    by_pc: Box<[u16]>,
    shapes: Box<[ConstructorShape]>,
}

/// The lazily computed shapes stored in a [`Chunk`].
///
/// It is a cache: clones start empty and two chunks compare equal whatever
/// they have computed. Copies made by [`Self::share`] share it instead.
#[derive(Default)]
pub struct ConstructorShapes {
    sites: OnceLock<Sites>,
    /// Where the sites are kept instead once [`Self::make_shareable`] ran:
    /// one slot for every copy [`Self::share`] makes.
    shared: Option<Arc<OnceLock<Sites>>>,
}

impl ConstructorShapes {
    fn slot(&self) -> &OnceLock<Sites> {
        self.shared.as_deref().unwrap_or(&self.sites)
    }

    /// Moves the sites, found or not, to a slot that the copies
    /// [`Self::share`] makes will share.
    pub fn make_shareable(&mut self) {
        if self.shared.is_none() {
            self.shared = Some(Arc::new(core::mem::take(&mut self.sites)));
        }
    }

    /// For an exact copy of the chunk: after [`Self::make_shareable`], the
    /// same sites, found or not, so the copies analyse the chunk once
    /// between them and share the shapes the VM builds from its sites;
    /// otherwise empty, as a clone. Those shapes are interned and pinned
    /// process-wide (`vm::registers::interned_shape`), so any copy would
    /// build the same one. The analysis reads the code, constants, upvalue
    /// names and debug locals; a copy that changes any of them must not
    /// share.
    #[must_use]
    pub fn share(&self) -> Self {
        Self {
            sites: OnceLock::new(),
            shared: self.shared.clone(),
        }
    }
}

impl Clone for ConstructorShapes {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for ConstructorShapes {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl core::fmt::Debug for ConstructorShapes {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("ConstructorShapes")
    }
}

impl Chunk {
    /// Whether the constructor shapes have been found.
    #[must_use]
    pub fn has_constructor_shapes(&self) -> bool {
        self.constructors.slot().get().is_some()
    }

    /// The shape of the `NewTable` at `pc`, if the instructions after it
    /// store two to eight distinct string keys into the new table (one key
    /// when the table is returned right after) before any
    /// branch, jump target or write to its register.
    ///
    /// Later stores of other keys are fine: the table converts when they
    /// arrive. The shape only has to name keys that are likely to be stored.
    #[must_use]
    #[inline]
    pub fn constructor_shape(&self, pc: usize) -> Option<&ConstructorShape> {
        let sites = self.constructors.slot().get_or_init(|| analyze(self));
        let index = usize::from(*sites.by_pc.get(pc)?).checked_sub(1)?;
        sites.shapes.get(index)
    }
}

/// Where control can arrive from somewhere other than the previous
/// instruction.
fn jump_targets(chunk: &Chunk) -> Vec<bool> {
    let mut targets = vec![false; chunk.code.len() + 1];
    for (pc, &instruction) in chunk.code.iter().enumerate() {
        let offset = match instruction.opcode() {
            Ok(OpCode::Jump) => instruction.jump_offset(),
            Ok(
                OpCode::JumpIfFalse
                | OpCode::NumericForPrep
                | OpCode::NumericForLoop
                | OpCode::ForInLoop,
            ) => i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes())),
            _ => continue,
        };
        let target = i64::try_from(pc).unwrap_or(i64::MAX) + 1 + i64::from(offset);
        if let Ok(target) = usize::try_from(target)
            && let Some(slot) = targets.get_mut(target)
        {
            *slot = true;
        }
    }
    targets
}

fn writes_register(instruction: Instruction, register: u8) -> bool {
    match written_by(instruction) {
        Written::None => false,
        Written::Range(start, count) => {
            let register = usize::from(register);
            (usize::from(start)..usize::from(start) + count).contains(&register)
        }
        Written::Any => true,
    }
}

/// Whether the scan must stop at `instruction`: anything not listed may branch,
/// leave the frame, or have effects the scan does not model.
fn leaves_straight_line(instruction: Instruction) -> bool {
    !matches!(
        instruction.opcode(),
        Ok(OpCode::SetField
            | OpCode::SetTable
            | OpCode::DeclareTable
            | OpCode::GetField
            | OpCode::GetTable
            | OpCode::LoadConstant
            | OpCode::LoadConstantWide
            | OpCode::LoadNil
            | OpCode::Move
            | OpCode::GetGlobal
            | OpCode::GetGlobalWide
            | OpCode::GetUpvalue
            | OpCode::GetTabUp
            | OpCode::SetTabUp
            | OpCode::Closure
            | OpCode::NewTable
            | OpCode::Call
            | OpCode::CallOne
            | OpCode::CallVararg
            | OpCode::CallOpen
            | OpCode::Vararg
            | OpCode::ExtraArgument
            | OpCode::Negate
            | OpCode::Not
            | OpCode::Length
            | OpCode::Concat
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK
            | OpCode::Add
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
            | OpCode::BitNot
            | OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::ClearRange
            | OpCode::TableVararg
            | OpCode::TableOpen)
    )
}

/// A set of registers.
#[derive(Default)]
struct Holders([u64; 4]);

impl Holders {
    fn insert(&mut self, register: u8) {
        self.0[usize::from(register >> 6)] |= 1 << (register & 63);
    }

    fn contains(&self, register: u8) -> bool {
        self.0[usize::from(register >> 6)] & (1 << (register & 63)) != 0
    }

    fn is_empty(&self) -> bool {
        self.0 == [0; 4]
    }

    fn remove(&mut self, register: usize) {
        self.0[register >> 6] &= !(1 << (register & 63));
    }

    /// The registers in the set, in ascending order.
    fn registers(&self) -> impl Iterator<Item = u8> + '_ {
        self.0.iter().enumerate().flat_map(|(word, &bits)| {
            let mut bits = bits;
            core::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                u8::try_from(word * 64 + bit as usize).ok()
            })
        })
    }

    /// Drops the registers `instruction` may write. Only the reference scan
    /// that debug builds check against uses it.
    #[cfg(debug_assertions)]
    fn remove_written(&mut self, instruction: Instruction) {
        let (start, count) = match written_by(instruction) {
            Written::None => return,
            Written::Range(start, count) => (usize::from(start), count),
            Written::Any => (0, 256),
        };
        for register in start..(start + count).min(256) {
            self.0[register >> 6] &= !(1 << (register & 63));
        }
    }
}

/// Whether `call`, at `pc`, is `setmetatable(table, ...)` with one result:
/// a `Call` whose first argument is one of `holders` and whose function
/// register was last loaded from the global `setmetatable` on the straight
/// line before it. A rebound global only makes the shape a worse guess.
fn is_setmetatable_call(
    chunk: &Chunk,
    targets: &[bool],
    pc: usize,
    call: Instruction,
    holders: &Holders,
) -> bool {
    if call.opcode() != Ok(OpCode::Call)
        || call.a() == u8::MAX
        || !holders.contains(call.a() + 1)
        || call.b() == 0
        || call.c() != 1
    {
        return false;
    }
    // The register to trace back and the pc to search below. A `Move` from a
    // temporary (the compiler loads the global before the constructor, then
    // copies it into the call register) moves the trace to its source.
    let mut function = call.a();
    let mut before = pc;
    'trace: loop {
        for writer in (0..before).rev() {
            let instruction = chunk.code[writer];
            if writes_register(instruction, function) {
                let constant = |index: usize| match chunk.constants.get(index) {
                    Some(crate::bytecode::Constant::String(name)) => Some(name),
                    _ => None,
                };
                // The global, or a local or upvalue named after it, as in
                // `local setmetatable = setmetatable`.
                let name = match instruction.opcode() {
                    Ok(OpCode::GetGlobal) => constant(usize::from(instruction.d())),
                    Ok(OpCode::GetTabUp) => constant(usize::from(instruction.c())),
                    Ok(OpCode::GetUpvalue) => chunk
                        .upvalues
                        .get(usize::from(instruction.d()))
                        .map(|upvalue| &upvalue.name),
                    Ok(OpCode::Move) => {
                        let local = chunk
                            .debug
                            .locals_at(writer)
                            .filter(|local| local.register() == instruction.b())
                            .last()
                            .map(crate::bytecode::LocalVariable::name);
                        if local.is_none() {
                            // As below: a join point at the copy may bring
                            // its source from another path.
                            if targets[writer] {
                                return false;
                            }
                            function = instruction.b();
                            before = writer;
                            continue 'trace;
                        }
                        local
                    }
                    _ => None,
                };
                return name.is_some_and(|name| name.as_bytes() == b"setmetatable");
            }
            // Control may arrive here with the register loaded elsewhere.
            if targets[writer] {
                return false;
            }
        }
        return false;
    }
}

fn analyze(chunk: &Chunk) -> Sites {
    let mut sites = shaped_sites(chunk);
    // A chunk with more sites than `u16` counts leaves the rest unshaped.
    sites.truncate(usize::from(u16::MAX));
    let mut by_pc = Vec::new();
    if !sites.is_empty() {
        by_pc.resize(chunk.code.len(), 0);
        for (index, (pc, _)) in sites.iter().enumerate() {
            by_pc[*pc] = u16::try_from(index + 1).unwrap_or(0);
        }
    }
    let result = Sites {
        by_pc: by_pc.into_boxed_slice(),
        shapes: sites.into_iter().map(|(_, shape)| shape).collect(),
    };
    #[cfg(debug_assertions)]
    {
        let reference = analyze_by_site(chunk);
        debug_assert_eq!(result.by_pc, reference.by_pc);
        debug_assert!(
            result
                .shapes
                .iter()
                .zip(reference.shapes.iter())
                .all(|(shape, other)| shape.keys == other.keys)
        );
    }
    result
}

/// One constructor the scan follows: the registers that hold its table and
/// the keys stored into it so far.
struct Constructor {
    pc: usize,
    holders: Holders,
    keys: Vec<LuaString>,
    open: bool,
}

/// The shaped `NewTable` sites of `chunk` in pc order.
///
/// Each constructor is followed from its `NewTable` to the end of its
/// straight line, a write that leaves no register holding the table, or a
/// key past [`MAX_KEYS`]. A straight line can hold thousands of
/// constructors whose tables stay in locals to its end, so the scan follows
/// all of them in one pass, with a per-register list of the constructors
/// whose table the register holds, instead of rescanning the line from each
/// `NewTable`.
fn shaped_sites(chunk: &Chunk) -> Vec<(usize, ConstructorShape)> {
    if !chunk
        .code
        .iter()
        .any(|instruction| instruction.opcode() == Ok(OpCode::NewTable))
    {
        return Vec::new();
    }
    let targets = jump_targets(chunk);
    let mut constructors: Vec<Constructor> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    let mut owners: Vec<Vec<usize>> = vec![Vec::new(); 256];
    let mut sites = Vec::new();
    let mut finish = |constructor: &mut Constructor, returned: bool| {
        constructor.open = false;
        let keys = core::mem::take(&mut constructor.keys);
        // A record that gains fields pays a shaped-to-ordered conversion that
        // the ordered table's small insert does not, so a lone key is shaped
        // only for `return { key = v }`, the leaf of a tree or list.
        let worth = keys.len() >= MIN_KEYS || (keys.len() == 1 && returned);
        if worth && keys.len() <= MAX_KEYS {
            sites.push((
                constructor.pc,
                ConstructorShape {
                    keys: keys.into_boxed_slice(),
                    shared: OnceLock::new(),
                },
            ));
        }
    };
    let mut touched = Vec::new();
    for (next, &instruction) in chunk.code.iter().enumerate() {
        if !open.is_empty() {
            if targets[next] || leaves_straight_line(instruction) {
                let returned = !targets[next]
                    && instruction.opcode() == Ok(OpCode::Return)
                    && instruction.b() == 1;
                for &index in &open {
                    let constructor = &mut constructors[index];
                    let returned = returned && constructor.holders.contains(instruction.a());
                    for register in constructor.holders.registers() {
                        owners[usize::from(register)].clear();
                    }
                    finish(constructor, returned);
                }
                open.clear();
            } else {
                if instruction.opcode() == Ok(OpCode::SetField)
                    && let Some(crate::bytecode::Constant::String(name)) =
                        chunk.constants.get(usize::from(instruction.b()))
                {
                    let table = usize::from(instruction.a());
                    let mut full = Vec::new();
                    for &index in &owners[table] {
                        let keys = &mut constructors[index].keys;
                        if !keys.contains(name) {
                            keys.push(name.clone());
                            if keys.len() > MAX_KEYS {
                                full.push(index);
                            }
                        }
                    }
                    for index in full {
                        let constructor = &mut constructors[index];
                        for register in constructor.holders.registers() {
                            owners[usize::from(register)].retain(|owner| *owner != index);
                        }
                        finish(constructor, false);
                        open.retain(|open| *open != index);
                    }
                }
                // `setmetatable({...}, mt)` hands the table back in the
                // call's register, where the fields the object gains next
                // are stored.
                let kept: Vec<usize> = match instruction.opcode() {
                    Ok(OpCode::Move) => owners[usize::from(instruction.b())].clone(),
                    Ok(OpCode::Call) if instruction.a() != u8::MAX => {
                        let argument = &owners[usize::from(instruction.a() + 1)];
                        match argument.first() {
                            Some(&index)
                                if is_setmetatable_call(
                                    chunk,
                                    &targets,
                                    next,
                                    instruction,
                                    &constructors[index].holders,
                                ) =>
                            {
                                argument.clone()
                            }
                            _ => Vec::new(),
                        }
                    }
                    _ => Vec::new(),
                };
                match written_by(instruction) {
                    Written::None => {}
                    Written::Range(start, count) => {
                        let start = usize::from(start);
                        for (register, holding) in
                            owners.iter_mut().enumerate().skip(start).take(count)
                        {
                            for index in holding.drain(..) {
                                constructors[index].holders.remove(register);
                                touched.push(index);
                            }
                        }
                    }
                    Written::Any => {
                        for &index in &open {
                            constructors[index].holders = Holders::default();
                            touched.push(index);
                        }
                        owners.iter_mut().for_each(Vec::clear);
                    }
                }
                let target = instruction.a();
                for index in kept {
                    let constructor = &mut constructors[index];
                    if constructor.open && !constructor.holders.contains(target) {
                        constructor.holders.insert(target);
                        owners[usize::from(target)].push(index);
                    }
                }
                for index in touched.drain(..) {
                    let constructor = &mut constructors[index];
                    if constructor.open && constructor.holders.is_empty() {
                        finish(constructor, false);
                        open.retain(|open| *open != index);
                    }
                }
            }
        }
        if instruction.opcode() == Ok(OpCode::NewTable) {
            let mut holders = Holders::default();
            holders.insert(instruction.a());
            owners[usize::from(instruction.a())].push(constructors.len());
            open.push(constructors.len());
            constructors.push(Constructor {
                pc: next,
                holders,
                keys: Vec::new(),
                open: true,
            });
        }
    }
    for index in open {
        finish(&mut constructors[index], false);
    }
    sites.sort_unstable_by_key(|(pc, _)| *pc);
    sites
}

#[cfg(debug_assertions)]
fn analyze_by_site(chunk: &Chunk) -> Sites {
    let mut targets = None;
    let mut sites = Vec::new();
    for (pc, &instruction) in chunk.code.iter().enumerate() {
        if instruction.opcode() != Ok(OpCode::NewTable) {
            continue;
        }
        let targets = targets.get_or_insert_with(|| jump_targets(chunk));
        // The registers that hold the new table: the constructor's, and the
        // copies `Move` and `setmetatable` make of it.
        let mut holders = Holders::default();
        holders.insert(instruction.a());
        let mut keys: Vec<LuaString> = Vec::new();
        let mut returned = false;
        for (next, &instruction) in chunk.code.iter().enumerate().skip(pc + 1) {
            if targets[next] || leaves_straight_line(instruction) {
                returned = !targets[next]
                    && instruction.opcode() == Ok(OpCode::Return)
                    && holders.contains(instruction.a())
                    && instruction.b() == 1;
                break;
            }
            if instruction.opcode() == Ok(OpCode::SetField)
                && holders.contains(instruction.a())
                && let Some(crate::bytecode::Constant::String(name)) =
                    chunk.constants.get(usize::from(instruction.b()))
                && !keys.contains(name)
            {
                keys.push(name.clone());
                if keys.len() > MAX_KEYS {
                    break;
                }
            }
            // `setmetatable({...}, mt)` hands the table back in the call's
            // register, where the fields the object gains next are stored.
            let kept = match instruction.opcode() {
                Ok(OpCode::Move) => holders.contains(instruction.b()),
                Ok(OpCode::Call) => {
                    is_setmetatable_call(chunk, targets, next, instruction, &holders)
                }
                _ => false,
            };
            holders.remove_written(instruction);
            if kept {
                holders.insert(instruction.a());
            }
            if holders.is_empty() {
                break;
            }
        }
        // A chunk with more sites than `u16` counts leaves the rest unshaped.
        // A record that gains fields pays a shaped-to-ordered conversion that
        // the ordered table's small insert does not, so a lone key is shaped
        // only for `return { key = v }`, the leaf of a tree or list.
        let worth = keys.len() >= MIN_KEYS || (keys.len() == 1 && returned);
        if worth && keys.len() <= MAX_KEYS && sites.len() < usize::from(u16::MAX) {
            sites.push((
                pc,
                ConstructorShape {
                    keys: keys.into_boxed_slice(),
                    shared: OnceLock::new(),
                },
            ));
        }
    }
    let mut by_pc = Vec::new();
    if !sites.is_empty() {
        by_pc.resize(chunk.code.len(), 0);
        for (index, (pc, _)) in sites.iter().enumerate() {
            by_pc[*pc] = u16::try_from(index + 1).unwrap_or(0);
        }
    }
    Sites {
        by_pc: by_pc.into_boxed_slice(),
        shapes: sites.into_iter().map(|(_, shape)| shape).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(constants: &[&str], code: Vec<Instruction>) -> Chunk {
        Chunk::with_parts(
            8,
            0,
            constants
                .iter()
                .map(|name| crate::bytecode::Constant::String(LuaString::from(*name)))
                .collect(),
            code,
            Vec::new(),
            Vec::new(),
        )
    }

    fn keys(chunk: &Chunk, pc: usize) -> Option<Vec<String>> {
        chunk.constructor_shape(pc).map(|shape| {
            shape
                .keys()
                .iter()
                .map(|key| String::from_utf8_lossy(key.as_bytes()).into_owned())
                .collect()
        })
    }

    fn new_table(register: u8) -> Instruction {
        Instruction::abc(OpCode::NewTable, register, 0, 0)
    }

    fn ret() -> Instruction {
        Instruction::abc(OpCode::Return, 0, 1, 0)
    }

    #[test]
    fn collects_keys_stored_into_the_new_register() {
        let chunk = chunk(
            &["value", "left", "right"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::set_field(1, 1, 3),
                // A store into another table is not a key of this one.
                Instruction::set_field(5, 2, 3),
                Instruction::set_field(1, 2, 3),
                ret(),
            ],
        );
        assert_eq!(
            keys(&chunk, 0),
            Some(vec!["value".into(), "left".into(), "right".into()])
        );
        assert!(chunk.constructor_shape(1).is_none());
    }

    #[test]
    fn constructors_open_at_once_keep_their_own_keys() {
        // Two tables live in locals side by side, one also copied: each
        // collects only the stores into a register that holds it, and a new
        // table in a register ends the older table's hold on it.
        let chunk = chunk(
            &["a", "b", "c", "d"],
            vec![
                new_table(1),
                new_table(2),
                Instruction::set_field(1, 0, 3),
                Instruction::set_field(2, 1, 3),
                Instruction::move_(4, 1),
                Instruction::set_field(4, 2, 3),
                Instruction::set_field(2, 3, 3),
                new_table(4),
                Instruction::set_field(4, 0, 3),
                Instruction::set_field(4, 1, 3),
                Instruction::set_field(1, 3, 3),
                ret(),
            ],
        );
        assert_eq!(
            keys(&chunk, 0),
            Some(vec!["a".into(), "c".into(), "d".into()])
        );
        assert_eq!(keys(&chunk, 1), Some(vec!["b".into(), "d".into()]));
        assert_eq!(keys(&chunk, 7), Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn duplicates_count_once() {
        let chunk = chunk(
            &["a", "b"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::set_field(1, 0, 3),
                ret(),
            ],
        );
        // One distinct key is below the minimum.
        assert!(chunk.constructor_shape(0).is_none());
        let chunk = self::chunk(
            &["a", "b"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::set_field(1, 0, 3),
                Instruction::set_field(1, 1, 3),
                ret(),
            ],
        );
        assert_eq!(keys(&chunk, 0), Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn a_lone_key_is_shaped_only_when_the_table_is_returned_at_once() {
        let returned = chunk(
            &["a"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::abc(OpCode::Return, 1, 1, 0),
            ],
        );
        assert_eq!(keys(&returned, 0), Some(vec!["a".into()]));
        // Another register, several values, or a longer scan may still grow the table.
        let other = chunk(
            &["a"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::abc(OpCode::Return, 2, 1, 0),
            ],
        );
        assert!(other.constructor_shape(0).is_none());
        let many = chunk(
            &["a"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::abc(OpCode::Return, 1, 2, 0),
            ],
        );
        assert!(many.constructor_shape(0).is_none());
    }

    #[test]
    fn one_key_or_more_than_eight_give_no_shape() {
        let names: Vec<String> = (0..9).map(|index| format!("k{index}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let build = |count: usize| {
            let mut code = vec![new_table(1)];
            code.extend((0..count).map(|index| Instruction::set_field(1, index as u8, 2)));
            code.push(ret());
            chunk(&names, code)
        };
        assert!(build(1).constructor_shape(0).is_none());
        assert_eq!(keys(&build(2), 0).map(|keys| keys.len()), Some(2));
        assert_eq!(keys(&build(8), 0).map(|keys| keys.len()), Some(8));
        assert!(build(9).constructor_shape(0).is_none());
    }

    #[test]
    fn stops_at_a_branch() {
        let chunk = chunk(
            &["a", "b", "c"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::set_field(1, 1, 2),
                Instruction::aj(OpCode::Jump, 0).unwrap(),
                Instruction::set_field(1, 2, 2),
                ret(),
            ],
        );
        assert_eq!(keys(&chunk, 0), Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn stops_at_a_jump_target() {
        let chunk = chunk(
            &["a", "b", "c"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::set_field(1, 1, 2),
                // Lands on pc 5 from a branch outside the straight line.
                Instruction::set_field(1, 2, 2),
                ret(),
                ret(),
            ],
        );
        assert_eq!(
            keys(&chunk, 0),
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
        let mut code = chunk.code.clone();
        // A jump at pc 4 back to pc 3 makes pc 3 a join point.
        code[4] = Instruction::aj(OpCode::Jump, -2).unwrap();
        let chunk = self::chunk(&["a", "b", "c"], code);
        assert_eq!(keys(&chunk, 0), Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn stops_when_the_register_is_overwritten() {
        let chunk = chunk(
            &["a", "b", "c"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::set_field(1, 1, 2),
                Instruction::abc(OpCode::LoadNil, 1, 0, 0),
                Instruction::set_field(1, 2, 2),
                ret(),
            ],
        );
        assert_eq!(keys(&chunk, 0), Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn a_call_over_the_register_ends_the_scan() {
        let chunk = chunk(
            &["a", "b", "c"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::set_field(1, 1, 2),
                Instruction::call(0, 0, 1),
                Instruction::set_field(1, 2, 2),
                ret(),
            ],
        );
        assert_eq!(keys(&chunk, 0), Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn follows_the_table_through_setmetatable() {
        // local c = setmetatable({name = v}, mt); c.x = v; c.y = v, as the
        // compiler emits it: the table is built in a scratch register,
        // moved into the argument, and the result moved into `c`.
        let chunk = chunk(
            &["setmetatable", "name", "x", "y"],
            vec![
                Instruction::get_global(5, 0),
                new_table(4),
                Instruction::set_field(4, 1, 0),
                Instruction::abc(OpCode::Move, 6, 4, 0),
                Instruction::abc(OpCode::Move, 7, 0, 0),
                Instruction::call(5, 2, 1),
                Instruction::abc(OpCode::Move, 2, 5, 0),
                Instruction::set_field(2, 2, 0),
                Instruction::set_field(2, 3, 0),
                ret(),
            ],
        );
        assert_eq!(
            keys(&chunk, 1),
            Some(vec!["name".into(), "x".into(), "y".into()])
        );
    }

    #[test]
    fn a_local_or_upvalue_named_setmetatable_counts() {
        let code = |load: Instruction| {
            vec![
                load,
                new_table(4),
                Instruction::set_field(4, 1, 0),
                Instruction::abc(OpCode::Move, 6, 4, 0),
                Instruction::call(5, 1, 1),
                Instruction::set_field(5, 2, 0),
                ret(),
            ]
        };
        let names = ["unused", "name", "x"];
        let expected = Some(vec!["name".into(), "x".into()]);
        let upvalue = Chunk::with_parts(
            8,
            0,
            names
                .iter()
                .map(|name| crate::bytecode::Constant::String(LuaString::from(*name)))
                .collect(),
            code(Instruction::get_upvalue(5, 0)),
            Vec::new(),
            vec![crate::bytecode::UpvalueDescriptor {
                name: LuaString::from("setmetatable"),
                source: crate::bytecode::UpvalueSource::Detached,
            }],
        );
        assert_eq!(keys(&upvalue, 1), expected);
        let mut local = chunk(&names, code(Instruction::abc(OpCode::Move, 5, 0, 0)));
        local.debug =
            local
                .debug
                .clone()
                .with_local_variables(vec![crate::bytecode::LocalVariable::new(
                    "setmetatable",
                    0,
                    0,
                    7,
                )]);
        assert_eq!(keys(&local, 1), expected);
        let mut other = chunk(&names, code(Instruction::abc(OpCode::Move, 5, 0, 0)));
        other.debug = other
            .debug
            .clone()
            .with_local_variables(vec![crate::bytecode::LocalVariable::new("make", 0, 0, 7)]);
        assert!(other.constructor_shape(1).is_none());
    }

    #[test]
    fn other_calls_end_the_scan() {
        let build = |function: Instruction, results: u8| {
            chunk(
                &["setmetatable", "name", "x", "other"],
                vec![
                    function,
                    new_table(2),
                    Instruction::set_field(2, 1, 0),
                    Instruction::call(1, 1, results),
                    Instruction::set_field(1, 2, 0),
                    ret(),
                ],
            )
        };
        // Another global, or every result kept.
        assert!(
            build(Instruction::get_global(1, 3), 1)
                .constructor_shape(1)
                .is_none()
        );
        assert!(
            build(Instruction::get_global(1, 0), 0)
                .constructor_shape(1)
                .is_none()
        );
        assert_eq!(
            keys(&build(Instruction::get_global(1, 0), 1), 1),
            Some(vec!["name".into(), "x".into()])
        );
        // A join point between the load and the call.
        let mut code = build(Instruction::get_global(1, 0), 1).code.clone();
        code.push(Instruction::aj(OpCode::Jump, -6).unwrap());
        let joined = chunk(&["setmetatable", "name", "x", "other"], code);
        assert!(joined.constructor_shape(1).is_none());
    }

    #[test]
    fn clones_recompute_and_compare_equal() {
        let chunk = chunk(
            &["a", "b"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::set_field(1, 1, 2),
                ret(),
            ],
        );
        let _ = chunk.constructor_shape(0);
        assert_eq!(chunk, chunk.clone());
        assert!(chunk.clone().constructor_shape(0).is_some());
    }

    #[test]
    fn shared_copies_analyse_once() {
        let chunk = chunk(
            &["a", "b"],
            vec![
                new_table(1),
                Instruction::set_field(1, 0, 2),
                Instruction::set_field(1, 1, 2),
                ret(),
            ],
        );
        let mut chunk = chunk;
        chunk.constructors.make_shareable();
        let mut copy = chunk.clone();
        copy.constructors = chunk.constructors.share();
        assert!(!chunk.has_constructor_shapes());
        assert_eq!(keys(&copy, 0), Some(vec!["a".to_owned(), "b".to_owned()]));
        assert!(chunk.has_constructor_shapes());
        assert!(core::ptr::eq(
            chunk.constructor_shape(0).expect("a shaped site"),
            copy.constructor_shape(0).expect("a shaped site")
        ));
        assert!(!chunk.clone().has_constructor_shapes());
    }
}
