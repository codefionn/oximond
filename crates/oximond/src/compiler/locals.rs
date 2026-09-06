//! The local bindings of the open scopes of one function, and register sets.

use crate::compiler::{Ident, Register};

/// A set of the 256 frame registers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RegisterBits([u64; 4]);

impl RegisterBits {
    pub(crate) const EMPTY: Self = Self([0; 4]);

    pub(crate) fn insert(&mut self, register: Register) {
        self.0[usize::from(register >> 6)] |= 1 << (register & 63);
    }

    pub(crate) fn remove(&mut self, register: Register) {
        self.0[usize::from(register >> 6)] &= !(1 << (register & 63));
    }

    pub(crate) fn contains(self, register: Register) -> bool {
        self.0[usize::from(register >> 6)] & (1 << (register & 63)) != 0
    }

    pub(crate) fn union(self, other: Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    /// The highest register in the set.
    pub(crate) fn max(self) -> Option<Register> {
        (0..4).rev().find(|&word| self.0[word] != 0).map(|word| {
            let bit = 63 - self.0[word].leading_zeros() as usize;
            Register::try_from(word * 64 + bit).expect("a register is below 256")
        })
    }

    /// The registers in ascending order.
    pub(crate) fn iter(self) -> impl Iterator<Item = Register> {
        (0..4_usize).flat_map(move |word| {
            let mut bits = self.0[word];
            core::iter::from_fn(move || {
                (bits != 0).then(|| {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    Register::try_from(word * 64 + bit).expect("a register is below 256")
                })
            })
        })
    }

    /// The registers from `first` up, in ascending order.
    pub(crate) fn from(self, first: Register) -> impl Iterator<Item = Register> {
        self.iter().filter(move |register| *register >= first)
    }
}

impl FromIterator<Register> for RegisterBits {
    fn from_iter<I: IntoIterator<Item = Register>>(registers: I) -> Self {
        let mut set = Self::EMPTY;
        for register in registers {
            set.insert(register);
        }
        set
    }
}

/// The local bindings of the open lexical scopes of one function.
///
/// Bindings sit in one vector in declaration order, so the bindings of a scope
/// are contiguous and those of inner scopes come later. A name is bound at
/// most once per scope: binding it again in the same scope replaces the
/// register. Lookups walk back from the innermost binding, as Lua's own
/// compiler does, which for the few dozen locals a function sees beats a
/// tree of owned names per scope, and opening a scope allocates nothing.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct LocalScopes {
    bindings: Vec<(Ident, Register)>,
    /// Where each open scope's bindings start in `bindings`.
    starts: Vec<usize>,
    /// One more than the highest register of a binding, or zero.
    end: u16,
}

impl LocalScopes {
    /// The number of open scopes.
    pub(crate) fn len(&self) -> usize {
        self.starts.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    pub(crate) fn push(&mut self) {
        self.starts.push(self.bindings.len());
    }

    /// Closes the innermost scope.
    pub(crate) fn pop(&mut self) {
        let start = self.starts.pop().expect("compiler scope exists");
        self.bindings.truncate(start);
        self.recount();
    }

    fn recount(&mut self) {
        self.end = self
            .bindings
            .iter()
            .map(|(_, register)| u16::from(*register) + 1)
            .max()
            .unwrap_or(0);
    }

    /// Binds `name` to `register` in the innermost scope.
    pub(crate) fn insert(&mut self, name: Ident, register: Register) {
        let start = *self.starts.last().expect("compiler scope exists");
        if let Some(binding) = self.bindings[start..]
            .iter_mut()
            .find(|(bound, _)| **bound == *name)
        {
            binding.1 = register;
            self.recount();
            return;
        }
        self.bindings.push((name, register));
        self.end = self.end.max(u16::from(register) + 1);
    }

    /// The register of the innermost binding of `name`.
    pub(crate) fn get(&self, name: &str) -> Option<Register> {
        self.bindings
            .iter()
            .rev()
            .find(|(bound, _)| &**bound == name)
            .map(|(_, register)| *register)
    }

    /// The bindings of the scope at `depth`, the outermost being zero.
    pub(crate) fn scope(&self, depth: usize) -> &[(Ident, Register)] {
        let start = self.starts[depth];
        let end = self
            .starts
            .get(depth + 1)
            .copied()
            .unwrap_or(self.bindings.len());
        &self.bindings[start..end]
    }

    /// The scopes, outermost first.
    pub(crate) fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = &[(Ident, Register)]> + ExactSizeIterator {
        (0..self.len()).map(|depth| self.scope(depth))
    }

    /// The registers of every binding.
    pub(crate) fn registers(&self) -> impl Iterator<Item = Register> + '_ {
        self.bindings.iter().map(|(_, register)| *register)
    }

    /// Moves every binding to the register `remap` gives for its own.
    pub(crate) fn remap(&mut self, remap: &[Register; 256]) {
        for (_, register) in &mut self.bindings {
            *register = remap[usize::from(*register)];
        }
        self.recount();
    }

    /// Whether the bindings fill exactly the registers below their count, so
    /// every register from the count up is free.
    pub(crate) fn packed(&self) -> bool {
        usize::from(self.end) <= self.bindings.len()
    }

    /// The number of bindings in all open scopes.
    /// One more than the highest register of a binding, or zero.
    pub(crate) fn end(&self) -> u16 {
        self.end
    }

    pub(crate) fn binding_count(&self) -> usize {
        self.bindings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_bindings_shadow_outer_ones_until_their_scope_closes() {
        let mut scopes = LocalScopes::default();
        scopes.push();
        scopes.insert("x".into(), 0);
        scopes.insert("y".into(), 1);
        scopes.push();
        scopes.insert("x".into(), 2);
        assert_eq!(scopes.get("x"), Some(2));
        assert_eq!(scopes.get("y"), Some(1));
        assert_eq!(scopes.scope(1), &[("x".into(), 2)]);
        // Rebinding in the same scope replaces the register.
        scopes.insert("x".into(), 3);
        assert_eq!(scopes.binding_count(), 3);
        assert!(!scopes.packed());
        scopes.pop();
        assert!(scopes.packed());
        assert_eq!(scopes.get("x"), Some(0));
        assert_eq!(scopes.len(), 1);
        assert_eq!(scopes.iter().next().map(<[_]>::len), Some(2));
    }

    #[test]
    fn register_bits_iterate_in_ascending_order() {
        let set = [200, 3, 64, 63].into_iter().collect::<RegisterBits>();
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![3, 63, 64, 200]);
        assert_eq!(set.from(64).collect::<Vec<_>>(), vec![64, 200]);
        assert_eq!(set.max(), Some(200));
        assert_eq!(RegisterBits::EMPTY.max(), None);
        let mut removed = set;
        removed.remove(200);
        assert_eq!(removed.max(), Some(64));
    }
}
