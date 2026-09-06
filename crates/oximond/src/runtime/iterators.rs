//! `string.gmatch` iterator state and its `Runtime` operations.

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::runtime::Runtime;
use crate::runtime::value::GmatchIteratorHandle;
use crate::string::LuaString;

/// The data an iterator needs to resume byte-pattern matching. Pattern
/// validation and matching stay outside the runtime so this crate does not
/// depend on the pattern engine.
#[derive(Clone)]
pub struct GmatchIteratorSnapshot {
    source: LuaString,
    pattern: LuaString,
    search: usize,
    last_match: Option<usize>,
    utf8_codes: bool,
}

impl GmatchIteratorSnapshot {
    #[must_use]
    pub fn source(&self) -> &LuaString {
        &self.source
    }

    #[must_use]
    pub fn pattern(&self) -> &LuaString {
        &self.pattern
    }

    #[must_use]
    pub const fn search(&self) -> usize {
        self.search
    }

    /// Where the last accepted match ended, when Lua 5.3+ empty-match
    /// skipping recorded one.
    #[must_use]
    pub const fn last_match(&self) -> Option<usize> {
        self.last_match
    }

    #[must_use]
    pub const fn is_utf8_codes(&self) -> bool {
        self.utf8_codes
    }
}

pub(crate) struct GmatchIterator {
    pub(crate) source: LuaString,
    pub(crate) pattern: LuaString,
    search: usize,
    last_match: Option<usize>,
    utf8_codes: bool,
}

impl Runtime {
    /// Creates state for one callable `string.gmatch` iterator.
    ///
    /// Pattern validation is done by the standard library before allocation.
    /// Keeping the source and cursor here gives the iterator normal Lua
    /// identity and lets the collector reject stale values safely.
    ///
    /// # Errors
    ///
    /// Returns an error if the iterator arena cannot allocate another handle.
    pub fn create_gmatch_iterator(
        &mut self,
        source: LuaString,
        pattern: LuaString,
    ) -> RuntimeResult<GmatchIteratorHandle> {
        let bytes = std::mem::size_of::<GmatchIterator>() + source.len() + pattern.len();
        let handle = self
            .gmatch_iterators
            .insert(GmatchIterator {
                source,
                pattern,
                search: 0,
                last_match: None,
                utf8_codes: false,
            })
            .map(GmatchIteratorHandle)
            .ok_or_else(|| RuntimeError::message("gmatch iterator handle space exhausted"))?;
        self.gc_charge(bytes);
        Ok(handle)
    }

    /// Copies the immutable matching inputs and current cursor for `handle`.
    ///
    /// # Errors
    ///
    /// Returns an error when `handle` is stale or invalid.
    pub fn gmatch_iterator_snapshot(
        &self,
        handle: GmatchIteratorHandle,
    ) -> RuntimeResult<GmatchIteratorSnapshot> {
        let iterator = self
            .gmatch_iterators
            .get(handle.0)
            .ok_or_else(|| RuntimeError::InvalidGmatchIterator(handle))?;
        Ok(GmatchIteratorSnapshot {
            source: iterator.source.clone(),
            pattern: iterator.pattern.clone(),
            search: iterator.search,
            last_match: iterator.last_match,
            utf8_codes: iterator.utf8_codes,
        })
    }

    /// The source, pattern, search position, last accepted match end and
    /// `utf8.codes` flag of an iterator, borrowed: [`Self::gmatch_iterator_snapshot`]
    /// without the string clones. `None` for a stale handle.
    #[must_use]
    pub fn gmatch_iterator_view(
        &self,
        handle: GmatchIteratorHandle,
    ) -> Option<(&LuaString, &LuaString, usize, Option<usize>, bool)> {
        let iterator = self.gmatch_iterators.get(handle.0)?;
        Some((
            &iterator.source,
            &iterator.pattern,
            iterator.search,
            iterator.last_match,
            iterator.utf8_codes,
        ))
    }

    /// Creates the callable iterator returned by `utf8.codes`.
    ///
    /// # Errors
    ///
    /// Returns an error when the source contains an invalid UTF-8 sequence.
    pub fn create_utf8_codes_iterator(
        &mut self,
        source: LuaString,
    ) -> RuntimeResult<GmatchIteratorHandle> {
        let bytes = std::mem::size_of::<GmatchIterator>() + source.len();
        let handle = self
            .gmatch_iterators
            .insert(GmatchIterator {
                source,
                pattern: LuaString::new([]),
                search: 0,
                last_match: None,
                utf8_codes: true,
            })
            .map(GmatchIteratorHandle)
            .ok_or_else(|| RuntimeError::message("UTF-8 iterator handle space exhausted"))?;
        self.gc_charge(bytes);
        Ok(handle)
    }

    /// Commits the cursor after a successful `string.gmatch` probe.
    ///
    /// # Errors
    ///
    /// Returns an error when `handle` is stale or invalid.
    pub fn set_gmatch_iterator_search(
        &mut self,
        handle: GmatchIteratorHandle,
        search: usize,
    ) -> RuntimeResult<()> {
        let iterator = self
            .gmatch_iterators
            .get_mut(handle.0)
            .ok_or_else(|| RuntimeError::InvalidGmatchIterator(handle))?;
        iterator.search = search;
        Ok(())
    }

    /// Commits an accepted match under Lua 5.3+ rules: the next search
    /// resumes at `end`, and an empty match ending there is skipped.
    ///
    /// # Errors
    ///
    /// Returns an error when `handle` is stale or invalid.
    pub fn set_gmatch_iterator_last_match(
        &mut self,
        handle: GmatchIteratorHandle,
        end: usize,
    ) -> RuntimeResult<()> {
        let iterator = self
            .gmatch_iterators
            .get_mut(handle.0)
            .ok_or_else(|| RuntimeError::InvalidGmatchIterator(handle))?;
        iterator.search = end;
        iterator.last_match = Some(end);
        Ok(())
    }
}
