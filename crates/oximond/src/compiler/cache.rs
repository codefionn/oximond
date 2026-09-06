//! Compiled chunks, shared by every Lua state of the process.
//!
//! An embedder such as Luanti runs one Lua state per thread (the server, the
//! map generator, async workers), and each state loads many of the same mod
//! files. Compiling is deterministic in the chunk's name, bytes and
//! compatibility profile, so the second state reuses the first state's
//! bytecode instead of parsing and compiling the file again.
//!
//! Strings loaded at run time repeat too: Luanti deserializes stored data
//! with `loadstring`, and Mineclonia's start loads 1905 strings (6.7 MB)
//! of which 494 differ, a 1.2 MB schematic among them in both the server
//! and the map generator state. Files (a chunk name that starts with `@`)
//! and strings have a budget each, charged with an estimate of everything an
//! entry retains. Past it the cache stops growing, so a program that builds
//! and loads endless distinct strings cannot grow it without bound.
//!
//! A string is kept from its first load while less than half the string
//! budget is spent: the schematic above is loaded exactly twice, and keeping
//! it only on its second load would compile it twice. Past half, a string
//! is kept only when it is loaded a second time, so a program that loads
//! many strings once cannot fill the rest with them before strings that
//! repeat come along. A small table of the strings seen once decides that;
//! it forgets an old string when a new one lands in its slot. Files are
//! kept from their first load. Each hit gets its
//! own copy of the function tree, so the per-chunk caches that execution
//! fills (field sites, the shapes the VM builds for constructors, capture
//! volatility) never cross from one state to another.
//!
//! The analyses a first call runs on a function, of its live registers and
//! of its constructors' keys, depend on the bytecode alone, so the copies
//! of a main function share them: the copy that runs first analyses it for
//! all, and a data string Luanti loads in several states, or loads again,
//! is not analysed again.
//!
//! Chunks are copied outside the cache's lock. A map generator thread that
//! copies the 1.2 MB schematic must not hold up every other state's load.

use std::sync::{Arc, Mutex, PoisonError};

use crate::bytecode::{Chunk, Constant};
use crate::compiler::CompilerConfig;
use crate::string::LuaString;

/// The most bytes the cache retains for files. A game's mods are a few
/// megabytes of source; past this the cache stops growing and later files
/// compile as before.
const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;

/// The most bytes the cache retains for strings, on the same terms.
/// Mineclonia's start keeps an estimated 23 MB of strings, the 1.2 MB
/// schematic alone 9.9 MB, which fits under half of this.
const MAX_STRING_BYTES: usize = 64 * 1024 * 1024;

/// The slots of the table of strings seen once. A slot holds a key; a new
/// string whose key selects a taken slot replaces the one there.
const SEEN_ONCE_SLOTS: usize = 1 << 14;

struct Entry {
    config: CompilerConfig,
    name: LuaString,
    source: Box<[u8]>,
    chunk: Arc<Chunk>,
}

struct Cache {
    entries: rustc_hash::FxHashMap<u64, Vec<Entry>>,
    file_bytes: usize,
    string_bytes: usize,
    file_budget: usize,
    string_budget: usize,
    /// Keys of strings loaded once, each in the slot its key selects. Zero
    /// marks a free slot.
    seen_once: Box<[u64]>,
}

impl Cache {
    fn new(file_budget: usize, string_budget: usize) -> Self {
        Self {
            entries: rustc_hash::FxHashMap::default(),
            file_bytes: 0,
            string_bytes: 0,
            file_budget,
            string_budget,
            seen_once: Box::default(),
        }
    }

    fn find(
        &self,
        key: u64,
        config: CompilerConfig,
        name: &LuaString,
        source: &[u8],
    ) -> Option<&Entry> {
        self.entries
            .get(&key)?
            .iter()
            .find(|entry| entry.config == config && entry.name == *name && *entry.source == *source)
    }

    /// Whether a string with `key` was loaded before. Remembers it if not.
    fn seen_before(&mut self, key: u64) -> bool {
        if self.seen_once.is_empty() {
            self.seen_once = vec![0; SEEN_ONCE_SLOTS].into_boxed_slice();
        }
        let tag = key.max(1);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the low bits of the key select the slot"
        )]
        let slot = &mut self.seen_once[key as usize & (SEEN_ONCE_SLOTS - 1)];
        let seen = *slot == tag;
        *slot = tag;
        seen
    }

    #[cfg(test)]
    fn entry_count(&self) -> usize {
        self.entries.values().map(Vec::len).sum()
    }
}

fn default_cache() -> Cache {
    Cache::new(MAX_FILE_BYTES, MAX_STRING_BYTES)
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

/// Whether a chunk named `name` is a file.
fn is_file(name: &LuaString) -> bool {
    name.as_bytes().first() == Some(&b'@')
}

fn key(config: CompilerConfig, name: &LuaString, source: &[u8]) -> u64 {
    use std::hash::{BuildHasher, Hash, Hasher};
    let mut hasher = rustc_hash::FxBuildHasher.build_hasher();
    config.hash(&mut hasher);
    name.as_bytes().hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
}

/// The compiled chunk `name` with `source`, compiled by `compile` only when
/// no state of the process compiled it before. `None` (a source `compile`
/// cannot finish) is not kept.
///
/// `complete` runs once a chunk is in hand, compiled or copied, and says
/// whether `source` is the whole chunk. A caller that holds only the start
/// of its input reads on there. A `false` keeps the chunk out of the cache,
/// since the source it belongs to is longer.
pub(crate) fn load(
    config: CompilerConfig,
    name: &LuaString,
    source: &[u8],
    compile: impl FnOnce() -> Option<Chunk>,
    complete: impl FnOnce() -> bool,
) -> Option<Chunk> {
    load_in(&CACHE, config, name, source, compile, complete)
}

fn load_in(
    cache: &Mutex<Option<Cache>>,
    config: CompilerConfig,
    name: &LuaString,
    source: &[u8],
    compile: impl FnOnce() -> Option<Chunk>,
    complete: impl FnOnce() -> bool,
) -> Option<Chunk> {
    let file = is_file(name);
    let key = key(config, name, source);
    let keep = {
        let mut guard = cache.lock().unwrap_or_else(PoisonError::into_inner);
        let cache = guard.get_or_insert_with(default_cache);
        if let Some(entry) = cache.find(key, config, name, source) {
            let chunk = Arc::clone(&entry.chunk);
            drop(guard);
            complete();
            return Some(fresh_copy(&chunk));
        }
        let budget = if file {
            cache.file_budget
        } else {
            cache.string_budget
        };
        source.len() <= budget
            && (file || cache.string_bytes < cache.string_budget / 2 || cache.seen_before(key))
    };
    let mut chunk = compile()?;
    if !complete() || !keep {
        return Some(chunk);
    }
    make_shareable(&mut chunk);
    let copy = Arc::new(fresh_copy(&chunk));
    let size = retained_bytes(&copy) + name.len() + source.len();
    let mut guard = cache.lock().unwrap_or_else(PoisonError::into_inner);
    let cache = guard.get_or_insert_with(default_cache);
    let (kept, budget) = if file {
        (&mut cache.file_bytes, cache.file_budget)
    } else {
        (&mut cache.string_bytes, cache.string_budget)
    };
    let fits = *kept + size <= budget;
    // Another state may have compiled the same chunk meanwhile.
    if fits && cache.find(key, config, name, source).is_none() {
        if file {
            cache.file_bytes += size;
        } else {
            cache.string_bytes += size;
        }
        cache.entries.entry(key).or_default().push(Entry {
            config,
            name: name.clone(),
            source: source.into(),
            chunk: copy,
        });
    } else {
        // Free the copy after unlocking.
        drop(guard);
        drop(copy);
    }
    Some(chunk)
}

/// An estimate of the bytes `chunk` and its nested functions keep alive: a
/// fixed share for the structures and allocations every function has, then
/// its instructions, constants, line numbers and other debug records.
fn retained_bytes(chunk: &Chunk) -> usize {
    use std::mem::size_of_val;
    const PER_FUNCTION: usize = 1024;
    let strings: usize = chunk
        .constants
        .iter()
        .map(|constant| match constant {
            Constant::String(string) => string.len(),
            _ => 0,
        })
        .sum();
    let debug = &chunk.debug;
    PER_FUNCTION
        + size_of_val(chunk.code.as_slice())
        + size_of_val(chunk.constants.as_slice())
        + strings
        + size_of_val(chunk.upvalues.as_slice())
        + size_of_val(debug.line_info())
        + size_of_val(debug.call_sites())
        + size_of_val(debug.diagnostic_sites())
        + size_of_val(debug.local_variables())
        + size_of_val(chunk.prototypes.as_slice())
        + chunk
            .prototypes
            .iter()
            .map(|prototype| retained_bytes(prototype))
            .sum::<usize>()
}

/// Lets the copies of `chunk`'s main function share its analyses (see
/// [`fresh_copy`]). The main function runs in every state that loads the
/// chunk, while most nested functions run in few, and a slot for each of
/// those would cost an allocation per function and copy.
fn make_shareable(chunk: &mut Chunk) {
    chunk.live.make_shareable();
    chunk.constructors.make_shareable();
}

/// A copy of `chunk` whose nested functions are copies too. Cloning a chunk
/// starts its execution caches empty, and new nested functions keep a
/// state's caches out of every other state's functions. The analyses of the
/// bytecode are shared, computed or not, once [`make_shareable`] ran.
fn fresh_copy(chunk: &Chunk) -> Chunk {
    let mut copy = chunk.clone();
    copy.live = chunk.live.share();
    copy.constructors = chunk.constructors.share();
    for prototype in &mut copy.prototypes {
        *prototype = Arc::new(fresh_copy(prototype));
    }
    copy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::CompatibilityProfile;
    use crate::compiler::CompilerLoader;

    /// Loads `source` through `cache`, counting the compiles in `compiles`.
    fn load_counted(
        cache: &Mutex<Option<Cache>>,
        loader: CompilerLoader,
        name: &LuaString,
        source: &[u8],
        compiles: &mut usize,
    ) -> Option<Chunk> {
        load_in(
            cache,
            loader.config,
            name,
            source,
            || {
                *compiles += 1;
                loader.try_load_complete_uncached(name, source)
            },
            || true,
        )
    }

    fn retained(cache: &Mutex<Option<Cache>>) -> (usize, usize) {
        let cache = cache.lock().unwrap();
        let cache = cache.as_ref().expect("cache in use");
        (cache.entry_count(), cache.string_bytes)
    }

    #[test]
    fn a_file_compiles_once_and_every_load_gets_its_own_functions() {
        let cache = Mutex::new(None);
        let loader = CompilerLoader::with_profile(CompatibilityProfile::Lua54);
        let name = LuaString::from("@cache-test-once.lua");
        let source = b"local n = 0 return function() n = n + 1 return n end";
        let mut compiles = 0;
        let first = load_counted(&cache, loader, &name, source, &mut compiles).unwrap();
        let second = load_counted(&cache, loader, &name, source, &mut compiles).unwrap();
        assert_eq!(compiles, 1);
        assert_eq!(first, second);
        assert!(!Arc::ptr_eq(&first.prototypes[0], &second.prototypes[0]));
    }

    #[test]
    fn a_changed_file_or_string_or_another_profile_compiles_again() {
        let cache = Mutex::new(None);
        let loader = CompilerLoader::with_profile(CompatibilityProfile::Lua54);
        let file = LuaString::from("@cache-test-changed.lua");
        let mut compiles = 0;
        for source in [&b"return 1"[..], b"return 2", b"return 1"] {
            load_counted(&cache, loader, &file, source, &mut compiles);
        }
        assert_eq!(compiles, 2);

        // Under half the budget spent, a string is kept from its first load.
        let string = LuaString::from("=cache-test-string");
        for source in [&b"return 3"[..], b"return 3", b"return 3", b"return 4"] {
            load_counted(&cache, loader, &string, source, &mut compiles);
        }
        assert_eq!(compiles, 4);

        // The same text under another name is another chunk: its name is in
        // its debug information.
        let renamed = LuaString::from("=cache-test-renamed");
        load_counted(&cache, loader, &renamed, b"return 3", &mut compiles);
        assert_eq!(compiles, 5);

        let other = CompilerLoader::with_profile(CompatibilityProfile::Lua51);
        load_counted(&cache, other, &file, b"return 1", &mut compiles);
        assert_eq!(compiles, 6);
    }

    #[test]
    fn a_string_chunk_compiles_once_and_a_failed_one_every_time() {
        let cache = Mutex::new(None);
        let loader = CompilerLoader::with_profile(CompatibilityProfile::LuaJit21);
        let name = LuaString::from("=(load)");
        let source = b"return {width = 3, nodes = {1, 2, 3}, meta = {{fields = {}}}}";
        let mut compiles = 0;
        let chunks: Vec<_> = (0..3)
            .map(|_| load_counted(&cache, loader, &name, source, &mut compiles).unwrap())
            .collect();
        assert_eq!(compiles, 1);
        assert_eq!(chunks[0], chunks[2]);
        // The copies share the analyses: the first one run computes them.
        assert!(!chunks[0].has_live_registers() && !chunks[0].has_constructor_shapes());
        assert_eq!(chunks[2].entry_register_count(), 0);
        assert_eq!(
            chunks[2]
                .constructor_shape(0)
                .map(|shape| shape.keys().len()),
            Some(3)
        );
        for chunk in &chunks {
            assert!(chunk.has_live_registers() && chunk.has_constructor_shapes());
        }

        let broken = b"return {width = ";
        for _ in 0..3 {
            assert!(load_counted(&cache, loader, &name, broken, &mut compiles).is_none());
        }
        assert_eq!(compiles, 4);
    }

    #[test]
    fn a_piece_the_reader_extends_is_not_kept() {
        let cache = Mutex::new(None);
        let loader = CompilerLoader::with_profile(CompatibilityProfile::Lua51);
        let name = LuaString::from("=cache-test-piece");
        let source = b"return 1";
        let mut compiles = 0;
        for ended in [false, false, false, true, true] {
            let mut asked = false;
            let chunk = load_in(
                &cache,
                loader.config,
                &name,
                source,
                || {
                    compiles += 1;
                    loader.try_load_complete_uncached(&name, source)
                },
                || {
                    asked = true;
                    ended
                },
            );
            assert!(chunk.is_some() && asked);
        }
        // Kept by the first load that ended.
        assert_eq!(compiles, 4);
        assert_eq!(retained(&cache).0, 1);
    }

    /// `for i = 1, 4e6 do loadstring("return " .. i) end` through the C API:
    /// each string loads once, so the cache keeps them until half its string
    /// budget is spent and no more after.
    #[test]
    fn strings_loaded_once_stop_being_kept_at_half_the_budget() {
        let cache = Mutex::new(None);
        let loader = CompilerLoader::with_profile(CompatibilityProfile::LuaJit21);
        let name = LuaString::from("=(load)");
        let template = loader
            .try_load_complete_uncached(&name, b"return 1")
            .unwrap();
        let mut source = String::new();
        for i in 1..=4_000_000 {
            source.clear();
            std::fmt::Write::write_fmt(&mut source, format_args!("return {i}")).unwrap();
            load_in(
                &cache,
                loader.config,
                &name,
                source.as_bytes(),
                || Some(template.clone()),
                || true,
            );
        }
        // Each entry is charged over 1 KB, and the last one kept may cross
        // the half.
        let (entries, bytes) = retained(&cache);
        let half = MAX_STRING_BYTES / 2;
        assert!(bytes >= half && bytes < half + 2048, "{bytes} bytes kept");
        assert!(entries < half / 1024, "{entries} entries kept");
    }

    /// Once half the string budget is spent, a string loaded once is not
    /// kept, and one loaded a second time is.
    #[test]
    fn past_half_the_budget_a_string_is_kept_on_its_second_load() {
        const BUDGET: usize = 64 * 1024;
        let cache = Mutex::new(Some(Cache::new(BUDGET, BUDGET)));
        let loader = CompilerLoader::with_profile(CompatibilityProfile::LuaJit21);
        let name = LuaString::from("=(load)");
        let mut compiles = 0;
        let mut sources = 0;
        while retained(&cache).1 < BUDGET / 2 {
            let source = format!("return {{{sources}}}");
            load_counted(&cache, loader, &name, source.as_bytes(), &mut compiles);
            sources += 1;
            assert_eq!(retained(&cache).0, sources, "kept from the first load");
        }
        let once = b"return {'once'}";
        load_counted(&cache, loader, &name, once, &mut compiles);
        assert_eq!(retained(&cache).0, sources);
        load_counted(&cache, loader, &name, once, &mut compiles);
        assert_eq!(retained(&cache).0, sources + 1);
        load_counted(&cache, loader, &name, once, &mut compiles);
        assert_eq!(compiles, sources + 2);
    }

    /// Strings loaded twice are kept until their estimated size spends the
    /// budget, however small each source is.
    #[test]
    fn strings_loaded_twice_are_kept_within_the_budget() {
        const BUDGET: usize = 256 * 1024;
        let cache = Mutex::new(Some(Cache::new(BUDGET, BUDGET)));
        let loader = CompilerLoader::with_profile(CompatibilityProfile::LuaJit21);
        let name = LuaString::from("=(load)");
        let mut compiles = 0;
        for i in 0..2000 {
            let source = format!("return {{{i}, 'x'}}");
            for _ in 0..2 {
                load_counted(&cache, loader, &name, source.as_bytes(), &mut compiles);
            }
        }
        let before = compiles;
        let (entries, bytes) = retained(&cache);
        assert!(bytes <= BUDGET, "{bytes} bytes kept");
        assert!(
            entries > 0 && entries <= BUDGET / 1024,
            "{entries} entries kept"
        );
        load_counted(&cache, loader, &name, b"return {0, 'x'}", &mut compiles);
        assert_eq!(compiles, before);
    }
}
