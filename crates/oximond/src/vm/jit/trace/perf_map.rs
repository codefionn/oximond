//! A perf map for the compiled traces.
//!
//! With `OXIMOND_PERF_MAP` set, every code range a trace installs is named in
//! `/tmp/perf-<pid>.map`, the file `perf report` reads to name samples that
//! fall in JIT memory. The name is `oximond trace#<index> <source>:<line>`
//! with the header's source line; a side compiled apart from its tree adds
//! `side<n>`. A recompile of a tree gives its new code the name again, and
//! `perf` keeps the last name written for an address. Off by default: the
//! check is one `OnceLock` read per install, and installs are rare.
//!
//! Cranelift writes the same file with a generic name (`oximond_trace`) for
//! each function when `PERF_BUILDID_DIR` is set, and `perf` keeps the first
//! name it reads for a range: unset `PERF_BUILDID_DIR` when using this map.
//!
//! Like the other debug switches, the variable is read once per process.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

struct Map {
    path: PathBuf,
    /// `(address, size, name)` already written, so a range is not repeated.
    written: HashSet<(usize, usize, String)>,
}

fn map() -> Option<&'static Mutex<Map>> {
    static MAP: OnceLock<Option<Mutex<Map>>> = OnceLock::new();
    MAP.get_or_init(|| {
        std::env::var_os("OXIMOND_PERF_MAP")?;
        Some(Mutex::new(Map {
            path: PathBuf::from(format!("/tmp/perf-{}.map", std::process::id())),
            written: HashSet::new(),
        }))
    })
    .as_ref()
}

/// Whether `OXIMOND_PERF_MAP` is set.
pub(crate) fn enabled() -> bool {
    map().is_some()
}

fn overlaps(a: usize, asize: usize, b: usize, bsize: usize) -> bool {
    a < b.saturating_add(bsize) && b < a.saturating_add(asize)
}

/// `old` without every line whose range overlaps `address..address + size`
/// (whatever its name: freed trace code hands its pages to later traces, and
/// `perf` picks among overlapping symbols by its own rules, not by recency),
/// plus the new line.
fn merge(old: &str, address: usize, size: usize, name: &str) -> String {
    let mut text = String::new();
    for line in old.lines() {
        let mut fields = line.split_whitespace();
        let range = (
            fields
                .next()
                .and_then(|f| usize::from_str_radix(f, 16).ok()),
            fields
                .next()
                .and_then(|f| usize::from_str_radix(f, 16).ok()),
        );
        if let (Some(a), Some(s)) = range
            && overlaps(a, s, address, size)
        {
            continue;
        }
        text.push_str(line);
        text.push('\n');
    }
    let _ = writeln!(text, "{address:x} {size:x} {name}");
    text
}

/// Names the code at `address..address + size`. The file is rewritten without
/// any line that overlaps the range, the generic line Cranelift wrote or the
/// name of a freed trace whose pages this code reuses (see the module docs). A line another thread appends meanwhile can be lost; its
/// trace is named when it installs.
pub(crate) fn note(name: &str, address: usize, size: usize) {
    let Some(map) = map() else { return };
    if size == 0 || address == 0 {
        return;
    }
    let mut map = map.lock().expect("perf map");
    if !map.written.insert((address, size, name.to_owned())) {
        return;
    }
    map.written
        .retain(|(a, s, _)| !overlaps(*a, *s, address, size));
    let old = std::fs::read_to_string(&map.path).unwrap_or_default();
    let text = merge(&old, address, size, name);
    let temporary = map.path.with_extension("map.tmp");
    if std::fs::write(&temporary, text).is_ok() {
        let _ = std::fs::rename(&temporary, &map.path);
    }
}

#[cfg(test)]
mod tests {
    use super::merge;

    #[test]
    fn reused_ranges_replace_stale_names() {
        let mut text = String::new();
        text = merge(&text, 0x1000, 0x1f0, "oximond trace#1 =a:4");
        text = merge(&text, 0x2000, 0x100, "oximond trace#2 =b:4");
        // freed and reused, with a different size and name
        text = merge(&text, 0x1000, 0x380, "oximond trace#3 =c:4");
        // a smaller one inside the old range
        text = merge(&text, 0x1100, 0x40, "oximond trace#4 =d:4");
        let mut ranges: Vec<(usize, usize)> = text
            .lines()
            .map(|l| {
                let mut f = l.split_whitespace();
                let a = usize::from_str_radix(f.next().unwrap(), 16).unwrap();
                (a, usize::from_str_radix(f.next().unwrap(), 16).unwrap())
            })
            .collect();
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0, "overlap in {text}");
        }
        assert!(!text.contains("=a:4") && !text.contains("=c:4"));
        assert!(text.contains("=b:4") && text.contains("=d:4"));
    }
}
