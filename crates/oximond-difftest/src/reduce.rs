//! Deterministic reduction of a failing program.
//!
//! The reducer deletes statements and blocks, replaces blocks with their
//! bodies and lowers loop counts, keeping each edit only if the caller's
//! check still reports the same failure. The check runs the oracle too, so
//! an edit that leaves invalid Lua (a dropped declaration, a loop counter
//! out of scope) is rejected rather than kept.

use std::time::Instant;

use crate::program::{Node, Program};

/// Names a statement declares in its own text: `local x, y = ...`,
/// `local function f`, `local x <const> = ...` and generated globals
/// (`G12 = ...`).
#[must_use]
pub fn declared_names(node: &Node) -> Vec<String> {
    let mut names = Vec::new();
    // Only unindented lines: the indented ones are inside a function the
    // statement defines, and their locals are not visible outside it.
    for line in node
        .head
        .lines()
        .filter(|line| !line.starts_with([' ', '\t']))
    {
        if let Some(rest) = line.strip_prefix("local function ") {
            names.extend(rest.split('(').next().map(|name| name.trim().to_owned()));
        } else if let Some(rest) = line.strip_prefix("local ") {
            let list = rest.split('=').next().unwrap_or_default();
            for name in list.split(',') {
                let name = name.split('<').next().unwrap_or_default().trim();
                if is_identifier(name) {
                    names.push(name.to_owned());
                }
            }
        } else if let Some((name, _)) = line.split_once(" = ")
            && name.starts_with('G')
            && is_identifier(name)
        {
            names.push(name.to_owned());
        }
    }
    names
}

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Whether `text` uses `name` as a whole identifier.
#[must_use]
pub fn mentions(text: &str, name: &str) -> bool {
    let bytes = text.as_bytes();
    let is_word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    text.match_indices(name).any(|(start, _)| {
        let end = start + name.len();
        (start == 0 || !is_word(bytes[start - 1])) && (end == bytes.len() || !is_word(bytes[end]))
    })
}

/// Deleting the node at `path` would leave a use of a name it declares.
/// Such a deletion turns the use into a nil global, which stock Lua
/// usually accepts, so the reduced program would test something else.
fn orphans_a_use(program: &Program, path: &[usize], candidate: &Program) -> bool {
    let Some(node) = program.node(path) else {
        return false;
    };
    let names = declared_names(node);
    if names.is_empty() {
        return false;
    }
    let text = candidate.render();
    names.iter().any(|name| mentions(&text, name))
}

/// The outcome of a reduction.
#[derive(Clone, Debug)]
pub struct Reduction {
    pub program: Program,
    pub attempts: usize,
    pub accepted: usize,
    /// One line per accepted edit.
    pub log: Vec<String>,
    /// The reducer stopped at the attempt or time limit.
    pub exhausted: bool,
}

/// Reduces `program` while `still_fails` holds, within `max_attempts`
/// checks and before `deadline`.
#[allow(clippy::too_many_lines, reason = "three passes over one candidate")]
pub fn reduce(
    program: &Program,
    mut still_fails: impl FnMut(&Program) -> bool,
    max_attempts: usize,
    deadline: Instant,
) -> Reduction {
    let mut current = program.clone();
    let mut attempts = 0;
    let mut log = Vec::new();
    let mut exhausted = false;
    let mut try_candidate =
        |candidate: Program, description: String, current: &mut Program| -> Option<bool> {
            if attempts >= max_attempts || Instant::now() >= deadline {
                return None;
            }
            attempts += 1;
            if still_fails(&candidate) {
                log.push(format!("{description} -> {} nodes", candidate.size()));
                *current = candidate;
                Some(true)
            } else {
                Some(false)
            }
        };

    'passes: loop {
        let mut progressed = false;
        // Deletion, parents first so whole blocks go in one step.
        let mut index = 0;
        loop {
            let paths = current.paths();
            let Some(path) = paths.get(index).cloned() else {
                break;
            };
            let mut accepted = false;
            if let Some(candidate) = current.without(&path)
                && !orphans_a_use(&current, &path, &candidate)
            {
                match try_candidate(candidate, format!("delete {path:?}"), &mut current) {
                    None => {
                        exhausted = true;
                        break 'passes;
                    }
                    Some(result) => accepted = result,
                }
            }
            if accepted {
                progressed = true;
            } else {
                index += 1;
            }
        }
        // Replace blocks by their bodies.
        let mut index = 0;
        loop {
            let paths = current.paths();
            let Some(path) = paths.get(index).cloned() else {
                break;
            };
            let mut accepted = false;
            if let Some(candidate) = current.unwrapped(&path) {
                match try_candidate(candidate, format!("unwrap {path:?}"), &mut current) {
                    None => {
                        exhausted = true;
                        break 'passes;
                    }
                    Some(result) => accepted = result,
                }
            }
            if accepted {
                progressed = true;
            } else {
                index += 1;
            }
        }
        // Lower loop counts by halving.
        for path in current.paths() {
            while let Some(count) = current.node(&path).and_then(|node| node.iterations) {
                if count <= 1 {
                    break;
                }
                let Some(candidate) = current.with_iterations(&path, count / 2) else {
                    break;
                };
                match try_candidate(
                    candidate,
                    format!("iterations {path:?} {count} -> {}", count / 2),
                    &mut current,
                ) {
                    None => {
                        exhausted = true;
                        break 'passes;
                    }
                    Some(true) => progressed = true,
                    Some(false) => break,
                }
            }
        }
        if !progressed {
            break;
        }
    }
    let accepted = log.len();
    Reduction {
        program: current,
        attempts,
        accepted,
        log,
        exhausted,
    }
}

#[cfg(test)]
mod tests {
    use super::reduce;
    use crate::program::{ITERATIONS, Node, Program};
    use std::time::{Duration, Instant};

    #[test]
    fn keeps_only_what_the_failure_needs() {
        let program = Program {
            header: Vec::new(),
            nodes: vec![
                Node::leaf("helper()").pinned(),
                Node::leaf("a()"),
                Node {
                    iterations: Some(64),
                    ..Node::block(
                        format!("for i = 1, {ITERATIONS} do"),
                        vec![Node::leaf("b()"), Node::leaf("BUG()"), Node::leaf("c()")],
                        "end",
                    )
                },
                Node::leaf("d()"),
            ],
        };
        // The "failure" needs BUG() inside a loop of at least 8 trips.
        let fails = |candidate: &Program| {
            let text = candidate.render();
            text.contains("BUG()")
                && text.lines().any(|line| {
                    line.strip_prefix("for i = 1, ")
                        .and_then(|rest| rest.strip_suffix(" do"))
                        .and_then(|count| count.parse::<u32>().ok())
                        .is_some_and(|count| count >= 8)
                })
        };
        let reduction = reduce(
            &program,
            fails,
            1000,
            Instant::now() + Duration::from_secs(10),
        );
        assert_eq!(
            reduction.program.render(),
            "helper()\nfor i = 1, 8 do\n  BUG()\nend\n"
        );
        assert!(!reduction.exhausted);
        assert_eq!(reduction.accepted, reduction.log.len());
    }

    #[test]
    fn declarations_stay_while_something_uses_them() {
        use super::{declared_names, mentions};
        assert_eq!(
            declared_names(&Node::leaf("local a, b <const> = 1, 2")),
            ["a", "b"]
        );
        assert_eq!(
            declared_names(&Node::leaf("local function f12(p)")),
            ["f12"]
        );
        assert_eq!(declared_names(&Node::leaf("G7 = 3")), ["G7"]);
        assert_eq!(
            declared_names(&Node::leaf("g1 = 3")),
            [] as [std::string::String; 0]
        );
        assert!(mentions("x = g1 + 1", "g1"));
        assert!(!mentions("x = g10 + 1", "g1"));
        let program = Program {
            header: Vec::new(),
            nodes: vec![
                Node::leaf("local g1 = 5"),
                Node::leaf("print(g1)"),
                Node::leaf("BUG()"),
            ],
        };
        // Any program with BUG() "fails"; the declaration may only go
        // together with its use.
        let reduction = reduce(
            &program,
            |candidate| candidate.render().contains("BUG()"),
            100,
            Instant::now() + Duration::from_secs(10),
        );
        assert_eq!(reduction.program.render(), "BUG()\n");
    }

    #[test]
    fn stops_at_the_attempt_limit() {
        let program = Program {
            header: Vec::new(),
            nodes: (0..10)
                .map(|index| Node::leaf(format!("s{index}()")))
                .collect(),
        };
        let reduction = reduce(
            &program,
            |_| false,
            3,
            Instant::now() + Duration::from_secs(10),
        );
        assert_eq!(reduction.attempts, 3);
        assert!(reduction.exhausted);
        assert_eq!(reduction.program, program);
    }
}
