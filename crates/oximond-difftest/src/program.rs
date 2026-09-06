//! Generated programs as a tree of statements, so the reducer can delete
//! whole statements and blocks and still print well-formed Lua.

use std::collections::BTreeSet;
use std::fmt::Write as _;

/// The placeholder in a loop header that [`Node::iterations`] replaces.
pub const ITERATIONS: &str = "{ITERATIONS}";

/// One statement, or one block with its nested statements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    /// The statement, or the block's opening line(s).
    pub head: String,
    /// Statements inside the block.
    pub body: Vec<Self>,
    /// The block's closing line, `None` for a plain statement.
    pub tail: Option<String>,
    /// A loop trip count the reducer may lower. Rendered in place of
    /// [`ITERATIONS`] in `head` and `tail`.
    pub iterations: Option<u32>,
    /// The reducer never removes a pinned node (the observation helpers).
    pub pinned: bool,
    /// Feature-ledger names this node's own text exercises.
    pub features: Vec<&'static str>,
}

impl Node {
    #[must_use]
    pub fn leaf(head: impl Into<String>) -> Self {
        Self {
            head: head.into(),
            body: Vec::new(),
            tail: None,
            iterations: None,
            pinned: false,
            features: Vec::new(),
        }
    }

    #[must_use]
    pub fn block(head: impl Into<String>, body: Vec<Self>, tail: impl Into<String>) -> Self {
        Self {
            tail: Some(tail.into()),
            body,
            ..Self::leaf(head)
        }
    }

    #[must_use]
    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }

    #[must_use]
    pub fn with_features(mut self, features: Vec<&'static str>) -> Self {
        self.features = features;
        self
    }

    fn render_into(&self, indent: usize, output: &mut String) {
        let pad = "  ".repeat(indent);
        let head = match self.iterations {
            Some(count) => self.head.replace(ITERATIONS, &count.to_string()),
            None => self.head.clone(),
        };
        for line in head.lines() {
            let _ = writeln!(output, "{pad}{line}");
        }
        for child in &self.body {
            child.render_into(indent + 1, output);
        }
        if let Some(tail) = &self.tail {
            let tail = match self.iterations {
                Some(count) => tail.replace(ITERATIONS, &count.to_string()),
                None => tail.clone(),
            };
            let _ = writeln!(output, "{pad}{tail}");
        }
    }

    fn collect_features(&self, into: &mut BTreeSet<&'static str>) {
        into.extend(self.features.iter().copied());
        for child in &self.body {
            child.collect_features(into);
        }
    }

    fn count(&self) -> usize {
        1 + self.body.iter().map(Self::count).sum::<usize>()
    }
}

/// A whole generated program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    /// Comment lines that identify the generator, seed and profile.
    pub header: Vec<String>,
    pub nodes: Vec<Node>,
}

impl Program {
    /// The Lua source.
    #[must_use]
    pub fn render(&self) -> String {
        let mut output = String::new();
        for line in &self.header {
            let _ = writeln!(output, "-- {line}");
        }
        for node in &self.nodes {
            node.render_into(0, &mut output);
        }
        output
    }

    /// Every feature-ledger name the program still contains.
    #[must_use]
    pub fn features(&self) -> BTreeSet<&'static str> {
        let mut features = BTreeSet::new();
        for node in &self.nodes {
            node.collect_features(&mut features);
        }
        features
    }

    /// The number of statements and blocks.
    #[must_use]
    pub fn size(&self) -> usize {
        self.nodes.iter().map(Node::count).sum()
    }

    /// Paths to every node, parents before children. A path lists child
    /// indexes from the top level down.
    #[must_use]
    pub fn paths(&self) -> Vec<Vec<usize>> {
        fn walk(nodes: &[Node], prefix: &mut Vec<usize>, into: &mut Vec<Vec<usize>>) {
            for (index, node) in nodes.iter().enumerate() {
                prefix.push(index);
                into.push(prefix.clone());
                walk(&node.body, prefix, into);
                prefix.pop();
            }
        }
        let mut paths = Vec::new();
        walk(&self.nodes, &mut Vec::new(), &mut paths);
        paths
    }

    /// The node at `path`.
    #[must_use]
    pub fn node(&self, path: &[usize]) -> Option<&Node> {
        let (first, rest) = path.split_first()?;
        let mut node = self.nodes.get(*first)?;
        for &index in rest {
            node = node.body.get(index)?;
        }
        Some(node)
    }

    /// The list that holds the node at `path`, and its index there.
    fn parent_list(&mut self, path: &[usize]) -> Option<(&mut Vec<Node>, usize)> {
        let (position, parents) = path.split_last()?;
        let mut list = &mut self.nodes;
        for &index in parents {
            list = &mut list.get_mut(index)?.body;
        }
        (*position < list.len()).then_some((list, *position))
    }

    /// A copy without the node at `path`.
    #[must_use]
    pub fn without(&self, path: &[usize]) -> Option<Self> {
        let mut copy = self.clone();
        let (list, index) = copy.parent_list(path)?;
        if list[index].pinned {
            return None;
        }
        list.remove(index);
        Some(copy)
    }

    /// A copy with the block at `path` replaced by its body.
    #[must_use]
    pub fn unwrapped(&self, path: &[usize]) -> Option<Self> {
        let mut copy = self.clone();
        let (list, index) = copy.parent_list(path)?;
        let node = &list[index];
        if node.pinned || node.tail.is_none() || node.body.is_empty() {
            return None;
        }
        let body = list.remove(index).body;
        for (offset, child) in body.into_iter().enumerate() {
            list.insert(index + offset, child);
        }
        Some(copy)
    }

    /// A copy with the loop at `path` running `iterations` times.
    #[must_use]
    pub fn with_iterations(&self, path: &[usize], iterations: u32) -> Option<Self> {
        let mut copy = self.clone();
        let (list, index) = copy.parent_list(path)?;
        let node = &mut list[index];
        if node.iterations? == iterations {
            return None;
        }
        node.iterations = Some(iterations);
        Some(copy)
    }
}

#[cfg(test)]
mod tests {
    use super::{ITERATIONS, Node, Program};

    fn sample() -> Program {
        Program {
            header: vec!["sample".into()],
            nodes: vec![
                Node::leaf("local x = 1").pinned(),
                Node {
                    iterations: Some(5),
                    ..Node::block(
                        format!("for i = 1, {ITERATIONS} do"),
                        vec![Node::leaf("x = x + i"), Node::leaf("print(x)")],
                        "end",
                    )
                },
            ],
        }
    }

    #[test]
    fn renders_nested_blocks_with_iterations() {
        assert_eq!(
            sample().render(),
            "-- sample\nlocal x = 1\nfor i = 1, 5 do\n  x = x + i\n  print(x)\nend\n"
        );
    }

    #[test]
    fn edits_keep_pinned_nodes_and_report_paths() {
        let program = sample();
        assert_eq!(
            program.paths(),
            vec![vec![0], vec![1], vec![1, 0], vec![1, 1]]
        );
        assert!(program.without(&[0]).is_none());
        let fewer = program.without(&[1, 0]).unwrap();
        assert_eq!(fewer.size(), 3);
        let flat = program.unwrapped(&[1]).unwrap();
        assert_eq!(
            flat.render(),
            "-- sample\nlocal x = 1\nx = x + i\nprint(x)\n"
        );
        let short = program.with_iterations(&[1], 2).unwrap();
        assert!(short.render().contains("for i = 1, 2 do"));
    }
}
