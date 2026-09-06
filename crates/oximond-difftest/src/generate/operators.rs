//! Classes with operator metamethods, driven by a hot loop.
//!
//! The section builds two classes whose handlers cover every event a trace
//! can run as an inlined call: arithmetic, bitwise operators (profiles with
//! integers), unary minus and complement, `..`, `==`, `<` and `<=`,
//! `tostring`, `#` and `pairs`. Objects of both classes sit in one array, and
//! the loop applies operators to them in the shapes a program writes: an
//! object on either side of an object, a number or a numeric string. Handlers
//! answer with numbers, strings, booleans, nothing, two values, or the object
//! itself, some after a store, so the loop meets every way a result flows on.
//! Part way through, objects change class and a handler is replaced, and some
//! sections run a handler that raises.
//!
//! Every result folds into a checksum, so the section prints a few numbers and
//! needs no output inside the loop.

use crate::program::{ITERATIONS, Node};

/// The shapes an operand pair takes.
#[derive(Clone, Copy)]
enum Shape {
    Objects,
    Same,
    ObjectNumber,
    NumberObject,
    ObjectString,
    StringObject,
}

impl super::Generator {
    /// The operand pair of an arithmetic or concatenation statement.
    fn operand_pair(&mut self, bitwise: bool) -> (String, String) {
        let shape = *self.rng.pick(&[
            Shape::Objects,
            Shape::Objects,
            Shape::Same,
            Shape::ObjectNumber,
            Shape::NumberObject,
            Shape::ObjectString,
            Shape::StringObject,
        ]);
        let number = if bitwise {
            *self.rng.pick(&["3", "7", "12"])
        } else {
            *self.rng.pick(&["3", "7", "2.5", "12", "0.5"])
        };
        let text = *self.rng.pick(&["\"10\"", "\"7\"", "\"2.5\""]);
        let text = if bitwise { "\"10\"" } else { text };
        match shape {
            Shape::Objects => ("o1".to_owned(), "o2".to_owned()),
            Shape::Same => ("o1".to_owned(), "o1".to_owned()),
            Shape::ObjectNumber => ("o1".to_owned(), number.to_owned()),
            Shape::NumberObject => (number.to_owned(), "o2".to_owned()),
            Shape::ObjectString => ("o2".to_owned(), text.to_owned()),
            Shape::StringObject => (text.to_owned(), "o1".to_owned()),
        }
    }

    /// The body of a binary handler around `core`, the value it computes.
    fn handler_body(&mut self, core: &str, allow_error: bool) -> String {
        match self
            .rng
            .weighted(&[6, 3, 2, 1, 2, 2, if allow_error { 3 } else { 0 }])
        {
            0 => format!("return {core}"),
            1 => format!(
                "local o = type(a) == \"table\" and a or b o.hits = o.hits + 1 return {core}"
            ),
            2 => format!("return {core}, \"extra\""),
            3 => "return".to_owned(),
            4 => format!("return {core} > 5"),
            5 => format!("return \"r\" .. ({core}) % 7"),
            _ => format!(
                "calls = calls + 1 if calls == LIMIT then error(\"boom\", 0) end return {core}"
            ),
        }
    }

    /// The text of the handler for one binary event; `symbol` is the
    /// operator the core computes.
    fn binary_handler(&mut self, symbol: &str, allow_error: bool) -> String {
        let core = match symbol {
            "/" | "%" | "//" => format!("num(a) {symbol} nz(num(b))"),
            "^" => "num(a) ^ (num(b) % 4)".to_owned(),
            "&" | "|" | "~" => format!("bit_operand(num(a)) {symbol} bit_operand(num(b))"),
            "<<" | ">>" => format!("bit_operand(num(a)) {symbol} (bit_operand(num(b)) % 8)"),
            _ => format!("num(a) {symbol} num(b)"),
        };
        let body = self.handler_body(&core, allow_error);
        format!("function(a, b) {body} end")
    }

    /// A handler for a unary event. `~x` calls it with the operand alone in
    /// oximond and with the operand twice in stock Lua, so only the first
    /// argument is read.
    fn unary_handler(&mut self, symbol: &str) -> String {
        let core = if symbol == "~" {
            "~bit_operand(num(a))".to_owned()
        } else {
            "-num(a)".to_owned()
        };
        let body = match self.rng.index(4) {
            0 => format!("return {core}"),
            1 => format!("a.hits = a.hits + 1 return {core}"),
            2 => format!("return {core}, 7"),
            _ => format!("return \"u\" .. ({core}) % 5"),
        };
        format!("function(a) {body} end")
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one arm per kind of statement the loop applies"
    )]
    pub(super) fn operator_section(&mut self, index: i64) -> Node {
        self.feature("operator_classes");
        let iterations = if self.rng.chance(1, 2) {
            self.rng.range(3_000, 8_000)
        } else {
            self.rng.range(10_000, 30_000)
        };
        let iterations = u32::try_from(iterations).expect("kernel trip counts fit u32");
        let integers = self.dialect.integers;
        let modern = self.dialect.len_metamethod;
        let mut events: Vec<(&str, &str)> = vec![
            ("__add", "+"),
            ("__sub", "-"),
            ("__mul", "*"),
            ("__div", "/"),
            ("__mod", "%"),
            ("__pow", "^"),
        ];
        if integers {
            events.extend([
                ("__idiv", "//"),
                ("__band", "&"),
                ("__bor", "|"),
                ("__bxor", "~"),
                ("__shl", "<<"),
                ("__shr", ">>"),
            ]);
        }
        // One handler of the first class may raise, part way through.
        let erroring = self.rng.chance(1, 4);
        let error_event = self.rng.index(events.len());
        let limit = self
            .rng
            .range(i64::from(iterations) / 4, i64::from(iterations) / 2);
        let mut lines = vec![
            "local calls = 0".to_owned(),
            "local function num(x) if type(x) == \"table\" then return x.v end return tonumber(x) or 0 end"
                .to_owned(),
            "local function fold(h, v)".to_owned(),
            "  local t = type(v)".to_owned(),
            "  if t == \"number\" then".to_owned(),
            "    if v ~= v or v == math.huge or v == -math.huge then return (h * 7 + 3) % 1000003 end"
                .to_owned(),
            "    return (h * 7 + math.floor(v) % 1009) % 1000003".to_owned(),
            "  elseif t == \"string\" then return (h * 7 + #v) % 1000003".to_owned(),
            "  elseif t == \"boolean\" then return (h * 7 + (v and 5 or 9)) % 1000003".to_owned(),
            "  elseif t == \"nil\" then return (h * 7 + 2) % 1000003".to_owned(),
            "  end".to_owned(),
            "  return (h * 7 + 11) % 1000003".to_owned(),
            "end".to_owned(),
            "local cmp = {}".to_owned(),
        ];
        // Comparison handlers are shared by both classes, which Lua 5.1 and
        // 5.2 need to call them for objects of different classes.
        let eq = match self.rng.index(3) {
            0 => "return a.v == b.v",
            1 => "return a.v - b.v",
            _ => "a.hits = a.hits + 1 return a.v == b.v",
        };
        let lt = match self.rng.index(3) {
            0 => "return a.v < b.v",
            1 => "if a.v < b.v then return a.v - b.v end",
            _ => "a.hits = a.hits + 1 return a.v < b.v",
        };
        lines.push(format!("cmp.eq = function(a, b) {eq} end"));
        lines.push(format!("cmp.lt = function(a, b) {lt} end"));
        // Lua 5.5 has no `__le` fallback on `__lt`.
        let le_handler = !matches!(self.profile, super::Profile::Lua55) && self.rng.chance(1, 2);
        let has_le = le_handler || matches!(self.profile, super::Profile::Lua55);
        if has_le {
            let le = match self.rng.index(2) {
                0 => "return a.v <= b.v",
                _ => "if a.v <= b.v then return 1 end",
            };
            lines.push(format!("cmp.le = function(a, b) {le} end"));
        }
        lines.push("local A, B = {}, {}".to_owned());
        if modern {
            lines.push("A.__name, B.__name = \"OpA\", \"OpB\"".to_owned());
        }
        for (class, first) in [("A", true), ("B", false)] {
            for (position, (event, symbol)) in events.iter().enumerate() {
                let allow_error = erroring && first && position == error_event;
                let handler = self.binary_handler(symbol, allow_error);
                lines.push(format!("{class}.{event} = {handler}"));
            }
            lines.push(format!("{class}.__unm = {}", self.unary_handler("-")));
            if integers {
                lines.push(format!("{class}.__bnot = {}", self.unary_handler("~")));
            }
            lines.push(format!("{class}.__eq, {class}.__lt = cmp.eq, cmp.lt"));
            if has_le {
                lines.push(format!("{class}.__le = cmp.le"));
            }
            let concat = match self.rng.index(3) {
                0 => "function(a, b) return tostring(num(a)) .. \"|\" .. tostring(num(b)) end",
                1 => "function(a, b) return num(a) + num(b) end",
                _ => "function(a, b) return type(a) == \"table\" and a or b end",
            };
            lines.push(format!("{class}.__concat = {concat}"));
            // The second class may answer with a number, which a trace leaves
            // to the interpreter (the first never does, so most loops run).
            let text = match self.rng.index(if first { 3 } else { 4 }) {
                0 => "function(o) return \"O\" .. o.v end",
                1 => "function(o) o.hits = o.hits + 1 return \"H\" .. o.hits end",
                2 => "function(o) return tostring(o.v) end",
                _ => "function(o) return o.v * 2 end",
            };
            lines.push(format!("{class}.__tostring = {text}"));
            if modern {
                let length = match self.rng.index(2) {
                    0 => "function(o) return o.v end",
                    _ => "function(o) return o.v + o.hits end",
                };
                lines.push(format!("{class}.__len = {length}"));
                lines.push(format!(
                    "{class}.__pairs = function(o) return piter, o, 0 end"
                ));
            }
        }
        if modern {
            // The iterator comes before the classes use it.
            let position = lines
                .iter()
                .position(|line| line == "local A, B = {}, {}")
                .expect("the classes are declared");
            lines.insert(
                position,
                "local function piter(t, k) if k < 3 then return k + 1, t.v + k end end".to_owned(),
            );
        }
        lines.push("local objs = {}".to_owned());
        // Most loops see one class, so a trace serves them all and the loop
        // runs natively; a fourth see both, which exits at every other object
        // until a side trace takes the second class.
        let class_of = if self.rng.chance(1, 4) {
            "k % 2 == 0 and A or B"
        } else {
            "A"
        };
        lines.push(format!(
            "for k = 1, 8 do objs[k] = setmetatable({{v = (k * 5) % 11 + 1, hits = 0}}, {class_of}) end"
        ));
        lines.push("local ck = 0".to_owned());
        let lines: Vec<String> = lines
            .into_iter()
            .map(|line| line.replace("LIMIT", &limit.to_string()))
            .collect();
        let mut body = vec![Node::leaf(lines.join("\n"))];

        // The statements of the loop.
        let mut statements = vec![
            Node::leaf("local o1 = objs[i % 8 + 1]".to_owned()).pinned(),
            Node::leaf("local o2 = objs[(i * 3 + 1) % 8 + 1]".to_owned()).pinned(),
        ];
        let count = self.rng.range(3, 7);
        for _ in 0..count {
            let text = match self.rng.index(10) {
                0..=3 => {
                    let (_, symbol) = *self.rng.pick(&events);
                    let bitwise = matches!(symbol, "&" | "|" | "~" | "<<" | ">>");
                    let (left, right) = self.operand_pair(bitwise);
                    format!("ck = fold(ck, {left} {symbol} {right})")
                }
                4 => {
                    if integers && self.rng.chance(1, 2) {
                        "ck = fold(ck, ~o1)".to_owned()
                    } else {
                        "ck = fold(ck, -o2)".to_owned()
                    }
                }
                5 => {
                    let operator = *self.rng.pick(&["<", "<=", ">", ">="]);
                    let pair = *self.rng.pick(&[("o1", "o2"), ("o2", "o1"), ("o1", "o1")]);
                    if self.rng.chance(1, 2) {
                        format!(
                            "if {} {operator} {} then ck = (ck * 3 + 1) % 1000003 end",
                            pair.0, pair.1
                        )
                    } else {
                        format!("ck = fold(ck, {} {operator} {})", pair.0, pair.1)
                    }
                }
                6 => {
                    let operator = *self.rng.pick(&["==", "~="]);
                    let right = *self.rng.pick(&["o2", "o2", "o1", "3"]);
                    if self.rng.chance(1, 2) {
                        format!("ck = fold(ck, o1 {operator} {right})")
                    } else {
                        format!("if o1 {operator} {right} then ck = (ck * 3 + 2) % 1000003 end")
                    }
                }
                7 => {
                    let (left, right) = self.operand_pair(false);
                    let text = *self.rng.pick(&["\"s\"", "\"tt\"", "4", "1.5"]);
                    match self.rng.index(3) {
                        0 => format!("ck = fold(ck, {left} .. {right})"),
                        1 => format!("ck = fold(ck, {text} .. {left} .. {right})"),
                        _ => format!("ck = fold(ck, {left} .. {text} .. {right} .. {text})"),
                    }
                }
                8 => {
                    if modern && self.rng.chance(1, 2) {
                        "ck = fold(ck, #o1)".to_owned()
                    } else {
                        "ck = fold(ck, tostring(o2))".to_owned()
                    }
                }
                _ => "ck = fold(ck, tostring(o1))".to_owned(),
            };
            statements.push(Node::leaf(text));
        }
        // Part way through, objects change class and a handler is replaced.
        let switch = self
            .rng
            .range(i64::from(iterations) / 2, i64::from(iterations) * 3 / 4);
        if self.rng.chance(2, 3) {
            statements.push(Node::leaf(format!(
                "if i == {switch} then setmetatable(objs[2], B) setmetatable(objs[3], A) end"
            )));
        }
        if self.rng.chance(1, 2) {
            let (event, symbol) = *self.rng.pick(&events);
            let handler = self.binary_handler(symbol, false);
            statements.push(Node::leaf(format!(
                "if i == {} then A.{event} = {handler} end",
                switch + 1
            )));
        }
        let loop_node = Node {
            iterations: Some(iterations),
            ..Node::block(format!("for i = 1, {ITERATIONS} do"), statements, "end")
        };
        let protected = erroring || self.rng.chance(1, 3);
        let run = if protected {
            body.push(Node::block("local function run()", vec![loop_node], "end"));
            "local ok, e = pcall(run)"
        } else {
            body.push(loop_node);
            "local ok, e = true, nil"
        };
        body.push(Node::leaf(run.to_owned()));
        if modern {
            // A `pairs` loop in the hot loop would make it an outer loop,
            // which no trace covers, so the handler has a loop of its own.
            let pairs_iterations = self.rng.range(2_000, 6_000);
            let pairs_iterations =
                u32::try_from(pairs_iterations).expect("loop trip counts fit u32");
            body.push(Node {
                iterations: Some(pairs_iterations),
                ..Node::block(
                    format!("for i = 1, {ITERATIONS} do"),
                    vec![
                        Node::leaf("local o1 = objs[i % 8 + 1]".to_owned()).pinned(),
                        Node::leaf(
                            "for k, v in pairs(o1) do ck = (ck * 3 + k + v) % 1000003 end"
                                .to_owned(),
                        ),
                    ],
                    "end",
                )
            });
        }
        body.push(Node::leaf(
            "local total = 0 for k = 1, 8 do total = total + objs[k].hits end".to_owned(),
        ));
        let tag = self.tag("OP");
        body.push(Node::leaf(format!("out({tag}, ck, total, ok, e)")));
        let features = self.take_features();
        Node::block(format!("do -- operator section {index}"), body, "end").with_features(features)
    }
}
