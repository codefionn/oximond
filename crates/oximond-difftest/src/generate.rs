//! The seeded program generator.
//!
//! Programs are built from typed, scope-aware choices rather than token
//! soup. Every variable has a fixed [`Kind`] that each assignment keeps, so
//! an operation only meets operands it accepts. Errors happen only where the
//! generator plants them, inside `pcall` or `xpcall`.
//!
//! A program prints observations through `out(tag, ...)`, which shows each
//! value's type, integer or float subtype and the result count. Tables print
//! through `arr` and `rec`, which walk array parts in order and sort record
//! keys, so traversal order never reaches the output. Nothing prints an
//! address.
//!
//! Lua leaves the evaluation order of operands, arguments and assignment
//! targets unspecified, so expressions only call functions without side
//! effects ("pure" functions). A call that writes state is a statement of
//! its own, assigned only to a plain variable.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::profile::{Dialect, Profile};
use crate::program::{ITERATIONS, Node, Program};
use crate::rng::Rng;

mod debug;
mod hot;
mod language;
mod operators;
mod runtime;
mod semantics;
mod stdlib;

#[cfg(test)]
mod probe_tests;

pub use hot::HotKnobs;

/// Bumped whenever the same seed would produce a different program.
/// Saved failures keep their source, so they replay across versions.
pub const GENERATOR_VERSION: u32 = 28;

// Kind::Int also includes integral floats from string arithmetic. Those can
// grow outside i64 in kernels, while actual integers must keep wrapping.
const BIT_OPERAND_HELPER: &str = "local function bit_operand(v)\n  local i = math.tointeger(v)\n  if i then return i end\n  if v ~= v or v == math.huge or v == -math.huge then return 0 end\n  return math.floor(v % 1048576)\nend";

// Equal numeric values can have different observable tags. Stock table.sort
// is unstable, so generated observations need a consistent subtype order.
const NUMERIC_LESS_HELPER: &str = "local function numeric_less(a, b)\n  return a < b or (a == b and math.type(a) < math.type(b))\nend";

// Arithmetic can turn an Int-kind expression into a nonfinite float.
// Array slots still need finite integral keys within the existing sequence.
const ARRAY_INDEX_HELPER: &str = "local function array_index(v, len)\n  if len < 1 or v ~= v or v == math.huge or v == -math.huge then return 1 end\n  local remainder = v % len\n  if remainder ~= remainder or remainder < 0 or remainder >= len then return 1 end\n  return math.floor(remainder) + 1\nend";

/// How large a program to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    Small,
    Medium,
    Large,
}

impl Size {
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "small" => Some(Self::Small),
            "medium" => Some(Self::Medium),
            "large" => Some(Self::Large),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }

    const fn limits(self) -> Limits {
        match self {
            Self::Small => Limits {
                sections: (2, 3),
                functions: (1, 3),
                statements: (2, 4),
                depth: 2,
            },
            Self::Medium => Limits {
                sections: (3, 5),
                functions: (2, 5),
                statements: (2, 6),
                depth: 3,
            },
            Self::Large => Limits {
                sections: (5, 8),
                functions: (3, 7),
                statements: (3, 8),
                depth: 3,
            },
        }
    }
}

#[derive(Clone, Copy)]
struct Limits {
    sections: (i64, i64),
    functions: (i64, i64),
    statements: (i64, i64),
    depth: u32,
}

/// Work one call or one loop iteration may do, in rough statement units.
/// Calls are only generated while the running estimate stays below these.
const FUNCTION_COST_LIMIT: u64 = 600;
const ITERATION_COST_LIMIT: u64 = 4000;

/// Array mutations may grow a sequence while an `ipairs` loop is active.
const ARRAY_LENGTH_LIMIT: u64 = 12;

/// Integer values in profiles without an integer subtype are reduced by
/// this modulus whenever they are stored, so they stay exact doubles and
/// valid `bit` library operands.
const DOUBLE_INT_MODULUS: &str = "1000003";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// An integer value: the integer subtype from Lua 5.3, an integral
    /// double before it.
    Int,
    /// Any number.
    Num,
    Str,
    Bool,
    /// A table whose array part `1..#t` holds `Int` values, never empty.
    Arr,
    /// A table with string keys holding `Num` values.
    Rec,
    /// An instance of one of the two generated classes.
    Obj,
}

const SCALARS: [Kind; 4] = [Kind::Int, Kind::Num, Kind::Str, Kind::Bool];
const ALL_KINDS: [Kind; 7] = [
    Kind::Int,
    Kind::Num,
    Kind::Str,
    Kind::Bool,
    Kind::Arr,
    Kind::Rec,
    Kind::Obj,
];

#[derive(Clone, Debug)]
struct Var {
    name: String,
    kind: Kind,
    assignable: bool,
    /// The function nesting level that declared it. A write from a deeper
    /// level is an upvalue write.
    level: usize,
    /// Declared inside a kernel section.
    kernel: bool,
}

#[derive(Clone, Debug)]
struct Function {
    name: String,
    params: Vec<Param>,
    results: Vec<Kind>,
    /// No writes outside its own locals, no impure calls.
    pure: bool,
    /// Estimated work per call.
    cost: u64,
}

#[derive(Clone, Copy, Debug)]
struct Param {
    kind: Kind,
    /// Arguments are reduced into `0..bound` (recursion depth limits).
    bound: Option<u32>,
}

/// What the statement being generated may do.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools, reason = "independent context flags")]
struct Frame {
    level: usize,
    pure: bool,
    /// Loop nesting inside the current function, for `break`.
    loops: u32,
    /// The innermost loop's `goto` continue label, if it has one.
    continue_label: Option<String>,
    /// Iterations of the enclosing loops in this function or section.
    multiplier: u64,
    /// The main loop counter, for phase changes.
    counter: Option<(String, u32)>,
    /// `coroutine.yield` is allowed here.
    in_coroutine: bool,
    depth: u32,
    /// Inside a numeric kernel loop: expressions stay on numbers, tables
    /// and objects, which the trace recorder can follow.
    kernel: bool,
    /// Inside a kernel that uses only its own locals, no upvalues or
    /// globals.
    kernel_locals: bool,
    /// Inside a generated function: no printing statements.
    quiet: bool,
}

/// The seed stream for one case. Profiles get distinct streams for the
/// same seed so a campaign over several profiles covers different
/// programs.
fn stream(seed: u64, profile: Profile) -> Rng {
    let index = Profile::ALL
        .iter()
        .position(|&candidate| candidate == profile)
        .expect("every profile is listed") as u64;
    let mut mixer = Rng::new(seed ^ (index << 56) ^ (u64::from(GENERATOR_VERSION) << 48));
    Rng::new(mixer.next_u64())
}

/// Builds the program for `seed` under `profile`.
#[must_use]
pub fn generate(seed: u64, profile: Profile, size: Size) -> Program {
    generate_excluding(seed, profile, size, &BTreeSet::new())
}

/// Builds the program for `seed` under `profile`, never choosing the
/// feature families named in `exclude` (see [`EXCLUDABLE`]). The same seed
/// with a different exclusion set is a different program.
#[must_use]
pub fn generate_excluding(
    seed: u64,
    profile: Profile,
    size: Size,
    exclude: &BTreeSet<String>,
) -> Program {
    let mut generator = Generator::new(stream(seed, profile), profile, size);
    generator.exclude.clone_from(exclude);
    let nodes = generator.program();
    let mut header = format!(
        "oximond-difftest generator={GENERATOR_VERSION} profile={profile} seed={seed} size={}",
        size.name()
    );
    if !exclude.is_empty() {
        let _ = write!(
            header,
            " exclude={}",
            exclude.iter().cloned().collect::<Vec<_>>().join(",")
        );
    }
    Program {
        header: vec![header],
        nodes,
    }
}

/// Builds a hot-loop program for `seed` with knobs drawn from the seed and
/// `size` (see [`HotKnobs::for_seed`]).
#[must_use]
pub fn generate_hot_seed(seed: u64, profile: Profile, size: Size) -> Program {
    generate_hot(seed, profile, &HotKnobs::for_seed(seed, size))
}

/// Builds a hot-loop program: the seed picks the details, `knobs` the shape.
#[must_use]
pub fn generate_hot(seed: u64, profile: Profile, knobs: &HotKnobs) -> Program {
    let mut generator = Generator::new(stream(seed, profile), profile, Size::Small);
    generator.limits = Limits {
        sections: (i64::from(knobs.sections), i64::from(knobs.sections)),
        functions: (1, 3),
        statements: (
            i64::from(knobs.statements.max(1)),
            i64::from(knobs.statements.max(1)),
        ),
        depth: knobs.depth,
    };
    generator.hot = Some(*knobs);
    let nodes = generator.program();
    Program {
        header: vec![format!(
            "oximond-difftest generator={GENERATOR_VERSION} profile={profile} seed={seed} hot={}",
            knobs.describe()
        )],
        nodes,
    }
}

/// Feature families a campaign can switch off, typically ones with a known
/// open oximond bug, so the campaign looks past them.
pub const EXCLUDABLE: &[&str] = &[
    "call_result_to_variable",
    "float_modulo",
    "format_c_zero",
    "tonumber_base",
    "math_random",
    "rep_separator",
    "format_q",
    "coroutine_isyieldable",
    "yield_across_pcall",
    "error_number",
    "error_level2",
    "error_no_integer",
    "error_bad_argument",
    "error_string_arith",
    "coroutine_error",
    "xpcall_lua_handler",
    "protected_kernel",
    "yield_boundaries",
    "stdlib_edges",
    "runtime_functions",
    "runtime_coroutines",
    "debug_probes",
    "runtime_types",
    "runtime_metatables",
    "runtime_control",
    "profile_libraries",
    "utf8",
    "patterns",
    "string_arithmetic",
    "text_calls",
    "table_traversal",
    "operator_classes",
];

// The frozen-seed toggles are independent feature switches of the generator.
#[allow(clippy::struct_excessive_bools)]
struct Generator {
    rng: Rng,
    dialect: Dialect,
    limits: Limits,
    scopes: Vec<Vec<Var>>,
    functions: Vec<Function>,
    /// Feature names used by the statement being built.
    features: Vec<&'static str>,
    next_name: u32,
    next_tag: u32,
    frame: Frame,
    /// Work estimated for the current function body or loop iteration.
    cost: u64,
    /// The fixed library functions (`ops`, `envf`, ...) are declared.
    library_ready: bool,
    exclude: BTreeSet<String>,
    // Frozen seed regressions can retain the probes from generator 14.
    expanded_probes: bool,
    // Frozen seed regressions can leave out the scalar library edges added
    // in generator 21: `math.random` draws, logarithms with bases, edge
    // indices and `tonumber` inputs.
    scalar_edges: bool,
    // Frozen seed regressions can keep the logarithm bases that generator 21
    // drew; generator 22 draws every base on every profile.
    exact_logs: bool,
    // Frozen seed regressions can leave out the statements generator 23 added:
    // `string.find`, `string.match`, `string.gmatch` and `string.gsub` with a
    // fixed set of patterns.
    pattern_calls: bool,
    // Frozen seed regressions can leave out the numeric strings that
    // generator 24 draws as arithmetic operands, from a table of them.
    string_arith: bool,
    // Frozen seed regressions can leave out the `utf8` and `string.pack`
    // statements generator 26 draws in loops, on the profiles that have them.
    text_calls: bool,
    // Frozen seed regressions can leave out the `pairs` and `next` traversals
    // generator 27 draws, with their clearing and updating writes.
    traversals: bool,
    // Frozen seed regressions can leave out the operator sections generator 28
    // draws: classes whose operator metamethods a hot loop applies.
    operator_classes: bool,
    // Opt-in hot-loop bias, off for every default entry point.
    hot: Option<HotKnobs>,
    hot_helpers: Vec<hot::Helper>,
    profile: Profile,
}

impl Generator {
    fn new(rng: Rng, profile: Profile, size: Size) -> Self {
        Self {
            rng,
            dialect: profile.features(),
            limits: size.limits(),
            scopes: vec![Vec::new()],
            functions: Vec::new(),
            features: Vec::new(),
            next_name: 0,
            next_tag: 0,
            frame: Frame {
                level: 0,
                pure: false,
                loops: 0,
                continue_label: None,
                multiplier: 1,
                counter: None,
                in_coroutine: false,
                depth: 0,
                kernel: false,
                kernel_locals: false,
                quiet: false,
            },
            cost: 0,
            library_ready: false,
            exclude: BTreeSet::new(),
            expanded_probes: true,
            scalar_edges: true,
            exact_logs: true,
            pattern_calls: true,
            string_arith: true,
            text_calls: true,
            traversals: true,
            operator_classes: true,
            hot: None,
            hot_helpers: Vec::new(),
            profile,
        }
    }

    fn allowed(&self, family: &str) -> bool {
        !self.exclude.contains(family)
    }

    /// `math.random` draws are emitted where the output can still be
    /// compared: profiles without a stock oracle, and the 5.4 and 5.5
    /// generators, which give the stock sequence. Oximond's Lua 5.1 and
    /// `LuaJIT` generators are not stock libc or stock `LuaJIT`, so their draws
    /// would bury real mismatches under known ones.
    fn random_draws(&self) -> bool {
        self.scalar_edges
            && self.allowed("math_random")
            && !matches!(self.profile, Profile::Lua51 | Profile::LuaJit21)
    }

    // ----- bookkeeping -------------------------------------------------

    fn fresh(&mut self, prefix: &str) -> String {
        self.next_name += 1;
        format!("{prefix}{}", self.next_name)
    }

    fn tag(&mut self, prefix: &str) -> String {
        self.next_tag += 1;
        format!("\"{prefix}{}\"", self.next_tag)
    }

    fn feature(&mut self, name: &'static str) {
        if !self.features.contains(&name) {
            self.features.push(name);
        }
    }

    fn take_features(&mut self) -> Vec<&'static str> {
        std::mem::take(&mut self.features)
    }

    /// A finished statement carrying the features used to build it.
    fn leaf(&mut self, text: impl Into<String>) -> Node {
        let features = self.take_features();
        Node::leaf(text).with_features(features)
    }

    fn declare(&mut self, name: &str, kind: Kind, assignable: bool) {
        let level = self.frame.level;
        let kernel = self.frame.kernel;
        self.scopes
            .last_mut()
            .expect("there is always a scope")
            .push(Var {
                name: name.to_owned(),
                kind,
                assignable,
                level,
                kernel,
            });
    }

    fn push_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn visible(&self, kind: Kind) -> Vec<Var> {
        // Inner declarations shadow outer ones with the same name; names
        // are unique, so every entry is visible.
        self.scopes
            .iter()
            .flatten()
            .filter(|var| var.kind == kind && (var.kernel || !self.frame.kernel_locals))
            .cloned()
            .collect()
    }

    /// Variables the current statement may assign.
    fn writable(&self, kind: Kind) -> Vec<Var> {
        self.visible(kind)
            .into_iter()
            .filter(|var| var.assignable && (!self.frame.pure || var.level == self.frame.level))
            .collect()
    }

    /// Variables a call's result may be assigned to. Excluding
    /// `call_result_to_variable` leaves only fresh locals, which steers
    /// clear of oximond's lost captured-local assignment.
    fn call_targets(&self, kind: Kind) -> Vec<Var> {
        if self.allowed("call_result_to_variable") {
            self.writable(kind)
        } else {
            Vec::new()
        }
    }

    /// Tables the current statement may mutate.
    fn mutable_tables(&self, kind: Kind) -> Vec<Var> {
        self.visible(kind)
            .into_iter()
            .filter(|var| !self.frame.pure || var.level == self.frame.level)
            .collect()
    }

    fn pick_var(&mut self, kind: Kind) -> Option<String> {
        let vars = self.visible(kind);
        if vars.is_empty() {
            None
        } else {
            Some(self.rng.pick(&vars).name.clone())
        }
    }

    fn charge(&mut self, units: u64) -> bool {
        let added = units.saturating_mul(self.frame.multiplier);
        let limit = if self.frame.level == 0 {
            ITERATION_COST_LIMIT
        } else {
            FUNCTION_COST_LIMIT
        };
        if self.cost.saturating_add(added) > limit {
            return false;
        }
        self.cost += added;
        true
    }

    // ----- the program -------------------------------------------------

    fn program(&mut self) -> Vec<Node> {
        let mut nodes = self.prelude();
        nodes.extend(self.classes());
        nodes.extend(self.state());
        nodes.extend(self.library_functions());
        self.library_ready = true;
        let (low, high) = self.limits.functions;
        for _ in 0..self.rng.range(low, high) {
            nodes.push(self.function());
        }
        if let Some(knobs) = self.hot {
            if knobs.anchor {
                nodes.push(Node::leaf(hot::ANCHOR_KERNEL.to_owned()));
            }
            nodes.extend(self.hot_sections(knobs));
            nodes.push(self.final_observation());
            return nodes;
        }
        // These run once outside hot loops. Keep each probe together so
        // reduction cannot split hook cleanup or a coroutine's resume steps.
        if self.allowed("runtime_functions") {
            nodes.push(self.function_probe());
        }
        if self.allowed("runtime_coroutines") {
            nodes.push(self.coroutine_probe());
        }
        if self.allowed("debug_probes") {
            nodes.push(self.debug_probe());
        }
        if self.allowed("runtime_types") {
            nodes.push(self.type_probe());
        }
        if self.allowed("runtime_metatables") {
            nodes.push(self.metatable_probe());
        }
        if self.allowed("runtime_control") {
            nodes.push(self.control_probe());
            if self.expanded_probes {
                nodes.push(self.language_probe());
            }
        }
        if self.allowed("profile_libraries") {
            nodes.push(self.profile_library_probe());
        }
        if self.dialect.utf8 && self.allowed("utf8") {
            nodes.push(self.utf8_probe());
        }
        let (low, high) = self.limits.sections;
        for index in 0..self.rng.range(low, high) {
            let section = if self.operator_classes
                && self.allowed("operator_classes")
                && self.rng.chance(1, 4)
            {
                self.operator_section(index)
            } else if self.expanded_probes
                && self.allowed("protected_kernel")
                && self.rng.chance(1, 5)
            {
                self.protected_kernel_section(index)
            } else if self.rng.chance(1, 2) {
                self.kernel_section(index)
            } else {
                self.section(index)
            };
            nodes.push(section);
        }
        nodes.push(self.final_observation());
        nodes
    }

    /// Observation helpers. The reducer keeps them while anything uses them.
    fn prelude(&mut self) -> Vec<Node> {
        // Floats print with 17 significant digits, enough to tell any two
        // doubles apart; `tostring`'s own format is tested where programs
        // call it. Every NaN prints as `nan`: its sign bit depends on how
        // the engine produced it (x86 makes 0/0 negative, a Rust constant
        // fold positive), and Lua does not define it.
        // Stock Lua 5.1 and 5.2 fold constant expressions and key their
        // constant table by value, so a folded `-0` and a `0` share one
        // constant and whichever came first wins. Those profiles print zero
        // without a sign.
        let number = if !self.dialect.integers && !self.dialect.bit_library {
            "if t == \"number\" then if v ~= v then return \"nnan\" end if v == 0 then return \"n0\" end return \"n\" .. string.format(\"%.17g\", v) end"
        } else if self.dialect.integers {
            "if t == \"number\" then if math.type(v) == \"integer\" then return \"i\" .. tostring(v) end if v ~= v then return \"fnan\" end return \"f\" .. string.format(\"%.17g\", v) end"
        } else {
            "if t == \"number\" then if v ~= v then return \"nnan\" end return \"n\" .. string.format(\"%.17g\", v) end"
        };
        let lines = [
            "local function show(v)".to_owned(),
            "  local t = type(v)".to_owned(),
            // Not `%q`, which is a feature under test: quote and escape
            // every byte outside printable ASCII as `\ddd`.
            "  if t == \"string\" then".to_owned(),
            "    return '\"' .. string.gsub(v, '[%c\"\\\\\\128-\\255]', function(c) return string.format('\\\\%03d', string.byte(c)) end) .. '\"'"
                .to_owned(),
            "  end".to_owned(),
            format!("  {number}"),
            "  if t == \"nil\" or t == \"boolean\" then return tostring(v) end".to_owned(),
            "  return t".to_owned(),
            "end".to_owned(),
            "local function out(tag, ...)".to_owned(),
            "  local n = select(\"#\", ...)".to_owned(),
            "  local parts = {}".to_owned(),
            "  for k = 1, n do parts[k] = show((select(k, ...))) end".to_owned(),
            "  print(tag .. \"#\" .. n .. \" \" .. table.concat(parts, \" \"))".to_owned(),
            "end".to_owned(),
            "local function arr(t)".to_owned(),
            "  local parts = {}".to_owned(),
            "  for k = 1, #t do parts[k] = show(t[k]) end".to_owned(),
            "  return \"[\" .. table.concat(parts, \",\") .. \"]\"".to_owned(),
            "end".to_owned(),
            "local function rec(r)".to_owned(),
            "  local keys = {}".to_owned(),
            "  for k in pairs(r) do keys[#keys + 1] = k end".to_owned(),
            "  table.sort(keys)".to_owned(),
            "  local parts = {}".to_owned(),
            "  for k = 1, #keys do parts[k] = keys[k] .. \"=\" .. show(r[keys[k]]) end".to_owned(),
            "  return \"{\" .. table.concat(parts, \",\") .. \"}\"".to_owned(),
            "end".to_owned(),
            "local function nz(x) if x == 0 then return 1 end return x end".to_owned(),
            "local function ek(e)".to_owned(),
            "  if type(e) == \"table\" then return \"obj:\" .. tostring(string_number(rawget(e, \"code\"))) end"
                .to_owned(),
            "  return e".to_owned(),
            "end".to_owned(),
        ];
        // Arithmetic NaN signs are unspecified. Keep formatting coverage by
        // passing a positive NaN to text conversions, without changing builtins.
        let mut source =
            "local function string_number(v)\n  if v ~= v then return math.abs(v) end\n".to_owned();
        if !self.dialect.integers && !self.dialect.bit_library {
            source.push_str("  if v == 0 then return 0 end\n");
        }
        source.push_str("  return v\nend\nlocal function string_numbers(t)\n  local copy = {}\n  for i = 1, #t do copy[i] = string_number(t[i]) end\n  return copy\nend\n");
        source.push_str(&lines.join("\n"));
        source.push('\n');
        source.push_str(ARRAY_INDEX_HELPER);
        if self.string_arith {
            // Every spelling converts in every profile.
            source.push_str(
                "\nlocal numeric_strings = {\"10\", \"42\", \"7\", \" 7 \", \"0x10\", \"2.5\", \"1e2\", \"-3\", \"0.5\", \"3.\"}",
            );
        }
        if self.dialect.integers {
            source.push('\n');
            source.push_str(BIT_OPERAND_HELPER);
            source.push('\n');
            source.push_str(NUMERIC_LESS_HELPER);
        }
        // `math.random` draws only after this fixed seed, so both modes see
        // the same sequence.
        if self.random_draws() {
            source.push_str("\nmath.randomseed(42)");
        }
        vec![Node::leaf(source)]
    }

    /// Two classes that share method names, so objects can switch
    /// metatables mid-loop, and a proxy table driven by `__index` and
    /// `__newindex` functions.
    fn classes(&mut self) -> Vec<Node> {
        let scale_a = self.rng.range(2, 5);
        let scale_b = self.rng.range(2, 7);
        // Match show()'s treatment of the Lua 5.1/5.2 constant-table zero
        // sign artifact. Other profiles retain the sign in class observations.
        let class_number = if !self.dialect.integers && !self.dialect.bit_library {
            "local function class_number(v) if v ~= v then return \"nan\" end if v == 0 then return \"0\" end return tostring(v) end"
        } else {
            "local function class_number(v) if v ~= v then return \"nan\" end return tostring(v) end"
        };
        let mut lines = vec![
            // NaN's sign is not specified. Class observations must match
            // show(), while explicit formatter tests keep raw inputs.
            class_number.to_owned(),
            "local cmp = {}".to_owned(),
            "function cmp.eq(a, b) return a.v == b.v end".to_owned(),
            "function cmp.lt(a, b) return a.v < b.v end".to_owned(),
            "function cmp.le(a, b) return a.v <= b.v end".to_owned(),
            "local ClassA = {}".to_owned(),
            "ClassA.__index = ClassA".to_owned(),
            "function ClassA.new(v) return setmetatable({v = v, n = 0}, ClassA) end".to_owned(),
            "function ClassA:bump(d) self.n = self.n + 1 self.v = self.v + d return self.v end"
                .to_owned(),
            "function ClassA:get() return self.v end".to_owned(),
            "function ClassA:name() return \"A\" .. self.n end".to_owned(),
            "ClassA.__add = function(a, b) return ClassA.new(a.v + b.v) end".to_owned(),
            "ClassA.__sub = function(a, b) return ClassA.new(a.v - b.v) end".to_owned(),
            "ClassA.__unm = function(a) return ClassA.new(-a.v) end".to_owned(),
            "ClassA.__eq, ClassA.__lt, ClassA.__le = cmp.eq, cmp.lt, cmp.le".to_owned(),
            format!("ClassA.__call = function(self, x) return self.v * {scale_a} + x end"),
            "ClassA.__tostring = function(self) return \"A(\" .. class_number(self.v) .. \")\" end"
                .to_owned(),
            "ClassA.__concat = function(a, b) return tostring(a) .. \"&\" .. tostring(b) end"
                .to_owned(),
            "local ClassB = setmetatable({}, {__index = ClassA})".to_owned(),
            "ClassB.__index = ClassB".to_owned(),
            "function ClassB.new(v) return setmetatable({v = v, n = 0}, ClassB) end".to_owned(),
            "function ClassB:bump(d) self.n = self.n + 2 self.v = self.v - d return self.v end"
                .to_owned(),
            "function ClassB:name() return \"B\" .. self.n end".to_owned(),
            "ClassB.__add = function(a, b) return ClassB.new(a.v * 2 + b.v) end".to_owned(),
            "ClassB.__sub = ClassA.__sub".to_owned(),
            "ClassB.__unm = function(a) return ClassB.new(a.v) end".to_owned(),
            "ClassB.__eq, ClassB.__lt, ClassB.__le = cmp.eq, cmp.lt, cmp.le".to_owned(),
            format!("ClassB.__call = function(self, x) return self.v - x * {scale_b} end"),
            "ClassB.__tostring = function(self) return \"B(\" .. class_number(self.v) .. \")\" end"
                .to_owned(),
            "ClassB.__concat = ClassA.__concat".to_owned(),
            "local Proxy = setmetatable({}, {".to_owned(),
            "  __index = function(t, k) return #k end,".to_owned(),
            "  __newindex = function(t, k, v) rawset(t, k, v % 1000) end,".to_owned(),
            "})".to_owned(),
        ];
        if self.dialect.len_metamethod {
            lines.push("ClassA.__len = function(self) return self.n end".to_owned());
            lines.push("ClassB.__len = function(self) return self.n * 10 end".to_owned());
        }
        vec![Node::leaf(lines.join("\n"))]
    }

    /// Program-wide state the functions capture as upvalues, plus a few
    /// globals.
    fn state(&mut self) -> Vec<Node> {
        let mut nodes = Vec::new();
        let mut kinds = vec![
            Kind::Int,
            Kind::Num,
            Kind::Str,
            Kind::Arr,
            Kind::Rec,
            Kind::Obj,
        ];
        for _ in 0..self.rng.range(1, 4) {
            kinds.push(*self.rng.pick(&ALL_KINDS));
        }
        for kind in kinds {
            let name = self.fresh("g");
            let value = self.initial(kind);
            self.feature("upvalues");
            let node = self.leaf(format!("local {name} = {value}"));
            nodes.push(node);
            self.declare(&name, kind, true);
        }
        for _ in 0..self.rng.range(0, 2) {
            let kind = *self.rng.pick(&[Kind::Int, Kind::Num, Kind::Str]);
            let name = self.fresh("G");
            let value = self.initial(kind);
            self.feature("globals");
            let node = self.leaf(format!("{name} = {value}"));
            nodes.push(node);
            self.declare(&name, kind, true);
        }
        nodes
    }

    /// A value to start a variable with.
    fn initial(&mut self, kind: Kind) -> String {
        match kind {
            Kind::Arr => self.array_constructor(),
            Kind::Rec => self.record_constructor(),
            Kind::Obj => self.object_constructor(),
            _ => self.expr(kind, 1),
        }
    }

    /// Fixed-shape functions every program can call: bounded recursion,
    /// a tail call, mutual recursion, varargs, multiple results and a
    /// polymorphic dispatch table.
    fn library_functions(&mut self) -> Vec<Node> {
        let mut nodes = Vec::new();
        let step = self.rng.range(1, 4);
        self.feature("recursion");
        self.feature("tail_call");
        nodes.push(self.leaf(format!(
            "local function sumdown(n, acc)\n  if n <= 0 then return acc end\n  return sumdown(n - 1, acc + n * {step})\nend"
        )));
        self.add_function(
            "sumdown",
            &[(Kind::Int, Some(40)), (Kind::Int, None)],
            &[Kind::Int],
            40,
        );
        self.feature("recursion");
        nodes.push(self.leaf(
            "local function fib(n)\n  if n < 2 then return n end\n  return fib(n - 1) + fib(n - 2)\nend",
        ));
        self.add_function("fib", &[(Kind::Int, Some(14))], &[Kind::Int], 1300);
        self.feature("mutual_recursion");
        nodes.push(self.leaf(
            "local isodd\nlocal function iseven(n)\n  if n == 0 then return true end\n  return isodd(n - 1)\nend\nisodd = function(n)\n  if n == 0 then return false end\n  return iseven(n - 1)\nend",
        ));
        self.add_function("iseven", &[(Kind::Int, Some(50))], &[Kind::Bool], 50);
        self.feature("varargs");
        self.feature("multiple_results");
        nodes.push(self.leaf(
            "local function va(...)\n  local n = select(\"#\", ...)\n  local first, second = ...\n  return n, first, second, select(n > 0 and n or 1, ...)\nend",
        ));
        let pack = if self.dialect.table_pack {
            self.feature("table_pack");
            "local function packn(...) local p = table.pack(...) return p.n, p[1] end"
        } else {
            "local function packn(...) return select(\"#\", ...), (...) end"
        };
        nodes.push(self.leaf(pack));
        // Two operators with one signature behind a dispatch table: a
        // polymorphic call site.
        let left = self.int_binary_text("a", "b");
        let right = self.int_binary_text("b", "a");
        self.feature("polymorphic_calls");
        nodes.push(self.leaf(format!(
            "local function opa(a, b) return {left} end\nlocal function opb(a, b) return {right} end\nlocal ops = {{opa, opb}}"
        )));
        self.feature("closures");
        let start = self.rng.range(0, 9);
        nodes.push(self.leaf(format!(
            "local counter = (function()\n  local n = {start}\n  return function(d) n = n + d return n end\nend)()"
        )));
        let factor = self.rng.range(2, 5);
        self.feature("coroutine_wrap");
        nodes.push(self.leaf(format!(
            "local gen = coroutine.wrap(function()\n  local k = 0\n  while true do\n    k = k + 1\n    coroutine.yield(k * {factor})\n  end\nend)"
        )));
        let env = if self.dialect.env_local {
            self.feature("env_local");
            "local function envf(x)\n  local _ENV = {v = 7, tostring = tostring, string_number = string_number}\n  return tostring(v) .. \":\" .. string_number(x)\nend"
                .to_owned()
        } else {
            self.feature("setfenv");
            "local envf = setfenv(function(x) return tostring(v) .. \":\" .. string_number(x) end, {v = 7, tostring = tostring, string_number = string_number})"
                .to_owned()
        };
        nodes.push(self.leaf(env));
        self.add_function("envf", &[(Kind::Int, None)], &[Kind::Str], 3);
        nodes
    }

    fn add_function(
        &mut self,
        name: &str,
        params: &[(Kind, Option<u32>)],
        results: &[Kind],
        cost: u64,
    ) {
        self.functions.push(Function {
            name: name.to_owned(),
            params: params
                .iter()
                .map(|&(kind, bound)| Param { kind, bound })
                .collect(),
            results: results.to_vec(),
            pure: true,
            cost,
        });
    }

    /// `a op b` over two integer names, kept in range for the profile.
    fn int_binary_text(&mut self, a: &str, b: &str) -> String {
        let text = match self.rng.index(4) {
            0 => format!("{a} + {b} * 3"),
            1 => format!("{a} - {b}"),
            2 => format!("{a} * 2 + {b}"),
            _ => format!("{a} % nz({b})"),
        };
        self.wrap_int(text)
    }

    /// Keeps an integer expression exact in profiles without integers.
    fn wrap_int(&self, text: String) -> String {
        if self.dialect.integers {
            text
        } else {
            format!("(({text}) % {DOUBLE_INT_MODULUS})")
        }
    }

    // ----- generated functions ------------------------------------------

    fn function(&mut self) -> Node {
        let name = self.fresh("f");
        let params: Vec<Param> = (0..self.rng.range(0, 3))
            .map(|_| Param {
                kind: *self.rng.pick(&[
                    Kind::Int,
                    Kind::Int,
                    Kind::Num,
                    Kind::Str,
                    Kind::Bool,
                    Kind::Arr,
                ]),
                bound: None,
            })
            .collect();
        let results: Vec<Kind> = if self.rng.chance(1, 4) {
            vec![*self.rng.pick(&SCALARS), *self.rng.pick(&SCALARS)]
        } else {
            vec![*self.rng.pick(&ALL_KINDS)]
        };
        let pure = self.rng.chance(1, 2);

        let saved_frame = self.frame.clone();
        let saved_cost = self.cost;
        self.frame = Frame {
            level: saved_frame.level + 1,
            pure,
            loops: 0,
            continue_label: None,
            multiplier: 1,
            counter: None,
            in_coroutine: false,
            depth: 0,
            kernel: false,
            kernel_locals: false,
            quiet: true,
        };
        self.cost = 0;
        self.push_scope();
        let mut names = Vec::new();
        for param in &params {
            let param_name = self.fresh("p");
            self.declare(&param_name, param.kind, true);
            names.push(param_name);
        }
        let mut body = self.statements();
        // Returned strings are cut like stored ones: a caller may feed the
        // result back in through a multiple assignment.
        let returned: Vec<String> = results
            .iter()
            .map(|&kind| {
                let value = self.expr(kind, 0);
                if kind == Kind::Str {
                    format!("string.sub({value}, 1, 48)")
                } else {
                    value
                }
            })
            .collect();
        if results.len() > 1 {
            self.feature("multiple_results");
        }
        let node = self.leaf(format!("return {}", returned.join(", ")));
        body.push(node);
        self.pop_scope();
        let cost = self.cost.max(1);
        self.frame = saved_frame;
        self.cost = saved_cost;

        self.functions.push(Function {
            name: name.clone(),
            params,
            results,
            pure,
            cost,
        });
        self.feature("functions");
        let features = self.take_features();
        Node::block(
            format!("local function {name}({})", names.join(", ")),
            body,
            "end",
        )
        .with_features(features)
    }

    // ----- sections ------------------------------------------------------

    /// One top-level section: a loop (hot, warm or cold), its locals and a
    /// closing observation.
    #[allow(clippy::too_many_lines, reason = "one section, read top to bottom")]
    fn section(&mut self, index: i64) -> Node {
        let shape = self.rng.weighted(&[6, 3, 2, 2, 2, 2, 2]);
        let iterations = match self.rng.weighted(&[5, 3, 2]) {
            0 => self.rng.range(100, 400),
            1 => self.rng.range(20, 90),
            _ => self.rng.range(1, 5),
        };
        let iterations = u32::try_from(iterations).expect("small iteration counts fit u32");
        self.push_scope();
        let saved_cost = self.cost;
        self.cost = 0;
        let mut body = Vec::new();
        let mut locals = Vec::new();
        for _ in 0..self.rng.range(1, 3) {
            let kind = *self
                .rng
                .pick(&[Kind::Int, Kind::Num, Kind::Str, Kind::Bool, Kind::Arr]);
            let name = self.fresh("s");
            let value = self.initial(kind);
            let node = self.leaf(format!("local {name} = {value}"));
            body.push(node);
            self.declare(&name, kind, true);
            locals.push((name, kind));
        }
        let counter = self.fresh("i");
        let label = format!("section {index}");
        let (head, tail, prefix, feature) = match shape {
            0 => (
                format!("for {counter} = 1, {ITERATIONS} do"),
                "end".to_owned(),
                None,
                "numeric_for",
            ),
            1 => (
                format!("for {counter} = {ITERATIONS}, 1, -1 do"),
                "end".to_owned(),
                None,
                "reverse_for",
            ),
            2 => (
                format!("while {counter} < {ITERATIONS} do"),
                "end".to_owned(),
                Some(format!("local {counter} = 0")),
                "while",
            ),
            3 => (
                "repeat".to_owned(),
                format!("until {counter} >= {ITERATIONS}"),
                Some(format!("local {counter} = 0")),
                "repeat",
            ),
            4 => {
                let source = self.fresh("src");
                let value = self.fresh("v");
                self.declare(&value, Kind::Int, false);
                (
                    format!("for {counter}, {value} in ipairs({source}) do"),
                    "end".to_owned(),
                    Some(format!(
                        "local {source} = {{}}\nfor k = 1, {ITERATIONS} do {source}[k] = k * 3 end"
                    )),
                    "ipairs",
                )
            }
            5 => {
                let float = self.fresh("x");
                self.declare(&float, Kind::Num, false);
                (
                    format!("for {float} = 0.5, {ITERATIONS} / 4 + 0.5, 0.25 do"),
                    "end".to_owned(),
                    Some(format!("local {counter} = 0")),
                    "float_for",
                )
            }
            _ => (
                format!("for {counter} = 1, {ITERATIONS} do"),
                "end".to_owned(),
                None,
                "nested_loops",
            ),
        };
        let counter_is_manual = matches!(shape, 2 | 3 | 5);
        if let Some(prefix) = prefix {
            body.push(Node {
                iterations: Some(iterations),
                ..Node::leaf(prefix)
            });
        }
        self.declare(&counter, Kind::Int, false);
        let multiplier = if shape == 4 || shape == 5 {
            // `ipairs` and the float loop run the same count, the float one
            // a little more; a cold count keeps the estimate honest.
            u64::from(iterations)
        } else {
            1
        };
        let _ = multiplier;

        let saved_frame = self.frame.clone();
        self.frame.loops += 1;
        self.frame.counter = Some((counter.clone(), iterations));
        self.frame.continue_label = None;
        self.push_scope();
        let mut loop_body = Vec::new();
        if counter_is_manual {
            loop_body.push(Node::leaf(format!("{counter} = {counter} + 1")));
        }
        let continue_label = if self.dialect.goto && self.rng.chance(1, 4) {
            let label = self.fresh("continue");
            self.frame.continue_label = Some(label.clone());
            Some(label)
        } else {
            None
        };
        if shape == 6 {
            loop_body.push(self.nested_loop());
        }
        loop_body.extend(self.statements());
        let mut observed = vec![counter.clone()];
        observed.extend(locals.iter().map(|(name, kind)| Self::observe(name, *kind)));
        for kind in [Kind::Int, Kind::Num, Kind::Str] {
            if self.rng.chance(1, 2)
                && let Some(var) = self.pick_var(kind)
            {
                observed.push(var);
            }
        }
        let tag = self.tag("L");
        loop_body.push(Node::leaf(format!("out({tag}, {})", observed.join(", "))));
        if let Some(label) = continue_label {
            // As in `inner_loop`: everything but the counter goes in a
            // `do` block so the `goto` skips no local's declaration.
            self.feature("goto_continue");
            let split = usize::from(counter_is_manual);
            let scoped = loop_body.split_off(split);
            loop_body.push(Node::block("do", scoped, "end"));
            loop_body.push(Node::leaf(format!("::{label}::")).pinned());
        }
        self.pop_scope();
        self.frame = saved_frame;
        self.feature(feature);
        let features = self.take_features();
        body.push(Node {
            iterations: Some(iterations),
            features,
            ..Node::block(head, loop_body, tail)
        });
        let finals: Vec<String> = locals
            .iter()
            .map(|(name, kind)| Self::observe(name, *kind))
            .collect();
        if !finals.is_empty() {
            let tag = self.tag("E");
            body.push(Node::leaf(format!("out({tag}, {})", finals.join(", "))));
        }
        self.pop_scope();
        self.cost = saved_cost;
        Node::block(format!("do -- {label}"), body, "end")
    }

    /// A hot loop over numbers, arrays and records with no output inside, so
    /// the trace recorder can compile it. A checksum folds in an integer
    /// every iteration, phase changes force guard exits part way and print
    /// the state when they fire, and everything prints after the loop.
    fn kernel_section(&mut self, index: i64) -> Node {
        // The trace compiler skips a loop whose remaining work cannot repay a
        // compile (about 262 000 guest instructions, policy.rs), so kernels
        // run thousands of trips.
        let iterations = if let Some(knobs) = self.hot {
            let base = i64::from(knobs.trips.max(HotKnobs::MIN_TRIPS));
            self.rng.range(base, base + base / 2)
        } else if self.rng.chance(1, 2) {
            self.rng.range(3_000, 8_000)
        } else {
            self.rng.range(10_000, 40_000)
        };
        let iterations = u32::try_from(iterations).expect("kernel trip counts fit u32");
        self.push_scope();
        let saved_cost = self.cost;
        self.cost = 0;
        let saved_frame = self.frame.clone();
        self.frame.kernel = true;
        let mut body = Vec::new();
        // Two kernels in three use only their own locals; the rest also
        // read and write upvalues and globals.
        let locals_only = self.rng.chance(2, 3);
        let mut locals = Vec::new();
        if self.string_arith && self.allowed("string_arithmetic") {
            // A local copy, so a locals-only kernel can read the strings.
            body.push(Node::leaf(
                "local numeric_strings = numeric_strings".to_owned(),
            ));
        }
        let mut kinds = vec![Kind::Int, Kind::Num];
        if let Some(knobs) = self.hot {
            let mut pool = vec![Kind::Int, Kind::Num];
            if knobs.types & HotKnobs::ARRAYS != 0 {
                pool.push(Kind::Arr);
            }
            if knobs.types & HotKnobs::RECORDS != 0 {
                pool.push(Kind::Rec);
            }
            for _ in 0..self.rng.range(1, 3) {
                kinds.push(*self.rng.pick(&pool));
            }
        } else {
            for _ in 0..self.rng.range(1, 3) {
                kinds.push(*self.rng.pick(&[Kind::Int, Kind::Num, Kind::Arr, Kind::Rec]));
            }
        }
        for kind in kinds {
            let name = self.fresh("k");
            let value = self.initial(kind);
            let node = self.leaf(format!("local {name} = {value}"));
            body.push(node);
            self.declare(&name, kind, true);
            locals.push((name, kind));
        }
        let checksum = self.fresh("ck");
        body.push(Node::leaf(format!("local {checksum} = 0")));
        if let Some(knobs) = self.hot {
            body.extend(self.hot_prefix(knobs));
        }
        let counter = self.fresh("i");
        let (head, prefix) = match self.rng.index(3) {
            0 => (format!("for {counter} = 1, {ITERATIONS} do"), None),
            1 => (format!("for {counter} = {ITERATIONS}, 1, -1 do"), None),
            _ => (
                format!("while {counter} < {ITERATIONS} do"),
                Some(format!("local {counter} = 0")),
            ),
        };
        if let Some(prefix) = &prefix {
            body.push(Node {
                iterations: Some(iterations),
                ..Node::leaf(prefix.clone())
            });
        }
        self.declare(&counter, Kind::Int, false);
        self.frame.loops += 1;
        self.frame.counter = Some((counter.clone(), iterations));
        self.frame.continue_label = None;
        self.frame.kernel_locals = locals_only;
        self.push_scope();
        let mut loop_body = Vec::new();
        if prefix.is_some() {
            loop_body.push(Node::leaf(format!("{counter} = {counter} + 1")).pinned());
        }
        let (low, high) = self.limits.statements;
        for _ in 0..self.rng.range(low + 1, high + 2) {
            if !self.charge(1) {
                break;
            }
            if self.hot.is_some()
                && self.rng.chance(1, 3)
                && let Some(statement) = self.hot_statement()
            {
                loop_body.push(statement);
            }
            loop_body.push(self.kernel_statement());
        }
        let folded = locals
            .iter()
            .find(|(_, kind)| *kind == Kind::Int)
            .map_or_else(|| counter.clone(), |(name, _)| name.clone());
        loop_body.push(Node::leaf(format!(
            "{checksum} = ({checksum} * 7 + {folded} % 1000003) % 1000003"
        )));
        self.pop_scope();
        self.frame = saved_frame;
        self.feature("kernel_loop");
        let features = self.take_features();
        body.push(Node {
            iterations: Some(iterations),
            features,
            ..Node::block(head, loop_body, "end")
        });
        let mut finals = vec![checksum];
        finals.extend(locals.iter().map(|(name, kind)| Self::observe(name, *kind)));
        let tag = self.tag("K");
        body.push(Node::leaf(format!("out({tag}, {})", finals.join(", "))));
        if self.hot.is_some() {
            body.extend(self.hot_suffix());
        }
        self.pop_scope();
        self.cost = saved_cost;
        Node::block(format!("do -- kernel section {index}"), body, "end")
    }

    /// A hot loop over named functions that `pcall` or `xpcall` calls and
    /// that raise on some iterations, or on every one: `error` at levels 0
    /// to 2 with strings, tables, nil and numbers, failed `assert`, and a few
    /// errors the runtime raises. The trace compiler inlines the named
    /// functions into the protected call, so the loop exercises the paths
    /// that catch an error without leaving native code, and the ones that
    /// fall back to the interpreter.
    #[allow(clippy::too_many_lines, reason = "one arm per statement family")]
    fn protected_kernel_section(&mut self, index: i64) -> Node {
        let iterations = if self.rng.chance(1, 2) {
            self.rng.range(3_000, 8_000)
        } else {
            self.rng.range(10_000, 30_000)
        };
        let iterations = u32::try_from(iterations).expect("kernel trip counts fit u32");
        let prefix = format!("pk{index}");
        // How often the function raises: every call, or every period-th.
        let period = *self.rng.pick(&[1_i64, 2, 3, 7, 11]);
        let mut raises: Vec<String> = vec![
            "error(\"fixed\")".to_owned(),
            "error(\"lvl0\", 0)".to_owned(),
            "error(\"lvl1\", 1)".to_owned(),
            "error(\"lvl2\", 2)".to_owned(),
            "error(\"computed\" .. n % 3)".to_owned(),
            "error(\"computed\" .. n % 3, 2)".to_owned(),
            "error(\"computed\" .. n % 3, 0)".to_owned(),
            format!("error({prefix}E)"),
            format!("error({prefix}E, 0)"),
            "error({code = n % 5})".to_owned(),
            "error(nil)".to_owned(),
            "error()".to_owned(),
            "error(7, 0)".to_owned(),
            "error(true, 0)".to_owned(),
            "assert(false)".to_owned(),
            "assert(nil, \"pk\")".to_owned(),
            "assert(n < 0, \"pk\" .. n % 3)".to_owned(),
            "local z = nil\n    return z + n".to_owned(),
            "local z = {}\n    return z.a.b".to_owned(),
        ];
        let raise = raises.swap_remove(self.rng.index(raises.len()));
        // A wrapper adds a level, so that level 2 names a Lua caller.
        let wrapped = self.rng.chance(1, 2);
        let entry = if wrapped {
            format!("{prefix}g")
        } else {
            format!("{prefix}f")
        };
        let protected = self.rng.index(3) != 0;
        // Lua 5.1's `xpcall` passes the function no arguments, so the
        // function reads the count from an upvalue there.
        let by_upvalue = protected && !self.dialect.xpcall_arguments;
        let parameter = if by_upvalue { "" } else { "n" };
        let mut lines = vec![
            format!("local {prefix}E = {{code = 3}}"),
            format!("local {prefix}n = 0"),
            format!("local function {prefix}f({parameter})"),
        ];
        if by_upvalue {
            lines.push(format!("  local n = {prefix}n"));
        }
        lines.extend([
            format!("  if n % {period} == 0 then"),
            format!("    {raise}"),
            "  end".to_owned(),
            "  return n * 2 + 1".to_owned(),
            "end".to_owned(),
            format!("local function {prefix}g({parameter})"),
            format!("  local r = {prefix}f({parameter})"),
            "  return r + 1".to_owned(),
            "end".to_owned(),
            format!("local function {prefix}h(m)"),
            "  if type(m) == \"table\" then return \"H:obj\" end".to_owned(),
            "  return \"H:\" .. tostring(m)".to_owned(),
            "end".to_owned(),
            format!("local {prefix}c, {prefix}e, {prefix}l = 0, 0, nil"),
        ]);
        let arguments = if by_upvalue {
            String::new()
        } else {
            ", i".to_owned()
        };
        let call = if protected {
            format!("xpcall({entry}, {prefix}h{arguments})")
        } else {
            format!("pcall({entry}{arguments})")
        };
        let mut loop_body = Vec::new();
        if by_upvalue {
            loop_body.push(Node::leaf(format!("{prefix}n = i")));
        }
        loop_body.push(Node::leaf(format!("local ok, v = {call}")));
        // The reducer may not drop the `else` alone.
        loop_body.push(Node::block(
            "if ok then",
            vec![
                Node::leaf(format!("{prefix}c = ({prefix}c * 7 + v) % 1000003")),
                Node::leaf("else").pinned(),
                Node::leaf(format!("{prefix}e = {prefix}e + 1")),
                Node::leaf(format!(
                    "if type(v) == \"string\" then\n  {prefix}c = ({prefix}c + #v) % 1000003\n  {prefix}l = v\nelseif type(v) == \"table\" then\n  {prefix}c = ({prefix}c + (v.code or 0)) % 1000003\n  {prefix}l = \"obj\"\nelse\n  {prefix}l = v\nend"
                )),
            ],
            "end",
        ));
        let tag = self.tag("PK");
        let mut body: Vec<Node> = lines.into_iter().map(Node::leaf).collect();
        self.feature("protected_kernel");
        let features = self.take_features();
        body.push(Node {
            iterations: Some(iterations),
            features,
            ..Node::block(format!("for i = 1, {ITERATIONS} do"), loop_body, "end")
        });
        body.push(Node::leaf(format!(
            "out({tag}, {prefix}c, {prefix}e, {prefix}l)"
        )));
        Node::block(
            format!("do -- protected kernel section {index}"),
            body,
            "end",
        )
    }

    fn kernel_statement(&mut self) -> Node {
        let nested = self.frame.depth < self.limits.depth;
        let weights = [
            8,                          // 0 assign
            4,                          // 1 compound update
            3,                          // 2 array write
            3,                          // 3 record write
            if nested { 3 } else { 0 }, // 4 if
            if nested { 2 } else { 0 }, // 5 inner numeric loop
            3,                          // 6 phase change with observation
            1,                          // 7 break
            1,                          // 8 swap
            0,                          // 9 method call (stops the recorder)
            if self.frame.pure || !self.random_draws() {
                0
            } else {
                2
            }, // 10 math.random draw
        ];
        let mut weights = weights;
        if let Some(knobs) = self.hot {
            weights[6] = if knobs.has(HotKnobs::PHASES) { 6 } else { 0 };
        }
        match self.rng.weighted(&weights) {
            10 => self.random_draw(),
            0 | 1 => {
                let kind = *self.rng.pick(&[Kind::Int, Kind::Num]);
                let targets = self.writable(kind);
                if targets.is_empty() {
                    return self.kernel_fallback();
                }
                let target = self.rng.pick(&targets).name.clone();
                let value = self.expr(kind, 0);
                let value = if self.rng.chance(1, 2) {
                    self.feature("compound_update");
                    format!("{target} + {value}")
                } else {
                    value
                };
                let value = self.store_int_or_num(kind, value);
                self.leaf(format!("{target} = {value}"))
            }
            2 => {
                let arrays = self.mutable_tables(Kind::Arr);
                if arrays.is_empty() {
                    return self.kernel_fallback();
                }
                let t = self.rng.pick(&arrays).name.clone();
                let index = self.expr(Kind::Int, 1);
                let value = self.stored(Kind::Int);
                self.feature("table_index_write");
                self.leaf(format!("{t}[array_index({index}, #{t})] = {value}"))
            }
            3 => {
                let records = self.mutable_tables(Kind::Rec);
                if records.is_empty() {
                    return self.kernel_fallback();
                }
                let r = self.rng.pick(&records).name.clone();
                let value = self.stored(Kind::Num);
                if self.rng.chance(1, 3) {
                    let key = self.dynamic_key();
                    self.feature("dynamic_key_write");
                    return self.leaf(format!("{r}[{key}] = {value}"));
                }
                let field = *self.rng.pick(&["x", "y", "z"]);
                self.feature("field_write");
                self.leaf(format!("{r}.{field} = {value}"))
            }
            4 => {
                let condition = self.condition();
                self.frame.depth += 1;
                self.push_scope();
                let mut body: Vec<Node> = (0..self.rng.range(1, 3))
                    .map(|_| self.kernel_statement())
                    .collect();
                self.pop_scope();
                self.push_scope();
                let otherwise: Vec<Node> = (0..self.rng.range(0, 2))
                    .map(|_| self.kernel_statement())
                    .collect();
                self.pop_scope();
                self.frame.depth -= 1;
                if !otherwise.is_empty() {
                    body.push(Node::leaf("else").pinned());
                    body.extend(otherwise);
                }
                self.feature("branches");
                let features = self.take_features();
                Node::block(format!("if {condition} then"), body, "end").with_features(features)
            }
            5 => {
                let var = self.fresh("j");
                let count = self.rng.range(1, 6);
                let saved = self.frame.clone();
                self.frame.multiplier = self
                    .frame
                    .multiplier
                    .saturating_mul(u64::try_from(count).unwrap_or(1));
                self.frame.loops += 1;
                self.frame.depth += 1;
                self.push_scope();
                self.declare(&var, Kind::Int, false);
                let body: Vec<Node> = (0..self.rng.range(1, 3))
                    .map(|_| self.kernel_statement())
                    .collect();
                self.pop_scope();
                self.frame = saved;
                self.feature("nested_loops");
                let features = self.take_features();
                Node::block(format!("for {var} = 1, {count} do"), body, "end")
                    .with_features(features)
            }
            6 => self.forced_exit(),
            7 => {
                let condition = self.expr(Kind::Bool, 1);
                self.feature("break");
                self.leaf(format!("if {condition} then break end"))
            }
            8 => {
                let kind = *self.rng.pick(&[Kind::Int, Kind::Num]);
                let vars = self.writable(kind);
                if vars.len() < 2 {
                    return self.kernel_fallback();
                }
                let a = self.rng.pick(&vars).name.clone();
                let b = self.rng.pick(&vars).name.clone();
                if a == b {
                    return self.kernel_fallback();
                }
                self.feature("multiple_assignment");
                self.leaf(format!("{a}, {b} = {b}, {a}"))
            }
            _ => {
                let objects = self.mutable_tables(Kind::Obj);
                let Some(o) = objects.first().map(|var| var.name.clone()) else {
                    return self.kernel_fallback();
                };
                let delta = self.expr(Kind::Num, 2);
                self.feature("method_call");
                self.leaf(format!("{o}:bump({delta})"))
            }
        }
    }

    /// Changes a number's subtype, replaces an array and drops a record
    /// field at one iteration of a kernel loop, so compiled code exits on
    /// a guard, and prints the loop state at that point.
    fn forced_exit(&mut self) -> Node {
        let (counter, iterations) = self
            .frame
            .counter
            .clone()
            .expect("kernel loops have a counter");
        let at = self.rng.range(1, i64::from(iterations.max(3)) - 1);
        let mut body = Vec::new();
        if let Some(var) = self.writable(Kind::Num).first().map(|var| var.name.clone()) {
            let delta = *self
                .rng
                .pick(&["0.5", "1", "-0.25", "2^53", "1e300", "0/0"]);
            body.push(Node::leaf(format!("{var} = {var} + {delta}")));
            self.feature("phase_number_subtype");
        }
        if let Some(var) = self
            .mutable_tables(Kind::Arr)
            .first()
            .map(|var| var.name.clone())
        {
            let fresh = self.array_constructor();
            body.push(Node::leaf(format!("{var} = {fresh}")));
            self.feature("phase_array_alias");
        }
        if let Some(var) = self
            .mutable_tables(Kind::Rec)
            .first()
            .map(|var| var.name.clone())
        {
            body.push(Node::leaf(format!("{var}.y = nil")));
            self.feature("phase_table_shape");
        }
        let mut seen = vec![counter.clone()];
        for kind in [Kind::Int, Kind::Num] {
            seen.extend(
                self.visible(kind)
                    .into_iter()
                    .filter(|var| var.name.starts_with('k') && var.assignable)
                    .map(|var| var.name),
            );
        }
        let tag = self.tag("F");
        body.push(Node::leaf(format!("out({tag}, {})", seen.join(", "))));
        self.feature("forced_exit");
        let features = self.take_features();
        Node::block(format!("if {counter} == {at} then"), body, "end").with_features(features)
    }

    fn kernel_fallback(&mut self) -> Node {
        let targets = self.writable(Kind::Int);
        let Some(target) = targets.first().map(|var| var.name.clone()) else {
            return self.leaf("local unused = 0");
        };
        let value = self.expr(Kind::Int, 1);
        let value = self.store_int_or_num(Kind::Int, value);
        self.leaf(format!("{target} = {value}"))
    }

    /// An inner loop that runs hot inside a short outer loop.
    fn nested_loop(&mut self) -> Node {
        let inner = self.fresh("j");
        let count = self.rng.range(30, 120);
        let saved = self.frame.clone();
        self.frame.multiplier = self
            .frame
            .multiplier
            .saturating_mul(u64::try_from(count).unwrap_or(1));
        self.frame.loops += 1;
        self.frame.continue_label = None;
        self.push_scope();
        self.declare(&inner, Kind::Int, false);
        let body = self.statements();
        self.pop_scope();
        self.frame = saved;
        self.feature("nested_loops");
        let features = self.take_features();
        Node::block(format!("for {inner} = 1, {count} do"), body, "end").with_features(features)
    }

    fn final_observation(&mut self) -> Node {
        let mut parts = Vec::new();
        for kind in ALL_KINDS {
            for var in self.visible(kind) {
                parts.push(Self::observe(&var.name, kind));
            }
        }
        Node::leaf(format!("out(\"END\", {})", parts.join(", ")))
    }

    /// A deterministic observation of a variable.
    fn observe(name: &str, kind: Kind) -> String {
        match kind {
            Kind::Arr => format!("arr({name})"),
            Kind::Rec => format!("rec({name})"),
            Kind::Obj => format!("tostring({name}), {name}.n"),
            _ => name.to_owned(),
        }
    }

    // ----- statements ----------------------------------------------------

    fn statements(&mut self) -> Vec<Node> {
        let (low, high) = self.limits.statements;
        let count = self.rng.range(low, high);
        let mut nodes = Vec::new();
        for _ in 0..count {
            if !self.charge(1) {
                break;
            }
            nodes.push(self.statement());
        }
        nodes
    }

    fn block(&mut self) -> Vec<Node> {
        self.frame.depth += 1;
        self.push_scope();
        let nodes = self.statements();
        self.pop_scope();
        self.frame.depth -= 1;
        nodes
    }

    fn statement(&mut self) -> Node {
        let nested = self.frame.depth < self.limits.depth;
        let main = self.frame.level == 0;
        let in_loop = self.frame.loops > 0;
        let phase = main && self.frame.counter.is_some();
        let weights = [
            12,                                               // 0 assign
            6,                                                // 1 local
            5,                                                // 2 observe
            if nested { 5 } else { 0 },                       // 3 if
            if nested { 3 } else { 0 },                       // 4 inner loop
            if self.frame.pure { 0 } else { 5 },              // 5 table/array mutation
            if self.frame.pure { 0 } else { 4 },              // 6 record mutation
            if self.frame.pure { 0 } else { 3 },              // 7 object call
            if self.frame.pure { 0 } else { 4 },              // 8 impure call
            if nested { 4 } else { 0 },                       // 9 pcall
            if main && nested { 2 } else { 0 },               // 10 coroutine
            if in_loop { 2 } else { 0 },                      // 11 break / goto
            if phase { 6 } else { 0 },                        // 12 phase change
            3,                                                // 13 string iteration
            if self.frame.pure { 0 } else { 2 },              // 14 closures in loops
            2,                                                // 15 swap
            if self.dialect.close_and_const { 2 } else { 0 }, // 16 close/const
            u32::from(main),                                  // 17 gc
            if self.frame.pure { 0 } else { 2 },              // 18 proxy
            u32::from(main && nested),                        // 19 integer edges
            if self.frame.in_coroutine { 3 } else { 0 },      // 20 yield
            if self.frame.pure { 0 } else { 2 },              // 21 table layout churn
            if main && nested && self.allowed("yield_boundaries") {
                2
            } else {
                0
            }, // 22 yield boundaries
            if main && self.allowed("stdlib_edges") {
                2
            } else {
                0
            }, // 23 stdlib edges
            if self.frame.pure || !self.random_draws() {
                0
            } else {
                3
            }, // 24 math.random draw
            if self.pattern_calls && self.allowed("patterns") {
                4
            } else {
                0
            }, // 25 pattern call
            if self.text_calls && self.dialect.utf8 && self.allowed("text_calls") {
                4
            } else {
                0
            }, // 26 utf8 and string.pack call
            if self.traversals && self.allowed("table_traversal") {
                4
            } else {
                0
            }, // 27 pairs and next traversal
        ];
        // Statements that print are left out of inner loops, where they
        // would print once per inner trip and flood the output.
        let mut weights = weights;
        // Function bodies print nothing either: a hot loop may call them
        // thousands of times.
        if self.frame.multiplier > 4 || self.frame.quiet {
            for printing in [2, 9, 10, 13, 14, 16, 17, 19, 21, 22, 23] {
                weights[printing] = 0;
            }
        }
        match self.rng.weighted(&weights) {
            0 => self.assignment(),
            1 => self.local_declaration(),
            2 => self.observation(),
            3 => self.if_statement(),
            4 => self.inner_loop(),
            5 => self.array_mutation(),
            6 => self.record_mutation(),
            7 => self.object_statement(),
            8 => self.impure_call(),
            9 => self.protected_call(),
            10 => self.coroutine_block(),
            11 => self.loop_exit(),
            12 => self.phase_change(),
            13 => self.string_iteration(),
            14 => self.closure_capture(),
            15 => self.swap(),
            16 => self.close_and_const(),
            17 => self.collect_garbage(),
            18 => self.proxy_write(),
            19 => self.integer_edges(),
            20 => self.yield_statement(),
            21 => self.layout_churn(),
            22 => self.yield_boundary(),
            24 => self.random_draw(),
            25 => self.pattern_call(),
            26 => self.text_call(),
            27 => self.traversal_call(),
            _ => self.stdlib_edge(),
        }
    }

    /// A call of `string.find`, `string.match`, `string.gmatch` or
    /// `string.gsub` with one of a fixed set of patterns, folded into an
    /// integer variable (a statement in a hot loop prints nothing). The
    /// patterns use the classes, captures, anchors, `%b` and `%f` every
    /// profile reads alike.
    fn pattern_call(&mut self) -> Node {
        let targets = self.writable(Kind::Int);
        if targets.is_empty() {
            return self.assignment();
        }
        let target = self.rng.pick(&targets).name.clone();
        let subject = if self.rng.chance(1, 2) {
            self.expr(Kind::Str, 1)
        } else {
            (*self.rng.pick(&[
                "\"key=value; other=thing\"",
                "\"  12 ab 345 \"",
                "\"f(a(b)c) x\"",
                "\"THE (quick) fox\"",
                "\"aaa bbb\"",
                "\"\"",
            ]))
            .to_owned()
        };
        let pattern = *self.rng.pick(&[
            "\"%a+\"",
            "\"(%d+)\"",
            "\"()%s()\"",
            "\"^(%w+)\"",
            "\"(%w+)$\"",
            "\"%b()\"",
            "\"%f[%w]%w+\"",
            "\"[%a_][%w_]*\"",
            "\"(.)(.)\"",
            "\"(%a+)=(%a+)\"",
            "\"x*\"",
            "\"%s*\"",
            "\"\"",
        ]);
        self.feature("patterns");
        let body = match self.rng.index(5) {
            0 => format!(
                "local a, b, c = string.find({subject}, {pattern}, {})\n  {target} = {}",
                self.rng.pick(&["1", "2", "-3", "20"]),
                self.store_int_or_num(
                    Kind::Int,
                    format!("{target} + (a or 0) + (b or 1) + #tostring(c)")
                )
            ),
            1 => format!(
                "local a, b = string.match({subject}, {pattern})\n  {target} = {}",
                self.store_int_or_num(Kind::Int, format!("{target} + #tostring(a) + #tostring(b)"))
            ),
            2 => format!(
                "local c = 0\n  for a, b in string.gmatch({subject}, {pattern}) do c = c + #tostring(a) + #tostring(b) + 1 end\n  {target} = {}",
                self.store_int_or_num(Kind::Int, format!("{target} + c"))
            ),
            3 => {
                let replacement = *self.rng.pick(&["\"#\"", "\"<%0>\"", "\"%%\"", "\"\""]);
                let limit = *self.rng.pick(&["", ", 1", ", 2", ", 0"]);
                format!(
                    "local r, c = string.gsub({subject}, {pattern}, {replacement}{limit})\n  {target} = {}",
                    self.store_int_or_num(Kind::Int, format!("{target} + #r + c"))
                )
            }
            _ => {
                let limit = *self.rng.pick(&["", ", 1", ", 3"]);
                format!(
                    "local r, c = string.gsub({subject}, {pattern}, {{a = \"A\", ab = false, [1] = \"one\", [\"12\"] = 7}}{limit})\n  {target} = {}",
                    self.store_int_or_num(Kind::Int, format!("{target} + #r + c"))
                )
            }
        };
        self.leaf(format!("do\n  {body}\nend"))
    }

    /// A call of the `utf8` library or of `string.pack`, `string.unpack` or
    /// `string.packsize`, with one of a fixed set of formats and edge strings,
    /// folded into an integer variable. Calls that may raise (a range, an
    /// invalid string, an overflow) go through `pcall`, so that the exit path
    /// of a trace is drawn too and the program goes on.
    #[allow(clippy::too_many_lines)]
    fn text_call(&mut self) -> Node {
        let targets = self.writable(Kind::Int);
        if targets.is_empty() {
            return self.assignment();
        }
        let target = self.rng.pick(&targets).name.clone();
        let subject = (*self.rng.pick(&[
            "\"\"",
            "\"abc\"",
            "\"h\\195\\169llo \\226\\130\\172\"",
            "\"\\240\\159\\152\\128x\"",
            "\"a\\255b\"",
            "\"\\226\\130\"",
            "\"\\128\"",
            "\"\\237\\160\\128\"",
            "\"\\244\\144\\128\\128\"",
            "\"\\195\\169\\195\\169\\195\\169\"",
        ]))
        .to_owned();
        let position = *self
            .rng
            .pick(&["1", "2", "3", "-1", "-2", "-3", "0", "5", "12"]);
        let other = *self.rng.pick(&["1", "2", "-1", "4", "-4", "0", "7"]);
        let code = *self.rng.pick(&[
            "0", "65", "127", "128", "2047", "2048", "8364", "65535", "65536", "128512", "1114111",
        ]);
        let number = *self.rng.pick(&[
            "0",
            "1",
            "-1",
            "127",
            "128",
            "-129",
            "255",
            "256",
            "65535",
            "-32768",
            "2147483647",
            "-2147483648",
            "4294967295",
        ]);
        self.feature("text_calls");
        let body = match self.rng.index(11) {
            0 => format!(
                "local a, b = utf8.len({subject})\n  {target} = {}",
                self.store_int_or_num(Kind::Int, format!("{target} + (a or -1) + (b or 0)"))
            ),
            1 => format!(
                "local ok, a, b = pcall(utf8.len, {subject}, {position}, {other})\n  {target} = {}",
                self.store_int_or_num(
                    Kind::Int,
                    format!("{target} + (ok and ((a or -1) + (b or 0)) or 7)")
                )
            ),
            2 => format!(
                "local ok, a, b, c = pcall(utf8.codepoint, {subject}, {position}, {other})\n  {target} = {}",
                self.store_int_or_num(
                    Kind::Int,
                    format!("{target} + (ok and ((a or 0) + (b or 0) + (c or 0)) or 5)")
                )
            ),
            3 => format!(
                "local s = utf8.char({code}, {})\n  {target} = {}",
                self.rng.pick(&["66", "233", "8364", "1114111"]),
                self.store_int_or_num(Kind::Int, format!("{target} + #s + utf8.len(s)"))
            ),
            4 => format!(
                "local ok, a, b = pcall(utf8.offset, {subject}, {other}, {position})\n  {target} = {}",
                self.store_int_or_num(
                    Kind::Int,
                    format!("{target} + (ok and (a or -1) or 3) + (ok and b or 0)")
                )
            ),
            5 => {
                let lax = if self.dialect.utf8_lax && self.rng.chance(1, 2) {
                    ", true"
                } else {
                    ""
                };
                format!(
                    "local c = 0\n  local ok = pcall(function() for p, cp in utf8.codes({subject}{lax}) do c = c + p + cp end end)\n  {target} = {}",
                    self.store_int_or_num(Kind::Int, format!("{target} + c + (ok and 1 or 2)"))
                )
            }
            6 => {
                // The format, its arguments and how many values it unpacks.
                let (format, arguments, values) = *self.rng.pick(&[
                    ("<i4", "N", 1),
                    (">I2", "N", 1),
                    ("<i1", "N", 1),
                    ("=i8", "N", 1),
                    ("<j", "N", 1),
                    ("<i2I2c3", "N, O, \"abc\"", 3),
                    (">!4 b I2 i4", "O, N, C", 3),
                    ("<i4 x i2", "N, O", 2),
                    ("<f", "1.5", 1),
                    (">d", "-2.25", 1),
                ]);
                let arguments = arguments
                    .replace('N', number)
                    .replace('O', other)
                    .replace('C', code);
                let names = ["a, p", "a, b, p", "a, b, c, p"][values - 1];
                let second = if values > 1 {
                    " + math.floor(b) % 100"
                } else {
                    ""
                };
                format!(
                    "local ok, x = pcall(string.pack, \"{format}\", {arguments})\n  if ok then local {names} = string.unpack(\"{format}\", x)\n    {target} = {} end",
                    self.store_int_or_num(
                        Kind::Int,
                        format!("{target} + #x + p + math.floor(a) % 1000{second}")
                    )
                )
            }
            7 => {
                let (format, value) = *self.rng.pick(&[
                    ("<s1", "\"ab\""),
                    (">s2", "\"\\0\\1\""),
                    ("<s4", "\"\""),
                    ("z", "\"abc\""),
                    ("z", "\"a\\0b\""),
                    ("c4", "\"abc\""),
                    ("c2", "\"abc\""),
                    ("<s", "\"x\""),
                ]);
                format!(
                    "local ok, x = pcall(string.pack, \"{format}\", {value})\n  if ok then local a, p = string.unpack(\"{format}\", x)\n    {target} = {} else {target} = {} end",
                    self.store_int_or_num(Kind::Int, format!("{target} + #x + p + #a")),
                    self.store_int_or_num(Kind::Int, format!("{target} + 11")),
                )
            }
            8 => {
                let data = *self.rng.pick(&[
                    "\"\\1\\2\\3\\4\\5\\6\\7\\8\"",
                    "\"\\255\\255\\255\\255\"",
                    "\"abc\\0def\"",
                    "\"\\3abcd\"",
                    "\"\"",
                ]);
                let (format, init) = *self.rng.pick(&[
                    ("<i4", "1"),
                    ("<i4", "2"),
                    (">I2", "-2"),
                    ("<i8", "1"),
                    ("<I3", "3"),
                    ("z", "1"),
                    ("<s1", "1"),
                    ("<i2i2", "1"),
                    ("<i1", "9"),
                    ("<i4", "0"),
                ]);
                format!(
                    "local ok, a, b, c = pcall(string.unpack, \"{format}\", {data}, {init})\n  {target} = {}",
                    self.store_int_or_num(
                        Kind::Int,
                        format!(
                            "{target} + (ok and ((type(a) == \"number\" and math.floor(a) % 1000 or #a) + (type(b) == \"number\" and b or 0)) or 13)"
                        )
                    )
                )
            }
            9 => {
                let format = *self.rng.pick(&[
                    "<i4i2",
                    "i1",
                    "<!4 i1 i4",
                    "!8 i1 d",
                    "c10",
                    "<j",
                    "x",
                    "z",
                    "s1",
                    "<!2 i1 Xi8",
                    "q",
                ]);
                format!(
                    "local ok, size = pcall(string.packsize, \"{format}\")\n  {target} = {}",
                    self.store_int_or_num(Kind::Int, format!("{target} + (ok and size or 17)"))
                )
            }
            _ => format!(
                "local s = utf8.char({code}, {})\n  local a, b = utf8.codepoint(s, 1, -1)\n  {target} = {}",
                self.rng.pick(&["66", "233", "8364", "1114111"]),
                self.store_int_or_num(Kind::Int, format!("{target} + a + b + utf8.len(s)"))
            ),
        };
        self.leaf(format!("do\n  {body}\nend"))
    }

    /// A traversal of a table built on the spot with `pairs` or `next`, folded
    /// into an integer variable. The table has an array part, string keys,
    /// sparse integer keys, a float and a boolean key, or a mix. The body may
    /// clear the key it reached (`t[k] = nil`), overwrite its value, copy it
    /// to another table, or step by hand with `next (t, k)`. Every write goes
    /// to the key just reached, which Lua allows, and every result is a sum or
    /// a count, so the order the keys come in (unspecified in Lua) never shows
    /// and the output can be compared with the stock interpreter.
    #[allow(clippy::too_many_lines)]
    fn traversal_call(&mut self) -> Node {
        let targets = self.writable(Kind::Int);
        if targets.is_empty() || !self.charge(48) {
            return self.assignment();
        }
        let target = self.rng.pick(&targets).name.clone();
        let count = self.rng.range(8, 40);
        let fill = match self.rng.index(6) {
            0 => format!("for i = 1, {count} do tq[i] = i * 3 % 11 end"),
            1 => format!("for i = 1, {count} do tq[\"k\" .. i] = i end"),
            2 => format!(
                "for i = 1, {count} do tq[i] = i % 5 tq[\"k\" .. i] = i end tq[1.5] = 7 tq[true] = 3"
            ),
            3 => format!("for i = 1, {count} do tq[i * 100] = i end"),
            4 => "tq = {10, 20, 30, a = 1, b = 2, [2.5] = 3, [false] = 4}".to_owned(),
            _ => format!(
                "for i = 1, {count} do tq[i] = i end for i = 2, {count}, 3 do tq[i] = nil end tq.tail = 9"
            ),
        };
        // What a key adds, whatever its type and the order it comes in.
        let weight = "(type(k) == \"number\" and math.floor(k * 2) % 7 or (type(k) == \"string\" and #k or 1))";
        let iterator = if self.rng.chance(1, 3) {
            "next, tq"
        } else {
            "pairs(tq)"
        };
        self.feature("table_traversal");
        let body = match self.rng.index(8) {
            0 => format!("for k, v in {iterator} do c = c + v + {weight} end"),
            1 => format!(
                "for k, v in {iterator} do c = c + v tq[k] = nil end\n  c = c + (next(tq) == nil and 1 or 100)"
            ),
            2 => format!(
                "for k, v in {iterator} do tq[k] = v * 2 + 1 c = c + 1 end\n  for k, v in pairs(tq) do c = c + v end"
            ),
            3 => format!(
                "for k, v in {iterator} do if v % 2 == 0 then tq[k] = nil else tq[k] = v + 1 end end\n  for k, v in pairs(tq) do c = c + v + {weight} end"
            ),
            4 => format!(
                "local tu = {{}}\n  for k, v in {iterator} do tu[k] = v c = c + 1 end\n  for k, v in pairs(tu) do c = c + v + {weight} tu[k] = nil end\n  c = c + (next(tu) == nil and 1 or 100)"
            ),
            5 => format!(
                "local tu = {{}}\n  for k, v in {iterator} do tu[k] = v end\n  for k, v in pairs(tq) do for k2, v2 in pairs(tu) do c = c + v2 % 3 end tq[k] = nil end\n  c = c + (next(tq) == nil and 1 or 100)"
            ),
            6 => format!(
                "local k, v = next(tq)\n  while k ~= nil do c = c + v + {weight} if v % 2 == 0 then tq[k] = nil end k, v = next(tq, k) end"
            ),
            _ => "for r = 1, 3 do for k, v in pairs(tq) do c = c + v if v > 3 then tq[k] = v - 3 else tq[k] = nil end end end".to_owned(),
        };
        let folded = self.store_int_or_num(Kind::Int, format!("{target} + c"));
        self.leaf(format!(
            "do\n  local tq, c = {{}}, 0\n  {fill}\n  {body}\n  {target} = {folded}\nend"
        ))
    }

    /// A draw from `math.random` into a variable. A draw changes the
    /// generator, so it is a statement of its own, never an operand. The
    /// prelude seeds the generator.
    fn random_draw(&mut self) -> Node {
        let kind = *self.rng.pick(&[Kind::Int, Kind::Num]);
        let targets = self.writable(kind);
        if targets.is_empty() {
            return self.kernel_fallback();
        }
        let target = self.rng.pick(&targets).name.clone();
        let draw = if kind == Kind::Num {
            if self.rng.chance(1, 2) {
                "math.random()".to_owned()
            } else {
                "(math.random() * 100)".to_owned()
            }
        } else {
            match self.rng.index(3) {
                0 => format!("math.random({})", self.rng.range(1, 1000)),
                1 => {
                    let low = self.rng.range(-50, 50);
                    format!("math.random({low}, {})", low + self.rng.range(0, 200))
                }
                _ => "math.random(7)".to_owned(),
            }
        };
        self.feature("math_random");
        let value = self.store_int_or_num(kind, draw);
        let features = self.take_features();
        Node::leaf(format!("{target} = {value}")).with_features(features)
    }

    fn assignment(&mut self) -> Node {
        let kind = *self.rng.pick(&ALL_KINDS);
        let targets = self.writable(kind);
        if targets.is_empty() {
            return self.local_declaration();
        }
        let target = self.rng.pick(&targets).name.clone();
        let value = self.stored(kind);
        if self.rng.chance(1, 4) && matches!(kind, Kind::Int | Kind::Num) {
            self.feature("compound_update");
            let delta = self.expr(kind, 1);
            let text = self.store_int_or_num(kind, format!("{target} + {delta}"));
            return self.leaf(format!("{target} = {text}"));
        }
        self.leaf(format!("{target} = {value}"))
    }

    /// An expression of `kind` in a form safe to store.
    fn stored(&mut self, kind: Kind) -> String {
        match kind {
            Kind::Arr | Kind::Rec | Kind::Obj if self.rng.chance(1, 2) => {
                self.feature("aliasing");
                self.pick_var(kind).unwrap_or_else(|| self.initial(kind))
            }
            Kind::Arr | Kind::Rec | Kind::Obj => self.initial(kind),
            Kind::Int | Kind::Num => {
                let value = self.expr(kind, 0);
                self.store_int_or_num(kind, value)
            }
            Kind::Str => {
                let value = self.expr(Kind::Str, 0);
                format!("string.sub({value}, 1, 48)")
            }
            Kind::Bool => self.expr(Kind::Bool, 0),
        }
    }

    fn store_int_or_num(&self, kind: Kind, text: String) -> String {
        if kind == Kind::Int {
            self.wrap_int(text)
        } else {
            text
        }
    }

    fn local_declaration(&mut self) -> Node {
        let kind = *self.rng.pick(&ALL_KINDS);
        let name = self.fresh("v");
        let value = if matches!(kind, Kind::Arr | Kind::Rec | Kind::Obj) {
            self.initial(kind)
        } else {
            self.stored(kind)
        };
        self.declare(&name, kind, true);
        self.leaf(format!("local {name} = {value}"))
    }

    fn observation(&mut self) -> Node {
        // Other statements fall back to an observation; where printing is
        // off, they get an assignment instead.
        if self.frame.multiplier > 4 || self.frame.quiet {
            return self.assignment();
        }
        let tag = self.tag("o");
        let mut values = Vec::new();
        match self.rng.index(6) {
            0 => {
                let callables: Vec<Function> =
                    self.functions.iter().filter(|f| f.pure).cloned().collect();
                if let Some(function) = self.pick_affordable(&callables) {
                    self.feature("multiple_results");
                    values.push(self.call(&function));
                    if self.rng.chance(1, 2) {
                        // A call that is not last is cut to one value.
                        values.insert(0, self.call(&function));
                        self.feature("result_adjustment");
                    }
                }
            }
            1 => {
                self.feature("varargs");
                let count = self.rng.index(4);
                let args: Vec<String> = (0..count).map(|_| self.vararg_value()).collect();
                let callee = if self.rng.chance(1, 2) { "va" } else { "packn" };
                values.push(format!("{callee}({})", args.join(", ")));
            }
            2 => {
                let kind = *self.rng.pick(&[Kind::Arr, Kind::Rec, Kind::Obj]);
                if let Some(var) = self.pick_var(kind) {
                    values.push(Self::observe(&var, kind));
                }
                if kind == Kind::Arr
                    && let Some(var) = self.pick_var(Kind::Arr)
                {
                    let unpack = if self.dialect.table_unpack {
                        "table.unpack"
                    } else {
                        "unpack"
                    };
                    self.feature("unpack");
                    values.push(format!("{unpack}({var}, 1, math.min(3, #{var}))"));
                }
            }
            _ => {
                for _ in 0..self.rng.range(1, 3) {
                    let kind = *self.rng.pick(&SCALARS);
                    values.push(self.expr(kind, 0));
                }
            }
        }
        if values.is_empty() {
            values.push(self.expr(Kind::Int, 0));
        }
        self.leaf(format!("out({tag}, {})", values.join(", ")))
    }

    fn vararg_value(&mut self) -> String {
        match self.rng.index(5) {
            0 => "nil".to_owned(),
            1 => self.expr(Kind::Str, 1),
            2 => self.expr(Kind::Bool, 1),
            _ => self.expr(Kind::Int, 1),
        }
    }

    fn if_statement(&mut self) -> Node {
        self.feature("branches");
        let condition = self.condition();
        let then = self.block();
        let mut node_text = format!("if {condition} then");
        let mut body = then;
        if self.rng.chance(1, 3) {
            let other = self.condition();
            let branch = self.block();
            body.push(Node::leaf(format!("elseif {other} then")));
            body.extend(branch);
            self.feature("elseif");
        }
        if self.rng.chance(1, 2) {
            let branch = self.block();
            body.push(Node::leaf("else"));
            body.extend(branch);
        }
        // `elseif` and `else` are rendered as indented leaves inside the
        // block; they are pinned so the reducer cannot drop them alone.
        for node in &mut body {
            if node.head == "else" || node.head.starts_with("elseif ") {
                node.pinned = true;
            }
        }
        let features = self.take_features();
        node_text.shrink_to_fit();
        Node::block(node_text, body, "end").with_features(features)
    }

    /// A boolean that short-circuits now and then.
    fn condition(&mut self) -> String {
        if self.rng.chance(1, 4) {
            self.feature("short_circuit");
            let left = self.expr(Kind::Bool, 1);
            let right = self.expr(Kind::Bool, 1);
            let op = if self.rng.chance(1, 2) { "and" } else { "or" };
            return format!("({left}) {op} ({right})");
        }
        self.expr(Kind::Bool, 0)
    }

    fn inner_loop_iterations(shape: usize, count: i64) -> u64 {
        if shape == 3 {
            // `count` only bounds the other loop shapes. `ipairs` reads
            // the live sequence, including elements inserted by its body
            // or an impure function, so charge its maximum length.
            ARRAY_LENGTH_LIMIT
        } else {
            u64::try_from(count.max(1)).unwrap_or(1)
        }
    }

    fn inner_loop(&mut self) -> Node {
        let shape = self.rng.index(4);
        let count = self.rng.range(0, 4);
        let saved = self.frame.clone();
        self.frame.multiplier = self
            .frame
            .multiplier
            .saturating_mul(Self::inner_loop_iterations(shape, count));
        self.frame.loops += 1;
        self.frame.continue_label = None;
        self.frame.depth += 1;
        self.push_scope();
        let (head, tail, prelude) = match shape {
            0 => {
                let var = self.fresh("k");
                self.declare(&var, Kind::Int, false);
                let step = *self.rng.pick(&[1, 1, 2, -1]);
                let (first, last) = if step > 0 { (1, count) } else { (count, 1) };
                self.feature("numeric_for");
                (
                    format!("for {var} = {first}, {last}, {step} do"),
                    "end".to_owned(),
                    None,
                )
            }
            1 => {
                let var = self.fresh("w");
                self.feature("while");
                self.declare(&var, Kind::Int, false);
                (
                    format!("while {var} < {count} do"),
                    "end".to_owned(),
                    Some((
                        var.clone(),
                        format!("local {var} = 0"),
                        format!("{var} = {var} + 1"),
                    )),
                )
            }
            2 => {
                let var = self.fresh("w");
                self.feature("repeat");
                self.declare(&var, Kind::Int, false);
                (
                    "repeat".to_owned(),
                    format!("until {var} >= {}", count.max(1)),
                    Some((
                        var.clone(),
                        format!("local {var} = 0"),
                        format!("{var} = {var} + 1"),
                    )),
                )
            }
            _ => {
                let key = self.fresh("k");
                let value = self.fresh("e");
                self.declare(&key, Kind::Int, false);
                self.declare(&value, Kind::Int, false);
                self.feature("ipairs");
                let array = self
                    .pick_var(Kind::Arr)
                    .unwrap_or_else(|| "{1, 2, 3}".to_owned());
                (
                    format!("for {key}, {value} in ipairs({array}) do"),
                    "end".to_owned(),
                    None,
                )
            }
        };
        let label = if self.dialect.goto && shape < 2 && self.rng.chance(1, 4) {
            let label = self.fresh("continue");
            self.frame.continue_label = Some(label.clone());
            Some(label)
        } else {
            None
        };
        let mut body = Vec::new();
        if let Some((_, _, increment)) = &prelude {
            body.push(Node::leaf(increment.clone()).pinned());
        }
        let statements = self.statements();
        if let Some(label) = label {
            // Locals stay inside a `do` block, so the `goto` never jumps
            // into their scope (Lua 5.5 rejects that even at a block's end).
            self.feature("goto_continue");
            body.push(Node::block("do", statements, "end"));
            body.push(Node::leaf(format!("::{label}::")).pinned());
        } else {
            body.extend(statements);
        }
        self.pop_scope();
        self.frame = saved;
        let features = self.take_features();
        let loop_node = Node::block(head, body, tail).with_features(features);
        match prelude {
            Some((_, declaration, _)) => Node::block(
                "do",
                vec![Node::leaf(declaration).pinned(), loop_node],
                "end",
            ),
            None => loop_node,
        }
    }

    fn array_mutation(&mut self) -> Node {
        let arrays = self.mutable_tables(Kind::Arr);
        if arrays.is_empty() {
            return self.assignment();
        }
        let t = self.rng.pick(&arrays).name.clone();
        let value = self.stored(Kind::Int);
        let index = self.expr(Kind::Int, 1);
        // The positional and observed forms are the expanded probes: a run
        // that turns them off keeps the stream of the older versions.
        let expanded = if self.expanded_probes { 8 } else { 0 };
        let text = match self.rng.index(9 + expanded) {
            0 | 1 => {
                self.feature("table_index_write");
                format!("{t}[array_index({index}, #{t})] = {value}")
            }
            // Positional calls in the middle of the list, a pop whose value
            // is observed, and a list that grows to its limit by appends.
            9 => {
                self.feature("table_insert");
                format!(
                    "if #{t} < {ARRAY_LENGTH_LIMIT} then table.insert({t}, array_index({index}, #{t}), {value}) end"
                )
            }
            10 => {
                self.feature("table_remove");
                format!("if #{t} > 1 then table.remove({t}, array_index({index}, #{t})) end")
            }
            11 if !self.frame.quiet && self.frame.multiplier <= 4 => {
                self.feature("table_remove");
                let tag = self.tag("R");
                format!("if #{t} > 1 then local popped = table.remove({t}) out({tag}, popped) end")
            }
            11 | 12 => {
                self.feature("table_insert");
                format!(
                    "for _ = 1, 3 do if #{t} < {ARRAY_LENGTH_LIMIT} then table.insert({t}, {value}) end end"
                )
            }
            // Stores at the two ends of the list, which a trace does in place:
            // a pop by nil whose value is read first, pops in a run, an
            // append next to a pop, and a plain overwrite of the last slot.
            13 => {
                self.feature("table_pop_nil");
                format!("if #{t} > 1 then {t}[#{t}] = nil end")
            }
            14 if !self.frame.quiet && self.frame.multiplier <= 4 => {
                self.feature("table_pop_nil");
                let tag = self.tag("N");
                format!(
                    "if #{t} > 1 then local top = {t}[#{t}] {t}[#{t}] = nil out({tag}, top) end"
                )
            }
            14 | 15 => {
                self.feature("table_pop_nil");
                format!("for _ = 1, 3 do if #{t} > 1 then {t}[#{t}] = nil end end")
            }
            16 => {
                self.feature("table_append");
                format!(
                    "if #{t} < {ARRAY_LENGTH_LIMIT} then {t}[#{t} + 1] = {value} if #{t} > 2 then {t}[#{t}] = nil end {t}[#{t} + 1] = {value} end"
                )
            }
            2 => {
                self.feature("table_insert");
                format!("if #{t} < {ARRAY_LENGTH_LIMIT} then table.insert({t}, {value}) end")
            }
            3 => {
                self.feature("table_insert");
                format!("if #{t} < {ARRAY_LENGTH_LIMIT} then table.insert({t}, 1, {value}) end")
            }
            4 => {
                self.feature("table_remove");
                format!("if #{t} > 1 then table.remove({t}) end")
            }
            5 => {
                self.feature("table_remove");
                format!("if #{t} > 1 then table.remove({t}, 1) end")
            }
            6 => {
                // A comparison callback runs many times per sort. Charge
                // the quadratic upper bound for our bounded arrays, rather
                // than treating the whole sort as one cheap statement.
                if !self.charge(ARRAY_LENGTH_LIMIT * ARRAY_LENGTH_LIMIT) {
                    self.feature("table_index_write");
                    return self.leaf(format!("{t}[1] = {value}"));
                }
                self.feature("table_sort");
                let ascending = self.rng.chance(1, 2);
                match (self.dialect.integers, ascending) {
                    (true, true) => format!("table.sort({t}, numeric_less)"),
                    (true, false) => {
                        format!("table.sort({t}, function(a, b) return numeric_less(b, a) end)")
                    }
                    (false, true) => format!("table.sort({t})"),
                    (false, false) => format!("table.sort({t}, function(a, b) return a > b end)"),
                }
            }
            7 => {
                self.feature("table_append");
                format!("if #{t} < {ARRAY_LENGTH_LIMIT} then {t}[#{t} + 1] = {value} end")
            }
            _ => {
                if self.dialect.table_move {
                    self.feature("table_move");
                    format!("table.move({t}, 1, math.min(2, #{t}), 2)")
                } else {
                    self.feature("table_index_write");
                    format!("{t}[1] = {value}")
                }
            }
        };
        self.leaf(text)
    }

    fn record_mutation(&mut self) -> Node {
        let records = self.mutable_tables(Kind::Rec);
        if records.is_empty() {
            return self.assignment();
        }
        let r = self.rng.pick(&records).name.clone();
        let value = self.stored(Kind::Num);
        let text = match self.rng.index(5) {
            0 | 1 => {
                let field = *self.rng.pick(&["x", "y", "z"]);
                self.feature("field_write");
                format!("{r}.{field} = {value}")
            }
            2 => {
                let key = self.dynamic_key();
                self.feature("dynamic_key_write");
                format!("{r}[{key}] = {value}")
            }
            3 => {
                let field = *self.rng.pick(&["w", "extra", "k2"]);
                self.feature("field_delete");
                format!("{r}.{field} = nil")
            }
            _ => {
                self.feature("rawset");
                format!("rawset({r}, \"w\", {value})")
            }
        };
        self.leaf(text)
    }

    /// A string key drawn from a small set, built at run time.
    fn dynamic_key(&mut self) -> String {
        // Kernels keep to constant keys and the `and`/`or` choice: a key
        // built with `..` stops the trace recorder.
        let choices = if self.hot_avoids_unsupported() {
            1
        } else if self.frame.kernel {
            2
        } else {
            3
        };
        match self.rng.index(choices) {
            0 => format!("\"{}\"", self.rng.pick(&["x", "y", "z", "w"])),
            1 => {
                let choice = self.expr(Kind::Bool, 1);
                format!("(({choice}) and \"x\" or \"y\")")
            }
            _ => {
                let index = self.expr(Kind::Int, 1);
                format!("\"k\" .. (({index}) % 4)")
            }
        }
    }

    fn object_statement(&mut self) -> Node {
        let objects = self.mutable_tables(Kind::Obj);
        if objects.is_empty() {
            return self.assignment();
        }
        let o = self.rng.pick(&objects).name.clone();
        let delta = self.expr(Kind::Num, 1);
        match self.rng.index(4) {
            0 => {
                self.feature("method_call");
                let targets = self.call_targets(Kind::Num);
                if let Some(target) = targets.first().map(|var| var.name.clone())
                    && self.rng.chance(1, 2)
                {
                    return self.leaf(format!("{target} = {o}:bump({delta})"));
                }
                self.leaf(format!("{o}:bump({delta})"))
            }
            1 => {
                self.feature("metatable_switch");
                let class = *self.rng.pick(&["ClassA", "ClassB"]);
                self.leaf(format!("setmetatable({o}, {class})"))
            }
            2 => {
                self.feature("field_write");
                self.leaf(format!("{o}.v = {delta}"))
            }
            _ => {
                self.feature("method_call");
                self.leaf(format!("{o}.bump({o}, {delta})"))
            }
        }
    }

    fn impure_call(&mut self) -> Node {
        let roll = self.rng.index(4);
        if roll == 0 && self.charge(2) {
            self.feature("closures");
            let delta = self.expr(Kind::Int, 1);
            let targets = self.call_targets(Kind::Int);
            if let Some(target) = targets.first().map(|var| var.name.clone()) {
                let value = self.wrap_int(format!("counter({delta})"));
                return self.leaf(format!("{target} = {value}"));
            }
            return self.leaf(format!("counter({delta})"));
        }
        if roll == 1 && self.frame.level == 0 && !self.frame.in_coroutine {
            self.feature("coroutine_wrap");
            let targets = self.call_targets(Kind::Int);
            if let Some(target) = targets.first().map(|var| var.name.clone()) {
                let value = self.wrap_int("gen()".to_owned());
                return self.leaf(format!("{target} = {value}"));
            }
        }
        let callables: Vec<Function> = self.functions.clone();
        let Some(function) = self.pick_affordable(&callables) else {
            return self.observation();
        };
        let call = self.call(&function);
        if function.results.len() > 1 {
            let targets: Vec<Option<String>> = function
                .results
                .iter()
                .map(|&kind| {
                    let vars = self.call_targets(kind);
                    (!vars.is_empty()).then(|| self.rng.pick(&vars).name.clone())
                })
                .collect();
            if let (Some(Some(a)), Some(Some(b))) = (targets.first(), targets.get(1))
                && a != b
            {
                self.feature("multiple_assignment");
                return self.leaf(format!("{a}, {b} = {call}"));
            }
            let first = self.fresh("r");
            let second = self.fresh("r");
            self.declare(&first, function.results[0], true);
            self.declare(&second, function.results[1], true);
            self.feature("multiple_results");
            return self.leaf(format!("local {first}, {second} = {call}"));
        }
        let kind = function.results[0];
        let targets = self.call_targets(kind);
        if !targets.is_empty() && kind != Kind::Int && self.rng.chance(2, 3) {
            let target = self.rng.pick(&targets).name.clone();
            let value = if kind == Kind::Str {
                format!("string.sub({call}, 1, 48)")
            } else {
                call
            };
            return self.leaf(format!("{target} = {value}"));
        }
        if kind == Kind::Int && !targets.is_empty() {
            let target = self.rng.pick(&targets).name.clone();
            let value = self.wrap_int(call);
            return self.leaf(format!("{target} = {value}"));
        }
        self.leaf(call)
    }

    fn pick_affordable(&mut self, functions: &[Function]) -> Option<Function> {
        let affordable: Vec<&Function> = functions
            .iter()
            .filter(|function| {
                let added = function.cost.saturating_mul(self.frame.multiplier);
                let limit = if self.frame.level == 0 {
                    ITERATION_COST_LIMIT
                } else {
                    FUNCTION_COST_LIMIT
                };
                self.cost.saturating_add(added) <= limit
                    && (!self.frame.pure || function.pure)
                    && (!self.frame.in_coroutine || function.cost < 100)
            })
            .collect();
        if affordable.is_empty() {
            return None;
        }
        let function = (*self.rng.pick(&affordable)).clone();
        self.cost += function.cost.saturating_mul(self.frame.multiplier);
        Some(function)
    }

    /// A call with arguments of the right kinds.
    fn call(&mut self, function: &Function) -> String {
        self.feature("calls");
        let args: Vec<String> = function
            .params
            .iter()
            .map(|param| {
                let value = self.expr(param.kind, 1);
                match param.bound {
                    // Int-kind values can become NaN or infinity through
                    // arithmetic. Modulo alone would leave NaN and let
                    // recursive counters run forever.
                    Some(bound) => format!("(array_index({value}, {bound}) - 1)"),
                    None => value,
                }
            })
            .collect();
        format!("{}({})", function.name, args.join(", "))
    }

    fn protected_call(&mut self) -> Node {
        let tag = self.tag("P");
        let saved = self.frame.clone();
        self.frame.depth += 1;
        self.frame.loops = 0;
        self.frame.continue_label = None;
        self.frame.level += 1;
        self.frame.pure = false;
        // A yield inside the protected function yields across `pcall`.
        self.frame.in_coroutine &=
            self.dialect.yield_across_pcall && self.allowed("yield_across_pcall");
        self.push_scope();
        let mut body = if self.rng.chance(1, 2) {
            self.statements()
        } else {
            Vec::new()
        };
        let raise = self.raise();
        let conditional = self.frame.counter.is_some() && self.rng.chance(1, 2);
        if conditional {
            let (counter, _) = saved.counter.clone().expect("checked above");
            let period = self.rng.range(2, 9);
            body.push(Node::block(
                format!("if {counter} % {period} == 0 then"),
                vec![Node::leaf(raise)],
                "end",
            ));
            body.push(Node::leaf(format!("return {}", self.expr(Kind::Int, 1))));
        } else {
            body.push(Node::leaf(raise));
        }
        self.pop_scope();
        self.frame = saved;
        self.feature("pcall");
        let features = self.take_features();
        if self.allowed("xpcall_lua_handler") && self.rng.chance(1, 4) {
            let handler = "function(m) return \"H:\" .. tostring(string_number(ek(m))) end";
            self.feature("xpcall");
            let mut node = Node::block(
                "local ok, e = xpcall(function()",
                body,
                format!("end, {handler})"),
            );
            node.features = features;
            return Node::block(
                "do",
                vec![node, Node::leaf(format!("out({tag}, ok, ek(e))")).pinned()],
                "end",
            );
        }
        let mut node = Node::block("local ok, e = pcall(function()", body, "end)");
        node.features = features;
        Node::block(
            "do",
            vec![node, Node::leaf(format!("out({tag}, ok, ek(e))")).pinned()],
            "end",
        )
    }

    /// A statement that raises an error, chosen from controlled families.
    fn raise(&mut self) -> String {
        let int = self.expr(Kind::Int, 1);
        let mut choices: Vec<(&'static str, String)> = vec![
            ("error_string", format!("error(\"boom\" .. tostring(string_number({int})))")),
            ("error_object", format!("error({{code = {int}}})")),
            ("error_level0", "error(\"plain\", 0)".to_owned()),
            ("error_level2", "error(\"caller\", 2)".to_owned()),
            ("error_number", "error(42)".to_owned()),
            ("error_nil_arith", "local z = nil\nlocal w = z + 1".to_owned()),
            ("error_nil_index", "local q = {}\nlocal w = q.a.b".to_owned()),
            ("error_call_field", "local q = {}\nq.nofn()".to_owned()),
            ("error_call_method", "local q = {}\nq:nometh()".to_owned()),
            ("error_call_global", "undefined_global(1)".to_owned()),
            ("error_concat", "local w = {} .. \"x\"".to_owned()),
            ("error_compare", "local w = {} < {}".to_owned()),
            ("error_compare_mixed", "local w = 1 < \"2\"".to_owned()),
            ("error_length", "local n = 5\nlocal w = #n".to_owned()),
            ("error_bad_argument", "string.rep()".to_owned()),
            ("error_bad_argument", "local w = (\"x\"):rep({})".to_owned()),
            ("error_bad_argument", "setmetatable(1, {})".to_owned()),
            ("error_bad_argument", "local w = string.char(256)".to_owned()),
            ("error_bad_argument", "local w = table.concat({{}})".to_owned()),
            ("error_string_arith", "local w = \"abc\" + 1".to_owned()),
            (
                "error_in_metamethod",
                "local w = setmetatable({}, {__index = function(t, k) error(\"idx \" .. k) end}).foo"
                    .to_owned(),
            ),
            ("error_call_table", "local w = setmetatable({}, {})()".to_owned()),
            ("error_tostring_object", "error(setmetatable({code = 7}, {__tostring = function() return \"TS\" end}))".to_owned()),
            ("assert", "assert(false, \"asserted\")".to_owned()),
            ("assert", "assert(nil)".to_owned()),
            ("nested_pcall", "error(select(2, pcall(error, \"inner\")))".to_owned()),
            ("error_in_callee", "local function deep(n) if n == 0 then error(\"deep\") end return deep(n - 1) end\ndeep(3)".to_owned()),
        ];
        if self.dialect.integers {
            choices.push(("error_idiv_zero", format!("local w = {int} // 0")));
            choices.push(("error_mod_zero", format!("local w = {int} % 0")));
            choices.push(("error_no_integer", "local w = 1.5 | 0".to_owned()));
            choices.push((
                "error_no_integer",
                "local w = string.format(\"%d\", 1.5)".to_owned(),
            ));
        }
        choices.retain(|(feature, _)| self.allowed(feature));
        let (feature, text) = choices.swap_remove(self.rng.index(choices.len()));
        self.feature(feature);
        text
    }

    fn coroutine_block(&mut self) -> Node {
        let tag = self.tag("C");
        let co = self.fresh("co");
        let yields = self.rng.range(1, 4);
        let saved = self.frame.clone();
        let saved_cost = self.cost;
        self.frame = Frame {
            level: saved.level + 1,
            pure: false,
            loops: 0,
            continue_label: None,
            multiplier: 1,
            counter: None,
            in_coroutine: true,
            depth: saved.depth + 1,
            kernel: false,
            kernel_locals: false,
            quiet: false,
        };
        self.cost = 0;
        self.push_scope();
        let a = self.fresh("a");
        let b = self.fresh("b");
        self.declare(&a, Kind::Int, true);
        self.declare(&b, Kind::Int, true);
        let mut body = self.statements();
        let k = self.fresh("k");
        let sum = self.wrap_int(format!("{a} + {k} * {b}"));
        let mut loop_body = vec![Node::leaf(format!(
            "local got = coroutine.yield({sum}, {k})"
        ))];
        loop_body.push(Node::leaf(format!(
            "if type(got) == \"number\" then {a} = {} end",
            self.wrap_int(format!("{a} + got"))
        )));
        if self.dialect.yield_across_pcall
            && self.allowed("yield_across_pcall")
            && self.rng.chance(1, 3)
        {
            self.feature("yield_across_pcall");
            loop_body.push(Node::leaf(
                "local okp, vp = pcall(function() return coroutine.yield(\"p\") end)",
            ));
        }
        body.push(Node::block(
            format!("for {k} = 1, {yields} do"),
            loop_body,
            "end",
        ));
        if self.allowed("coroutine_error") && self.rng.chance(1, 4) {
            self.feature("coroutine_error");
            body.push(Node::leaf("error(\"co fail\")"));
        }
        body.push(Node::leaf(format!("return \"done\", {a}")));
        self.pop_scope();
        self.frame = saved;
        self.cost = saved_cost;
        let start_a = self.expr(Kind::Int, 1);
        let start_b = self.expr(Kind::Int, 1);
        let mut nodes = vec![Node::block(
            format!("local {co} = coroutine.create(function({a}, {b})"),
            body,
            "end)",
        )];
        nodes.push(Node::leaf(format!(
            "out({tag}, coroutine.resume({co}, {start_a}, {start_b}))"
        )));
        for _ in 0..(yields + 2) {
            let send = self.expr(Kind::Int, 1);
            nodes.push(Node::leaf(format!(
                "out({tag}, coroutine.status({co}), coroutine.resume({co}, {send}))"
            )));
        }
        self.feature("coroutines");
        if self.dialect.integers && self.allowed("coroutine_isyieldable") {
            self.feature("coroutine_isyieldable");
            nodes.push(Node::leaf(format!("out({tag}, coroutine.isyieldable())")));
        }
        let features = self.take_features();
        Node::block("do", nodes, "end").with_features(features)
    }

    /// A coroutine that yields from inside a metamethod, an iterator or a C
    /// function's callback, then is resumed until it finishes. Stock 5.2+
    /// and `LuaJIT` allow some of these and reject others with an exact
    /// message; 5.1 rejects them all.
    fn yield_boundary(&mut self) -> Node {
        let tag = self.tag("Y");
        let co = self.fresh("co");
        let mut bodies: Vec<(&'static str, &'static str)> = vec![
            (
                "yield_index",
                "local o = setmetatable({}, {__index = function(t, k) return coroutine.yield(k) end})\nlocal v = o.field\nreturn \"index\", v",
            ),
            (
                "yield_newindex",
                "local log = {}\nlocal o = setmetatable({}, {__newindex = function(t, k, v) log[#log + 1] = coroutine.yield(v) end})\no.f = a\no.g = a + 1\nreturn \"newindex\", log[1], log[2]",
            ),
            (
                "yield_arith",
                "local o = setmetatable({}, {__add = function(x, y) return coroutine.yield(\"add\") end, __unm = function(x) return coroutine.yield(\"unm\") end})\nlocal r = o + a\nlocal s = -o\nreturn \"arith\", r, s",
            ),
            (
                "yield_compare",
                "local mt = {}\nmt.__lt = function(x, y) return coroutine.yield(\"lt\") end\nmt.__le = function(x, y) return coroutine.yield(\"le\") end\nmt.__eq = function(x, y) return coroutine.yield(\"eq\") end\nlocal o, p = setmetatable({}, mt), setmetatable({}, mt)\nreturn \"compare\", o < p, o <= p, o == p",
            ),
            (
                "yield_concat",
                "local o = setmetatable({}, {__concat = function(x, y) return coroutine.yield(\"concat\") end})\nlocal r = o .. \"s\"\nreturn \"concat\", r",
            ),
            (
                "yield_call",
                "local o = setmetatable({}, {__call = function(self, x) return coroutine.yield(x) end})\nlocal r = o(a)\nreturn \"call\", r",
            ),
            (
                "yield_iterator",
                "local n = 0\nfor v in function(_, c) c = (c or 0) + 1 if c > 3 then return nil end coroutine.yield(c) return c end do n = n + v end\nreturn \"iter\", n",
            ),
            (
                "yield_in_pcall_metamethod",
                "local o = setmetatable({}, {__index = function(t, k) return select(2, pcall(coroutine.yield, k)) end})\nreturn \"pcallmeta\", o.key",
            ),
            (
                "yield_sort_comparator",
                "local t = {3, 1, 2}\ntable.sort(t, function(x, y) coroutine.yield(\"sort\") return x < y end)\nreturn \"sort\", t[1]",
            ),
            (
                "yield_gsub_callback",
                "local s = string.gsub(\"ab\", \"%a\", function(c) return coroutine.yield(c) end)\nreturn \"gsub\", s",
            ),
            (
                "yield_tostring",
                "local o = setmetatable({}, {__tostring = function() return coroutine.yield(\"ts\") end})\nreturn \"tostring\", tostring(o)",
            ),
            (
                "yield_load_reader",
                "local f = load(function() return coroutine.yield(\"reader\") end)\nreturn \"load\", type(f)",
            ),
        ];
        if self.dialect.len_metamethod {
            bodies.push(("yield_len", "local o = setmetatable({}, {__len = function() return coroutine.yield(\"len\") end})\nreturn \"len\", #o"));
        }
        let (feature, body) = *self.rng.pick(&bodies);
        let body = if self.dialect.integers && feature == "yield_sort_comparator" {
            body.replace("return x < y", "return numeric_less(x, y)")
        } else {
            body.to_owned()
        };
        self.feature("yield_boundaries");
        self.feature(feature);
        let start = self.rng.range(1, 9);
        let mut lines = vec![format!("local {co} = coroutine.create(function(a)")];
        lines.extend(body.lines().map(|line| format!("  {line}")));
        lines.push("end)".to_owned());
        lines.push(format!("out({tag}, coroutine.resume({co}, {start}))"));
        for send in 0..4 {
            let value = self.rng.range(10, 99) + send;
            lines.push(format!(
                "out({tag}, coroutine.status({co}), coroutine.resume({co}, {value}))"
            ));
        }
        let text = format!(
            "do\n{}\nend",
            lines
                .iter()
                .map(|line| format!("  {line}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        self.leaf(text)
    }

    /// Standard library corner cases, each observed through `pcall`.
    fn stdlib_edge(&mut self) -> Node {
        let tag = self.tag("S");
        let mut cases: Vec<(&'static str, String)> = vec![
            ("gsub_bad_capture", "string.gsub(\"abc\", \"(a)\", \"%2\")".to_owned()),
            ("gsub_bad_escape", "string.gsub(\"abc\", \"a\", \"%\")".to_owned()),
            ("format_s_tostring", "string.format(\"%s|%5s\", setmetatable({}, {__tostring = function() return \"TS\" end}), \"x\")".to_owned()),
            ("gmatch_malformed", "(function() local n = 0 for w in string.gmatch(\"abc\", \"[a\") do n = n + 1 end return n end)()".to_owned()),
            ("gmatch_caret", "(function() local n = 0 for w in string.gmatch(\"aaa\", \"^a\") do n = n + 1 end return n end)()".to_owned()),
            ("find_malformed", "string.find(\"abc\", \"%\")".to_owned()),
            ("rep_negative", "string.rep(\"ab\", -1)".to_owned()),
        ];
        if self.dialect.xpcall_arguments {
            cases.push((
                "xpcall_load_reader",
                "(function() local parts = {\"return \", \"4\", \"2\"} local i = 0 local ok, f = xpcall(load, function(m) return \"H\" end, function() i = i + 1 return parts[i] end) return ok, type(f), f and f() end)()".to_owned(),
            ));
        }
        if !self.dialect.table_unpack {
            cases.push((
                "table_foreach",
                "table.foreach({10, 20, 30}, function(k, v) if k == 2 then return v * 3 end end)"
                    .to_owned(),
            ));
            cases.push((
                "table_foreachi",
                "table.foreachi({10, 20}, function(i, v) return nil end)".to_owned(),
            ));
            cases.push(("table_getn", "table.getn({1, 2, 3})".to_owned()));
        }
        if self.dialect.integers {
            let proxy = "setmetatable({}, {__index = function(t, i) return i * 10 end, __len = function() return 3 end})";
            cases.push(("table_concat_meta", format!("table.concat({proxy}, \",\")")));
            cases.push(("table_unpack_meta", format!("table.unpack({proxy})")));
            cases.push((
                "table_move_meta",
                format!("table.concat(table.move({proxy}, 1, 3, 1, {{}}), \",\")"),
            ));
            cases.push((
                "table_insert_meta",
                "(function() local store = {} local p = setmetatable({}, {__index = store, __newindex = function(t, k, v) rawset(store, k, v) end, __len = function() return #store end}) table.insert(p, 5) table.insert(p, 1, 6) return store[1], store[2], #store end)()".to_owned(),
            ));
            cases.push((
                "table_remove_meta",
                "(function() local store = {7, 8, 9} local p = setmetatable({}, {__index = store, __newindex = function(t, k, v) rawset(store, k, v) end, __len = function() return #store end}) return table.remove(p), table.remove(p, 1), store[1], store[2] end)()".to_owned(),
            ));
        }
        let (feature, expression) = cases.swap_remove(self.rng.index(cases.len()));
        self.feature("stdlib_edges");
        self.feature(feature);
        self.leaf(format!(
            "out({tag}, pcall(function() return {expression} end))"
        ))
    }

    fn yield_statement(&mut self) -> Node {
        self.feature("coroutine_yield");
        let value = self.expr(Kind::Int, 1);
        self.leaf(format!("coroutine.yield({value})"))
    }

    fn loop_exit(&mut self) -> Node {
        let condition = self.expr(Kind::Bool, 1);
        if let Some(label) = self.frame.continue_label.clone()
            && self.rng.chance(1, 2)
        {
            self.feature("goto_continue");
            return self.leaf(format!("if {condition} then goto {label} end"));
        }
        self.feature("break");
        self.leaf(format!("if {condition} then break end"))
    }

    /// A statement that runs once a hot loop has had time to compile, so
    /// the compiled code meets a new type, shape or metatable.
    fn phase_change(&mut self) -> Node {
        let (counter, iterations) = self.frame.counter.clone().expect("only called in a loop");
        let at = if iterations > 2 {
            self.rng.range(1, i64::from(iterations) - 1)
        } else {
            1
        };
        let guard = match self.rng.index(3) {
            0 => format!("{counter} == {at}"),
            1 => format!("{counter} > {at}"),
            _ => format!("{counter} % {} == 0", self.rng.range(2, 40)),
        };
        let mut body = Vec::new();
        match self.rng.index(6) {
            0 => {
                if let Some(var) = self.writable(Kind::Num).first().map(|v| v.name.clone()) {
                    self.feature("phase_number_subtype");
                    let delta = *self
                        .rng
                        .pick(&["0.5", "1", "-0.25", "2^53", "1e300", "0/0"]);
                    body.push(Node::leaf(format!("{var} = {var} + {delta}")));
                }
            }
            1 => {
                if let Some(var) = self
                    .mutable_tables(Kind::Obj)
                    .first()
                    .map(|v| v.name.clone())
                {
                    self.feature("metatable_switch");
                    let class = *self.rng.pick(&["ClassA", "ClassB"]);
                    body.push(Node::leaf(format!("setmetatable({var}, {class})")));
                }
            }
            2 => {
                if let Some(var) = self
                    .mutable_tables(Kind::Rec)
                    .first()
                    .map(|v| v.name.clone())
                {
                    self.feature("phase_table_shape");
                    let key = self.dynamic_key();
                    body.push(Node::leaf(format!("{var}[{key}] = {counter} / 2")));
                    body.push(Node::leaf(format!("{var}.z = nil")));
                }
            }
            3 => {
                if let Some(var) = self.writable(Kind::Str).first().map(|v| v.name.clone()) {
                    self.feature("phase_string");
                    let value =
                        *self
                            .rng
                            .pick(&["\"12\"", "\"0x1F\"", "\" 3.5 \"", "\"\"", "\"\\0z\""]);
                    body.push(Node::leaf(format!("{var} = {value}")));
                }
            }
            4 => {
                if let Some(var) = self
                    .mutable_tables(Kind::Arr)
                    .first()
                    .map(|v| v.name.clone())
                {
                    self.feature("phase_array_alias");
                    let fresh = self.array_constructor();
                    body.push(Node::leaf(format!("{var} = {fresh}")));
                }
            }
            _ => {
                self.feature("phase_global");
                body.push(Node::leaf(format!("G_phase = {counter}")));
                body.push(Node::leaf("G_phase = nil".to_owned()));
            }
        }
        if body.is_empty() {
            return self.assignment();
        }
        self.feature("phase_change");
        let features = self.take_features();
        Node::block(format!("if {guard} then"), body, "end").with_features(features)
    }

    fn string_iteration(&mut self) -> Node {
        let count = self.fresh("n");
        let word = self.fresh("w");
        let source = self.expr(Kind::Str, 1);
        let pattern = *self
            .rng
            .pick(&["\"%a+\"", "\"%d+\"", "\"(%w)(%w)\"", "\".\"", "\"[^,]+\""]);
        self.feature("gmatch");
        let observe = self.tag("g");
        self.leaf(format!(
            "do\n  local {count} = 0\n  for {word} in string.gmatch({source}, {pattern}) do {count} = {count} + #{word} end\n  out({observe}, {count})\nend"
        ))
    }

    fn closure_capture(&mut self) -> Node {
        let Some((counter, _)) = self.frame.counter.clone() else {
            return self.closure_cell();
        };
        let list = self.fresh("fns");
        let tag = self.tag("K");
        let scale = self.rng.range(2, 5);
        self.feature("closures_per_iteration");
        self.leaf(format!(
            "do\n  local {list} = {{}}\n  for k = 1, 3 do {list}[k] = function() return {counter} * {scale} + k end end\n  out({tag}, {list}[1](), {list}[3]())\nend"
        ))
    }

    /// A closure that shares a mutable cell with the enclosing scope.
    fn closure_cell(&mut self) -> Node {
        let cell = self.fresh("c");
        let tag = self.tag("K");
        let start = self.expr(Kind::Int, 1);
        self.feature("captured_cells");
        self.leaf(format!(
            "do\n  local {cell} = {start}\n  local function bump() {cell} = {} return {cell} end\n  bump()\n  {cell} = {}\n  out({tag}, bump(), {cell})\nend",
            self.wrap_int(format!("{cell} + 1")),
            self.wrap_int(format!("{cell} * 2"))
        ))
    }

    fn swap(&mut self) -> Node {
        let kind = *self.rng.pick(&ALL_KINDS);
        let vars = self.writable(kind);
        if vars.len() < 2 {
            return self.assignment();
        }
        let a = self.rng.pick(&vars).name.clone();
        let b = self.rng.pick(&vars).name.clone();
        if a == b {
            return self.assignment();
        }
        self.feature("multiple_assignment");
        self.leaf(format!("{a}, {b} = {b}, {a}"))
    }

    fn close_and_const(&mut self) -> Node {
        let tag = self.tag("X");
        let name = self.fresh("cv");
        let value = self.expr(Kind::Int, 1);
        if self.rng.chance(1, 2) {
            self.feature("const_local");
            self.leaf(format!(
                "do\n  local {name} <const> = {value}\n  out({tag}, {name} + 1)\nend"
            ))
        } else {
            self.feature("close_local");
            self.leaf(format!(
                "do\n  local {name} <close> = setmetatable({{}}, {{__close = function(_, err) out({tag}, \"closed\", err) end}})\n  out({tag}, \"open\", {value})\nend"
            ))
        }
    }

    fn collect_garbage(&mut self) -> Node {
        let tag = self.tag("W");
        self.feature("gc");
        if self.rng.chance(1, 2) {
            let objects = self.visible(Kind::Obj);
            let keep = objects
                .first()
                .map_or_else(|| "{}".to_owned(), |var| var.name.clone());
            self.feature("weak_tables");
            return self.leaf(format!(
                "do\n  local weak = setmetatable({{}}, {{__mode = \"k\"}})\n  local live = {{{keep}, {{}}, {{}}}}\n  for k = 1, #live do weak[live[k]] = k end\n  collectgarbage()\n  local sum = 0\n  for _, v in pairs(weak) do sum = sum + v end\n  out({tag}, sum, rawequal(live[1], {keep}))\nend"
            ));
        }
        match &self.frame.counter {
            Some((counter, _)) => {
                let counter = counter.clone();
                self.leaf(format!("if {counter} % 97 == 0 then collectgarbage() end"))
            }
            None => self.leaf("collectgarbage()"),
        }
    }

    fn proxy_write(&mut self) -> Node {
        let key = *self.rng.pick(&["k1", "kk2", "key3"]);
        let value = self.stored(Kind::Int);
        self.feature("newindex_metamethod");
        self.leaf(format!("Proxy.{key} = {value}"))
    }

    fn integer_edges(&mut self) -> Node {
        let tag = self.tag("N");
        let count = self.fresh("c");
        let lines = if self.dialect.integer_for_edges {
            self.feature("integer_for_edges");
            let edge = *self.rng.pick(&[
                "for k = math.maxinteger - 2, math.maxinteger do",
                "for k = math.mininteger, math.mininteger + 2 do",
                "for k = math.mininteger + 2, math.mininteger, -1 do",
                "for k = 1, 3.5 do",
                "for k = 3, 1 do",
            ]);
            format!(
                "do\n  local {count} = 0\n  {edge} {count} = {count} + 1 end\n  out({tag}, {count})\nend"
            )
        } else {
            self.feature("float_for");
            let edge = *self.rng.pick(&[
                "for k = 1, 2, 0.25 do",
                "for k = 3, 1 do",
                "for k = 1, 3.5 do",
                "for k = -1, -2.5, -0.5 do",
            ]);
            format!(
                "do\n  local {count} = 0\n  {edge} {count} = {count} + k end\n  out({tag}, {count})\nend"
            )
        };
        self.leaf(lines)
    }

    /// Adds and removes many keys, so a record's layout changes under
    /// field reads that compiled code may have specialised.
    fn layout_churn(&mut self) -> Node {
        let records = self.mutable_tables(Kind::Rec);
        if records.is_empty() {
            return self.assignment();
        }
        let r = self.rng.pick(&records).name.clone();
        let tag = self.tag("R");
        let count = self.rng.range(4, 12);
        self.feature("table_layout_churn");
        self.leaf(format!(
            "do\n  for k = 1, {count} do {r}[\"t\" .. k] = k end\n  local seen = ({r}.x or 0)\n  for k = 1, {count} do {r}[\"t\" .. k] = nil end\n  out({tag}, seen, {r}.y, rec({r}))\nend"
        ))
    }

    // ----- expressions ---------------------------------------------------

    fn expr(&mut self, kind: Kind, depth: u32) -> String {
        let leaf = depth >= 3 || self.rng.chance(u64::from(depth) + 1, 4);
        match kind {
            Kind::Int => self.int_expr(depth, leaf),
            Kind::Num => self.num_expr(depth, leaf),
            Kind::Str => self.str_expr(depth, leaf),
            Kind::Bool => self.bool_expr(depth, leaf),
            // A constructor used as a prefix (`({1})[1]`) needs parentheses.
            Kind::Arr => self
                .pick_var(Kind::Arr)
                .unwrap_or_else(|| format!("({})", self.array_constructor())),
            Kind::Rec => self
                .pick_var(Kind::Rec)
                .unwrap_or_else(|| format!("({})", self.record_constructor())),
            Kind::Obj => self.obj_expr(depth),
        }
    }

    fn int_literal(&mut self) -> String {
        match self.rng.weighted(&[6, 2, 2, 2]) {
            0 => self.rng.range(0, 12).to_string(),
            1 => format!("({})", self.rng.range(-12, -1)),
            2 => self.rng.range(100, 5000).to_string(),
            _ => {
                self.feature("boundary_values");
                let edges: &[&str] = if self.dialect.integers {
                    &[
                        "math.maxinteger",
                        "math.mininteger",
                        "0x7fffffff",
                        "2147483648",
                        "0xff",
                        "(-0x80000000)",
                        "9007199254740993",
                    ]
                } else if self.allowed("float_modulo") {
                    &[
                        "2147483647",
                        "4294967296",
                        "0xff",
                        "9007199254740992",
                        "(-2147483648)",
                    ]
                } else {
                    // Products of these stay below 2^53, where the double
                    // modulo that keeps integers in range is exact under
                    // every rounding (stock 5.1 built with FMA differs above).
                    &["65535", "0xff", "(-65536)"]
                };
                (*self.rng.pick(edges)).to_owned()
            }
        }
    }

    fn int_leaf(&mut self) -> String {
        if self.rng.chance(2, 3)
            && let Some(var) = self.pick_var(Kind::Int)
        {
            return var;
        }
        self.int_literal()
    }

    /// A small integer leaf. Int-kind variables can hold overflowing floats,
    /// so keep variable operands within the same range as literal operands.
    fn small_int_leaf(&mut self) -> String {
        if self.rng.chance(1, 2)
            && let Some(var) = self.pick_var(Kind::Int)
        {
            return format!("(array_index({var}, 321) - 21)");
        }
        self.rng.range(-20, 300).to_string()
    }

    #[allow(clippy::too_many_lines, reason = "one arm per expression family")]
    fn int_expr(&mut self, depth: u32, leaf: bool) -> String {
        if leaf {
            return self.int_leaf();
        }
        let next = depth + 1;
        let other = u32::from(!self.frame.kernel);
        // Streams from before the expanded probes keep the old gate: the
        // trace recorder did not follow the 5.3+ library calls then.
        let expanded = u32::from(self.expanded_probes);
        let integers = u32::from(self.dialect.integers) * expanded;
        let library =
            u32::from(self.expanded_probes || !(self.frame.kernel && self.dialect.integers));
        let metatables = u32::from(self.library_ready) * expanded * other;
        let choice = self.rng.weighted(&[
            8, // arithmetic
            3, // division and modulo
            if self.dialect.integers || self.dialect.bit_library || self.dialect.bit32_library {
                3
            } else {
                0
            },
            3 * other,                             // lengths
            2 * other,                             // string bytes and searches
            3 * library,                           // math library
            3,                                     // array read
            2 * other,                             // pure call
            other,                                 // proxy read
            other,                                 // string coercion
            2,                                     // ternary
            u32::from(self.library_ready) * other, // object length or polymorphic dispatch
            2 * integers,                          // math.tointeger
            2 * expanded,                          // select over values
            expanded,                              // assert returning its argument
            2 * expanded,                          // rawget of an array or a record
            2 * expanded,                          // unpack
            metatables,                            // getmetatable
        ]);
        match choice {
            0 => {
                let a = self.expr(Kind::Int, next);
                let b = self.expr(Kind::Int, next);
                let op = *self.rng.pick(&["+", "-", "*"]);
                self.feature("integer_arithmetic");
                if self.rng.chance(1, 6) {
                    self.feature("unary_minus");
                    return self.wrap_int(format!("-({a})"));
                }
                self.wrap_int(format!("({a} {op} {b})"))
            }
            1 => {
                let a = self.expr(Kind::Int, next);
                let b = self.expr(Kind::Int, next);
                if self.rng.chance(1, 2) {
                    self.feature("modulo");
                    format!("({a} % nz({b}))")
                } else if self.dialect.integers {
                    self.feature("floor_division");
                    format!("({a} // nz({b}))")
                } else {
                    self.feature("floor_division");
                    format!("math.floor({a} / nz({b}))")
                }
            }
            2 => self.bitwise(),
            3 => {
                if self.dialect.utf8 && self.allowed("utf8") && self.rng.chance(1, 4) {
                    self.feature("utf8");
                    let s = self.expr(Kind::Str, next);
                    return format!("(utf8.len({s}) or -1)");
                }
                self.feature("length_operator");
                match self.rng.index(3) {
                    // Parenthesized: `#t ^ 2` would mean `#(t ^ 2)`.
                    0 => format!("(#{})", self.expr(Kind::Str, next)),
                    1 => format!("(#{})", self.expr(Kind::Arr, next)),
                    _ if self.dialect.rawlen => {
                        self.feature("rawlen");
                        format!("rawlen({})", self.expr(Kind::Arr, next))
                    }
                    _ => format!("string.len({})", self.expr(Kind::Str, next)),
                }
            }
            4 => {
                let s = self.expr(Kind::Str, next);
                self.feature("string_library");
                match self.rng.index(3) {
                    0 => {
                        let at = self.small_int_leaf();
                        format!("(string.byte({s}, {at}) or -1)")
                    }
                    1 => {
                        let needle = self.string_literal();
                        format!("(string.find({s}, {needle}, 1, true) or 0)")
                    }
                    _ => format!("(select(\"#\", string.byte({s}, 1, 3)))"),
                }
            }
            5 => {
                let a = self.expr(Kind::Int, next);
                let b = self.expr(Kind::Int, next);
                self.feature("math_library");
                // Kernels of the 5.3+ profiles keep to what the trace
                // recorder follows.
                if self.expanded_probes && self.frame.kernel && self.dialect.integers {
                    return format!("math.abs({a})");
                }
                match self.rng.index(3) {
                    0 => format!("math.max({a}, {b})"),
                    1 => format!("math.min({a}, {b})"),
                    _ => format!("math.abs({a})"),
                }
            }
            6 => {
                let t = self.expr(Kind::Arr, next);
                let index = self.expr(Kind::Int, next);
                self.feature("table_index_read");
                if !self.frame.kernel && self.rng.chance(1, 4) {
                    self.feature("float_key_normalization");
                    return format!("({t}[1.0] or 0)");
                }
                format!("({t}[array_index({index}, #{t})] or 0)")
            }
            7 => self
                .pure_call_of(Kind::Int)
                .unwrap_or_else(|| self.int_leaf()),
            8 => {
                self.feature("index_metamethod");
                let key = *self.rng.pick(&["k1", "kk2", "key3"]);
                format!("Proxy.{key}")
            }
            9 => {
                self.feature("string_coercion");
                let text = *self.rng.pick(&["\"10\"", "\"0x10\"", "\" 7 \""]);
                let a = self.small_int_leaf();
                format!("({text} + {a})")
            }
            10 => {
                self.feature("ternary");
                let condition = self.expr(Kind::Bool, next);
                let a = self.expr(Kind::Int, next);
                let b = self.expr(Kind::Int, next);
                format!("({condition} and {a} or {b})")
            }
            12 => {
                let a = self.expr(Kind::Num, next);
                self.feature("math_tointeger");
                format!("(math.tointeger({a}) or -1)")
            }
            13 => {
                let a = self.expr(Kind::Int, next);
                let b = self.expr(Kind::Int, next);
                let c = self.expr(Kind::Int, next);
                self.feature("select");
                match self.rng.index(5) {
                    0 => format!("select(\"#\", {a}, {b}, {c})"),
                    1 => format!("(select(2, {a}, {b}, {c}))"),
                    2 => format!("(select(-1, {a}, {b}, {c}))"),
                    3 => format!("(select(3, {a}, {b}, {c}))"),
                    _ => format!("select(\"#\", select(2, {a}, {b}, {c}))"),
                }
            }
            14 => {
                let a = self.expr(Kind::Int, next);
                self.feature("assert");
                if self.rng.chance(1, 2) {
                    format!("(assert({a}))")
                } else {
                    format!("(assert({a}, \"message\"))")
                }
            }
            15 => {
                if self.hot_avoids_unsupported() {
                    let t = self.expr(Kind::Arr, next);
                    let index = self.expr(Kind::Int, next);
                    self.feature("table_index_read");
                    return format!("({t}[array_index({index}, #{t})] or 0)");
                }
                self.feature("rawget");
                if self.rng.chance(1, 3) {
                    let r = self.expr(Kind::Rec, next);
                    let key = *self.rng.pick(&["\"x\"", "\"y\"", "\"missing\""]);
                    return format!("(rawget({r}, {key}) or 0)");
                }
                let t = self.expr(Kind::Arr, next);
                let index = self.expr(Kind::Int, next);
                format!("(rawget({t}, array_index({index}, #{t})) or 0)")
            }
            16 => {
                let unpack = if self.dialect.table_unpack {
                    "table.unpack"
                } else {
                    "unpack"
                };
                let t = self.expr(Kind::Arr, next);
                self.feature("unpack");
                match self.rng.index(4) {
                    0 => format!("(select(\"#\", {unpack}({t})))"),
                    1 => format!("(({unpack}({t})) or 0)"),
                    2 => format!("((select(2, {unpack}({t}, 1, 3))) or 0)"),
                    _ => format!("(select(\"#\", {unpack}({t}, 2, 3)))"),
                }
            }
            17 => {
                // The metatable of an object is its class, or whatever a
                // phase change switched it to.
                let o = self.expr(Kind::Obj, next);
                self.feature("getmetatable");
                match self.rng.index(3) {
                    0 => format!("(rawequal(getmetatable({o}), ClassA) and 1 or 0)"),
                    1 => format!("(getmetatable({o}) == ClassB and 1 or 0)"),
                    _ => {
                        let s = self.expr(Kind::Str, next);
                        format!("(getmetatable({s}) ~= nil and 1 or 0)")
                    }
                }
            }
            _ => {
                if self.dialect.len_metamethod
                    && self.rng.chance(1, 2)
                    && let Some(o) = self.pick_var(Kind::Obj)
                {
                    self.feature("len_metamethod");
                    return format!("(#{o})");
                }
                self.feature("polymorphic_calls");
                let selector = self.small_int_leaf();
                let a = self.expr(Kind::Int, next);
                let b = self.expr(Kind::Int, next);
                format!("ops[array_index({selector}, 2)]({a}, {b})")
            }
        }
    }

    /// A call of the `bit` or `bit32` library with operands at the edges of
    /// their conversions: word boundaries, ties, signs of zero, NaN and the
    /// infinities (where the library reads them), and shift counts around 32.
    fn bit_library_call(&mut self, a: String, b: String) -> String {
        self.feature("bit_library_edges");
        let luajit = self.dialect.bit_library;
        let library = if luajit { "bit" } else { "bit32" };
        let whole: &[&str] = &[
            "0x7fffffff",
            "0x80000000",
            "0xffffffff",
            "(-0x80000000)",
            "4294967296",
            "4294967297",
            "(-1)",
            "0",
            "255",
        ];
        // Lua 5.3's `bit32` raises for a fraction, and for a number out of
        // the range of an integer.
        let fractions: &[&str] = &[
            "0.5",
            "1.5",
            "2.5",
            "(-1.5)",
            "(-0.0)",
            "(2^53)",
            "(2^51 + 0.5)",
            "3.75",
        ];
        let non_finite: &[&str] = &["(0/0)", "math.huge", "(-math.huge)"];
        let reads_fractions = luajit || !self.dialect.integers;
        let operand = |this: &mut Self, small: String| -> String {
            match this.rng.index(4) {
                0 | 1 => small,
                2 => (*this.rng.pick(whole)).to_owned(),
                _ if reads_fractions => {
                    let pool = if luajit && this.rng.chance(1, 4) {
                        non_finite
                    } else {
                        fractions
                    };
                    (*this.rng.pick(pool)).to_owned()
                }
                _ => (*this.rng.pick(whole)).to_owned(),
            }
        };
        let a = operand(self, a);
        let b = operand(self, b);
        let count = (*self
            .rng
            .pick(&["0", "1", "31", "32", "33", "(-1)", "5", "64"]))
        .to_owned();
        if luajit {
            match self.rng.index(10) {
                0 => format!("{library}.tobit({a})"),
                1 => format!("{library}.bnot({a})"),
                2 => format!("{library}.band({a}, {b})"),
                3 => format!("{library}.bor({a}, {b}, 0x0f0f)"),
                4 => format!("{library}.bxor({a}, {b})"),
                5 => format!("{library}.lshift({a}, {count})"),
                6 => format!("{library}.rshift({a}, {count})"),
                7 => format!("{library}.arshift({a}, {count})"),
                8 => format!("{library}.rol({a}, {count})"),
                _ => format!("{library}.ror({a}, {count})"),
            }
        } else {
            let (field, width) =
                *self
                    .rng
                    .pick(&[(0, 1), (3, 5), (8, 8), (24, 8), (31, 1), (0, 32)]);
            match self.rng.index(11) {
                0 => format!("{library}.bnot({a})"),
                1 => format!("{library}.band({a}, {b})"),
                2 => format!("{library}.bor({a}, {b}, 0x0f0f)"),
                3 => format!("{library}.bxor({a}, {b})"),
                4 => format!("{library}.lshift({a}, {count})"),
                5 => format!("{library}.rshift({a}, {count})"),
                6 => format!("{library}.arshift({a}, {count})"),
                7 => format!("{library}.lrotate({a}, {count})"),
                8 => format!("{library}.rrotate({a}, {count})"),
                9 => format!("({library}.btest({a}, {b}) and 1 or 0)"),
                _ if self.rng.chance(1, 2) => {
                    format!("{library}.extract({a}, {field}, {width})")
                }
                _ => format!("{library}.replace({a}, {b}, {field}, {width})"),
            }
        }
    }

    fn bitwise(&mut self) -> String {
        let a = self.small_int_leaf();
        let b = self.small_int_leaf();
        self.feature("bitwise");
        if self.expanded_probes
            && (self.dialect.bit_library || self.dialect.bit32_library)
            && self.rng.chance(1, 3)
        {
            return self.bit_library_call(a, b);
        }
        if self.dialect.integers {
            let a = format!("bit_operand({a})");
            let b = format!("bit_operand({b})");
            match self.rng.index(6) {
                0 => format!("({a} & {b})"),
                1 => format!("({a} | {b})"),
                2 => format!("({a} ~ {b})"),
                3 => format!("({a} << ({b} % 70))"),
                4 => format!("({a} >> ({b} % 70))"),
                _ => format!("(~{a})"),
            }
        } else {
            let library = if self.dialect.bit_library {
                "bit"
            } else {
                "bit32"
            };
            let shift_mask = 31;
            match self.rng.index(6) {
                0 => format!("{library}.band({a}, {b})"),
                1 => format!("{library}.bor({a}, {b})"),
                2 => format!("{library}.bxor({a}, {b})"),
                3 => format!("{library}.lshift({a}, {b} % {shift_mask})"),
                4 => format!("{library}.rshift({a}, {b} % {shift_mask})"),
                _ => format!("{library}.bnot({a})"),
            }
        }
    }

    fn num_literal(&mut self) -> String {
        self.feature("float_literals");
        let mut literals = vec![
            "0.5", "1.5", "(-2.25)", "0.1", "3.14159", "1e10", "1e-3", "100.0",
        ];
        // Stock Lua 5.1 and 5.2 key their constant tables by number value, so
        // a `-0.0` literal and a later `0` share one constant and whichever
        // came first wins. That is an artefact of stock's compiler, not
        // language semantics; those profiles only meet -0 at run time.
        if self.dialect.integers || self.dialect.bit_library {
            literals.push("(-0.0)");
        }
        if self.rng.chance(1, 4) {
            self.feature("nonfinite");
            literals.extend(["math.huge", "(-math.huge)", "(1e308 * 10)", "math.pi"]);
            if self.expanded_probes {
                // Where a float stops fitting an integer, for `math.floor`.
                literals.extend(["(2^63)", "(-2^63)", "(2^53)", "(-0.5)"]);
            }
        }
        if self.dialect.hex_escapes {
            literals.push("0x1p-2");
        }
        (*self.rng.pick(&literals)).to_owned()
    }

    #[allow(clippy::too_many_lines, reason = "one arm per expression family")]
    fn num_expr(&mut self, depth: u32, leaf: bool) -> String {
        if leaf {
            return match self.rng.index(3) {
                0 => self.num_literal(),
                1 => self
                    .pick_var(Kind::Num)
                    .unwrap_or_else(|| self.num_literal()),
                _ => self.int_leaf(),
            };
        }
        let next = depth + 1;
        let other = u32::from(!self.frame.kernel);
        let library =
            u32::from(self.expanded_probes || !(self.frame.kernel && self.dialect.integers));
        match self.rng.weighted(&[
            3,
            6,
            2,
            1,
            4 * library,
            2 * other,
            2,
            2 * other,
            other,
            other,
            if self.string_arith && self.allowed("string_arithmetic") {
                2
            } else {
                0
            },
        ]) {
            10 => {
                self.feature("string_arithmetic");
                let i = self.small_int_leaf();
                let text = format!("numeric_strings[({i}) % 10 + 1]");
                let b = self.expr(Kind::Num, next);
                match self.rng.index(7) {
                    0 => format!("({text} {} {b})", self.rng.pick(&["+", "-", "*", "/"])),
                    1 => format!("({b} {} {text})", self.rng.pick(&["+", "-", "*", "/"])),
                    2 => format!("({text} {} {text})", self.rng.pick(&["+", "-", "*"])),
                    // Float `%` is a family of its own, which a campaign may
                    // leave out for its known differences.
                    3 if self.allowed("float_modulo") => format!("({text} % nz({b}))"),
                    4 if self.allowed("float_modulo") => format!("({b} % nz({text}))"),
                    3 | 4 => format!("({text} + {b})"),
                    5 if self.dialect.integers => format!("({text} // nz({b}))"),
                    5 => format!("({text} ^ {})", self.rng.range(0, 3)),
                    _ => format!("(-{text})"),
                }
            }
            1 => {
                let a = self.expr(Kind::Num, next);
                let b = self.expr(Kind::Num, next);
                let op = *self.rng.pick(&["+", "-", "*", "/"]);
                self.feature("float_arithmetic");
                format!("({a} {op} {b})")
            }
            2 if self.allowed("float_modulo") => {
                let a = self.expr(Kind::Num, next);
                let b = self.expr(Kind::Num, next);
                self.feature("float_modulo");
                format!("({a} % nz({b}))")
            }
            // An integer, or a float modulo the campaign excluded.
            0 | 2 => self.int_expr(depth, false),
            3 => {
                let a = self.expr(Kind::Num, next);
                let exponent = self.rng.range(0, 3);
                self.feature("power");
                format!("({a} ^ {exponent})")
            }
            4 => {
                let a = self.expr(Kind::Num, next);
                self.feature("math_library");
                // Kernels of the 5.3+ profiles keep to what the trace
                // recorder follows: `floor`, `ceil` and `abs`.
                let pick = if self.expanded_probes && self.frame.kernel && self.dialect.integers {
                    if self.scalar_edges {
                        [0, 1, 8, 9, 10][self.rng.index(5)]
                    } else {
                        [0, 1, 8][self.rng.index(3)]
                    }
                } else if self.scalar_edges {
                    self.rng.index(12)
                } else {
                    self.rng.index(9)
                };
                match pick {
                    0 => format!("math.floor({a})"),
                    1 => format!("math.ceil({a})"),
                    2 => format!("math.sqrt(math.abs({a}))"),
                    3 => format!("math.fmod({a}, nz({}))", self.expr(Kind::Num, next)),
                    4 => format!("(math.modf({a}))"),
                    5 => format!("math.max({a}, {})", self.expr(Kind::Num, next)),
                    6 => format!("math.sin({a})"),
                    7 => format!("math.exp(({a}) % 8)"),
                    9 => {
                        // Bases 1, 0 and negative ones are the logarithm's
                        // own edges, with the natural form and 2 and 10.
                        // Since generator 22 Oximond matches stock 5.4, 5.5
                        // and LuaJIT for every base (`log2` and `log10` for
                        // 2 and 10, LuaJIT's reciprocal form), so all of
                        // them are drawn. Older streams kept the bases and
                        // the unary LuaJIT form that Oximond then differed on.
                        let bases: &[&str] = if self.exact_logs {
                            &["2", "10", "2.718281828459045", "1", "0", "-2"]
                        } else {
                            match self.profile {
                                Profile::Lua54 | Profile::Lua55 => {
                                    &["2.718281828459045", "1", "0", "-2"]
                                }
                                _ => &["2", "10", "2.718281828459045", "1", "0", "-2"],
                            }
                        };
                        let base = *self.rng.pick(bases);
                        if self.profile == Profile::LuaJit21 && !self.exact_logs {
                            format!("math.log({a})")
                        } else {
                            format!("math.log({a}, {base})")
                        }
                    }
                    10 => format!("select(2, math.modf({a}))"),
                    // Lua 5.5 has no `math.pow`, and neither has Oximond's
                    // profile since generator 22.
                    11 => format!("(math.pow and math.pow({a}, 2) or 0)"),
                    _ => format!("math.abs({a})"),
                }
            }
            5 => {
                let s = self.expr(Kind::Str, next);
                self.feature("tonumber");
                if self.scalar_edges && self.rng.chance(1, 4) {
                    let edge = *self.rng.pick(&[
                        " 0x1p4 ",
                        "1e",
                        "10",
                        "9223372036854775808",
                        "  12  ",
                        "0x",
                        "1e5",
                        "-0x10",
                        "inf",
                        "nan",
                        "",
                        ".5",
                        "5.",
                    ]);
                    format!("(tonumber(\"{edge}\") or -1)")
                } else if self.rng.chance(1, 3) && self.allowed("tonumber_base") {
                    self.feature("tonumber_base");
                    format!("(tonumber({s}, 16) or -1)")
                } else {
                    format!("(tonumber({s}) or 0)")
                }
            }
            6 => {
                let r = self.expr(Kind::Rec, next);
                if self.rng.chance(1, 2) {
                    self.feature("field_read");
                    let field = *self.rng.pick(&["x", "y", "z"]);
                    format!("({r}.{field} or 0)")
                } else {
                    self.feature("dynamic_key_read");
                    let key = self.dynamic_key();
                    format!("({r}[{key}] or 0)")
                }
            }
            7 => {
                let o = self.expr(Kind::Obj, next);
                match self.rng.index(3) {
                    0 => {
                        self.feature("method_call");
                        format!("{o}:get()")
                    }
                    1 => {
                        self.feature("call_metamethod");
                        let x = self.small_int_leaf();
                        format!("{o}({x})")
                    }
                    _ => {
                        self.feature("inherited_field_read");
                        format!("{o}.v")
                    }
                }
            }
            8 => {
                self.feature("string_coercion");
                let text = *self.rng.pick(&["\"2.5\"", "\"1e2\"", "\"-0.5\""]);
                let a = self.expr(Kind::Num, next);
                format!("({text} * {a})")
            }
            _ => self
                .pure_call_of(Kind::Num)
                .unwrap_or_else(|| self.num_literal()),
        }
    }

    fn string_literal(&mut self) -> String {
        let mut pool = vec![
            "\"\"",
            "\"a\"",
            "\"hello\"",
            "\"10\"",
            "\" 3.5 \"",
            "\"0x1F\"",
            "\"abc def\"",
            "\"x,y,,z\"",
            "\"nan\"",
            "\"1e3\"",
            "\"-0\"",
            "\"tab\\tnew\\nline\"",
            "\"quote\\\"q\"",
            "\"back\\\\slash\"",
            "\"\\65\\066\"",
            "\"nul\\0byte\"",
            "\"\\200\\255high\"",
            "'single'",
            "[[long [=[x]=] str]]",
            "[==[a]]b]==]",
        ];
        if self.dialect.hex_escapes {
            pool.extend(["\"\\x41\\x7a\"", "\"skip \\z\n      spaces\""]);
        }
        if self.dialect.unicode_escapes {
            pool.extend(["\"\\u{48}\\u{e4}\\u{20AC}\""]);
        }
        let text = (*self.rng.pick(&pool)).to_owned();
        if text.contains('\\') || text.starts_with('[') {
            self.feature("string_escapes");
        }
        text
    }

    #[allow(clippy::too_many_lines, reason = "one arm per expression family")]
    fn str_expr(&mut self, depth: u32, leaf: bool) -> String {
        if leaf {
            if self.rng.chance(1, 2)
                && let Some(var) = self.pick_var(Kind::Str)
            {
                return var;
            }
            return self.string_literal();
        }
        let next = depth + 1;
        let library = u32::from(self.library_ready);
        match self
            .rng
            .weighted(&[5, 3, 4, 4, 2, 3, 2, 2, 1, 2 * library, 1])
        {
            0 => {
                let a = self.expr(Kind::Str, next);
                let b = if self.rng.chance(1, 3) {
                    self.feature("number_concat");
                    format!("string_number({})", self.expr(Kind::Num, next))
                } else {
                    self.expr(Kind::Str, next)
                };
                self.feature("concat");
                format!("string.sub({a} .. {b}, 1, 40)")
            }
            1 => {
                self.feature("tostring");
                if self.scalar_edges && self.rng.chance(1, 8) {
                    return "tostring(nil)".to_owned();
                }
                let kind = *self.rng.pick(&[Kind::Int, Kind::Num, Kind::Bool]);
                let value = self.expr(kind, next);
                let value = if matches!(kind, Kind::Int | Kind::Num) {
                    format!("string_number({value})")
                } else {
                    value
                };
                format!("tostring({value})")
            }
            2 => {
                let s = self.expr(Kind::Str, next);
                self.feature("string_library");
                match self.rng.index(7) {
                    0 => format!("string.upper({s})"),
                    1 => format!("({s}):lower()"),
                    2 => format!("string.reverse({s})"),
                    3 => {
                        let i = self.small_int_leaf();
                        let j = if self.scalar_edges && self.rng.chance(1, 3) {
                            (*self.rng.pick(&["0", "-1", "-3", "100", "-100"])).to_owned()
                        } else {
                            self.small_int_leaf()
                        };
                        format!("string.sub({s}, {i}, {j})")
                    }
                    4 => {
                        let n = self.small_int_leaf();
                        format!("string.rep({s}, {n} % 3)")
                    }
                    5 if self.dialect.rep_separator && self.allowed("rep_separator") => {
                        self.feature("rep_separator");
                        let n = self.small_int_leaf();
                        format!("string.rep({s}, {n} % 3, \",\")")
                    }
                    _ => format!("({s}):sub(2)"),
                }
            }
            3 => self.format_expr(next),
            4 => {
                let n = self.small_int_leaf();
                self.feature("string_char");
                format!("string.char({n} % 256, 65 + {n} % 26)")
            }
            5 => {
                let s = self.expr(Kind::Str, next);
                self.feature("patterns");
                let (pattern, replacement) = *self.rng.pick(&[
                    ("\"%a\"", "\"<%0>\""),
                    ("\"%d+\"", "\"#\""),
                    ("\"(%w)(%w)\"", "\"%2%1\""),
                    ("\"[aeiou]\"", "{a = \"4\", e = \"3\"}"),
                    ("\"%s\"", "\"_\""),
                    ("\"^%s*(.-)%s*$\"", "\"%1\""),
                    ("\"x*\"", "\"-\""),
                    ("\"%p\"", "function(c) return c .. c end"),
                    ("\"%f[%w]%w+\"", "string.upper"),
                ]);
                if self.rng.chance(1, 3) {
                    format!("(string.gsub({s}, {pattern}, {replacement}, 2))")
                } else if self.rng.chance(1, 2) {
                    format!("(string.gsub({s}, {pattern}, {replacement}))")
                } else {
                    format!("(string.match({s}, {pattern}) or \"?\")")
                }
            }
            6 => {
                self.feature("table_concat");
                let t = self.expr(Kind::Arr, next);
                if self.rng.chance(1, 2) {
                    format!("table.concat(string_numbers({t}), \",\")")
                } else {
                    format!("table.concat(string_numbers({t}), \"-\", 1, math.min(2, #{t}))")
                }
            }
            7 => {
                self.feature("type");
                let kind = *self.rng.pick(&ALL_KINDS);
                format!("type({})", self.expr(kind, next))
            }
            8 => {
                let o = self.expr(Kind::Obj, next);
                match self.rng.index(3) {
                    0 => {
                        self.feature("tostring_metamethod");
                        format!("tostring({o})")
                    }
                    1 => {
                        self.feature("concat_metamethod");
                        let s = self.expr(Kind::Str, next);
                        format!("string.sub({o} .. {s}, 1, 40)")
                    }
                    _ => {
                        self.feature("method_call");
                        format!("{o}:name()")
                    }
                }
            }
            9 => {
                if self.dialect.utf8 && self.allowed("utf8") && self.rng.chance(1, 2) {
                    self.feature("utf8");
                    let n = self.small_int_leaf();
                    let cp = *self.rng.pick(&[
                        0,
                        0x7F,
                        0x80,
                        0x7FF,
                        0x800,
                        0xD7FF,
                        0xE000,
                        0xFFFF,
                        0x10000,
                        0x0010_FFFF,
                    ]);
                    return format!("utf8.char(72, {n} % 2048, {cp})");
                }
                let int = self.expr(Kind::Int, next);
                self.feature("envf");
                format!("envf({int})")
            }
            _ => self
                .pure_call_of(Kind::Str)
                .unwrap_or_else(|| self.string_literal()),
        }
    }

    fn format_expr(&mut self, next: u32) -> String {
        self.feature("string_format");
        match self.rng.index(7) {
            0 => {
                let value = self.expr(Kind::Int, next);
                let value = if self.dialect.integers {
                    // Preserve integer boundaries while converting large
                    // floats and nonfinite values to valid integer arguments.
                    format!("bit_operand({value})")
                } else {
                    format!("(array_index({value}, 1048576) - 524289)")
                };
                format!("string.format(\"%d\", {value})")
            }
            1 => format!(
                "string.format(\"%5.2f\", string_number({}))",
                self.expr(Kind::Num, next)
            ),
            2 => {
                let number = self.expr(Kind::Num, next);
                let int = self.small_int_leaf();
                // Stock 5.1 converts a negative %x argument to an unsigned
                // integer, which C leaves undefined (x86 gives all ones).
                let int = if self.dialect.integers || self.dialect.bit_library {
                    int
                } else {
                    format!("math.abs({int})")
                };
                format!("string.format(\"%g|%x\", string_number({number}), {int})")
            }
            3 if self.allowed("format_q") => {
                self.feature("format_q");
                format!("string.format(\"%q\", {})", self.expr(Kind::Str, next))
            }
            4 => {
                // A width on `%s` rejects strings with NUL bytes in 5.2+.
                let zero =
                    if self.dialect.integers || self.dialect.goto && !self.dialect.bit_library {
                        "\"\\0\""
                    } else {
                        "\"%z\""
                    };
                let a = self.expr(Kind::Str, next);
                let b = self.expr(Kind::Str, next);
                format!(
                    "string.format(\"%5s|%-5s|\", (string.gsub({a}, {zero}, \"\")), (string.gsub({b}, {zero}, \"\")))"
                )
            }
            5 => format!(
                "string.format(\"%s %s %%\", string_number({}), tostring({}))",
                self.expr(Kind::Int, next),
                self.expr(Kind::Bool, next)
            ),
            _ => {
                // `%c` of 0 adds nothing in stock 5.1 (a C string).
                let low = i32::from(!self.allowed("format_c_zero"));
                format!(
                    "string.format(\"%c%c\", 65 + {} % 26, {low} + {} % 127)",
                    self.small_int_leaf(),
                    self.small_int_leaf()
                )
            }
        }
    }

    fn bool_expr(&mut self, depth: u32, leaf: bool) -> String {
        if leaf {
            if self.rng.chance(1, 2)
                && let Some(var) = self.pick_var(Kind::Bool)
            {
                return var;
            }
            return (*self.rng.pick(&["true", "false"])).to_owned();
        }
        let next = depth + 1;
        let other = u32::from(!self.frame.kernel);
        let expanded = u32::from(self.expanded_probes);
        match self.rng.weighted(&[
            6,
            2 * other,
            2,
            3,
            2 * other,
            other,
            other,
            other,
            other,
            2 * expanded,
            expanded,
        ]) {
            0 => {
                let kind = *self.rng.pick(&[Kind::Int, Kind::Num]);
                let a = self.expr(kind, next);
                let b = self.expr(kind, next);
                let op = *self.rng.pick(&["<", "<=", ">", ">=", "==", "~="]);
                self.feature("comparisons");
                format!("({a} {op} {b})")
            }
            1 => {
                let a = self.expr(Kind::Str, next);
                let b = self.expr(Kind::Str, next);
                let op = *self.rng.pick(&["<", "==", "~=", ">="]);
                self.feature("string_comparison");
                format!("({a} {op} {b})")
            }
            2 => {
                self.feature("not");
                format!("(not {})", self.expr(Kind::Bool, next))
            }
            3 => {
                let a = self.expr(Kind::Bool, next);
                let b = self.expr(Kind::Bool, next);
                let op = *self.rng.pick(&["and", "or"]);
                self.feature("short_circuit");
                format!("({a} {op} {b})")
            }
            4 => {
                self.feature("truthiness");
                let kind = *self.rng.pick(&ALL_KINDS);
                let value = self.expr(kind, next);
                format!("(not not {value})")
            }
            choice @ (5..=7 | 9 | 10) => self.bool_probe(choice, next),
            _ => self
                .pure_call_of(Kind::Bool)
                .unwrap_or_else(|| "true".to_owned()),
        }
    }

    /// The boolean probes of [`Self::bool_expr`] that exercise one library
    /// function or metamethod: `choice` picks the probe.
    fn bool_probe(&mut self, choice: usize, next: u32) -> String {
        match choice {
            5 => {
                let o = self.expr(Kind::Obj, next);
                let p = self.expr(Kind::Obj, next);
                match self.rng.index(3) {
                    0 => {
                        self.feature("eq_metamethod");
                        format!("({o} == {p})")
                    }
                    1 => {
                        self.feature("lt_metamethod");
                        format!("({o} < {p})")
                    }
                    _ => {
                        self.feature("missing_key");
                        format!("({o}.missing == nil)")
                    }
                }
            }
            6 => {
                self.feature("rawequal");
                let a = self.expr(Kind::Arr, next);
                let b = self.expr(Kind::Arr, next);
                format!("rawequal({a}, {b})")
            }
            9 => {
                // The type of any value is a constant name.
                self.feature("type_check");
                let kind = *self
                    .rng
                    .pick(&[Kind::Int, Kind::Num, Kind::Bool, Kind::Arr]);
                let value = self.expr(kind, next);
                let name = *self
                    .rng
                    .pick(&["number", "table", "boolean", "string", "nil"]);
                let op = *self.rng.pick(&["==", "~="]);
                format!("(type({value}) {op} \"{name}\")")
            }
            10 => {
                // Raw equality of numbers: subtypes compare by value.
                self.feature("rawequal");
                let kind = *self.rng.pick(&[Kind::Int, Kind::Num]);
                let a = self.expr(kind, next);
                let b = self.expr(kind, next);
                format!("rawequal({a}, {b})")
            }
            7 => {
                if self.dialect.integers {
                    self.feature("math_type");
                    let n = self.expr(Kind::Num, next);
                    let name = if self.expanded_probes {
                        *self.rng.pick(&["float", "integer"])
                    } else {
                        "float"
                    };
                    format!("(math.type({n}) == \"{name}\")")
                } else {
                    let n = self.expr(Kind::Num, next);
                    self.feature("nan_check");
                    format!("({n} ~= {n})")
                }
            }
            _ => unreachable!("bool_expr picks only the probes it names"),
        }
    }

    fn obj_expr(&mut self, depth: u32) -> String {
        if depth >= 2 || self.rng.chance(3, 4) {
            return self
                .pick_var(Kind::Obj)
                .unwrap_or_else(|| self.object_constructor());
        }
        let next = depth + 1;
        let a = self.expr(Kind::Obj, next);
        let b = self.expr(Kind::Obj, next);
        self.feature("arithmetic_metamethods");
        match self.rng.index(3) {
            0 => format!("({a} + {b})"),
            1 => format!("({a} - {b})"),
            _ => format!("(-{a})"),
        }
    }

    fn array_constructor(&mut self) -> String {
        let count = self.rng.range(1, 5);
        let items: Vec<String> = (0..count).map(|_| self.expr(Kind::Int, 2)).collect();
        self.feature("table_constructor");
        format!("{{{}}}", items.join(", "))
    }

    fn record_constructor(&mut self) -> String {
        let x = self.expr(Kind::Num, 2);
        let y = self.expr(Kind::Num, 2);
        self.feature("table_constructor");
        if self.rng.chance(1, 3) {
            self.feature("nested_tables");
            return format!("{{x = {x}, y = {y}, [\"z\"] = 1, inner = nil}}");
        }
        format!("{{x = {x}, y = {y}}}")
    }

    fn object_constructor(&mut self) -> String {
        let v = self.expr(Kind::Num, 2);
        let class = *self.rng.pick(&["ClassA", "ClassB"]);
        self.feature("metatables");
        format!("{class}.new({v})")
    }

    /// A call of a pure function returning `kind` first, if one is
    /// affordable here.
    fn pure_call_of(&mut self, kind: Kind) -> Option<String> {
        let candidates: Vec<Function> = self
            .functions
            .iter()
            .filter(|function| function.pure && function.results.first() == Some(&kind))
            .cloned()
            .collect();
        let function = self.pick_affordable(&candidates)?;
        let call = self.call(&function);
        Some(if function.results.len() > 1 {
            format!("({call})")
        } else {
            call
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Size, generate};
    use crate::profile::Profile;

    /// Protected kernels raise inside `pcall` and `xpcall`, in every profile,
    /// and the interpreter and the tracing mode agree on their output.
    #[test]
    fn programs_raise_errors_in_protected_kernels_that_both_modes_agree_on() {
        for (profile, compatibility) in Profile::ALL
            .into_iter()
            .zip(oximond::CompatibilityProfile::ALL)
        {
            let mut seen = 0;
            for seed in 0..200 {
                let program = generate(seed, profile, Size::Small);
                if !program.features().contains("protected_kernel") {
                    continue;
                }
                seen += 1;
                if seen > 3 {
                    continue;
                }
                let source = program.render();
                assert!(
                    source.contains("protected kernel section"),
                    "{profile:?} seed {seed}"
                );
                let mut outputs = Vec::new();
                for mode in [
                    oximond::ExecutionMode::Interpreter,
                    oximond::ExecutionMode::Tracing,
                ] {
                    let mut lua = oximond::Lua::builder()
                        .profile(compatibility)
                        .execution_mode(mode)
                        .build()
                        .unwrap();
                    lua.vm_mut().use_synchronous_jit_compilation();
                    let wrapped = format!(
                        "local observations = {{}}\nlocal print = function(line) observations[#observations + 1] = line end\n{source}\nreturn table.concat(observations, '\\n')"
                    );
                    outputs.push(format!("{:?}", lua.exec(&wrapped)));
                }
                assert_eq!(outputs[0], outputs[1], "{profile:?} seed {seed}");
            }
            assert!(seen > 0, "{profile:?} never generates a protected kernel");
        }
    }

    /// Programs of every profile call the table and bit functions a trace
    /// records as operations, with edge operands, and the interpreter and
    /// the tracing mode agree on them.
    #[test]
    fn programs_call_rawget_unpack_getmetatable_and_the_bit_libraries() {
        type Applies = fn(Profile) -> bool;
        let wanted: [(&str, Applies); 4] = [
            ("rawget", |_| true),
            ("unpack", |_| true),
            ("getmetatable", |_| true),
            ("bit_library_edges", |profile| {
                matches!(profile, Profile::LuaJit21 | Profile::Lua52)
            }),
        ];
        for (profile, compatibility) in Profile::ALL
            .into_iter()
            .zip(oximond::CompatibilityProfile::ALL)
        {
            let mut seen = std::collections::BTreeSet::new();
            let mut checked = 0;
            for seed in 0..120 {
                let program = generate(seed, profile, Size::Small);
                let features = program.features();
                let new: Vec<_> = wanted
                    .iter()
                    .filter(|(name, _)| features.contains(name))
                    .map(|(name, _)| *name)
                    .collect();
                if new.is_empty() {
                    continue;
                }
                seen.extend(new);
                if checked >= 3 {
                    continue;
                }
                checked += 1;
                let source = program.render();
                let mut outputs = Vec::new();
                for mode in [
                    oximond::ExecutionMode::Interpreter,
                    oximond::ExecutionMode::Tracing,
                ] {
                    let mut lua = oximond::Lua::builder()
                        .profile(compatibility)
                        .execution_mode(mode)
                        .build()
                        .unwrap();
                    lua.vm_mut().use_synchronous_jit_compilation();
                    let wrapped = format!(
                        "local observations = {{}}\nlocal print = function(line) observations[#observations + 1] = line end\n{source}\nreturn table.concat(observations, '\\n')"
                    );
                    outputs.push(format!("{:?}", lua.exec(&wrapped)));
                }
                assert_eq!(outputs[0], outputs[1], "{profile:?} seed {seed}");
            }
            for (name, applies) in wanted {
                if applies(profile) {
                    assert!(seen.contains(name), "{profile:?} never generates {name}");
                }
            }
        }
    }

    /// Generator 24 draws numeric strings as arithmetic operands in every
    /// profile. A program that does runs the same interpreted and traced.
    #[test]
    fn programs_do_arithmetic_on_numeric_strings_and_both_modes_agree() {
        for (profile, compatibility) in Profile::ALL
            .into_iter()
            .zip(oximond::CompatibilityProfile::ALL)
        {
            let mut checked = 0;
            for seed in 0..120 {
                let program = generate(seed, profile, Size::Small);
                if !program.features().contains("string_arithmetic") {
                    continue;
                }
                if checked >= 3 {
                    break;
                }
                checked += 1;
                let source = program.render();
                let mut outputs = Vec::new();
                for mode in [
                    oximond::ExecutionMode::Interpreter,
                    oximond::ExecutionMode::Tracing,
                ] {
                    let mut lua = oximond::Lua::builder()
                        .profile(compatibility)
                        .execution_mode(mode)
                        .build()
                        .unwrap();
                    lua.vm_mut().use_synchronous_jit_compilation();
                    let wrapped = format!(
                        "local observations = {{}}\nlocal print = function(line) observations[#observations + 1] = line end\n{source}\nreturn table.concat(observations, '\\n')"
                    );
                    outputs.push(format!("{:?}", lua.exec(&wrapped)));
                }
                assert_eq!(outputs[0], outputs[1], "{profile:?} seed {seed}");
            }
            assert!(checked > 0, "{profile:?} never generates string arithmetic");
        }
    }

    #[test]
    fn lua54_seed_2949_keeps_exact_strings_with_the_version_13_stream() {
        let profile = Profile::Lua54;
        let index = Profile::ALL.iter().position(|&p| p == profile).unwrap() as u64;
        let mut mixer = super::Rng::new(0x0b85 ^ (index << 56) ^ (13_u64 << 48));
        let mut generator =
            super::Generator::new(super::Rng::new(mixer.next_u64()), profile, Size::Medium);
        generator.expanded_probes = false;
        generator.scalar_edges = false;
        generator.exact_logs = false;
        generator.pattern_calls = false;
        generator.string_arith = false;
        generator.text_calls = false;
        generator.traversals = false;
        generator.operator_classes = false;
        let program = super::Program {
            header: vec![],
            nodes: generator.program(),
        };
        let source = format!(
            "local observations = {{}}\nlocal print = function(line) observations[#observations + 1] = line end\n{}\nreturn table.concat(observations, '\\n')",
            program.render()
        );
        if let Ok(oracle) = crate::case::find_oracle(profile, None) {
            let command = [
                oracle.command.into_os_string(),
                "-e".into(),
                format!("io.write((function() {source} end)())").into(),
            ];
            let outcome = crate::run::run(
                &command,
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
                &[],
                crate::run::Limits::default(),
            );
            assert!(
                outcome.status.success(),
                "{}",
                String::from_utf8_lossy(&outcome.stderr)
            );
            let expected = String::from_utf8(outcome.stdout).unwrap();
            for mode in [
                oximond::ExecutionMode::Interpreter,
                oximond::ExecutionMode::Tracing,
            ] {
                let mut lua = oximond::Lua::builder()
                    .profile(oximond::CompatibilityProfile::Lua54)
                    .execution_mode(mode)
                    .build()
                    .unwrap();
                lua.vm_mut().use_synchronous_jit_compilation();
                assert_eq!(
                    lua.exec(&source).unwrap(),
                    vec![oximond::Value::String(oximond::LuaString::from(
                        expected.as_str()
                    ))],
                    "{mode:?}"
                );
            }
        }
    }

    #[test]
    fn generated_numeric_text_conversions_canonicalize_arithmetic_nan() {
        for (profile, compatibility) in Profile::ALL
            .into_iter()
            .zip(oximond::CompatibilityProfile::ALL)
        {
            let mut generator =
                super::Generator::new(super::stream(1, profile), profile, Size::Small);
            generator.declare("value", super::Kind::Num, false);
            generator.declare("intvalue", super::Kind::Int, false);
            let mut expressions = std::collections::BTreeMap::new();
            for _ in 0..2048 {
                let expression = generator.str_expr(2, false).replace("intvalue", "value");
                if !expression.contains("value") {
                    continue;
                }
                let route = if expression.starts_with("tostring(") {
                    "tostring"
                } else if expression.contains(" .. string_number(value)") {
                    "concat"
                } else if expression.starts_with("string.format(\"%5.2f") {
                    "fixed"
                } else if expression.starts_with("string.format(\"%g|%x") {
                    "general"
                } else if expression.starts_with("string.format(\"%s %s") {
                    "string"
                } else {
                    continue;
                };
                expressions.entry(route).or_insert(expression);
            }
            assert_eq!(
                expressions.len(),
                5,
                "exercise every numeric conversion route"
            );
            let mut nodes = generator.prelude();
            nodes.extend(generator.classes());
            nodes.extend(generator.library_functions());
            generator.declare("arrayvalue", super::Kind::Arr, false);
            let array_expression = (0..2048)
                .map(|_| generator.str_expr(2, false))
                .find(|expression| {
                    expression.starts_with("table.concat(") && expression.contains("arrayvalue")
                })
                .expect("exercise generated numeric table concatenation");
            let array_expected = array_expression.replace("arrayvalue", "positive_numbers");
            nodes.push(super::Node::leaf(format!("local nan = 0/0\nlocal z = tonumber('0') / -2.25\nlocal arrayvalue = {{nan, -nan, math.huge, -math.huge, 0, z}}\nlocal positive_numbers = {{math.abs(nan), math.abs(nan), math.huge, -math.huge, 0, z}}\nassert(({array_expression}) == ({array_expected}))\nassert(arrayvalue[1] ~= arrayvalue[1] and arrayvalue[2] ~= arrayvalue[2] and 1/arrayvalue[6] < 0)")));
            nodes.push(super::Node::leaf(
                "for _, v in ipairs({0/0, -(0/0)}) do assert(envf(v) == envf(math.abs(v))) assert(ek({code = v}) == ek({code = math.abs(v)})) end",
            ));
            nodes.push(super::Node::leaf(format!(
                "assert(string_number(math.huge) == math.huge)\nassert(string_number(-math.huge) == -math.huge)\nlocal z = tonumber('0') / -2.25\nassert(1/string_number(z) {} 0)",
                if matches!(profile, Profile::Lua51 | Profile::Lua52) { ">" } else { "<" }
            )));
            if generator.dialect.integers {
                nodes.push(super::Node::leaf("for _, v in ipairs({math.mininteger, math.maxinteger, 0}) do assert(math.type(string_number(v)) == 'integer' and string_number(v) == v) end"));
            }
            for expression in expressions.into_values() {
                let expected = expression.replace("value", "math.abs(value)");
                nodes.push(super::Node::leaf(format!("for _, value in ipairs({{0/0, -(0/0)}}) do assert(({expression}) == ({expected}), {expression:?}) end")));
            }
            nodes.push(super::Node::leaf("return true"));
            let program = super::Program {
                header: vec![],
                nodes,
            };
            let mut lua = oximond::Lua::builder()
                .profile(compatibility)
                .build()
                .unwrap();
            assert_eq!(
                lua.exec(program.render()).unwrap(),
                vec![oximond::Value::Boolean(true)]
            );
            if let Ok(oracle) = crate::case::find_oracle(profile, None) {
                let command = [
                    oracle.command.into_os_string(),
                    "-e".into(),
                    program.render().into(),
                ];
                let outcome = crate::run::run(
                    &command,
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
                    &[],
                    crate::run::Limits::default(),
                );
                assert!(
                    outcome.status.success(),
                    "{profile}: {}",
                    String::from_utf8_lossy(&outcome.stderr)
                );
            }
        }
    }

    #[test]
    fn integer_string_arguments_handle_overflow_and_preserve_tagged_integers() {
        for (profile, compatibility) in Profile::ALL
            .into_iter()
            .zip(oximond::CompatibilityProfile::ALL)
        {
            let mut generator =
                super::Generator::new(super::stream(1, profile), profile, Size::Small);
            generator.declare("value", super::Kind::Int, false);
            let small = (0..128)
                .map(|_| generator.small_int_leaf())
                .find(|expr| expr.contains("value"))
                .expect("exercise a variable operand");
            let formatted = (0..128)
                .map(|_| generator.format_expr(3))
                .find(|expr| expr.starts_with("string.format(\"%d\"") && expr.contains("value"))
                .expect("exercise a variable integer format argument");
            let mut nodes = generator.prelude();
            nodes.push(super::Node::leaf(format!(
                concat!(
                    "for _,value in ipairs({{1e100, -1e100, math.huge, -math.huge, 0/0, 1.5}}) do\n",
                    "  local index = {}\n",
                    "  assert(index >= -20 and index <= 300 and index == math.floor(index))\n",
                    "  string.byte('abc', index)\n",
                    "  string.sub('abc', index, index)\n",
                    "  string.rep('abc', index % 3)\n",
                    "  string.char(index % 256)\n",
                    "  string.format('%x%c', index, 65 + index % 26)\n",
                    "  assert(type({}) == 'string')\n",
                    "end\n",
                ),
                small, formatted
            )));
            if generator.dialect.integers {
                nodes.push(super::Node::leaf(format!(
                    "for _,value in ipairs({{math.mininteger, math.maxinteger}}) do assert({formatted} == tostring(value)) end"
                )));
            }
            nodes.push(super::Node::leaf("return true"));
            let program = super::Program {
                header: vec![],
                nodes,
            };
            let mut lua = oximond::Lua::builder()
                .profile(compatibility)
                .build()
                .unwrap();
            assert_eq!(
                lua.exec(program.render()).unwrap(),
                vec![oximond::Value::Boolean(true)]
            );
        }
    }

    #[test]
    fn bitwise_operands_preserve_integers_and_bound_unrepresentable_floats() {
        for profile in [
            oximond::CompatibilityProfile::Lua53,
            oximond::CompatibilityProfile::Lua54,
            oximond::CompatibilityProfile::Lua55,
        ] {
            let mut lua = oximond::Lua::builder().profile(profile).build().unwrap();
            let source = format!(
                concat!(
                    "{}\nreturn bit_operand(math.maxinteger), bit_operand(math.mininteger),\n",
                    "bit_operand(9007199254740992.0), bit_operand(1e100),\n",
                    "bit_operand(-1e100), bit_operand(1.5),\n",
                    "bit_operand(math.huge), bit_operand(-math.huge), bit_operand(0/0)",
                ),
                super::BIT_OPERAND_HELPER,
            );
            let values = lua.exec(&source).unwrap();
            assert_eq!(
                values,
                vec![
                    oximond::Value::Integer(i64::MAX),
                    oximond::Value::Integer(i64::MIN),
                    oximond::Value::Integer(9_007_199_254_740_992),
                    oximond::Value::Integer(0),
                    oximond::Value::Integer(0),
                    oximond::Value::Integer(1),
                    oximond::Value::Integer(0),
                    oximond::Value::Integer(0),
                    oximond::Value::Integer(0),
                ]
            );
            lua.exec(format!(
                "{}\nreturn bit_operand(1e100) ~ 1, ~bit_operand(0/0)",
                super::BIT_OPERAND_HELPER
            ))
            .unwrap();
        }
    }

    #[test]
    fn sorting_equal_numeric_values_orders_subtypes_consistently() {
        for profile in [
            oximond::CompatibilityProfile::Lua53,
            oximond::CompatibilityProfile::Lua54,
            oximond::CompatibilityProfile::Lua55,
        ] {
            let mut lua = oximond::Lua::builder().profile(profile).build().unwrap();
            let source = format!(
                concat!(
                    "{}\nlocal t = {{2, 1.0, 2.0, 1, 1.0, 2}}\n",
                    "table.sort(t, numeric_less)\n",
                    "local ascending = {{}}\n",
                    "for k, v in ipairs(t) do ascending[k] = math.type(v) .. v end\n",
                    "table.sort(t, function(a,b) return numeric_less(b,a) end)\n",
                    "local descending = {{}}\n",
                    "for k, v in ipairs(t) do descending[k] = math.type(v) .. v end\n",
                    "return table.concat(ascending, ','), table.concat(descending, ',')",
                ),
                super::NUMERIC_LESS_HELPER
            );
            let values = lua.exec(&source).unwrap();
            assert_eq!(
                values,
                vec![
                    oximond::Value::String(oximond::LuaString::from(
                        "float1.0,float1.0,integer1,float2.0,integer2,integer2"
                    )),
                    oximond::Value::String(oximond::LuaString::from(
                        "integer2,integer2,float2.0,integer1,float1.0,float1.0"
                    )),
                ]
            );
        }
    }

    #[test]
    fn array_indices_stay_in_sequence_after_float_overflow() {
        for profile in oximond::CompatibilityProfile::ALL {
            let mut lua = oximond::Lua::builder().profile(profile).build().unwrap();
            let source = format!(
                concat!(
                    "{}\nlocal t = {{10, 20, 30}}\n",
                    "for _,v in ipairs({{1e100, -1e100, 1.5, -1.5, math.huge, -math.huge, 0/0}}) do\n",
                    "  local i = array_index(v, #t)\n",
                    "  assert(i >= 1 and i <= #t and i == math.floor(i))\n",
                    "  t[i] = t[i] + 1\nend\n",
                    "assert(#t == 3)\n",
                    "assert(array_index(5, 3) == 3 and array_index(-1, 3) == 3)\n",
                    "assert(array_index(0/0, 3) == 1 and array_index(math.huge, 3) == 1)\n",
                    "assert(array_index(1e100, 2) == 1 and array_index(-1, 2) == 2)\n",
                    "assert(array_index(0/0, 2) == 1 and array_index(-math.huge, 2) == 1)\n",
                    "return true",
                ),
                super::ARRAY_INDEX_HELPER
            );
            assert_eq!(
                lua.exec(&source).unwrap(),
                vec![oximond::Value::Boolean(true)]
            );
        }
    }

    #[test]
    fn class_tostring_canonicalizes_nan_and_profile_specific_zero_sign() {
        for (profile, compatibility) in Profile::ALL
            .into_iter()
            .zip(oximond::CompatibilityProfile::ALL)
        {
            let mut generator =
                super::Generator::new(super::stream(1, profile), profile, Size::Small);
            let mut nodes = vec![super::Node::leaf(format!(
                "local canonical_zero = {}",
                matches!(profile, Profile::Lua51 | Profile::Lua52)
            ))];
            nodes.extend(generator.classes());
            nodes.push(super::Node::leaf(concat!(
                "local nan = 0/0\n",
                "assert(tostring(ClassA.new(nan)) == 'A(nan)')\n",
                "assert(tostring(ClassA.new(-nan)) == 'A(nan)')\n",
                "assert(tostring(ClassB.new(nan)) == 'B(nan)')\n",
                "assert(tostring(ClassB.new(-nan)) == 'B(nan)')\n",
                "assert(tostring(ClassA.new(7)) == 'A(7)')\n",
                "assert(tostring(ClassB.new(7)) == 'B(7)')\n",
                // Produce negative zero at runtime, so stock constant-table
                // deduplication cannot erase the sign before the assertion.
                "local z = tonumber('0') / (-2.25)\n",
                "assert(1/z < 0)\n",
                "assert(string.sub(tostring(z), 1, 1) == '-')\n",
                "if canonical_zero then\n",
                "  assert(tostring(ClassA.new(z)) == 'A(0)')\n",
                "  assert(tostring(ClassB.new(z)) == 'B(0)')\n",
                "else\n",
                "  assert(tostring(ClassA.new(z)) == 'A(' .. tostring(z) .. ')')\n",
                "  assert(tostring(ClassB.new(z)) == 'B(' .. tostring(z) .. ')')\n",
                "end\n",
                "return true",
            )));
            let program = super::Program {
                header: vec![],
                nodes,
            };
            let mut lua = oximond::Lua::builder()
                .profile(compatibility)
                .build()
                .unwrap();
            assert_eq!(
                lua.exec(program.render()).unwrap(),
                vec![oximond::Value::Boolean(true)]
            );
            if let Ok(oracle) = crate::case::find_oracle(profile, None) {
                let command = [
                    oracle.command.into_os_string(),
                    "-e".into(),
                    program.render().into(),
                ];
                let outcome = crate::run::run(
                    &command,
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
                    &[],
                    crate::run::Limits::default(),
                );
                assert!(
                    outcome.status.success(),
                    "{profile}: {:?}\n{}",
                    outcome.status,
                    String::from_utf8_lossy(&outcome.stderr)
                );
            }
        }
    }

    #[test]
    fn a_seed_always_yields_the_same_program() {
        for profile in Profile::ALL {
            for seed in 0..20 {
                let first = generate(seed, profile, Size::Medium).render();
                let second = generate(seed, profile, Size::Medium).render();
                assert_eq!(first, second, "{profile} seed {seed}");
            }
        }
    }

    #[test]
    fn seeds_and_profiles_yield_different_programs() {
        let a = generate(1, Profile::Lua54, Size::Small).render();
        let b = generate(2, Profile::Lua54, Size::Small).render();
        let c = generate(1, Profile::Lua51, Size::Small).render();
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn profile_gated_syntax_stays_in_its_profile() {
        for seed in 0..200 {
            let lua51 = generate(seed, Profile::Lua51, Size::Medium).render();
            assert!(!lua51.contains("//"), "seed {seed} uses // in 5.1");
            assert!(!lua51.contains("goto"), "seed {seed} uses goto in 5.1");
            assert!(
                !lua51.contains("<close>"),
                "seed {seed} uses <close> in 5.1"
            );
            assert!(!lua51.contains("\\x"), "seed {seed} uses \\x in 5.1");
            let lua53 = generate(seed, Profile::Lua53, Size::Medium).render();
            assert!(
                !lua53.contains("<const>"),
                "seed {seed} uses <const> in 5.3"
            );
        }
    }
}
