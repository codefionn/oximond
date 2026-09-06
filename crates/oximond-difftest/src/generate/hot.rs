//! Opt-in bias toward hot loops, for testing the tracing JIT against the
//! interpreter.
//!
//! The default generator mixes cold sections, probes that run once and hot
//! kernels. This bias drops the one-shot probes and spends the program on
//! loops that pass the trace hot threshold: kernels with thousands of trips
//! that read and write arrays and records, call functions, bump closure
//! upvalues, build strings and string keys, switch types part way (guard
//! exits and side traces), nest short loops and branch. A [`HotKnobs`] value
//! sets the shape, so a property test can shrink it. Programs from the
//! default entry points stay as they were.

use super::{Generator, Kind, Size};
use crate::program::Node;
use crate::rng::Rng;

/// Shape of a hot program. Every field shrinks toward a smaller, simpler
/// program when its value gets smaller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HotKnobs {
    /// Loop sections after the helper functions (at least one).
    pub sections: u32,
    /// Statements per kernel loop body, before nested ones.
    pub statements: u32,
    /// Base trip count of a kernel loop. The generator adds up to half again.
    pub trips: u32,
    /// Nesting depth of branches and inner loops in a kernel.
    pub depth: u32,
    /// Value types in a kernel besides integers and floats: [`Self::ARRAYS`],
    /// [`Self::RECORDS`].
    pub types: u8,
    /// Statement and section families: the `*` constants below.
    pub features: u8,
    /// Puts [`ANCHOR_KERNEL`] before the other sections, a loop the trace
    /// compiler always compiles, so the program enters native code however
    /// the random sections turn out.
    pub anchor: bool,
}

/// A loop of integer arithmetic, one array read and one array write, a branch
/// and no calls: the recorder follows all of it, and it runs enough guest
/// instructions (`12000` trips of about 30 instructions) to repay a compile
/// (`COMPILE_REPAYING_GUEST_WORK` is 262144). It is plain text, so adding it
/// does not change what the seed draws for the other sections.
pub const ANCHOR_KERNEL: &str = "do -- anchor kernel
  local a, b, c, n = 7, 0, 0, 0
  local t = {1, 2, 3, 4, 5, 6, 7, 8}
  for i = 1, 12000 do
    a = (a * 31 + i) % 1000003
    local x = t[a % 8 + 1]
    b = (b + x * (a % 13)) % 1000003
    if a % 3 == 0 then c = c + 1 else n = n + a % 5 end
    t[i % 8 + 1] = (x + a) % 1000
  end
  out(\"ANCHOR\", a, b, c, n, t[1], t[8])
end";

impl HotKnobs {
    /// The anchor kernel and no random section. A property run measures what
    /// this does under each variant, so it can tell native code the anchor
    /// entered from native code the random sections entered.
    #[must_use]
    pub const fn anchor_only() -> Self {
        Self {
            sections: 0,
            statements: 1,
            trips: 2000,
            depth: 0,
            types: 0,
            features: 0,
            anchor: true,
        }
    }

    pub const ARRAYS: u8 = 1;
    pub const RECORDS: u8 = 2;

    /// String building, `tostring` and string-keyed table writes.
    pub const STRINGS: u8 = 1;
    /// Calls to local functions with a branch inside.
    pub const CALLS: u8 = 2;
    /// Upvalue writes through closures, open and closed.
    pub const CLOSURES: u8 = 4;
    /// Kernels that `pcall` raising functions.
    pub const PROTECTED: u8 = 8;
    /// Operator metamethod classes.
    pub const OPERATORS: u8 = 16;
    /// Cold and warm sections between the kernels: loops that rejoin.
    pub const COLD: u8 = 32;
    /// Mid-loop type changes that force guard exits.
    pub const PHASES: u8 = 64;
    /// Constructs the trace recorder rejects (`rawget`, a key chosen with
    /// `and`/`or`). A loop with one never compiles, so without this bit the
    /// kernels avoid them and most cases reach native code.
    pub const UNSUPPORTED: u8 = 128;
    pub const ALL_FEATURES: u8 = 255;

    pub const MIN_TRIPS: u32 = 2_000;

    /// The smallest program the knobs describe.
    #[must_use]
    pub const fn minimal() -> Self {
        Self {
            sections: 1,
            statements: 2,
            trips: Self::MIN_TRIPS,
            depth: 1,
            types: 0,
            features: 0,
            anchor: false,
        }
    }

    /// Knobs drawn from `seed` within the bounds of `size`, so a campaign
    /// seed alone replays the program.
    ///
    /// # Panics
    ///
    /// Never: the drawn values are small and fit their types.
    #[must_use]
    pub fn for_seed(seed: u64, size: Size) -> Self {
        let mut rng = Rng::new(seed ^ 0x686f_742d_6b6e_6f62);
        let (low, high) = match size {
            Size::Small => (1, 2),
            Size::Medium => (2, 4),
            Size::Large => (4, 6),
        };
        let trips = if rng.chance(1, 2) {
            rng.range(3_000, 8_000)
        } else {
            rng.range(10_000, 30_000)
        };
        let mut features = 0;
        for bit in 0..7 {
            if rng.chance(3, 4) {
                features |= 1 << bit;
            }
        }
        if rng.chance(1, 4) {
            features |= Self::UNSUPPORTED;
        }
        Self {
            sections: u32::try_from(rng.range(low, high)).expect("small"),
            statements: u32::try_from(rng.range(3, 8)).expect("small"),
            trips: u32::try_from(trips).expect("trip counts fit u32"),
            depth: u32::try_from(rng.range(1, 3)).expect("small"),
            types: u8::try_from(rng.range(1, 3)).expect("small"),
            features: features | Self::STRINGS | Self::PHASES,
            anchor: false,
        }
    }

    /// The knobs in a form that can sit in a program header and be parsed
    /// back.
    #[must_use]
    pub fn describe(&self) -> String {
        let anchor = if self.anchor { ",anchor=1" } else { "" };
        format!(
            "sections={},statements={},trips={},depth={},types={},features={}{anchor}",
            self.sections, self.statements, self.trips, self.depth, self.types, self.features
        )
    }

    pub(super) const fn has(&self, feature: u8) -> bool {
        self.features & feature != 0
    }
}

/// A helper a kernel declares before its loop, for `hot_statement` to use.
#[derive(Clone, Debug)]
pub(super) enum Helper {
    Call(String),
    /// An open upvalue: the counter and the function that bumps it.
    Upvalue(String, String),
    /// A closed upvalue: a closure from a factory.
    Closure(String),
    Hash(String),
}

impl Generator {
    /// Whether a kernel being built leaves out what the recorder rejects.
    pub(super) fn hot_avoids_unsupported(&self) -> bool {
        self.frame.kernel
            && self
                .hot
                .is_some_and(|knobs| !knobs.has(HotKnobs::UNSUPPORTED))
    }

    /// The sections of a hot program: kernels first, other families by
    /// the knobs.
    pub(super) fn hot_sections(&mut self, knobs: HotKnobs) -> Vec<Node> {
        let mut nodes = Vec::new();
        // Zero sections is the anchor alone (see `HotKnobs::anchor_only`).
        let sections = if knobs.anchor && knobs.sections == 0 {
            0
        } else {
            knobs.sections.max(1)
        };
        for index in 0..i64::from(sections) {
            let weights = [
                6,
                if knobs.has(HotKnobs::PROTECTED) && self.allowed("protected_kernel") {
                    2
                } else {
                    0
                },
                if knobs.has(HotKnobs::OPERATORS)
                    && self.operator_classes
                    && self.allowed("operator_classes")
                {
                    2
                } else {
                    0
                },
                if knobs.has(HotKnobs::COLD) { 2 } else { 0 },
            ];
            let choice = if index == 0 {
                0
            } else {
                self.rng.weighted(&weights)
            };
            nodes.push(match choice {
                0 => self.kernel_section(index),
                1 => self.protected_kernel_section(index),
                2 => self.operator_section(index),
                _ => self.section(index),
            });
        }
        nodes
    }

    /// Declarations at the top of a kernel for the families in `knobs`.
    pub(super) fn hot_prefix(&mut self, knobs: HotKnobs) -> Vec<Node> {
        self.hot_helpers.clear();
        let mut nodes = Vec::new();
        if knobs.has(HotKnobs::CALLS) {
            let name = self.fresh("hf");
            nodes.push(Node::leaf(format!(
                "local function {name}(a, b) if a > b then return (a - b) % 1000003 end return (a * 3 + b) % 1000003 end"
            )));
            self.hot_helpers.push(Helper::Call(name));
        }
        if knobs.has(HotKnobs::CLOSURES) {
            let counter = self.fresh("hc");
            let bump = self.fresh("hb");
            nodes.push(Node::leaf(format!("local {counter} = 0")));
            nodes.push(Node::leaf(format!(
                "local function {bump}(d) {counter} = {counter} + d return {counter} end"
            )));
            self.hot_helpers.push(Helper::Upvalue(counter, bump));
            let factory = self.fresh("hm");
            let closure = self.fresh("hg");
            nodes.push(Node::leaf(format!(
                "local function {factory}() local n = 0 return function(d) n = n + d return n end end"
            )));
            nodes.push(Node::leaf(format!("local {closure} = {factory}()")));
            self.hot_helpers.push(Helper::Closure(closure));
        }
        if knobs.has(HotKnobs::STRINGS) {
            let name = self.fresh("hh");
            nodes.push(Node::leaf(format!("local {name} = {{}}")));
            self.hot_helpers.push(Helper::Hash(name));
        }
        nodes
    }

    /// Observations after a kernel loop for the helpers it declared.
    pub(super) fn hot_suffix(&mut self) -> Vec<Node> {
        let helpers = std::mem::take(&mut self.hot_helpers);
        let mut nodes = Vec::new();
        for helper in helpers {
            match helper {
                Helper::Call(_) => {}
                Helper::Upvalue(counter, _) => {
                    let tag = self.tag("U");
                    nodes.push(Node::leaf(format!("out({tag}, {counter})")));
                }
                Helper::Closure(closure) => {
                    let tag = self.tag("U");
                    nodes.push(Node::leaf(format!("out({tag}, {closure}(0))")));
                }
                Helper::Hash(name) => {
                    let tag = self.tag("H");
                    nodes.push(Node::leaf(format!(
                        "do local total, count = 0, 0 for _, v in pairs({name}) do total = total + v count = count + 1 end out({tag}, total, count) end"
                    )));
                }
            }
        }
        nodes
    }

    /// One extra kernel statement from the helpers' families, or `None`
    /// when no family is on.
    pub(super) fn hot_statement(&mut self) -> Option<Node> {
        if self.hot_helpers.is_empty() {
            return None;
        }
        let helper = self.rng.pick(&self.hot_helpers.clone()).clone();
        let targets = self.writable(Kind::Int);
        if targets.is_empty() {
            return None;
        }
        let target = self.rng.pick(&targets).name.clone();
        let e = self.expr(Kind::Int, 1);
        let text = match helper {
            Helper::Call(name) => {
                self.feature("hot_call");
                let value = self.store_int_or_num(Kind::Int, format!("{name}({target}, {e})"));
                format!("{target} = {value}")
            }
            Helper::Upvalue(_, bump) => {
                self.feature("hot_upvalue");
                let value = self.store_int_or_num(Kind::Int, format!("{bump}({e}) + {target}"));
                format!("{target} = {value}")
            }
            Helper::Closure(closure) => {
                self.feature("hot_closure");
                let value = self.store_int_or_num(Kind::Int, format!("{closure}({e}) + {target}"));
                format!("{target} = {value}")
            }
            Helper::Hash(name) => match self.rng.index(3) {
                0 => {
                    self.feature("hot_string_key");
                    let other = self.expr(Kind::Int, 1);
                    format!(
                        "{name}[\"k\" .. ({e} % 7)] = ({name}[\"k\" .. ({other} % 7)] or 0) + 1"
                    )
                }
                1 => {
                    self.feature("hot_tostring");
                    let value =
                        self.store_int_or_num(Kind::Int, format!("#tostring({e}) + {target}"));
                    format!("{target} = {value}")
                }
                _ => {
                    self.feature("hot_string_byte");
                    let value = self.store_int_or_num(
                        Kind::Int,
                        format!("string.byte(tostring({e}), 1) + {target}"),
                    );
                    format!("{target} = {value}")
                }
            },
        };
        Some(self.leaf(text))
    }
}

#[cfg(test)]
mod tests {
    use super::HotKnobs;
    use crate::generate::{Size, generate, generate_hot, generate_hot_seed};
    use crate::profile::Profile;

    #[test]
    fn a_hot_seed_always_yields_the_same_program() {
        for profile in Profile::ALL {
            let first = generate_hot_seed(7, profile, Size::Medium).render();
            assert_eq!(first, generate_hot_seed(7, profile, Size::Medium).render());
            assert_ne!(first, generate_hot_seed(8, profile, Size::Medium).render());
        }
    }

    #[test]
    fn the_bias_is_opt_in_and_leaves_default_programs_alone() {
        let hot = generate_hot_seed(3, Profile::Lua54, Size::Medium).render();
        let plain = generate(3, Profile::Lua54, Size::Medium).render();
        assert_ne!(hot, plain);
        assert!(hot.lines().next().unwrap().contains(" hot="));
        assert!(!plain.lines().next().unwrap().contains(" hot="));
        // The default program for a seed is pinned by
        // `a_seed_always_yields_the_same_program`; it must not depend on the
        // hot generator having run first.
        assert_eq!(plain, generate(3, Profile::Lua54, Size::Medium).render());
    }

    #[test]
    fn knobs_shape_the_program() {
        let minimal = generate_hot(5, Profile::Lua54, &HotKnobs::minimal());
        let all = HotKnobs {
            sections: 4,
            statements: 8,
            trips: 9_000,
            depth: 3,
            types: HotKnobs::ARRAYS | HotKnobs::RECORDS,
            features: HotKnobs::ALL_FEATURES,
            anchor: false,
        };
        let big = generate_hot(5, Profile::Lua54, &all);
        assert!(big.size() > minimal.size());
        let text = big.render();
        assert!(text.contains("kernel section"));
        assert_eq!(
            minimal.render().matches("kernel section").count(),
            1,
            "one section"
        );
    }

    #[test]
    fn every_hot_family_shows_up_over_a_few_seeds() {
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..40 {
            let knobs = HotKnobs {
                sections: 3,
                statements: 8,
                trips: 5_000,
                depth: 3,
                types: 3,
                features: HotKnobs::ALL_FEATURES,
                anchor: false,
            };
            seen.extend(generate_hot(seed, Profile::Lua54, &knobs).features());
        }
        for family in [
            "hot_call",
            "hot_upvalue",
            "hot_closure",
            "hot_string_key",
            "hot_tostring",
            "phase_number_subtype",
            "nested_loops",
            "protected_kernel",
        ] {
            assert!(seen.contains(family), "{family} missing from {seen:?}");
        }
    }

    #[test]
    fn the_anchor_kernel_leaves_the_other_sections_as_they_were() {
        let plain = HotKnobs::for_seed(11, Size::Medium);
        let anchored = HotKnobs {
            anchor: true,
            ..plain
        };
        let without = generate_hot(11, Profile::Lua54, &plain).render();
        let with = generate_hot(11, Profile::Lua54, &anchored).render();
        assert!(!without.contains("anchor kernel"));
        assert!(with.contains("do -- anchor kernel"));
        assert!(with.lines().next().unwrap().ends_with(",anchor=1"));
        // Only the anchor and the header's knob list differ.
        let body = |text: &str| text.lines().skip(1).collect::<Vec<_>>().join("\n");
        let removed = with.replacen(super::ANCHOR_KERNEL, "", 1);
        assert_eq!(body(&removed).replace("\n\n", "\n"), body(&without));
    }
}
