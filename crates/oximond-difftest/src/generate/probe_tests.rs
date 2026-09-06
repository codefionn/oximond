//! Probe generation must stay valid for stock Lua, even when oximond
//! disagrees with an observation. A mismatch belongs in a campaign report.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use super::{Generator, Size, generate, generate_excluding, stream};
use crate::case::find_oracle;
use crate::profile::Profile;
use crate::program::Program;
use crate::run::{Limits, run};

const FAMILIES: [&str; 7] = [
    "runtime_functions",
    "runtime_coroutines",
    "debug_probes",
    "runtime_types",
    "runtime_metatables",
    "runtime_control",
    "profile_libraries",
];

#[test]
fn overflowing_string_position_seed_finishes_on_stock_lua53() {
    let profile = Profile::Lua53;
    let seed = 3048;
    // Keep the version-8 stream that passed a huge float to string.byte.
    let mut mixer = crate::rng::Rng::new(seed ^ (2 << 56) ^ (8 << 48));
    let mut generator =
        Generator::new(crate::rng::Rng::new(mixer.next_u64()), profile, Size::Large);
    generator.expanded_probes = false;
    generator.pattern_calls = false;
    generator.string_arith = false;
    generator.text_calls = false;
    generator.traversals = false;
    generator.operator_classes = false;
    generator.scalar_edges = false;
    generator.exact_logs = false;
    let program = Program {
        header: vec![],
        nodes: generator.program(),
    };
    assert!(program.features().contains("string_library"));
    let local_oracle = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/lua-libraries/source/lua-5.3.6/src/lua");
    let oracle = find_oracle(profile, None).or_else(|_| find_oracle(profile, Some(&local_oracle)));
    let Ok(oracle) = oracle else {
        eprintln!("Skipping {profile} seed validation: stock interpreter unavailable");
        return;
    };
    let command = [
        oracle.command.into_os_string(),
        "-e".into(),
        program.render().into(),
    ];
    let outcome = run(
        &command,
        Path::new(env!("CARGO_MANIFEST_DIR")),
        &[],
        Limits::default(),
    );
    assert!(
        outcome.status.success(),
        "seed 3048: {:?}\n{}",
        outcome.status,
        String::from_utf8_lossy(&outcome.stderr),
    );
}

#[test]
fn mutable_ipairs_seed_stays_within_the_output_budget() {
    fn assert_quiet(node: &crate::program::Node) {
        assert!(!node.head.contains("out("), "{}", node.head);
        for child in &node.body {
            assert_quiet(child);
        }
    }

    let profile = Profile::Lua55;
    // Pin the version-7 RNG stream that produced >8 MiB at seed 1003.
    // Changing the generator version must not quietly replace the repro.
    let seed = 1003;
    let mut mixer = crate::rng::Rng::new(seed ^ (4 << 56) ^ (7 << 48));
    let mut generator = Generator::new(
        crate::rng::Rng::new(mixer.next_u64()),
        profile,
        Size::Medium,
    );
    generator.expanded_probes = false;
    generator.pattern_calls = false;
    generator.string_arith = false;
    generator.text_calls = false;
    generator.traversals = false;
    generator.operator_classes = false;
    generator.scalar_edges = false;
    generator.exact_logs = false;
    let program = Program {
        header: vec![],
        nodes: generator.program(),
    };

    let mutable_loops: Vec<_> = program
        .paths()
        .into_iter()
        .filter_map(|path| program.node(&path))
        .filter(|node| node.head.contains(" in ipairs(g"))
        .collect();
    assert!(
        !mutable_loops.is_empty(),
        "repro must traverse captured arrays"
    );
    for node in mutable_loops {
        assert_quiet(node);
    }

    let Ok(oracle) = find_oracle(profile, None) else {
        eprintln!("Skipping {profile} output validation: stock interpreter unavailable");
        return;
    };
    let command = [
        oracle.command.into_os_string(),
        "-e".into(),
        program.render().into(),
    ];
    let outcome = run(
        &command,
        Path::new(env!("CARGO_MANIFEST_DIR")),
        &[],
        Limits::default(),
    );
    assert!(
        outcome.status.success(),
        "seed 1003: {:?}, {} output bytes\n{}",
        outcome.status,
        outcome.stdout.len(),
        String::from_utf8_lossy(&outcome.stderr),
    );
}

#[test]
fn generated_programs_do_not_call_filesystem_or_os_functions() {
    for profile in Profile::ALL {
        for seed in 0..64 {
            let source = generate(seed, profile, Size::Medium).render();
            for forbidden in ["io.", "os.", "dofile(", "loadfile(", "require(", "package."] {
                assert!(
                    !source.contains(forbidden),
                    "{profile}/{seed} uses {forbidden}"
                );
            }
        }
    }
}

#[test]
fn every_program_exercises_the_new_families_unless_excluded() {
    let exclude = FAMILIES.into_iter().map(str::to_owned).collect();
    for profile in Profile::ALL {
        for seed in 0..16 {
            let program = generate(seed, profile, Size::Small);
            let features = program.features();
            for family in FAMILIES {
                assert!(features.contains(family), "{profile}/{seed}: {family}");
            }
            let excluded = generate_excluding(seed, profile, Size::Small, &exclude);
            for family in FAMILIES {
                assert!(
                    !excluded.features().contains(family),
                    "{profile}/{seed}: {family}"
                );
            }
        }
    }
}

#[test]
fn seed_window_covers_each_probe_shape() {
    for profile in Profile::ALL {
        let mut features = BTreeSet::new();
        for seed in 0..512 {
            features.extend(generate(seed, profile, Size::Small).features());
        }
        for feature in [
            "function_recursive_results",
            "function_mutual_results",
            "function_vararg_holes",
            "function_shared_upvalues",
            "function_callable_table",
            "function_loaded_environment",
            "coroutine_resume_holes",
            "coroutine_nested_resume",
            "coroutine_wrap_holes",
            "coroutine_resume_rejections",
            "coroutine_running_identity",
            "debug_generic_for_hooks",
            "debug_numeric_for_hooks",
            "debug_while_hooks",
            "debug_repeat_hooks",
            "debug_call_hooks",
            "debug_tail_hooks",
            "debug_getinfo",
            "debug_named_locals",
            "debug_named_upvalues",
            "debug_count_hooks",
            "debug_thread_locals",
            "debug_thread_hooks",
            "debug_break_hooks",
            "types_mixed_arguments",
            "types_mixed_results",
            "types_identity_keys",
            "types_truthiness",
            "types_scalar_conversion",
            "metatable_index_newindex",
            "metatable_call_results",
            "metatable_arithmetic_concat",
            "metatable_order_equality_raw",
            "metatable_globals_environment",
            "metatable_protected_errors",
            "control_nested_break",
            "control_generic_false",
            "control_branch_short_circuit",
            "control_nested_yield",
            "control_break_closures",
        ] {
            assert!(
                features.contains(feature),
                "{profile} never generates {feature}"
            );
        }
        let dialect = profile.features();
        for (feature, expected) in [
            ("coroutine_close_cleanup", dialect.close_and_const),
            ("coroutine_recursive_yield", !dialect.close_and_const),
            ("types_full_userdata", dialect.setfenv),
            ("types_light_userdata", !dialect.setfenv),
            ("control_goto_scopes", dialect.goto),
            ("metatable_length", dialect.len_metamethod),
            ("metatable_mixed_equality", dialect.mixed_eq_metamethod),
        ] {
            assert_eq!(features.contains(feature), expected, "{profile}: {feature}");
        }
    }
}

#[test]
fn reducer_removes_whole_probes_including_their_cleanup() {
    for profile in Profile::ALL {
        let program = generate(0, profile, Size::Small);
        for family in FAMILIES {
            let path = program
                .paths()
                .into_iter()
                .find(|path| {
                    program
                        .node(path)
                        .is_some_and(|node| node.features.contains(&family))
                })
                .expect("probe exists");
            let reduced = program.without(&path).expect("probe can be removed");
            assert!(!reduced.features().contains(family));
            assert!(
                program.unwrapped(&path).is_none(),
                "{family} cannot be split"
            );
        }
    }
}

#[test]
fn probes_finish_and_print_deterministically_on_available_stock_interpreters() {
    let limits = Limits {
        timeout: Duration::from_secs(5),
        output_cap: 64 * 1024,
        ..Limits::default()
    };
    for profile in Profile::ALL {
        let local = match profile {
            Profile::Lua52 => Some("lua-5.2.4"),
            Profile::Lua53 => Some("lua-5.3.6"),
            _ => None,
        };
        let oracle = find_oracle(profile, None).or_else(|err| {
            let Some(version) = local else {
                return Err(err);
            };
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/lua-libraries/source")
                .join(version)
                .join("src/lua");
            find_oracle(profile, Some(&path))
        });
        let Ok(oracle) = oracle else {
            eprintln!("Skipping {profile} probe validation: stock interpreter unavailable");
            continue;
        };
        let mut features = BTreeSet::new();
        for seed in 0..48 {
            let mut generator = Generator::new(stream(seed, profile), profile, Size::Small);
            let mut nodes = generator.prelude();
            nodes.push(generator.function_probe());
            nodes.push(generator.coroutine_probe());
            nodes.push(generator.debug_probe());
            nodes.push(generator.type_probe());
            nodes.push(generator.metatable_probe());
            nodes.push(generator.control_probe());
            nodes.push(generator.language_probe());
            nodes.push(generator.profile_library_probe());
            if profile.features().utf8 {
                nodes.push(generator.utf8_probe());
            }
            let program = Program {
                header: vec![],
                nodes,
            };
            features.extend(program.features());
            let command = [
                oracle.command.clone().into_os_string(),
                "-e".into(),
                format!(
                    "io=nil;os=nil;dofile=nil;loadfile=nil;require=nil;package=nil\n{}",
                    program.render()
                )
                .into(),
            ];
            let first = run(&command, Path::new(env!("CARGO_MANIFEST_DIR")), &[], limits);
            assert!(
                first.status.success(),
                "{profile}/{seed}: {:?}\n{}\n{}",
                first.status,
                String::from_utf8_lossy(&first.stderr),
                program.render()
            );
            let second = run(&command, Path::new(env!("CARGO_MANIFEST_DIR")), &[], limits);
            assert!(
                second.status.success(),
                "{profile}/{seed}: {:?}",
                second.status
            );
            assert_eq!(first.stdout, second.stdout, "{profile}/{seed}");
            assert!(
                !first.stdout.is_empty(),
                "{profile}/{seed} has no observations"
            );
        }
        assert!(
            features.contains("debug_line_hooks"),
            "{profile} misses line-hook probes"
        );
    }
}

#[test]
fn version_specific_libraries_follow_the_profile() {
    for profile in Profile::ALL {
        let program = generate(0, profile, Size::Small);
        let features = program.features();
        let dialect = profile.features();
        for (feature, expected) in [
            ("utf8_roundtrip", dialect.utf8),
            ("utf8_invalid", dialect.utf8),
            ("utf8_iterator_control", dialect.utf8),
            ("utf8_partial_range", dialect.utf8),
            ("utf8_lax", dialect.utf8_lax),
            ("library_string_pack", dialect.integers),
            ("library_string_pack_variants", dialect.integers),
            ("library_string_dump", true),
            ("library_string_dump_upvalues", true),
            ("library_string_dump_strip", dialect.integers),
            ("library_table_create", dialect.lua55),
            ("lua55_global_declarations", dialect.lua55),
            ("lua55_named_varargs", dialect.lua55),
            ("const_capture", dialect.close_and_const),
            ("close_control_flow", dialect.close_and_const),
            ("goto_backward", dialect.goto),
            ("weak_tables_reclaimed", true),
            ("library_integer_math", dialect.integers),
            ("library_table_move", dialect.table_move),
            ("library_legacy_math", !dialect.integers),
            ("library_legacy_table", !dialect.table_pack),
            ("library_bit32", dialect.bit32_library),
            ("library_luajit_bit", dialect.bit_library),
        ] {
            assert_eq!(features.contains(feature), expected, "{profile}: {feature}");
        }
        let excluded = BTreeSet::from(["utf8".to_owned(), "text_calls".to_owned()]);
        for seed in 0..64 {
            let program = generate_excluding(seed, profile, Size::Medium, &excluded);
            assert!(!program.features().contains("utf8"), "{profile}/{seed}");
            assert!(!program.render().contains("utf8."), "{profile}/{seed}");
        }
    }
}

#[test]
fn repeated_sort_seed_uses_a_bounded_write_in_the_inner_loop() {
    fn inspect(node: &crate::program::Node, writes: &mut bool) {
        assert!(!node.head.contains("table.sort("), "{}", node.head);
        *writes |= node.head.starts_with("g4[1] = ");
        for child in &node.body {
            inspect(child, writes);
        }
    }
    // Pin the version-11 stream. Seed 310 used to sort a 12-element table
    // 12 times in each of 95 inner trips, across 366 outer trips. Charging
    // those sorts as single statements let comparator work evade the budget.
    let mut mixer = crate::rng::Rng::new(0x0136 ^ (1 << 56) ^ (11 << 48));
    let mut generator = Generator::new(
        crate::rng::Rng::new(mixer.next_u64()),
        Profile::Lua52,
        Size::Medium,
    );
    generator.expanded_probes = false;
    generator.pattern_calls = false;
    generator.string_arith = false;
    generator.text_calls = false;
    generator.traversals = false;
    generator.operator_classes = false;
    generator.scalar_edges = false;
    generator.exact_logs = false;
    let program = Program {
        header: vec![],
        nodes: generator.program(),
    };
    let inner = program
        .paths()
        .into_iter()
        .find_map(|path| {
            program
                .node(&path)
                .filter(|node| node.head == "for j116 = 1, 95 do")
        })
        .expect("the original 95-trip loop must remain in the repro");
    let mut bounded_write = false;
    inspect(inner, &mut bounded_write);
    assert!(
        bounded_write,
        "expensive sort must become a bounded array write"
    );
}

#[test]
fn nonfinite_recursion_seed_normalizes_depth_arguments() {
    // The version-12 stream made g1 infinite in a kernel, then passed
    // g1 % 40 (NaN) to sumdown. Its exit comparison never became true.
    let mut mixer = crate::rng::Rng::new(0x002d ^ (2 << 56) ^ (12 << 48));
    let mut generator = Generator::new(
        crate::rng::Rng::new(mixer.next_u64()),
        Profile::Lua53,
        Size::Medium,
    );
    generator.expanded_probes = false;
    generator.pattern_calls = false;
    generator.string_arith = false;
    generator.text_calls = false;
    generator.traversals = false;
    generator.operator_classes = false;
    generator.scalar_edges = false;
    generator.exact_logs = false;
    let program = Program {
        header: vec![],
        nodes: generator.program(),
    };
    let source = program.render();
    assert!(
        source.contains("g1 = g1 + (g1 +"),
        "retain overflow-producing kernel"
    );
    assert!(source.contains("sumdown((array_index(g1, 40) - 1), v89)"));
    assert!(!source.contains("sumdown((g1) % 40"));
    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/lua-libraries/source/lua-5.3.6/src/lua");
    let Ok(oracle) =
        find_oracle(Profile::Lua53, None).or_else(|_| find_oracle(Profile::Lua53, Some(&local)))
    else {
        eprintln!("Skipping Lua53 recursion repro: stock interpreter unavailable");
        return;
    };
    let outcome = run(
        &[oracle.command.into_os_string(), "-e".into(), source.into()],
        Path::new(env!("CARGO_MANIFEST_DIR")),
        &[],
        Limits {
            timeout: Duration::from_secs(5),
            output_cap: 1024 * 1024,
            ..Limits::default()
        },
    );
    assert!(
        outcome.status.success(),
        "{:?}: {}",
        outcome.status,
        String::from_utf8_lossy(&outcome.stderr)
    );
    let stdout = String::from_utf8_lossy(&outcome.stdout);
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("K11#") && line.contains("finf")),
        "retain infinite state before the formerly unbounded recursion"
    );
    assert!(
        stdout.lines().any(|line| line.starts_with("END#")),
        "repro must finish"
    );
}

#[test]
fn recursion_depths_stay_finite_integral_and_within_bounds() {
    let mut lua = oximond::Lua::builder()
        .profile(oximond::CompatibilityProfile::Lua53)
        .build()
        .unwrap();
    let source = format!(
        "{}\nlocal inputs = {{math.huge, -math.huge, 0/0, 1.5, -1.5, 1e100, -1e100, 1.7976931348623157e308, -1.7976931348623157e308, 0, 39, 40}}\n\
         for _, bound in ipairs({{12, 14, 40, 50, 321}}) do\n\
           for _, value in ipairs(inputs) do\n\
             local index = array_index(value, bound)\n\
             assert(index >= 1 and index <= bound and index == math.floor(index))\n\
             local depth = index - 1\n\
             assert(depth >= 0 and depth < bound and depth == math.floor(depth))\n\
           end\n\
         end\n\
         return array_index(math.huge, 40) - 1, array_index(-math.huge, 40) - 1, array_index(0/0, 40) - 1",
        super::ARRAY_INDEX_HELPER,
    );
    assert_eq!(
        lua.exec(&source).unwrap(),
        vec![oximond::Value::Integer(0); 3]
    );
    // Older Lua's floor-based modulo can overflow even for finite inputs.
    // Judge the bounds with every available stock version, not just oximond.
    for profile in Profile::ALL {
        let local = match profile {
            Profile::Lua52 => Some("lua-5.2.4"),
            Profile::Lua53 => Some("lua-5.3.6"),
            _ => None,
        };
        let oracle = find_oracle(profile, None).or_else(|err| {
            let Some(version) = local else {
                return Err(err);
            };
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/lua-libraries/source")
                .join(version)
                .join("src/lua");
            find_oracle(profile, Some(&path))
        });
        let Ok(oracle) = oracle else {
            eprintln!("Skipping {profile} depth bound validation: stock interpreter unavailable");
            continue;
        };
        let outcome = run(
            &[
                oracle.command.into_os_string(),
                "-e".into(),
                source.clone().into(),
            ],
            Path::new(env!("CARGO_MANIFEST_DIR")),
            &[],
            Limits::default(),
        );
        assert!(
            outcome.status.success(),
            "{profile}: {:?}: {}",
            outcome.status,
            String::from_utf8_lossy(&outcome.stderr)
        );
    }
}

#[test]
fn text_calls_appear_only_where_the_libraries_exist() {
    for profile in Profile::ALL {
        let dialect = profile.features();
        let seen = (0..60).any(|seed| {
            generate(seed, profile, Size::Medium)
                .features()
                .contains("text_calls")
        });
        assert_eq!(seen, dialect.utf8, "{profile}");
    }
}

/// FNV-1a over the bytes of a program, to pin a stream without keeping the
/// program.
fn fingerprint(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Generators 27 and 28 added the traversals behind `traversals` and the
/// operator sections behind `operator_classes`. Without them, the version-26
/// stream of a seed is the program generator 26
/// made, which these fingerprints (taken from the generator 26 binary) pin.
#[test]
fn the_version_26_streams_survive_the_traversal_and_operator_switches() {
    let cases: [(Profile, u64, Size, u64); 12] = [
        (Profile::Lua51, 7, Size::Small, 2_618_446_608_833_497_340),
        (Profile::Lua51, 91, Size::Medium, 16_829_954_938_915_608_574),
        (Profile::Lua52, 7, Size::Small, 6_159_398_584_791_010_493),
        (Profile::Lua52, 91, Size::Medium, 8_562_529_869_826_376_848),
        (Profile::Lua53, 7, Size::Small, 4_965_578_771_302_658_482),
        (Profile::Lua53, 91, Size::Medium, 7_577_150_440_263_480_546),
        (Profile::Lua54, 7, Size::Small, 15_462_953_585_270_054_099),
        (Profile::Lua54, 91, Size::Medium, 825_188_384_583_041_533),
        (Profile::Lua55, 7, Size::Small, 18_248_886_637_194_670_383),
        (Profile::Lua55, 91, Size::Medium, 8_030_379_454_665_270_706),
        (Profile::LuaJit21, 7, Size::Small, 686_841_563_378_735_533),
        (
            Profile::LuaJit21,
            91,
            Size::Medium,
            17_689_880_551_331_421_956,
        ),
    ];
    for (profile, seed, size, expected) in cases {
        let index = Profile::ALL
            .iter()
            .position(|&candidate| candidate == profile)
            .unwrap() as u64;
        let mut mixer = crate::rng::Rng::new(seed ^ (index << 56) ^ (26 << 48));
        let mut generator = Generator::new(crate::rng::Rng::new(mixer.next_u64()), profile, size);
        generator.traversals = false;
        generator.operator_classes = false;
        let program = Program {
            header: vec![],
            nodes: generator.program(),
        };
        assert_eq!(
            fingerprint(&program.render()),
            expected,
            "{profile} seed {seed}"
        );
    }
}

#[test]
fn operator_sections_appear_in_every_profile_and_can_be_excluded() {
    for profile in Profile::ALL {
        let seen = (0..60).any(|seed| {
            generate(seed, profile, Size::Medium)
                .features()
                .contains("operator_classes")
        });
        assert!(seen, "{profile} never generates an operator section");
        let excluded = BTreeSet::from(["operator_classes".to_owned()]);
        assert!(
            (0..60).all(|seed| {
                !generate_excluding(seed, profile, Size::Medium, &excluded)
                    .features()
                    .contains("operator_classes")
            }),
            "{profile}"
        );
    }
}

/// An operator section is valid Lua for stock Lua of its profile, and ends
/// without an error the generator did not plant.
#[test]
fn operator_sections_run_on_the_stock_interpreters() {
    for profile in Profile::ALL {
        let Ok(oracle) = find_oracle(profile, None) else {
            eprintln!("Skipping {profile} operator sections: stock interpreter unavailable");
            continue;
        };
        let mut checked = 0;
        for seed in 0..80 {
            let program = generate(seed, profile, Size::Medium);
            if !program.features().contains("operator_classes") {
                continue;
            }
            checked += 1;
            let source = format!(
                "io.write((function() local observations = {{}} local print = function(line) observations[#observations + 1] = line end\n{}\nreturn table.concat(observations, '\\n') end)())",
                program.render()
            );
            let outcome = run(
                &[
                    oracle.command.clone().into_os_string(),
                    "-e".into(),
                    source.into(),
                ],
                Path::new(env!("CARGO_MANIFEST_DIR")),
                &[],
                Limits {
                    timeout: Duration::from_secs(60),
                    ..Limits::default()
                },
            );
            assert!(
                outcome.status.success(),
                "{profile} seed {seed}: {:?}\n{}",
                outcome.status,
                String::from_utf8_lossy(&outcome.stderr)
            );
            if checked >= 6 {
                break;
            }
        }
        assert!(checked > 0, "{profile}");
    }
}

#[test]
fn table_traversals_appear_on_every_profile_and_can_be_left_out() {
    for profile in Profile::ALL {
        let seen = (0..60).any(|seed| {
            generate(seed, profile, Size::Medium)
                .features()
                .contains("table_traversal")
        });
        assert!(seen, "{profile}");
        let excluded = BTreeSet::from(["table_traversal".to_owned()]);
        for seed in 0..60 {
            let program = generate_excluding(seed, profile, Size::Medium, &excluded);
            assert!(
                !program.features().contains("table_traversal"),
                "{profile}/{seed}"
            );
            assert!(!program.render().contains("local tq"), "{profile}/{seed}");
        }
    }
}
