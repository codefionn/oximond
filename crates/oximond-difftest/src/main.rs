use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use oximond_difftest::campaign::{
    ArtifactContext, Seeds, Settings, add_to_corpus, check_corpus, program_from_source,
    run_campaign, save_artifact,
};
use oximond_difftest::case::{
    Engines, Expectation, Mode, Oracle, Verdict, find_oracle, first_difference_with, run_case,
    run_case_with_expectation,
};
use oximond_difftest::compare::{
    Comparison, DEFAULT_ABSOLUTE_TOLERANCE, DEFAULT_RELATIVE_TOLERANCE,
};
use oximond_difftest::generate::{EXCLUDABLE, Size, generate_excluding, generate_hot_seed};
use oximond_difftest::jit::{DEFAULT_MIN_NATIVE_SHARE, check_non_vacuity, summarize};
use oximond_difftest::profile::Profile;
use oximond_difftest::program::Program;
use oximond_difftest::reduce::reduce;
use oximond_difftest::run::Limits;

const HELP: &str = "\
oximond-difftest: seeded Lua programs, run by stock Lua and by oximond, and by
oximond's JIT variants against its own interpreter

USAGE
  oximond-difftest gen --seed N --profile P [--size S]
  oximond-difftest run [SEEDS] [OPTIONS]
  oximond-difftest run-jit [SEEDS] [OPTIONS]
                      hot-loop programs; oximond's interpreter is the oracle and
                      tracing-mir (the shipped JIT), tracing-mir-profile,
                      tracing-mir-only and tracing (Cranelift only) are the
                      variants. Exits 1 on a mismatch or when too few cases
                      ran native code
  oximond-difftest replay --seed N --profile P [OPTIONS]
  oximond-difftest replay-file FILE [--profile P] [OPTIONS]
  oximond-difftest reject-file FILE [--profile P] [OPTIONS]
                      compare compiler rejection without running the source
  oximond-difftest reduce-file FILE [--profile P] [OPTIONS]
  oximond-difftest reduce-modes --seed N --profile P [OPTIONS]
                      reduce while the interpreter and tracing mode both
                      finish and print different output (no oracle needed)
  oximond-difftest corpus-add DIR --seed N --profile P [--name NAME] [OPTIONS]
  oximond-difftest corpus-check DIR [OPTIONS]
  oximond-difftest run-fixtures [OPTIONS]
  oximond-difftest run-workloads [OPTIONS]
  oximond-difftest run-rejections [OPTIONS]

SEEDS
  --seed N[,N...]     one seed or a list
  --seeds A..B        seeds A through B-1
  --count N           N seeds starting at --start (default 0)
  --duration D        stop starting cases after D (90, 90s, 30m, 2h, 1d); without
                      a seed list above, run random seeds until then
  --master-seed N     seed of the random seed stream (default: fresh each run)

OPTIONS
  --profiles LIST     comma list, `all` (default) or `oracle` (lua51,lua54,lua55,luajit21)
  --profile P         one profile
  --size S            small | medium (default) | large
  --float-compare MODE exact (default) | approximate for tagged numeric observations
  --float-abs-tol N   absolute tolerance (approximate mode, default 1e-12)
  --float-rel-tol N   relative tolerance (approximate mode, default 1e-12)
  --exclude LIST      feature families to leave out (changes the programs):
                      call_result_to_variable,float_modulo,format_c_zero,
                      tonumber_base,math_random,rep_separator,format_q,coroutine_isyieldable,yield_across_pcall,
                      error_number,error_level2,error_no_integer,error_bad_argument,
                      error_string_arith,coroutine_error,xpcall_lua_handler,
                      yield_boundaries,stdlib_edges,runtime_functions,
                      runtime_coroutines,debug_probes,runtime_types,
                      runtime_metatables,runtime_control,profile_libraries,utf8,patterns,string_arithmetic,text_calls,table_traversal,
                      operator_classes
  --modes LIST        default interpreter,tracing-mir,tracing-mir-profile,
                      tracing-mir-only,tracing; also tracing-async,
                      tracing-profile,tracing-mir-crossprofile,
                      tracing-mir-async (run-jit default: the same without
                      interpreter). With a stock oracle, the JIT modes are
                      also compared with the interpreter mode
  --variants LIST     run-jit: same as --modes
  --interpreter-oracle  oximond --execution interpreter judges, not stock Lua
  --hot               hot-loop programs (gen, replay; always on in run-jit)
  --min-native-share F  run-jit: fail if fewer cases than F (default 0.6) of a
                      variant entered native code, or a MIR variant compiled
                      no tree through MIR (0 turns the share check off)
  --oximond PATH      binary under test (default target/release/oximond)
  --oracle P=CMD      stock interpreter for profile P (default lua5.1, lua5.4, ...)
  --jobs N            parallel cases (default 4)
  --out DIR           artifacts and reports (default target/difftest)
  --trace-profile-record-dir DIR  record one profile per tracing case
  --trace-profile-load-dir DIR    load profiles for tracing cases
  --timeout SECS      per process (default 20)
  --memory MIB        address-space cap per process (default 2048, 0 = none)
  --reduce            reduce each mismatch
  --reduce-attempts N (default 400)   --reduce-seconds S (default 300)
  --reruns N          reruns that decide whether a mismatch is stable (default 2)
";

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    match run(&arguments) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("oximond-difftest: {message}");
            ExitCode::from(2)
        }
    }
}

#[derive(Default)]
struct Options {
    positional: Vec<String>,
    values: BTreeMap<String, Vec<String>>,
    flags: Vec<String>,
}

impl Options {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut options = Self::default();
        let mut index = 0;
        while index < arguments.len() {
            let argument = &arguments[index];
            if let Some(name) = argument.strip_prefix("--") {
                if matches!(name, "reduce" | "help" | "interpreter-oracle" | "hot") {
                    options.flags.push(name.to_owned());
                } else {
                    let value = arguments
                        .get(index + 1)
                        .ok_or_else(|| format!("--{name} needs a value"))?;
                    options
                        .values
                        .entry(name.to_owned())
                        .or_default()
                        .push(value.clone());
                    index += 1;
                }
            } else {
                options.positional.push(argument.clone());
            }
            index += 1;
        }
        Ok(options)
    }

    fn value(&self, name: &str) -> Option<&str> {
        self.values
            .get(name)
            .and_then(|values| values.last())
            .map(String::as_str)
    }

    fn number<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        self.value(name).map_or(Ok(default), |text| {
            text.parse()
                .map_err(|_| format!("--{name}: not a number: {text}"))
        })
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|flag| flag == name)
    }
}

fn run(arguments: &[String]) -> Result<ExitCode, String> {
    let Some((command, rest)) = arguments.split_first() else {
        print!("{HELP}");
        return Ok(ExitCode::SUCCESS);
    };
    let mut options = Options::parse(rest)?;
    if command == "run-jit" {
        options.flags.push("interpreter-oracle".to_owned());
        options.flags.push("hot".to_owned());
    }
    if options.flag("help") || command == "help" || command == "--help" {
        print!("{HELP}");
        return Ok(ExitCode::SUCCESS);
    }
    match command.as_str() {
        "gen" => {
            let profile = single_profile(&options, None)?;
            let seed = options.number("seed", 0_u64)?;
            let program = if options.flag("hot") {
                generate_hot_seed(seed, profile, size(&options)?)
            } else {
                generate_excluding(seed, profile, size(&options)?, &exclude(&options)?)
            };
            print!("{}", program.render());
            Ok(ExitCode::SUCCESS)
        }
        "run" => campaign(&options),
        "run-jit" => run_jit(&options),
        "run-fixtures" => run_suite(&options, false),
        "run-workloads" => run_suite(&options, true),
        "run-rejections" => run_rejections(&options),
        "replay" => {
            let profile = single_profile(&options, None)?;
            let seed: u64 = options
                .value("seed")
                .ok_or("replay needs --seed")?
                .parse()
                .map_err(|_| "--seed: not a number")?;
            let size = size(&options)?;
            let program = if options.flag("hot") {
                generate_hot_seed(seed, profile, size)
            } else {
                generate_excluding(seed, profile, size, &exclude(&options)?)
            };
            replay(&options, profile, &program, Some((seed, size)))
        }
        "replay-file" | "reduce-file" | "reject-file" => {
            let file = options.positional.first().ok_or("which file?")?;
            let source = fs::read_to_string(file).map_err(|error| format!("{file}: {error}"))?;
            let profile = single_profile(&options, Some(&source))?;
            let program = program_from_source(&source);
            if command == "reduce-file" {
                reduce_file(&options, profile, &program)
            } else if command == "reject-file" {
                reject_file(&options, profile, &source)
            } else {
                replay(&options, profile, &program, None)
            }
        }
        "reduce-modes" => {
            // Interpreter against tracing, for profiles without an oracle.
            let profile = single_profile(&options, None)?;
            let seed = options.number("seed", 0_u64)?;
            let program = generate_excluding(seed, profile, size(&options)?, &exclude(&options)?);
            reduce_modes(&options, profile, &program)
        }
        "corpus-add" => {
            let directory = corpus_directory(&options)?;
            let profile = single_profile(&options, None)?;
            let seed = options.number("seed", 0_u64)?;
            let size = size(&options)?;
            let name = options.value("name").map_or_else(
                || format!("{profile}-{}-{seed}", size.name()),
                str::to_owned,
            );
            let oracle = oracle_for(&options, profile)?;
            fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
            add_to_corpus(
                &directory,
                &name,
                &generate_excluding(seed, profile, size, &exclude(&options)?).render(),
                &oracle,
                limits(&options)?,
            )?;
            println!("added {}", directory.join(format!("{name}.lua")).display());
            Ok(ExitCode::SUCCESS)
        }
        "corpus-check" => {
            let directory = corpus_directory(&options)?;
            let engines = engines(&options)?;
            let scratch = out_directory(&options).join("corpus-work");
            let problems = check_corpus(&directory, &engines, &scratch)?;
            for problem in &problems {
                println!("{problem}");
            }
            println!("{} problems", problems.len());
            Ok(if problems.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        _ => Err(format!(
            "unknown command `{command}`; see `oximond-difftest help`"
        )),
    }
}

fn corpus_directory(options: &Options) -> Result<PathBuf, String> {
    options
        .positional
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| "which corpus directory?".to_owned())
}

fn size(options: &Options) -> Result<Size, String> {
    options.value("size").map_or(Ok(Size::Medium), |text| {
        Size::parse(text).ok_or_else(|| format!("unknown size `{text}`"))
    })
}

fn exclude(options: &Options) -> Result<std::collections::BTreeSet<String>, String> {
    let Some(list) = options.value("exclude") else {
        return Ok(std::collections::BTreeSet::new());
    };
    list.split(',')
        .filter(|name| !name.is_empty())
        .map(|name| {
            if EXCLUDABLE.contains(&name) {
                Ok(name.to_owned())
            } else {
                Err(format!(
                    "cannot exclude `{name}`; excludable: {}",
                    EXCLUDABLE.join(",")
                ))
            }
        })
        .collect()
}

fn single_profile(options: &Options, source: Option<&str>) -> Result<Profile, String> {
    if let Some(text) = options.value("profile") {
        return Profile::parse(text).ok_or_else(|| format!("unknown profile `{text}`"));
    }
    source
        .and_then(|source| source.lines().next())
        .and_then(|line| {
            line.split_whitespace()
                .find_map(|field| field.strip_prefix("profile="))
        })
        .and_then(Profile::parse)
        .ok_or_else(|| "which --profile?".to_owned())
}

fn profiles(options: &Options) -> Result<Vec<Profile>, String> {
    if let Some(profile) = options.value("profile") {
        return Ok(vec![
            Profile::parse(profile).ok_or_else(|| format!("unknown profile `{profile}`"))?,
        ]);
    }
    match options.value("profiles").unwrap_or("all") {
        "oracle" => Ok(vec![
            Profile::Lua51,
            Profile::Lua54,
            Profile::Lua55,
            Profile::LuaJit21,
        ]),
        "all" => Ok(Profile::ALL.to_vec()),
        list => list
            .split(',')
            .map(|name| Profile::parse(name).ok_or_else(|| format!("unknown profile `{name}`")))
            .collect(),
    }
}

fn seeds(options: &Options) -> Result<Seeds, String> {
    let listed = ["seed", "seeds", "count", "start"]
        .iter()
        .any(|name| options.value(name).is_some());
    if let Some(master) = options.value("master-seed") {
        if listed || options.value("duration").is_none() {
            return Err("--master-seed needs --duration and no seed list".to_owned());
        }
        let master = master
            .parse()
            .map_err(|_| format!("--master-seed: not a number: {master}"))?;
        return Ok(Seeds::Random { master });
    }
    if !listed && options.value("duration").is_some() {
        return Ok(Seeds::Random {
            master: fresh_master_seed(),
        });
    }
    if let Some(list) = options.value("seed") {
        return list
            .split(',')
            .map(|seed| {
                seed.parse()
                    .map_err(|_| format!("--seed: not a number: {seed}"))
            })
            .collect::<Result<_, _>>()
            .map(Seeds::List);
    }
    if let Some(range) = options.value("seeds") {
        let (start, end) = range.split_once("..").ok_or("--seeds takes A..B")?;
        let start: u64 = start.parse().map_err(|_| "--seeds: bad start")?;
        let end: u64 = end.parse().map_err(|_| "--seeds: bad end")?;
        return Ok(Seeds::List((start..end).collect()));
    }
    let start = options.number("start", 0_u64)?;
    let count = options.number("count", 100_u64)?;
    Ok(Seeds::List((start..start + count).collect()))
}

/// A master seed that differs between runs: std's per-process hash keys
/// mixed with the clock and the process id.
fn fresh_master_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    hasher.write_u128(now.as_nanos());
    hasher.write_u32(std::process::id());
    hasher.finish()
}

/// `90`, `90s`, `1.5m`, `2h` or `1d`; a bare number counts seconds.
fn duration(options: &Options) -> Result<Option<Duration>, String> {
    let Some(text) = options.value("duration") else {
        return Ok(None);
    };
    let (number, unit) = match text.char_indices().last() {
        Some((index, letter)) if letter.is_ascii_alphabetic() => (&text[..index], letter),
        _ => (text, 's'),
    };
    let scale = match unit {
        's' => 1.0,
        'm' => 60.0,
        'h' => 3600.0,
        'd' => 86_400.0,
        _ => {
            return Err(format!(
                "--duration: unknown unit in `{text}`; use s, m, h or d"
            ));
        }
    };
    number
        .parse::<f64>()
        .ok()
        .and_then(|number| Duration::try_from_secs_f64(number * scale).ok())
        .map(Some)
        .ok_or_else(|| format!("--duration: not a duration: {text}"))
}

fn limits(options: &Options) -> Result<Limits, String> {
    Ok(Limits {
        timeout: Duration::from_secs_f64(options.number("timeout", 20.0)?),
        memory_mib: options.number("memory", 2048)?,
        ..Limits::default()
    })
}

fn oximond_binary(options: &Options) -> Result<PathBuf, String> {
    let oximond = options.value("oximond").map_or_else(
        || {
            let release = PathBuf::from("target/release/oximond");
            if release.exists() {
                release
            } else {
                PathBuf::from("target/debug/oximond")
            }
        },
        PathBuf::from,
    );
    fs::canonicalize(&oximond).map_err(|error| format!("{}: {error}", oximond.display()))
}

fn engines(options: &Options) -> Result<Engines, String> {
    let oximond = oximond_binary(options)?;
    let modes = modes(options)?;
    let trace_profile_record_dir = options.value("trace-profile-record-dir").map(PathBuf::from);
    let trace_profile_load_dir = options.value("trace-profile-load-dir").map(PathBuf::from);
    let trace_profile_record_dir = if let Some(directory) = trace_profile_record_dir {
        fs::create_dir_all(&directory)
            .map_err(|error| format!("{}: {error}", directory.display()))?;
        Some(fs::canonicalize(&directory).map_err(|error| error.to_string())?)
    } else {
        None
    };
    let trace_profile_load_dir = trace_profile_load_dir
        .map(|directory| {
            fs::canonicalize(&directory)
                .map_err(|error| format!("{}: {error}", directory.display()))
        })
        .transpose()?;
    Ok(Engines {
        oximond,
        limits: limits(options)?,
        modes,
        trace_profile_record_dir,
        trace_profile_load_dir,
        comparison: comparison(options)?,
    })
}

fn modes(options: &Options) -> Result<Vec<Mode>, String> {
    let Some(list) = options.value("variants").or_else(|| options.value("modes")) else {
        return Ok(
            if options.flag("interpreter-oracle") && options.flag("hot") {
                Mode::JIT.to_vec()
            } else {
                Mode::ALL.to_vec()
            },
        );
    };
    list.split(',')
        .map(|name| Mode::parse(name).ok_or_else(|| format!("unknown mode `{name}`")))
        .collect()
}

fn comparison(options: &Options) -> Result<Comparison, String> {
    match options.value("float-compare").unwrap_or("exact") {
        "exact" => {
            if options.value("float-abs-tol").is_some() || options.value("float-rel-tol").is_some()
            {
                return Err("float tolerances require --float-compare approximate".to_owned());
            }
            Ok(Comparison::Exact)
        }
        "approximate" => {
            let absolute: f64 = options.number("float-abs-tol", DEFAULT_ABSOLUTE_TOLERANCE)?;
            let relative: f64 = options.number("float-rel-tol", DEFAULT_RELATIVE_TOLERANCE)?;
            let comparison = Comparison::Approximate { absolute, relative };
            comparison.validate().map_err(str::to_owned)?;
            Ok(comparison)
        }
        mode => Err(format!("unknown --float-compare mode `{mode}`")),
    }
}

fn oracle_for(options: &Options, profile: Profile) -> Result<Oracle, String> {
    if options.flag("interpreter-oracle") {
        return Ok(Oracle::interpreter(&oximond_binary(options)?, profile));
    }
    let custom = options.values.get("oracle").and_then(|values| {
        values.iter().find_map(|value| {
            let (name, command) = value.split_once('=')?;
            (name == profile.name()).then(|| PathBuf::from(command))
        })
    });
    find_oracle(profile, custom.as_deref())
}

/// The artifacts directory, absolute: a case runs with its work directory as
/// the current directory, so a relative path such as the profile a profile
/// variant records would resolve under the work directory instead.
fn out_directory(options: &Options) -> PathBuf {
    let out = PathBuf::from(options.value("out").unwrap_or("target/difftest"));
    std::path::absolute(&out).unwrap_or(out)
}

fn campaign(options: &Options) -> Result<ExitCode, String> {
    let profiles = profiles(options)?;
    let oracles = profiles
        .iter()
        .map(|&profile| (profile, oracle_for(options, profile)))
        .collect();
    let seeds = seeds(options)?;
    let duration = duration(options)?;
    if let (Seeds::Random { master }, Some(duration)) = (&seeds, duration) {
        eprintln!(
            "random seeds for {:.0}s from master seed {master}; `--master-seed {master}` replays the order",
            duration.as_secs_f64()
        );
    }
    let settings = Settings {
        profiles,
        seeds,
        duration,
        size: size(options)?,
        engines: engines(options)?,
        oracles,
        out: out_directory(options),
        jobs: options.number("jobs", 4)?,
        reduce: options.flag("reduce"),
        reduce_attempts: options.number("reduce-attempts", 400)?,
        reduce_time: Duration::from_secs(options.number("reduce-seconds", 300)?),
        reruns: options.number("reruns", 2)?,
        exclude: exclude(options)?,
        hot: options.flag("hot"),
    };
    let records = run_campaign(&settings)?;
    let bad = records
        .iter()
        .filter(|record| !matches!(record.verdict, Verdict::Pass | Verdict::Inconclusive))
        .count();
    if options.flag("interpreter-oracle") && options.flag("hot") {
        let min_share: f64 = options.number("min-native-share", DEFAULT_MIN_NATIVE_SHARE)?;
        if min_share > 0.0
            && let Err(problem) = check_non_vacuity(&summarize(&records), min_share)
        {
            eprintln!("vacuous run: {problem}");
            return Ok(ExitCode::FAILURE);
        }
    }
    Ok(if bad == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// `run-jit`: [`campaign`] with oximond's interpreter as the oracle and hot
/// programs. Profiles default to the four that have a stock oracle, so the
/// same programs can also be checked against stock Lua.
fn run_jit(options: &Options) -> Result<ExitCode, String> {
    if options.value("profiles").is_none() && options.value("profile").is_none() {
        let mut options = Options {
            positional: options.positional.clone(),
            values: options.values.clone(),
            flags: options.flags.clone(),
        };
        options
            .values
            .insert("profiles".to_owned(), vec!["oracle".to_owned()]);
        return campaign(&options);
    }
    campaign(options)
}

fn run_suite(options: &Options, workloads: bool) -> Result<ExitCode, String> {
    let engines = engines(options)?;
    let suite_name = if workloads { "workloads" } else { "fixtures" };
    let mut bad = 0;
    for profile in profiles(options)? {
        let oracle = oracle_for(options, profile);
        if let Err(error) = &oracle {
            eprintln!("{profile}: no oracle: {error}; results are inconclusive");
        }
        let oracle = oracle.ok();
        let cases = if workloads {
            oximond_difftest::workloads::workloads(profile)
        } else {
            oximond_difftest::fixtures::fixtures(profile)
        };
        for (name, source) in cases {
            let directory = out_directory(options)
                .join(suite_name)
                .join(profile.name())
                .join(name);
            let result = run_case(&source, &directory, profile, oracle.as_ref(), &engines)?;
            println!("{suite_name}/{profile}/{name}: {}", result.verdict.name());
            if !result.signature.is_empty() {
                println!("  signature: {}", result.signature.join(","));
            }
            if workloads {
                for run in &result.modes {
                    if let Some(jit) = &run.jit {
                        println!(
                            "  {}: native_entries={} traces_compiled={} iterations={}",
                            run.label, jit.native_entries, jit.traces_compiled, jit.iterations
                        );
                    }
                }
            }
            if !matches!(result.verdict, Verdict::Pass | Verdict::Inconclusive) {
                bad += 1;
                let context = ArtifactContext {
                    hot: false,
                    profile,
                    seed: None,
                    size: None,
                    oracle: oracle.as_ref(),
                    engines: &engines,
                    stable: None,
                };
                save_artifact(
                    &directory,
                    &source,
                    &program_from_source(&source),
                    &result,
                    &context,
                )?;
                println!("  artifacts: {}", directory.display());
            }
        }
    }
    Ok(if bad == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn run_rejections(options: &Options) -> Result<ExitCode, String> {
    let engines = engines(options)?;
    let mut bad = 0;
    for profile in profiles(options)? {
        let oracle = oracle_for(options, profile);
        if let Err(error) = &oracle {
            eprintln!("{profile}: no oracle: {error}; results are inconclusive");
        }
        let oracle = oracle.ok();
        for (name, source) in oximond_difftest::rejections::rejections(profile) {
            let directory = out_directory(options)
                .join("rejections")
                .join(profile.name())
                .join(name);
            let result = run_case_with_expectation(
                source,
                &directory,
                profile,
                oracle.as_ref(),
                &engines,
                Expectation::Reject,
            )?;
            println!("rejections/{profile}/{name}: {}", result.verdict.name());
            if !result.signature.is_empty() {
                println!("  signature: {}", result.signature.join(","));
            }
            if !matches!(result.verdict, Verdict::Pass | Verdict::Inconclusive) {
                bad += 1;
                save_rejection_artifact(
                    &directory,
                    source,
                    profile,
                    oracle.as_ref(),
                    &engines,
                    &result,
                )?;
                println!("  artifacts: {}", directory.display());
            }
        }
    }
    Ok(if bad == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn replay(
    options: &Options,
    profile: Profile,
    program: &Program,
    seed: Option<(u64, Size)>,
) -> Result<ExitCode, String> {
    let engines = engines(options)?;
    let oracle = oracle_for(options, profile);
    if let Err(error) = &oracle {
        eprintln!("no oracle: {error}; the result is inconclusive");
    }
    let oracle = oracle.ok();
    let name = seed.map_or_else(|| "file".to_owned(), |(seed, _)| seed.to_string());
    let directory = out_directory(options)
        .join("replay")
        .join(format!("{profile}-{name}"));
    let source = program.render();
    let result = run_case(&source, &directory, profile, oracle.as_ref(), &engines)?;
    println!("verdict: {}", result.verdict.name());
    if !result.signature.is_empty() {
        println!("signature: {}", result.signature.join(","));
    }
    if let Some(oracle_run) = &result.oracle {
        println!(
            "oracle: {} in {}ms",
            oracle_run.observation.end.describe(),
            oracle_run.outcome.elapsed.as_millis()
        );
    }
    for run in &result.modes {
        let jit = run.jit.as_ref().map_or_else(String::new, |jit| {
            format!(
                ", jit productive={} traces_compiled={} native_entries={} side_exits={}",
                jit.productive(),
                jit.traces_compiled,
                jit.native_entries,
                jit.side_exits
            )
        });
        println!(
            "{}: {} in {}ms{jit}",
            run.label,
            run.observation.end.describe(),
            run.outcome.elapsed.as_millis()
        );
        if let Some(oracle_run) = &result.oracle
            && let Some(line) = first_difference_with(
                &oracle_run.observation.stdout,
                &run.observation.stdout,
                engines.comparison,
            )
        {
            println!(
                "  first difference at line {}:\n    oracle: {}\n    {}: {}",
                line.line, line.expected, run.label, line.actual
            );
        }
        if let Some(interpreter) = result
            .modes
            .iter()
            .find(|other| other.label == Mode::Interpreter.name() && other.label != run.label)
            && let Some(line) = first_difference_with(
                &interpreter.observation.stdout,
                &run.observation.stdout,
                Comparison::Exact,
            )
        {
            println!(
                "  first difference from the interpreter at line {}:\n    interpreter: {}\n    {}: {}",
                line.line, line.expected, run.label, line.actual
            );
        }
    }
    let context = ArtifactContext {
        profile,
        seed: seed.map(|(seed, _)| seed),
        size: seed.map(|(_, size)| size),
        oracle: oracle.as_ref(),
        engines: &engines,
        stable: None,
        hot: options.flag("hot"),
    };
    save_artifact(&directory, &source, program, &result, &context)?;
    println!("artifacts: {}", directory.display());
    Ok(match result.verdict {
        Verdict::Pass | Verdict::Inconclusive => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    })
}

fn reject_file(options: &Options, profile: Profile, source: &str) -> Result<ExitCode, String> {
    let engines = engines(options)?;
    let oracle = oracle_for(options, profile);
    if let Err(error) = &oracle {
        eprintln!("no oracle: {error}; the result is inconclusive");
    }
    let oracle = oracle.ok();
    let directory = out_directory(options)
        .join("reject-file")
        .join(profile.name());
    let result = run_case_with_expectation(
        source,
        &directory,
        profile,
        oracle.as_ref(),
        &engines,
        Expectation::Reject,
    )?;
    println!("verdict: {}", result.verdict.name());
    if !result.signature.is_empty() {
        println!("signature: {}", result.signature.join(","));
    }
    if let Some(run) = &result.oracle {
        println!(
            "oracle: {}",
            String::from_utf8_lossy(&run.observation.stdout).trim()
        );
    }
    for run in &result.modes {
        println!(
            "{}: {} ({})",
            run.label,
            String::from_utf8_lossy(&run.observation.stdout).trim(),
            run.observation.end.describe()
        );
    }
    save_rejection_artifact(
        &directory,
        source,
        profile,
        oracle.as_ref(),
        &engines,
        &result,
    )?;
    println!("artifacts: {}", directory.display());
    Ok(match result.verdict {
        Verdict::Pass | Verdict::Inconclusive => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    })
}

fn save_rejection_artifact(
    directory: &Path,
    source: &str,
    profile: Profile,
    oracle: Option<&Oracle>,
    engines: &Engines,
    result: &oximond_difftest::case::CaseResult,
) -> Result<(), String> {
    let context = ArtifactContext {
        profile,
        seed: None,
        size: None,
        oracle,
        engines,
        stable: None,
        hot: false,
    };
    save_artifact(
        directory,
        source,
        &program_from_source(source),
        result,
        &context,
    )?;
    let meta_path = directory.join("meta.txt");
    let meta = fs::read_to_string(&meta_path).map_err(|error| error.to_string())?;
    let meta = meta.replace(
        "replay_file: oximond-difftest replay-file ",
        "replay_file: oximond-difftest reject-file ",
    );
    fs::write(&meta_path, format!("expectation: reject\n{meta}"))
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn reduce_modes(
    options: &Options,
    profile: Profile,
    program: &Program,
) -> Result<ExitCode, String> {
    let mut engines = engines(options)?;
    engines.modes = vec![Mode::Interpreter, Mode::Tracing];
    let directory = out_directory(options)
        .join("reduce-modes")
        .join(profile.name());
    let work = directory.join("work");
    let disagrees = |candidate: &Program| {
        run_case(&candidate.render(), &work, profile, None, &engines).is_ok_and(|result| {
            !result.modes_agree
                && result
                    .modes
                    .iter()
                    .all(|run| run.observation.end == oximond_difftest::case::End::Success)
        })
    };
    if !disagrees(program) {
        return Err("the modes agree on this program".to_owned());
    }
    let attempts = options.number("reduce-attempts", 400)?;
    let seconds = options.number("reduce-seconds", 300)?;
    let reduction = reduce(
        program,
        disagrees,
        attempts,
        Instant::now() + Duration::from_secs(seconds),
    );
    let path = directory.join("reduced.lua");
    fs::write(&path, reduction.program.render()).map_err(|error| error.to_string())?;
    fs::write(directory.join("reduction.log"), reduction.log.join("\n"))
        .map_err(|error| error.to_string())?;
    println!(
        "{} attempts, {} accepted: {} -> {} nodes\n{}",
        reduction.attempts,
        reduction.accepted,
        program.size(),
        reduction.program.size(),
        path.display()
    );
    Ok(ExitCode::SUCCESS)
}

fn reduce_file(options: &Options, profile: Profile, program: &Program) -> Result<ExitCode, String> {
    let engines = engines(options)?;
    let oracle = oracle_for(options, profile)?;
    let directory = out_directory(options)
        .join("reduce-file")
        .join(profile.name());
    let original = run_case(
        &program.render(),
        &directory.join("original"),
        profile,
        Some(&oracle),
        &engines,
    )?;
    if original.verdict != Verdict::Mismatch {
        return Err(format!(
            "the file does not fail (verdict {})",
            original.verdict.name()
        ));
    }
    let target = options.value("signature").map_or_else(
        || original.signature.clone(),
        |text| text.split(',').map(str::to_owned).collect(),
    );
    println!("reducing for {}", target.join(","));
    let work = directory.join("work");
    let check = |candidate: &Program| {
        run_case(&candidate.render(), &work, profile, Some(&oracle), &engines).is_ok_and(|result| {
            result.verdict == Verdict::Mismatch
                && result.signature.iter().any(|entry| target.contains(entry))
        })
    };
    let attempts = options.number("reduce-attempts", 400)?;
    let seconds = options.number("reduce-seconds", 300)?;
    let reduction = reduce(
        program,
        check,
        attempts,
        Instant::now() + Duration::from_secs(seconds),
    );
    let reduced = reduction.program.render();
    let path = directory.join("reduced.lua");
    fs::write(&path, &reduced).map_err(|error| error.to_string())?;
    let log_path = directory.join("reduction.log");
    fs::write(&log_path, reduction.log.join("\n")).map_err(|error| error.to_string())?;
    println!(
        "{} attempts, {} accepted{}: {} -> {} lines\n{}",
        reduction.attempts,
        reduction.accepted,
        if reduction.exhausted {
            " (limit reached)"
        } else {
            ""
        },
        program.size(),
        reduction.program.size(),
        path.display()
    );
    let _ = Path::new(&work);
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(args: &[&str]) -> Result<Comparison, String> {
        comparison(&Options::parse(
            &args.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>(),
        )?)
    }

    #[test]
    fn run_jit_defaults_to_the_jit_variants_and_other_commands_keep_theirs() {
        let mut options = Options::default();
        options.flags.push("interpreter-oracle".to_owned());
        options.flags.push("hot".to_owned());
        let names: Vec<&str> = modes(&options).unwrap().iter().map(|m| m.name()).collect();
        assert_eq!(
            names,
            [
                "tracing-mir",
                "tracing-mir-profile",
                "tracing-mir-only",
                "tracing",
            ]
        );
        let options = Options::parse(&["--variants".into(), "tracing-mir-async".into()]).unwrap();
        assert_eq!(modes(&options).unwrap(), vec![Mode::TracingMirAsync]);
        assert!(
            Options::parse(&["--interpreter-oracle".into()])
                .unwrap()
                .flag("interpreter-oracle")
        );
        assert_eq!(modes(&Options::default()).unwrap(), Mode::ALL.to_vec());
    }

    #[test]
    fn defaults_include_every_profile_and_execution_mode() {
        let options = Options::default();
        assert_eq!(profiles(&options).unwrap(), Profile::ALL.to_vec());
        assert_eq!(modes(&options).unwrap(), Mode::ALL.to_vec());
    }

    fn parse(args: &[&str]) -> Options {
        Options::parse(&args.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn durations_take_a_unit_or_count_seconds() {
        let of = |text: &str| duration(&parse(&["--duration", text]));
        assert_eq!(of("90").unwrap(), Some(Duration::from_secs(90)));
        assert_eq!(of("90s").unwrap(), Some(Duration::from_secs(90)));
        assert_eq!(of("1.5m").unwrap(), Some(Duration::from_secs(90)));
        assert_eq!(of("2h").unwrap(), Some(Duration::from_hours(2)));
        assert_eq!(of("1d").unwrap(), Some(Duration::from_hours(24)));
        assert_eq!(duration(&parse(&[])).unwrap(), None);
        for text in ["", "m", "5x", "-1s", "NaN", "inf"] {
            assert!(of(text).is_err(), "{text}");
        }
    }

    #[test]
    fn a_duration_without_a_seed_list_draws_random_seeds() {
        assert_eq!(
            seeds(&parse(&["--duration", "1m", "--master-seed", "9"])).unwrap(),
            Seeds::Random { master: 9 }
        );
        assert!(matches!(
            seeds(&parse(&["--duration", "1m"])).unwrap(),
            Seeds::Random { .. }
        ));
        assert_eq!(
            seeds(&parse(&["--duration", "1m", "--seeds", "3..5"])).unwrap(),
            Seeds::List(vec![3, 4])
        );
        assert_eq!(seeds(&parse(&[])).unwrap(), Seeds::List((0..100).collect()));
        for args in [
            vec!["--master-seed", "9"],
            vec!["--duration", "1m", "--count", "5", "--master-seed", "9"],
        ] {
            assert!(seeds(&parse(&args)).is_err(), "{args:?}");
        }
    }

    #[test]
    fn float_comparison_requires_an_explicit_mode_and_valid_tolerances() {
        assert_eq!(policy(&[]).unwrap(), Comparison::Exact);
        assert_eq!(
            policy(&["--float-compare", "approximate"]).unwrap(),
            Comparison::approximate()
        );
        assert_eq!(
            policy(&[
                "--float-compare",
                "approximate",
                "--float-abs-tol",
                "0",
                "--float-rel-tol",
                "1e-9"
            ])
            .unwrap(),
            Comparison::Approximate {
                absolute: 0.0,
                relative: 1e-9
            }
        );
        for args in [
            vec!["--float-compare", "unknown"],
            vec!["--float-abs-tol", "1e-9"],
            vec!["--float-compare", "approximate", "--float-abs-tol", "-1"],
            vec!["--float-compare", "approximate", "--float-rel-tol", "NaN"],
            vec!["--float-compare", "approximate", "--float-rel-tol", "inf"],
        ] {
            assert!(policy(&args).is_err(), "{args:?}");
        }
    }
}
