//! The failure pipeline end to end, with fake engines whose behaviour the
//! test controls: capture, artifacts, replay from the saved source,
//! reduction, missing oracles, crashes, timeouts, harness errors and the
//! corpus's known-failure matching.
//!
//! The fake oracle prints the text after `print ` on each line and rejects
//! a file that uses `needs-decl` without a `declare` line, the way stock
//! Lua rejects a reduced program that lost a declaration. The fake oximond
//! prints the same, except that it turns `SAME` into `OTHER` in every mode,
//! in tracing mode it turns `BUG` into `WRONG`, and it hangs on `HANG` and
//! crashes on `CRASH`.

#![cfg(unix)]

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use oximond_difftest::campaign::{
    ArtifactContext, Seeds, Settings, check_corpus, run_campaign, save_artifact,
};
use oximond_difftest::case::{
    CASE_FILE, Engines, Expectation, Mode, Verdict, find_oracle, run_case,
    run_case_with_expectation,
};
use oximond_difftest::generate::Size;
use oximond_difftest::profile::Profile;
use oximond_difftest::program::{Node, Program};
use oximond_difftest::reduce::reduce;
use oximond_difftest::run::Limits;

const FAKE_LUA: &str = r#"#!/bin/sh
if [ "$1" = "-v" ]; then echo "Lua 5.4.9  fake"; exit 0; fi
file="$1"
if [ "$file" = "compile-only.lua" ]; then
  if grep -q REJECT case.lua; then echo rejected; else echo accepted; fi
  exit 0
fi
if grep -q "needs-decl" "$file" && ! grep -q "^declare" "$file"; then
  echo "fake-lua: $file:1: syntax error near 'needs-decl'" >&2
  exit 1
fi
while IFS= read -r line; do
  case "$line" in
    "print "*) printf '%s\n' "${line#print }" ;;
  esac
done < "$file"
"#;

const FAKE_OXIMOND: &str = r#"#!/bin/sh
mode=""
file=""
while [ $# -gt 0 ]; do
  case "$1" in
    --execution) mode="$2"; shift 2 ;;
    --compat) shift 2 ;;
    *) file="$1"; shift ;;
  esac
done
if [ "$file" = "compile-only.lua" ]; then
  if grep -q REJECT-CANDIDATE-CRASH case.lua; then kill -SEGV $$; fi
  if grep -q REJECT-CANDIDATE-HANG case.lua; then sleep 30; fi
  if grep -q REJECT case.lua; then
    if grep -q CANDIDATE-ACCEPT case.lua; then echo accepted; else echo rejected; fi
  else
    echo accepted
  fi
  exit 0
fi
while IFS= read -r line; do
  case "$line" in
    *HANG*) sleep 30 ;;
    *CRASH*) kill -SEGV $$ ;;
    "print "*SAME*|"print "*BUG*)
      text="${line#print }"
      text=$(printf '%s' "$text" | sed 's/SAME/OTHER/')
      if [ "$mode" = tracing ]; then text=$(printf '%s' "$text" | sed 's/BUG/WRONG/'); fi
      printf '%s\n' "$text" ;;
    "print "*) printf '%s\n' "${line#print }" ;;
  esac
done < "$file"
"#;

struct Fixture {
    root: PathBuf,
    oracle: PathBuf,
    engines: Engines,
}

fn fixture(name: &str) -> Fixture {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("difftest-pipeline")
        .join(name);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let write_script = |file: &str, text: &str| {
        let path = root.join(file);
        fs::write(&path, text).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    };
    let oracle = write_script("fake-lua", FAKE_LUA);
    let oximond = write_script("fake-oximond", FAKE_OXIMOND);
    Fixture {
        root,
        oracle,
        engines: Engines {
            oximond,
            limits: Limits {
                timeout: Duration::from_secs(2),
                memory_mib: 0,
                ..Limits::default()
            },
            modes: vec![Mode::Interpreter, Mode::Tracing],
            trace_profile_record_dir: None,
            trace_profile_load_dir: None,
            comparison: oximond_difftest::compare::Comparison::Exact,
        },
    }
}

fn program(lines: &[&str]) -> Program {
    Program {
        header: vec!["fake profile=lua54".into()],
        nodes: lines
            .iter()
            .map(|line| Node::leaf(*line))
            .chain([Node::block(
                "do",
                vec![Node::leaf("print o9#1 inner")],
                "end",
            )])
            .collect(),
    }
}

#[test]
fn a_mismatch_is_captured_saved_replayed_and_reduced() {
    let fixture = fixture("capture");
    let oracle = find_oracle(Profile::Lua54, Some(&fixture.oracle)).unwrap();
    let program = program(&[
        "print o1#1 a",
        "declare x",
        "print o2#1 needs-decl BUG",
        "print o3#1 c",
    ]);
    let source = program.render();
    let result = run_case(
        &source,
        &fixture.root.join("work"),
        Profile::Lua54,
        Some(&oracle),
        &fixture.engines,
    )
    .unwrap();
    assert_eq!(result.verdict, Verdict::Mismatch);
    assert_eq!(result.signature, vec!["tracing:stdout@o2".to_owned()]);

    // The artifact holds the source, every engine's output and a replay
    // command.
    let artifact = fixture.root.join("artifact");
    let context = ArtifactContext {
        profile: Profile::Lua54,
        seed: Some(7),
        size: Some(Size::Small),
        oracle: Some(&oracle),
        engines: &fixture.engines,
        stable: Some(true),
        hot: false,
    };
    save_artifact(&artifact, &source, &program, &result, &context).unwrap();
    for file in [
        "case.lua",
        "oracle.stdout",
        "interpreter.stdout",
        "tracing.stdout",
        "tracing.stderr",
        "diff.txt",
        "meta.txt",
    ] {
        assert!(artifact.join(file).exists(), "{file} missing");
    }
    let meta = fs::read_to_string(artifact.join("meta.txt")).unwrap();
    assert!(meta.contains("signature: tracing:stdout@o2"), "{meta}");
    assert!(
        meta.contains("replay_file: oximond-difftest replay-file"),
        "{meta}"
    );
    assert!(meta.contains("stable_on_rerun: true"), "{meta}");
    let diff = fs::read_to_string(artifact.join("diff.txt")).unwrap();
    assert!(
        diff.contains("oracle: o2#1 needs-decl BUG")
            && diff.contains("tracing: o2#1 needs-decl WRONG"),
        "{diff}"
    );

    // The saved source replays to the same failure without the generator.
    let saved = fs::read_to_string(artifact.join(CASE_FILE)).unwrap();
    let replayed = run_case(
        &saved,
        &fixture.root.join("replay"),
        Profile::Lua54,
        Some(&oracle),
        &fixture.engines,
    )
    .unwrap();
    assert_eq!(replayed.signature, result.signature);

    // Reduction keeps the failing statement and the declaration the oracle
    // needs, and drops everything else.
    let work = fixture.root.join("reduce");
    let target = result.signature.clone();
    let check = |candidate: &Program| {
        run_case(
            &candidate.render(),
            &work,
            Profile::Lua54,
            Some(&oracle),
            &fixture.engines,
        )
        .is_ok_and(|result| result.verdict == Verdict::Mismatch && result.signature == target)
    };
    let reduction = reduce(
        &program,
        check,
        100,
        Instant::now() + Duration::from_secs(60),
    );
    assert_eq!(
        reduction.program.render(),
        "-- fake profile=lua54\ndeclare x\nprint o2#1 needs-decl BUG\n"
    );
}

#[test]
fn engine_failures_get_their_own_categories() {
    let fixture = fixture("categories");
    let oracle = find_oracle(Profile::Lua54, Some(&fixture.oracle)).unwrap();
    let crash = run_case(
        &program(&["print o1#1 x", "print CRASH"]).render(),
        &fixture.root.join("crash"),
        Profile::Lua54,
        Some(&oracle),
        &fixture.engines,
    )
    .unwrap();
    assert_eq!(crash.verdict, Verdict::Mismatch);
    assert_eq!(
        crash.signature,
        vec!["interpreter:crash".to_owned(), "tracing:crash".to_owned()]
    );
    let hang = run_case(
        &program(&["print HANG"]).render(),
        &fixture.root.join("hang"),
        Profile::Lua54,
        Some(&oracle),
        &fixture.engines,
    )
    .unwrap();
    assert_eq!(
        hang.signature,
        vec![
            "interpreter:timeout".to_owned(),
            "tracing:timeout".to_owned()
        ]
    );
    let invalid = run_case(
        &program(&["print needs-decl"]).render(),
        &fixture.root.join("invalid"),
        Profile::Lua54,
        Some(&oracle),
        &fixture.engines,
    )
    .unwrap();
    assert_eq!(invalid.verdict, Verdict::OracleSyntaxError);
    assert!(
        invalid.modes.is_empty(),
        "oximond does not run when the oracle rejects the program"
    );
}

#[test]
fn expected_rejection_checks_compilation_without_running_source() {
    let fixture = fixture("rejection");
    let oracle = find_oracle(Profile::Lua54, Some(&fixture.oracle)).unwrap();
    let checked = |name: &str, source: &str, oracle: Option<&_>| {
        run_case_with_expectation(
            source,
            &fixture.root.join(name),
            Profile::Lua54,
            oracle,
            &fixture.engines,
            Expectation::Reject,
        )
        .unwrap()
    };
    let pass = checked("pass", "REJECT\nprint HANG\n", Some(&oracle));
    assert_eq!(pass.verdict, Verdict::Pass);
    assert_eq!(pass.modes.len(), 2);
    assert!(fixture.root.join("pass").join("compile-only.lua").exists());
    assert_eq!(
        fs::read_to_string(fixture.root.join("pass").join(CASE_FILE)).unwrap(),
        "REJECT\nprint HANG\n"
    );

    let mismatch = checked("mismatch", "REJECT CANDIDATE-ACCEPT", Some(&oracle));
    assert_eq!(mismatch.verdict, Verdict::Mismatch);
    assert_eq!(mismatch.signature.len(), 2);

    let accepted = checked("accepted", "print harmless", Some(&oracle));
    assert_eq!(accepted.verdict, Verdict::OracleUnexpectedAcceptance);
    assert!(accepted.modes.is_empty());

    let missing = checked("missing", "REJECT", None);
    assert_eq!(missing.verdict, Verdict::Inconclusive);

    let crash = checked("crash", "REJECT-CANDIDATE-CRASH", Some(&oracle));
    assert_eq!(crash.verdict, Verdict::Mismatch);
    assert_eq!(
        crash.signature,
        vec!["interpreter:crash".to_owned(), "tracing:crash".to_owned()]
    );
    let timeout = checked("timeout", "REJECT-CANDIDATE-HANG", Some(&oracle));
    assert_eq!(timeout.verdict, Verdict::Mismatch);
    assert_eq!(
        timeout.signature,
        vec![
            "interpreter:timeout".to_owned(),
            "tracing:timeout".to_owned()
        ]
    );
}

#[test]
fn a_missing_or_wrong_oracle_leaves_the_interpreter_as_judge() {
    let fixture = fixture("oracle");
    assert!(find_oracle(Profile::Lua54, Some(Path::new("/nonexistent/lua5.4"))).is_err());
    // The fake reports 5.4, so it cannot stand in for 5.1.
    let error = find_oracle(Profile::Lua51, Some(&fixture.oracle)).unwrap_err();
    assert!(error.contains("not Lua 5.1"), "{error}");
    let without_oracle = |lines: &[&str]| {
        run_case(
            &program(lines).render(),
            &fixture.root.join("work"),
            Profile::Lua54,
            None,
            &fixture.engines,
        )
        .unwrap()
    };
    // Tracing differs from the interpreter: a mismatch with no oracle.
    let result = without_oracle(&["print o1#1 BUG"]);
    assert_eq!(result.verdict, Verdict::Mismatch);
    assert!(!result.modes_agree, "the fake modes disagree on BUG");
    assert_eq!(result.signature, ["tracing:vs-interpreter:stdout@o1"]);
    // Both modes agree, so nothing judges them.
    let result = without_oracle(&["print o1#1 SAME"]);
    assert_eq!(result.verdict, Verdict::Inconclusive);
    assert_eq!(result.signature, [] as [std::string::String; 0]);
}

#[test]
fn jit_modes_are_judged_by_the_interpreter_when_it_differs_from_the_oracle() {
    let fixture = fixture("vs-interpreter");
    let oracle = find_oracle(Profile::Lua54, Some(&fixture.oracle)).unwrap();
    let case = |name: &str, lines: &[&str]| {
        run_case(
            &program(lines).render(),
            &fixture.root.join(name),
            Profile::Lua54,
            Some(&oracle),
            &fixture.engines,
        )
        .unwrap()
    };
    // The interpreter matches the oracle: the oracle's signature says it all.
    let result = case("matching", &["print o1#1 BUG"]);
    assert_eq!(result.signature, ["tracing:stdout@o1"]);
    // Both modes differ from the oracle the same way: no JIT bug.
    let result = case("shared", &["print o1#1 SAME"]);
    assert_eq!(
        result.signature,
        ["interpreter:stdout@o1", "tracing:stdout@o1"]
    );
    // Both differ from the oracle, and tracing from the interpreter too.
    let result = case("both", &["print o1#1 SAME BUG"]);
    assert_eq!(result.verdict, Verdict::Mismatch);
    assert_eq!(
        result.signature,
        [
            "interpreter:stdout@o1",
            "tracing:stdout@o1",
            "tracing:vs-interpreter:stdout@o1"
        ]
    );
}

#[test]
fn a_case_the_harness_cannot_run_is_recorded() {
    let fixture = fixture("harness");
    let out = fixture.root.join("out");
    fs::create_dir_all(&out).unwrap();
    // A file where the work directory should go.
    fs::write(out.join("work"), "").unwrap();
    let oracle = find_oracle(Profile::Lua54, Some(&fixture.oracle));
    let settings = Settings {
        profiles: vec![Profile::Lua54],
        seeds: Seeds::List(vec![1, 2, 3]),
        duration: None,
        size: Size::Small,
        engines: fixture.engines.clone(),
        oracles: [(Profile::Lua54, oracle)].into_iter().collect(),
        out: out.clone(),
        jobs: 2,
        reduce: false,
        reduce_attempts: 0,
        reduce_time: Duration::ZERO,
        reruns: 0,
        exclude: BTreeSet::new(),
        hot: false,
    };
    let records = run_campaign(&settings).unwrap();
    assert_eq!(records.len(), 3);
    assert!(
        records
            .iter()
            .all(|record| record.verdict == Verdict::HarnessError)
    );
    let tsv = fs::read_to_string(out.join("cases.tsv")).unwrap();
    assert_eq!(
        tsv.lines()
            .filter(|line| line.contains("harness_error"))
            .count(),
        3
    );
}

#[test]
fn a_timed_campaign_runs_random_seeds_until_its_deadline() {
    let fixture = fixture("timed");
    let oracle = find_oracle(Profile::Lua54, Some(&fixture.oracle));
    let settings = Settings {
        profiles: vec![Profile::Lua51, Profile::Lua54],
        seeds: Seeds::Random { master: 11 },
        duration: Some(Duration::from_secs(1)),
        size: Size::Small,
        engines: fixture.engines.clone(),
        oracles: [(Profile::Lua54, oracle)].into_iter().collect(),
        out: fixture.root.join("campaign"),
        jobs: 2,
        reduce: false,
        reduce_attempts: 0,
        reduce_time: Duration::ZERO,
        reruns: 0,
        exclude: BTreeSet::new(),
        hot: false,
    };
    let started = Instant::now();
    let records = run_campaign(&settings).unwrap();
    assert!(started.elapsed() < Duration::from_secs(15));
    assert!(records.len() >= 2, "{} cases", records.len());
    // Cases go seed by seed, each on every profile, through the master
    // seed's stream in order.
    let seeds: BTreeSet<u64> = records.iter().map(|record| record.seed).collect();
    let expected: BTreeSet<u64> = (0..records.len().div_ceil(2))
        .map(|n| Seeds::Random { master: 11 }.nth(n).unwrap())
        .collect();
    assert_eq!(seeds, expected);
    let lua51 = records
        .iter()
        .filter(|record| record.profile == Profile::Lua51)
        .count();
    assert!(lua51.abs_diff(records.len() - lua51) <= 1);
    let report = fs::read_to_string(settings.out.join("report.md")).unwrap();
    assert!(report.contains("random, master seed 11"), "{report}");
    assert!(report.contains("- duration: 1s"), "{report}");
    fs::remove_dir_all(&fixture.root).unwrap();
}

#[test]
fn random_seeds_need_a_duration() {
    let fixture = fixture("endless");
    let settings = Settings {
        profiles: vec![Profile::Lua54],
        seeds: Seeds::Random { master: 1 },
        duration: None,
        size: Size::Small,
        engines: fixture.engines.clone(),
        oracles: std::collections::BTreeMap::new(),
        out: fixture.root.join("campaign"),
        jobs: 1,
        reduce: false,
        reduce_attempts: 0,
        reduce_time: Duration::ZERO,
        reruns: 0,
        exclude: BTreeSet::new(),
        hot: false,
    };
    assert!(run_campaign(&settings).is_err());
    let _ = fs::remove_dir_all(&fixture.root);
}

#[test]
fn known_failures_must_match_exactly() {
    let fixture = fixture("corpus");
    let corpus = fixture.root.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    let source = "-- fake profile=lua54\nprint o1#1 ok\nprint o2#1 BUG\n";
    fs::write(corpus.join("bug.lua"), source).unwrap();
    fs::write(corpus.join("bug.stdout"), "o1#1 ok\no2#1 BUG\n").unwrap();
    fs::write(
        corpus.join("fine.lua"),
        "-- fake profile=lua54\nprint o1#1 ok\n",
    )
    .unwrap();
    fs::write(corpus.join("fine.stdout"), "o1#1 ok\n").unwrap();
    let scratch = fixture.root.join("scratch");

    // Without a known-failure entry the bug is a problem.
    let problems = check_corpus(&corpus, &fixture.engines, &scratch).unwrap();
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].starts_with("bug tracing stdout@o2"),
        "{problems:?}"
    );

    // The exact signature is accepted.
    fs::write(
        corpus.join("bug.known-failure"),
        "# float modulo\ntracing:stdout@o2\n",
    )
    .unwrap();
    assert_eq!(
        check_corpus(&corpus, &fixture.engines, &scratch).unwrap(),
        Vec::<String>::new()
    );

    // A different failing mode is reported, not absorbed.
    fs::write(corpus.join("bug.known-failure"), "interpreter:stdout@o2\n").unwrap();
    let problems = check_corpus(&corpus, &fixture.engines, &scratch).unwrap();
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("known failure changed")),
        "{problems:?}"
    );

    // A fixed bug is reported so its entry gets updated.
    fs::write(corpus.join("bug.known-failure"), "tracing:stdout@o2\n").unwrap();
    fs::write(corpus.join("bug.lua"), source.replace("BUG", "FIXED")).unwrap();
    fs::write(corpus.join("bug.stdout"), "o1#1 ok\no2#1 FIXED\n").unwrap();
    let problems = check_corpus(&corpus, &fixture.engines, &scratch).unwrap();
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("now passes")),
        "{problems:?}"
    );
}

const FAKE_LUAJIT: &str = r#"#!/bin/sh
if [ "$1" = "-v" ]; then echo "LuaJIT 2.1.0 fake"; exit 0; fi
joff=""
if [ "$1" = "-joff" ]; then joff=1; shift; fi
while IFS= read -r line; do
  case "$line" in
    "print "*)
      text="${line#print }"
      if [ -n "$joff" ]; then text=$(printf '%s' "$text" | sed 's/BUG/WRONG/'); fi
      printf '%s\n' "$text" ;;
  esac
done < "$1"
"#;

#[test]
fn a_mode_matching_luajits_interpreter_run_passes() {
    let fixture = fixture("alternate");
    let path = fixture.root.join("fake-luajit");
    fs::write(&path, FAKE_LUAJIT).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    let oracle = find_oracle(Profile::LuaJit21, Some(&path)).unwrap();
    assert_eq!(oracle.alternate, vec![std::ffi::OsString::from("-joff")]);
    // Tracing prints WRONG, which the oracle's -joff run prints too.
    let result = run_case(
        &program(&["print o1#1 BUG"]).render(),
        &fixture.root.join("work"),
        Profile::LuaJit21,
        Some(&oracle),
        &fixture.engines,
    )
    .unwrap();
    assert_eq!(result.verdict, Verdict::Pass, "{:?}", result.signature);
    assert!(result.oracle_alternate.is_some());
}

#[test]
fn approximate_floats_preserve_replay_policy_and_find_the_first_real_mismatch() {
    use oximond_difftest::compare::Comparison;

    let mut fixture = fixture("approximate-floats");
    let numeric_cases = r#"    "print o1#1 f1"*)
      if [ "$mode" = tracing ]; then printf 'o1#1 f1.0000000000001\n';
      else printf 'o1#1 f1\n'; fi ;;
    "print o2#1 f2"*)
      if [ "$mode" = tracing ]; then printf 'o2#1 f3\n';
      else printf 'o2#1 f2\n'; fi ;;
"#;
    let script = FAKE_OXIMOND.replace(
        "    \"print \"*)",
        &format!("{numeric_cases}    \"print \"*)"),
    );
    fs::write(&fixture.engines.oximond, script).unwrap();
    let oracle = find_oracle(Profile::Lua54, Some(&fixture.oracle)).unwrap();
    let source = "print o1#1 f1\n";
    let work = fixture.root.join("work");
    let strict = run_case(
        source,
        &work,
        Profile::Lua54,
        Some(&oracle),
        &fixture.engines,
    )
    .unwrap();
    assert_eq!(strict.verdict, Verdict::Mismatch);
    assert_eq!(strict.signature, ["tracing:stdout@o1"]);

    fixture.engines.comparison = Comparison::approximate();
    let tolerant = run_case(
        source,
        &work,
        Profile::Lua54,
        Some(&oracle),
        &fixture.engines,
    )
    .unwrap();
    assert_eq!(tolerant.verdict, Verdict::Pass);
    assert!(
        !tolerant.modes_agree,
        "mode agreement must remain byte exact"
    );
    let context = ArtifactContext {
        profile: Profile::Lua54,
        seed: Some(1),
        size: Some(Size::Small),
        oracle: Some(&oracle),
        engines: &fixture.engines,
        stable: None,
        hot: false,
    };
    let artifacts = fixture.root.join("artifacts");
    save_artifact(
        &artifacts,
        source,
        &program(&["print o1#1 f1"]),
        &tolerant,
        &context,
    )
    .unwrap();
    let meta = fs::read_to_string(artifacts.join("meta.txt")).unwrap();
    assert!(meta.contains("float_comparison: approximate"));
    for command in meta.lines().filter(|line| line.starts_with("replay_")) {
        assert!(command.contains("--float-compare approximate --float-abs-tol"));
        assert!(command.contains("--float-rel-tol"));
    }
    assert_eq!(fs::read(artifacts.join("diff.txt")).unwrap(), []);

    let source = "print o1#1 f1\nprint o2#1 f2\n";
    let mismatch = run_case(
        source,
        &work,
        Profile::Lua54,
        Some(&oracle),
        &fixture.engines,
    )
    .unwrap();
    assert_eq!(mismatch.signature, ["tracing:stdout@o2"]);
    save_artifact(&artifacts, source, &program(&[source]), &mismatch, &context).unwrap();
    let diff = fs::read_to_string(artifacts.join("diff.txt")).unwrap();
    assert!(diff.contains("stdout line 2"));
    assert!(!diff.contains("stdout line 1"));
    fs::remove_dir_all(&fixture.root).unwrap();
}

#[test]
fn campaigns_count_passes_that_required_float_tolerance() {
    let mut fixture = fixture("approximate-campaign");
    fs::write(
        &fixture.oracle,
        "#!/bin/sh\nif [ \"$1\" = -v ]; then echo 'Lua 5.4.9 fake'; else echo 'o1#1 f1'; fi\n",
    )
    .unwrap();
    fs::write(
        &fixture.engines.oximond,
        "#!/bin/sh\necho 'o1#1 f1.0000000000001'\n",
    )
    .unwrap();
    fixture.engines.comparison = oximond_difftest::compare::Comparison::approximate();
    let oracle = find_oracle(Profile::Lua54, Some(&fixture.oracle)).unwrap();
    let settings = Settings {
        profiles: vec![Profile::Lua54],
        seeds: Seeds::List(vec![1]),
        duration: None,
        size: Size::Small,
        engines: fixture.engines.clone(),
        oracles: [(Profile::Lua54, Ok(oracle))].into_iter().collect(),
        out: fixture.root.join("campaign"),
        jobs: 1,
        reduce: false,
        reduce_attempts: 0,
        reduce_time: Duration::ZERO,
        reruns: 0,
        exclude: BTreeSet::new(),
        hot: false,
    };
    let records = run_campaign(&settings).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].verdict, Verdict::Pass);
    assert!(records[0].approximate_match);
    let report = fs::read_to_string(settings.out.join("report.md")).unwrap();
    assert!(report.contains("float comparison: approximate"));
    assert!(report.contains("approximate matches"));
    assert!(report.contains("| lua54 | 1 | 1 | 0 | 0 | 0 / 0 | 0 | 1 |"));
    let cases = fs::read_to_string(settings.out.join("cases.tsv")).unwrap();
    assert!(cases.lines().next().unwrap().ends_with("approximate_match"));
    assert!(cases.lines().nth(1).unwrap().ends_with("true"));
    fs::remove_dir_all(&fixture.root).unwrap();
}
