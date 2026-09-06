//! `pcall` of a Lua closure and calls through a `__call` closure, which the
//! plain interpreter pushes from its fast loop. Tracing mode and hooked runs
//! keep the general frames, so the two modes check each other: every case in
//! `fixtures/protected_inloop.lua` must print the same lines in both.

use std::path::PathBuf;
use std::process::Command;

fn run(profile: &str, mode: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args([
            "--compat",
            profile,
            "--execution",
            mode,
            "protected_inloop.lua",
        ])
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"))
        .output()
        .expect("run oximond");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{profile}/{mode}: {output:?}"
    );
    String::from_utf8(output.stdout).expect("utf-8 output")
}

#[test]
fn the_interpreter_agrees_with_general_frames_in_every_profile() {
    for profile in ["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"] {
        let interpreter = run(profile, "interpreter");
        assert_eq!(interpreter, run(profile, "tracing"), "{profile}");
        for line in [
            "zero\ttrue\n",
            "many\ttrue\t1\t2\t1\t4\n",
            "va\ttrue\t3\t1\tnil\t1\n",
            "short\ttrue\tnil\n",
            "nonstring\tfalse\ttable\t4\n",
            "nested\ttrue\tfalse\tin1\n",
            "deep\ttrue\ttrue\n",
            "hookerr\t",
            "c3\t10\t2\t3\t4\n",
            "v\t0\t3\t1\tnil\tnil\n",
            "tail\t2\n",
            "err\tfalse\tboom\n",
            "rec\tdone\n",
        ] {
            assert!(interpreter.contains(line), "{profile}: missing {line:?}");
        }
        let yields: &[&str] = if profile == "lua51" {
            &["yield\ttrue\tfalse\tattempt to yield across metamethod/C-call boundary\n"]
        } else {
            &["yield\ttrue\t1\n", "yield2\ttrue\ttrue\tback\n"]
        };
        for line in yields {
            assert!(interpreter.contains(line), "{profile}: missing {line:?}");
        }
        if profile == "lua51" || profile == "luajit21" {
            assert!(
                interpreter.contains("ud\tuserdata\t1\t2\n"),
                "{profile}: __call on userdata"
            );
        }
        assert!(
            interpreter.contains("'pcall'"),
            "{profile}: a traceback names the pcall frame"
        );
    }
}
