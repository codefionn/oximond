//! Background compilation installs traces at scheduling-dependent points.
//! These tests sweep deterministic install delays so every such point runs.
//! Most programs here once broke the removed region JIT at one install point;
//! they stay as sweeps over the trace engine's install points.

use oximond::compat::{CompatibilityProfile, ExecutionMode};
use oximond::compiler::{CompilerConfig, compile_with_config};
use oximond::stdlib::install_base_with_profile;
use oximond::vm::Vm;

fn assert_every_install_point_matches_the_interpreter(
    source: &str,
    delays: impl Iterator<Item = u32>,
) {
    assert_profile_install_points_match(CompatibilityProfile::Lua51, source, delays);
}

fn assert_profile_install_points_match(
    profile: CompatibilityProfile,
    source: &str,
    delays: impl Iterator<Item = u32>,
) {
    let chunk = compile_with_config(source, CompilerConfig::new(profile)).expect("compile");
    let mut interpreter = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    install_base_with_profile(interpreter.runtime_mut(), profile).unwrap();
    let expected = interpreter.execute(&chunk).expect("interpret");
    let delays: Vec<u32> = delays.collect();
    for &delay in &delays {
        let mut tracing = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        tracing.use_delayed_synchronous_jit_compilation(delay);
        install_base_with_profile(tracing.runtime_mut(), profile).unwrap();
        let actual = tracing.execute(&chunk);
        assert_eq!(
            actual.as_ref().ok(),
            Some(&expected),
            "install delay {delay}: {actual:?}"
        );
    }
}

#[test]
fn table_concat_builder_survives_every_install_point() {
    assert_every_install_point_matches_the_interpreter(
        r#"
        local total = 0
        for _ = 1, 12 do
            local parts = {}
            for i = 1, 300 do
                parts[i] = tostring(i % 101)
            end
            total = total + #table.concat(parts, ",")
        end
        return total
        "#,
        0..64,
    );
}

#[test]
fn intrinsic_results_outside_the_store_shape_replay_in_the_vm() {
    // Each loop keeps a table in a moved temporary and calls `tostring`
    // natively. A string result the region cannot consume must not resume
    // after a call that never committed the frame.
    assert_every_install_point_matches_the_interpreter(
        r#"
        local hits, parts, lengths = 0, {}, 0
        for i = 1, 200 do
            local s = tostring(i % 7)
            if s == "3" then hits = hits + 1 end
            parts[i] = s
            local t = parts
            lengths = lengths + #t[i] + #tostring(i)
        end
        return hits, #parts, lengths
        "#,
        0..32,
    );
    // A string result that feeds a concatenation, not a table store.
    assert_every_install_point_matches_the_interpreter(
        r#"
        local t = {}
        for i = 1, 3000 do
            local x = i * 0.5
            local y = i / 8
            t[#t + 1] = tostring(x) .. "," .. y .. "," .. (x + 0.25)
        end
        return #t, t[#t]
        "#,
        0..8,
    );
    for (outer, inner) in [(1, 3), (2, 2), (4, 50)] {
        assert_every_install_point_matches_the_interpreter(
            &format!(
                r#"
                local total = 0
                for _ = 1, {outer} do
                    local parts = {{}}
                    for i = 1, {inner} do
                        parts[i] = tostring(i % 101)
                    end
                    total = total + #table.concat(parts, ",")
                end
                return total
                "#
            ),
            0..8,
        );
    }
}

#[test]
fn materialized_upvalue_does_not_clobber_a_live_register() {
    // In Lua 5.2 globals load through the `_ENV` upvalue. The region that
    // resumes after `math.floor` reuses the register holding `math.fmod`
    // for a later `_ENV` load; materializing the upvalue at entry must not
    // overwrite the live function.
    assert_profile_install_points_match(
        CompatibilityProfile::Lua52,
        r#"
        c = 1
        local function f(t, i)
            local b = t.n
            local res = math.fmod(math.floor(i / c), b) + 1
            c = c * b
            return t[res]
        end
        local neg = {" ", " not "; n = 2}
        local arg = {"a", "b", "c", "d"; n = 4}
        local count = 0
        local i = 0
        repeat
            c = 1
            local s = f(neg, i) .. f(arg, i) .. f(neg, i) .. f(arg, i) .. f(neg, i) .. f(arg, i)
            count = count + #s
            i = i + 1
        until i == c
        return count, i
        "#,
        0..8,
    );
}

#[test]
fn region_exit_rewrites_registers_a_call_commit_left_stale_in_the_frame() {
    // A native call site commits the whole register file to the frame. The
    // inner loop's limit register then holds `math.abs` there. When the
    // region later exits with that register back at its entry value, the exit
    // must still write it, or the interpreter's loop latch reads a function.
    assert_every_install_point_matches_the_interpreter(
        r"
        local total = 0
        for i = 1, 150000 do
            for j = 1, 2 do total = total + j end
            if i > 2000 then total = total + math.abs(i) end
        end
        return total
        ",
        0..16,
    );
}

#[test]
fn region_exit_after_a_call_commit_keeps_silent_values_right() {
    // The same stale-frame bug without a type error: a three-level nest with
    // `math.abs` in the middle level, and a `while` inner loop before
    // `math.floor`. A stale register gives a wrong sum, not an error.
    assert_every_install_point_matches_the_interpreter(
        r"
        local t = 0
        for i = 1, 2000 do
            for j = 1, 100 do
                for k = 1, 2 do t = t + k end
                t = t + math.abs(j - 50)
            end
        end
        return t
        ",
        0..16,
    );
    assert_every_install_point_matches_the_interpreter(
        r"
        local t = 0
        for i = 1, 100000 do
            local j = 0
            while j < 3 do j = j + 1; t = t + j end
            t = t + math.floor(i / 2)
        end
        return t
        ",
        0..16,
    );
}

#[test]
fn call_committed_frame_does_not_leak_into_table_fields_after_a_region_exit() {
    // Removes the script even when an assertion panics.
    struct RemoveOnDrop(std::path::PathBuf);
    impl Drop for RemoveOnDrop {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    // Generated reproducer (seed 136). A field that is only ever read came
    // back as -3, the argument of the `abs` call, because a region exit
    // skipped a write-back after the call site had committed the frame. It
    // needs real background installs, so it runs the binary.
    let source = r"
        local abs = math.abs
        local m = {} for r = 1, 8 do local row = {} for c = 1, 16 do row[c] = c end m[r] = row end
        local t = {} for k = 1, 32 do t[k] = k end
        local p = {x = 1, y = 2}
        local acc = 0
        local function bump(v) acc = acc + v end
        local s, z = 0, 0
        for i = 1, 3000 do
            local row = m[(i % 8) + 1]; s = s + row[(i % 16) + 1]
            local j = 0
            while j < 12 do
                j = j + 1
                if j % 3 == 0 then s = s + p.y else z = z - 1 end
                for k = j % 4, 9 do
                    s = s + (i * s)
                    if s > 1e6 then s = s - 1e6 end
                    if z > 100000 then break end
                end
                if i == 2995 then s = s + abs(-3) end
            end
        end
        local q = 0 for k = 1, 32 do q = q + t[k] end
        print(s, z, q, acc, p.x, p.y)
    ";
    let path = std::env::temp_dir().join(format!("oximond-call-commit-{}.lua", std::process::id()));
    let _cleanup = RemoveOnDrop(path.clone());
    std::fs::write(&path, source).expect("write script");
    let run = |extra: &[&str]| {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_oximond"))
            .arg("--compat=lua51")
            .args(extra)
            .arg(&path)
            .output()
            .expect("run oximond");
        assert!(output.status.success(), "{extra:?}: {output:?}");
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    let expected = run(&["--execution=interpreter"]);
    let actual = run(&["--execution=tracing"]);
    assert_eq!(actual, expected);
}
