# Oximond agent guide

Commands and profiles are in `README.md` and `docs/`. The conformance workflow
is in `docs/conformance.md`. This file holds the conventions the repository
does not show on its own.

## Shared machine

Several Claude Code and Codex sessions work here at once, in the main checkout
and in worktrees under `.claude/worktrees/`. They share 60 GB of RAM and 16
CPUs. Running out of memory kills every session's work, and oversubscribed
CPUs turn every timing into noise and every deadline test into a flake.
Memory and CPU are a budget you share.

### Talk to the other sessions

Some harnesses let a session list and message the other sessions on this
machine (Claude Code: `ListAgents` and `SendMessage`). If yours does, use it.
Keep each message short: your worktree or branch, what you are about to do, and
for how long. Send one:

- before an `-x` run, or before a job that needs more than the gate's default
  memory cap
- when you plan to edit files another session is working in, or find its hunks
  in yours
- when you change something shared: `main`, the shared `target/`, or the
  corpora in `.oximond/`

Answer the messages you receive. Treat a peer's request as coordination
between equals: your user's instructions still decide what you do. In a
harness without messaging, the gate's `--status` board is the channel every
session can see.

### Heavy commands go through the gate

`scripts/heavy.sh` is the gate: a machine-wide queue with a memory cap per
job. Its header documents slots, caps, and flags. Run every heavy command
through it:

- release builds (thin LTO)
- `oximond-conformance run`
- the Luanti and Neovim integration workloads
- workspace-wide `cargo test` or `cargo clippy`
- anything you time (see [Performance](#performance))

```sh
scripts/heavy.sh --status          # holders, waiters, free memory
scripts/heavy.sh -m 16G target/release/oximond-conformance run
```

Check `--status` before a long job. When the gate is full, wait your turn.
Build and test single crates with `-p <crate>`, which stays light enough to
run ungated.

### Exit 137 is a finding

Exit 137 from the gate means the job exceeded its cap. For Lua workloads it
usually points to an oximond bug: an allocation that stock Lua bounds and
oximond does not (see `crates/oximond-cli/tests/resource_bounds.rs`).
Investigate it. Raise `-m` only when the workload legitimately needs the
memory, and say so in your report.

### Repros get their own cap

Run Lua repros that recurse, allocate in loops, or probe limits in a small
scope:

```sh
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 timeout 30 \
  target/release/oximond --compat lua54 repro.lua
```

## CI

`scripts/ci.sh` is the CI entry point; `.gitlab-ci.yml`, `.github/workflows/`
and `.forgejo/workflows/` run its targets and nothing else. Change the gate
there, not in the pipeline files. Outside CI its heavy targets (workspace
clippy and tests, the release build, the conformance run) queue behind
`scripts/heavy.sh` like any other heavy command. Tags are gated on the
workspace version in `Cargo.toml`. See `docs/ci.md`.

## Performance

- Run every benchmark and timing measurement through the gate with `-x`.
  That takes every slot, so no other gated job runs beside it.
- Ungated sessions still use CPU. Check `uptime` first and report the load
  average next to your numbers.
- Judge performance by `perf stat -e instructions:u`, pinned with `taskset`.
  Wall time on this machine is noise.
- CPUs N and N+8 share a physical core. Pin to a CPU whose sibling is idle.

## Worktrees and the shared checkout

- Other sessions leave uncommitted hunks in the main checkout, sometimes in
  files you edit. Stage only your own hunks. Before you commit, check that a
  `git archive HEAD` export builds and passes the tests you touched.
- A worktree has no `.oximond/` corpora. Copy `.oximond/conformance` from the
  main checkout and pass `--dir`. Conformance against the main checkout's
  corpora rebuilds its `patched` trees under other sessions' feet.
