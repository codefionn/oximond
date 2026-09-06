---
name: bench
description: Run the Lua benchmark suite on Lua, LuaJIT, and both Oximond modes (interpreter, tracing) and report the table in chat. Use when asked to run, rerun, or compare benchmarks.
---

Arguments pass through to `benchmarks/run.py` (`--filter REGEX`, `--quick`,
`--repeat N`). With none, run the full suite.

1. Build the binary the suite runs. `target/release/oximond` is shared and
   often older than HEAD:

   ```sh
   scripts/heavy.sh -m 16G cargo build --release -p oximond-cli
   ```

   In a worktree, run from the worktree so its own `target/` is built and
   measured.

2. Pick a CPU. CPUs N and N+8 share a core, so choose N where both N and N+8
   are idle (`top -1 -bn2 -d1`, second sample; CPU 4 when nothing stands out). Record
   `uptime` and the HEAD revision.

3. Run the suite exclusively, in the background (the full suite takes a few
   minutes):

   ```sh
   scripts/heavy.sh -x python3 benchmarks/run.py --cpu N \
     --json target/benchmarks/results.json \
     --markdown target/benchmarks/results.md [ARGS]
   ```

   Exit 1 is a checksum mismatch or a missing runtime. Report it with the
   runner's output; it is a correctness bug, not a slow benchmark. Record
   `uptime` again when it finishes.

4. Report in chat. Done when the message holds:
   - the revision, the CPU, and the load average before and after. Above
     about 2, say that differences under 10% are noise.
   - the full table from `target/benchmarks/results.md`, with times
     shortened to milliseconds at three significant figures and columns
     renamed `interp` and `tracing`.
   - findings, each tied to rows: tracing slower than the interpreter,
     benchmarks where tracing gains nothing over the interpreter, and the
     largest gaps to LuaJIT.
   - when a previous run's numbers are in the conversation, the rows that
     moved by more than 10%.

Wall time is noisy on this shared machine. To confirm a single finding,
count instructions instead: `perf stat -e instructions:u` under `taskset`,
through `scripts/heavy.sh -x`.
