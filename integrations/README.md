Run smoke tests for integration builds already present in `target/`:

```sh
python3 integrations/test.py
python3 integrations/test.py --list
```

The runner checks Lua libraries, Valkey, Luanti, and Neovim. Its default run
skips integrations with missing prerequisites. It does not download sources
or build binaries. Each integration's guide explains how to prepare its builds.
Named selections and `--all` fail if their prerequisites are missing.

```sh
python3 integrations/test.py lua-libraries valkey
python3 integrations/test.py lua-libraries valkey --test full
python3 integrations/test.py --all
```

Smoke checks run four Lua library cases under Lua 5.4 interpreter mode,
Valkey's scripting workload under the same profile, Luanti's Lua tests with
both runtimes, and Neovim's core Kickstart checks with all three runtimes.
Full checks run the library and Valkey profile matrices, Luanti devtest worlds,
and Neovim's full configuration. Full Neovim tests can let its plugin manager
download missing plugins.

The runner uses `scripts/heavy.sh` for each workload. Invoke it directly.
It also recognizes an enclosing gate started with the current script.
Use `--already-gated` if an older gate already holds the slot.

Results go to `target/integrations/report.json`, with separate logs and
detailed reports beside it. Change the destination with `--report PATH`.
Use `--oximond PATH` to test another CLI build with the Lua library harness.
Executed failures return a nonzero exit code. The report records skips
separately from passes.

Run the orchestration tests with:

```sh
python3 integrations/test_runner_test.py
```
