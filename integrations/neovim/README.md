# Neovim engine comparison

This integration builds Neovim v0.12.5 at commit
`5885a30e1e1225349079e7a1c4a3848aa8e43e42` against Lua 5.1,
LuaJIT 2.1, and Oximond. All three builds use [kickstart.nvim](https://github.com/nvim-lua/kickstart.nvim)
at commit `80743df53d8f7058fc5b60e41f1081d11df9c880` as their configuration.
The driver keeps each editor's config, data, cache, and state under
`target/neovim/user-<engine>`. It does not use your personal Neovim setup.

Run from the Oximond repository root:

```sh
python3 integrations/neovim/neovim.py prepare
python3 integrations/neovim/neovim.py build all
python3 integrations/neovim/neovim.py smoke all
python3 integrations/neovim/neovim.py full-smoke oximond
python3 integrations/neovim/neovim.py benchmark all --samples 7
python3 integrations/neovim/neovim.py run luajit
python3 integrations/neovim/neovim.py run --full-config luajit
python3 integrations/neovim/neovim.py run --full-config oximond
```

For the Oximond build, the driver also downloads Lua 5.1.5 source from
[lua.org](https://www.lua.org/ftp/) and verifies its published SHA-256. It
compiles `lauxlib.c` and Oximond's C API bridge, not Lua's interpreter.
`build lua` and `build luajit` use the host's development libraries. Build
requirements on Linux are CMake, Ninja, a C/C++ compiler, Rust/Cargo for Oximond, and
Neovim's system development dependencies. The [Neovim build guide](https://github.com/neovim/neovim/blob/v0.12.5/BUILD.md)
lists the latter. The complete kickstart configuration also needs its own
[external tools](https://github.com/nvim-lua/kickstart.nvim#install-external-dependencies).
The Oximond link options currently target Linux.

`smoke` loads kickstart's options, keymaps, and plugin build hook in headless
Neovim. It edits a scratch buffer through `vim.api` and checks a fixed result.
`benchmark` repeats the same work in shuffled engine order and reports median
wall time. That time includes editor startup and the common kickstart core.
It is an editor workload, not a pure Lua VM microbenchmark.

The checked-in kickstart patch stops before plugin setup in comparison mode.
This gives all three engines the same workload and keeps plugin downloads out
of the timing. `full-smoke` checks the complete kickstart config separately.
It passes with LuaJIT and Oximond in the pinned setup. Their first full launch
downloads plugins and may start parser downloads. Stock Lua 5.1 cannot parse
`goto` statements in the pinned guess-indent and Blink plugins, so its full
config fails before setup finishes. The benchmark uses the common core for
that reason.

Use `run lua`, `run luajit`, or `run oximond` to try the editor interactively.
Editor arguments can follow `--`, for example
`python3 integrations/neovim/neovim.py run oximond -- README.md`.

The Neovim and kickstart patches only change isolated checkouts under
`target/neovim`. `prepare` refuses to overwrite unrelated changes there.
