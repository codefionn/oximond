# Third-party notices

## LuaJIT test-cleanup

The compiler and CLI tests include the source of `test/lang/assignment.lua` from:

- Repository: https://github.com/LuaJIT/LuaJIT-test-cleanup
- Commit: `014708bceb70550a3ab8d539cff14d9085ca9cb8`

The repository README says Mike Pall placed the Lua and LuaJIT tests and benchmarks he wrote in the public domain. It also warns that some benchmark files have other or uncertain copyrights. Oximond does not copy the benchmark tree. The full corpus is fetched as an external conformance input and is not included in Oximond release artifacts.

## Reference implementations

Oximond uses the Lua 5.1 and LuaJIT source code and documentation as behavioral references. Their code is not linked into Oximond.

- Lua 5.1: Copyright 1994-2012 Lua.org, PUC-Rio. Lua license.
- LuaJIT: Copyright 2005-2026 Mike Pall. MIT license.
