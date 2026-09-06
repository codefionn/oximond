The Lua 5.1 public headers and auxiliary library here come from the pinned
Luanti integration checkout's `lib/lua/src` directory at commit
`c0e6812b1a4260bb25a1f606f70f55f4962bb97d`. The auxiliary library
uses only Lua's public API. No stock interpreter, VM, or standard libraries
are compiled into Oximond.

Lua's MIT license and copyright notice are preserved at the end of `lua.h`.
The files are unchanged upstream copies. `modern-module.h` is Oximond's source
compatibility header for the official conformance fixtures. It uses the Lua
5.1 ABI exported by Oximond's native host.
