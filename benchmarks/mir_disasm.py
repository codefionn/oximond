#!/usr/bin/env python3
"""Disassemble the machine code the MIR backend printed.

    OXIMOND_JIT_BACKEND=mir OXIMOND_JIT_DISASM=1 oximond prog.lua 2>&1 | python3 benchmarks/mir_disasm.py

Reads stderr of a run with OXIMOND_JIT_DISASM set, finds each
`jit-disasm: mir machine code N bytes` record and runs it through objdump.
"""
import re
import subprocess
import sys
import tempfile

text = sys.stdin.read()
for k, m in enumerate(re.finditer(r"jit-disasm: mir machine code (\d+) bytes\n([0-9a-f]+)", text)):
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(bytes.fromhex(m.group(2)))
        f.flush()
        out = subprocess.run(
            ["objdump", "-D", "-b", "binary", "-mi386:x86-64", "-M", "intel", f.name],
            capture_output=True,
            text=True,
        ).stdout
    print(f"# tree {k}: {m.group(1)} bytes")
    print("\n".join(out.splitlines()[7:]))
