#!/usr/bin/env python3
"""Counts the instructions of a loop in a raw x86-64 image.

usage: disasm.py IMAGE.bin

The loop is found as the last backward `jmp`: it runs from the jump's target to
the jump. Reports instruction count, spill traffic (memory operands based on
rsp), register-to-register moves, and loads/stores through other bases.
"""
import re
import subprocess
import sys

out = subprocess.run(
    ["objdump", "-D", "-b", "binary", "-mi386:x86-64", "-M", "intel", sys.argv[1]],
    capture_output=True, text=True, check=True,
).stdout
insts = []
for line in out.splitlines():
    m = re.match(r"\s*([0-9a-f]+):\t([0-9a-f ]+?)\s*\t(.*)", line)
    if m:
        insts.append((int(m.group(1), 16), m.group(3).strip()))
    else:
        m = re.match(r"\s*([0-9a-f]+):\t([0-9a-f ]+)$", line)  # continuation of long encodings
        if m and insts:
            pass
loop = None
for addr, text in insts:
    m = re.match(r"jmp\s+0x([0-9a-f]+)", text)
    if m and int(m.group(1), 16) < addr:
        loop = (int(m.group(1), 16), addr)
if loop is None:
    print("no backward jmp found")
    sys.exit(1)
body = [(a, t) for a, t in insts if loop[0] <= a <= loop[1]]
regs = r"(?:r[a-d]x|r[sd]i|r[sb]p|r\d+[dwb]?|e[a-d]x|xmm\d+)"
spill = sum(1 for _, t in body if "[rsp" in t)
rr = sum(1 for _, t in body if re.match(rf"(?:mov|movaps|movapd|movsd|movq|movdqa|movups)\s+{regs},{regs}$", t))
mem = sum(1 for _, t in body if "[" in t and "[rsp" not in t and not t.startswith("lea"))
print(f"loop 0x{loop[0]:x}..0x{loop[1]:x}: {len(body)} instructions, {spill} with an rsp operand (spill or reload), {rr} register moves, {mem} other memory operands")
