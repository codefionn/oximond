"""Turns the addresses `trace` printed into the functions that contain them,
in the order a run first runs them. usage: functions.py BINARY ADDRESSES...
Each further file lists the functions of another run, after those of the
files before it, without the ones they listed."""
import bisect
import subprocess
import sys

binary, *address_files = sys.argv[1:]
symbols = []
listing = subprocess.run(
    ["nm", "-S", "-n", "--defined-only", binary], capture_output=True, text=True, check=True
).stdout
for line in listing.splitlines():
    parts = line.split(None, 3)
    if len(parts) == 4 and parts[2] in "tTwW":
        symbols.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
starts = [symbol[0] for symbol in symbols]
seen = set()
lines = [line for path in address_files for line in open(path)]
for line in lines:
    address = int(line, 16)
    index = bisect.bisect_right(starts, address) - 1
    if index >= 0 and address < symbols[index][0] + max(symbols[index][1], 1):
        # Identical functions are folded into one address, and the linker may
        # keep either name's section, so list every name at the address.
        first = index
        while first > 0 and symbols[first - 1][0] == symbols[index][0]:
            first -= 1
        last = index
        while last + 1 < len(symbols) and symbols[last + 1][0] == symbols[index][0]:
            last += 1
        for name in (symbol[2] for symbol in symbols[first : last + 1]):
            if name not in seen:
                seen.add(name)
                print(name)
