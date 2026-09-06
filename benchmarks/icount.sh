#!/bin/bash
# Count instructions:u per benchmark, pinned and with a clean environment.
# Run it through the gate: scripts/heavy.sh -x benchmarks/icount.sh [options] [NAME...]
#
#   -b BIN       oximond binary (default target/release/oximond)
#   -r RUNTIMES  comma list of lua,interp,tracing (default all three)
#   -s SCALE     workload scale (default 3, as benchmarks/run.py)
#   -c CPU       CPU to pin to (default 4)
#   -o LEVEL     pass --opt-level=LEVEL to the tracing runtime (default: no flag)
#   NAME...      suite names such as algorithm/nbody; none means the whole suite
#
# Prints one line per benchmark and runtime: name, runtime, millions of
# instructions. Copy binaries you compare to paths of equal length: the
# initial stack alignment follows argv and env size and moves some counts ~5%.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
bin=target/release/oximond runtimes=lua,interp,tracing scale=3 cpu=4 level=
while getopts b:r:s:c:o: opt; do
    case $opt in
        b) bin=$OPTARG ;;
        r) runtimes=$OPTARG ;;
        s) scale=$OPTARG ;;
        c) cpu=$OPTARG ;;
        o) level=$OPTARG ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))
python3 -I - "$@" <<'EOF' >"${TMPDIR:-/tmp}/icount.$$"
import json, sys
suite = json.load(open("benchmarks/suite.json"))
names = sys.argv[1:] or [e["name"] for e in suite]
by_name = {e["name"]: e for e in suite}
for name in names:
    e = by_name[name]
    print(name, "benchmarks/" + e["path"], e.get("compat", "lua51"))
EOF
while read -r name path compat; do
    for runtime in ${runtimes//,/ }; do
        case $runtime in
            lua) cmd=("$([ "$compat" = lua54 ] && echo lua5.4 || echo lua5.1)") ;;
            interp) cmd=("$bin" "--compat=$compat" --execution=interpreter -E) ;;
            tracing) cmd=("$bin" "--compat=$compat" --execution=tracing -E ${level:+"--opt-level=$level"}) ;;
        esac
        count=$(env -i PATH=/usr/bin:/bin taskset -c "$cpu" perf stat -x, -e instructions:u \
            "${cmd[@]}" "$path" "$scale" 2>&1 >/dev/null | awk -F, '/instructions/ {print $1}')
        printf '%-34s %-8s %9.1fM\n' "$name" "$runtime" "$(echo "$count / 1000000" | bc -l)"
    done
done <"${TMPDIR:-/tmp}/icount.$$"
rm -f "${TMPDIR:-/tmp}/icount.$$"
