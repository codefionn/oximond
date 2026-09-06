#!/bin/bash
# Collect every raw measurement for latency.md. Run through the gate:
#
#   scripts/heavy.sh -x -m 16G docs/impl/native-backend/mir/tools/collect.sh OUTDIR
#
# Environment:
#   PLAIN   release binary of the measured commit        (target/release/oximond)
#   INSTR   release binary with instrument.patch applied (exact per-thread counters)
#   AP      release binary built with --features alloc-phases
#   SO      mallocprof.so (see mallocprof.c)
#   BENCHES suite names to profile in depth
#
# Sampling profiles are in sample.sh.
set -uo pipefail
export LC_ALL=C
out=${1:?outdir}
root=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "$root"
PLAIN=${PLAIN:-$root/target/release/oximond}
INSTR=${INSTR:?}
AP=${AP:?}
SO=${SO:?}
BENCHES=${BENCHES:-"algorithm/nbody algorithm/mandelbrot algorithm/spectral_norm opt/arith_mixed table/hash_read_indexed"}
mkdir -p "$out"
uptime > "$out/uptime.start"
ENV=(env -i PATH=/usr/bin:/bin)

path_of() { python3 -I -c "
import json,sys
for e in json.load(open('benchmarks/suite.json')):
    if e['name']==sys.argv[1]: print('benchmarks/'+e['path'], e.get('compat','lua51'))" "$1"; }

for b in $BENCHES; do
    tag=${b//\//_}
    read -r file compat < <(path_of "$b")
    args=(--compat="$compat" --execution=tracing -E "$file" 3)
    uptime > "$out/$tag.uptime"
    benchmarks/icount.sh -b "$PLAIN" -r tracing,interp "$b" > "$out/$tag.icount" 2>&1
    "${ENV[@]}" OXIMOND_JIT_SYNC=1 taskset -c 4 perf stat -x, -e instructions:u "$PLAIN" "${args[@]}" 2>&1 >/dev/null |
        awk -F, '/instructions/ {print $1}' > "$out/$tag.total"
    for n in 1 2 3; do
        "${ENV[@]}" OXIMOND_JIT_TRACE=1 taskset -c 4,5 "$INSTR" "${args[@]}" > /dev/null 2> "$out/$tag.lat$n.log"
    done
    "${ENV[@]}" OXIMOND_JIT_SYNC=1 OXIMOND_JIT_TRACE=1 taskset -c 4 "$INSTR" "${args[@]}" > /dev/null 2> "$out/$tag.latsync.log"
    "${ENV[@]}" OXIMOND_JIT_SYNC=1 OXIMOND_ALLOC_PHASES="$out/$tag.alloc_phases.json" \
        taskset -c 4 "$AP" "${args[@]}" > /dev/null 2>&1
    "${ENV[@]}" OXIMOND_JIT_SYNC=1 MALLOCPROF_OUT="$out/$tag.malloc.txt" LD_PRELOAD="$SO" \
        taskset -c 4 "$PLAIN" "${args[@]}" > /dev/null 2>&1
    echo "done $b" >&2
done

# Fit data: every suite benchmark with the instrumented build, compile inline.
mkdir -p "$out/fit"
python3 -I -c "
import json
for e in json.load(open('benchmarks/suite.json')): print(e['name'], 'benchmarks/'+e['path'], e.get('compat','lua51'))" |
while read -r name file compat; do
    tag=${name//\//_}
    "${ENV[@]}" OXIMOND_JIT_SYNC=1 OXIMOND_JIT_TRACE=1 taskset -c 4 \
        "$INSTR" --compat="$compat" --execution=tracing -E "$file" 3 > /dev/null 2> "$out/fit/$tag.log"
done
uptime > "$out/uptime.end"
