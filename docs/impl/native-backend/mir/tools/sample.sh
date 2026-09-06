#!/bin/bash
# Sampling profiles for phases.py. Run through the gate:
#
#   scripts/heavy.sh -x -m 16G docs/impl/native-backend/mir/tools/sample.sh OUTDIR
#
# Environment: PLAIN (release binary), BENCHES, RUNS (default 250 for nbody, half
# of it for the others).
#
# Why many short runs with a long period: this machine throttles perf. Below
# about 2000 samples a second every overflow is kept, above that the kernel
# keeps a few samples per millisecond and, after about 5 ms, none, and that
# biases a profile toward the start of the run. A period of one 24th of the
# run's instructions stays under the limit (22 samples in nbody's 33.6M
# instructions, checked against the expected count). Each run uses a slightly
# different period so the samples of many runs land at different points of the
# same deterministic instruction stream.
set -uo pipefail
export LC_ALL=C
out=${1:?outdir}
root=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "$root"
PLAIN=${PLAIN:-$root/target/release/oximond}
RUNS=${RUNS:-250}
BENCHES=${BENCHES:-"algorithm/nbody algorithm/mandelbrot algorithm/spectral_norm opt/arith_mixed table/hash_read_indexed"}
mkdir -p "$out"
uptime > "$out/uptime.start"
ENV=(env -i PATH=/usr/bin:/bin)
for b in $BENCHES; do
    tag=${b//\//_}
    read -r file compat < <(python3 -I -c "
import json,sys
for e in json.load(open('benchmarks/suite.json')):
    if e['name']==sys.argv[1]: print('benchmarks/'+e['path'], e.get('compat','lua51'))" "$b")
    args=(--compat="$compat" --execution=tracing -E "$file" 3)
    mkdir -p "$out/$tag.bg" "$out/$tag.sync"
    # Exact instruction count of one run, then a period that gives about 22
    # samples a run, jittered by up to 12% from run to run.
    total=$("${ENV[@]}" OXIMOND_JIT_SYNC=1 taskset -c 4 perf stat -x, -e instructions:u "$PLAIN" "${args[@]}" 2>&1 >/dev/null | awk -F, '/instructions/ {print $1}')
    echo "$total" > "$out/$tag.total"
    base=$(( total / 24 ))
    runs=$RUNS
    [ "$b" != algorithm/nbody ] && runs=$(( RUNS / 2 ))
    for i in $(seq 1 "$runs"); do
        period=$(( base + base * i * 12 / (100 * runs) ))
        # Background compile thread: two CPUs.
        "${ENV[@]}" taskset -c 4,5 perf record -q -e instructions:u -c "$period" --call-graph dwarf,24576 \
            -o "$out/$tag.bg/$i.data" -- "$PLAIN" "${args[@]}" > /dev/null 2>&1
        # The same without call chains. A sample whose IP is in generated code
        # has no callchain in the dwarf recordings above, and so has one whose
        # unwinding failed; the flat recordings keep the IP and tell them apart.
        "${ENV[@]}" taskset -c 4,5 perf record -q -e instructions:u -c "$period" \
            -o "$out/$tag.bg/$i.flat" -- "$PLAIN" "${args[@]}" > /dev/null 2>&1
        "${ENV[@]}" OXIMOND_JIT_SYNC=1 taskset -c 4 perf record -q -e instructions:u -c "$period" \
            -o "$out/$tag.sync/$i.flat" -- "$PLAIN" "${args[@]}" > /dev/null 2>&1
        # Inline compile: one CPU, as benchmarks/icount.sh measures.
        "${ENV[@]}" OXIMOND_JIT_SYNC=1 taskset -c 4 perf record -q -e instructions:u -c "$period" --call-graph dwarf,24576 \
            -o "$out/$tag.sync/$i.data" -- "$PLAIN" "${args[@]}" > /dev/null 2>&1
    done
    echo "done $b" >&2
done
uptime > "$out/uptime.end"
