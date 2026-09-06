#!/bin/bash
# Repeated background-compile runs of the instrumented build for the
# submission to first native entry numbers. Run through the gate:
#
#   INSTR=... scripts/heavy.sh -x -m 16G docs/impl/native-backend/mir/tools/latruns.sh OUTDIR
#
# Two CPUs (4,5), default broker (background worker). Scale 3 is the suite's
# scale; scale 10 lets the sides finish before the run ends.
set -uo pipefail
export LC_ALL=C
out=${1:?outdir}
root=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "$root"
INSTR=${INSTR:?}
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
    for scale in 3 10; do
        runs=10; [ "$scale" = 10 ] && runs=4
        for n in $(seq 1 $runs); do
            "${ENV[@]}" OXIMOND_JIT_TRACE=1 taskset -c 4,5 "$INSTR" --compat="$compat" --execution=tracing -E "$file" "$scale" \
                > /dev/null 2> "$out/$tag.s$scale.bg$n.log"
        done
        "${ENV[@]}" OXIMOND_JIT_SYNC=1 OXIMOND_JIT_TRACE=1 taskset -c 4 "$INSTR" --compat="$compat" --execution=tracing -E "$file" "$scale" \
            > /dev/null 2> "$out/$tag.s$scale.sync.log"
    done
done
uptime > "$out/uptime.end"
