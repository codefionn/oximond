#!/bin/bash
# Compile cost and loop run cost of the MIR prototype against Cranelift's
# Optimized tier (opt_level=none, backtracking). Run it through the gate:
#
#   scripts/heavy.sh -m 8G cargo build --release -p oximond-mir --example bench
#   scripts/heavy.sh -x -m 8G crates/oximond-mir/bench/bench.sh
#
# Counts instructions:u and cycles:u with perf stat (all threads; the bench is
# single-threaded), pinned with taskset to CPU $CPU, under env -i. Each marginal
# cost is the difference of two runs of the same binary with different repeat
# counts, divided by the difference, so process start and one-time setup cancel.
# Pick a CPU whose SMT sibling (CPU+8) is idle.
#
# Output: one TSV line per measurement on stdout, for bench/summarize.py.
set -euo pipefail
cd "$(dirname "$0")/../../.."
cpu=${CPU:-4}
bin=${BIN:-target/release/examples/bench}
mkdir -p ~/.cache/agent-tmp/mir-research/proto
tmp=$(mktemp -d ~/.cache/agent-tmp/mir-research/proto/bench.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
echo "# uptime start: $(uptime)" >&2
echo "# cpu $cpu, sibling $((cpu + 8)); bin $bin" >&2

# measure MODE KERNEL N -> "instructions cycles" on stdout, program output to $tmp/out
measure() {
    env -i PATH=/usr/bin:/bin taskset -c "$cpu" perf stat -x, -e instructions:u,cycles:u -o "$tmp/perf" \
        "$bin" "$@" >"$tmp/out"
    awk -F, '/instructions:u/ {i=$1} /cycles:u/ {c=$1} END {print i, c}' "$tmp/perf"
}

# marginal MODE KERNEL N1 N2 -> prints "instr_per_unit cycles_per_unit"
marginal() {
    local mode=$1 kernel=$2 n1=$3 n2=$4
    read -r i1 c1 < <(measure "$mode" "$kernel" "$n1")
    read -r i2 c2 < <(measure "$mode" "$kernel" "$n2")
    awk -v i1="$i1" -v i2="$i2" -v c1="$c1" -v c2="$c2" -v n1="$n1" -v n2="$n2" \
        'BEGIN {printf "%.1f %.1f\n", (i2 - i1) / (n2 - n1), (c2 - c1) / (n2 - n1)}'
}

kernels=${KERNELS:-"tiny nbody int_side swap pressure guard_restore"}
for k in $kernels; do
    measure mir-allocs "$k" 200 >/dev/null; mir_allocs=$(cat "$tmp/out")
    measure clif-allocs "$k" 20 >/dev/null; clif_allocs=$(cat "$tmp/out")
    measure mir-compile "$k" 10 >/dev/null; mir_info=$(cat "$tmp/out")
    measure clif-compile "$k" 5 >/dev/null; clif_info=$(cat "$tmp/out")
    read -r mb _ < <(marginal mir-build "$k" 200 2200)
    read -r mc mcc < <(marginal mir-compile "$k" 200 2200)
    read -r ct _ < <(marginal clif-translate "$k" 20 220)
    read -r cc ccc < <(marginal clif-compile "$k" 20 220)
    case $k in
        nbody) f1=2000 f2=12000 ;;
        int_side) f1=20000 f2=120000 ;;
        pressure | guard_restore) f1=20000 f2=120000 ;;
        *) f1=100000 f2=1100000 ;;
    esac
    read -r mr mrc < <(marginal mir-run "$k" "$f1" "$f2")
    measure mir-run "$k" "$f2" >/dev/null; mir_exit=$(cat "$tmp/out")
    read -r cr crc < <(marginal clif-run "$k" "$f1" "$f2")
    measure clif-run "$k" "$f2" >/dev/null; clif_exit=$(cat "$tmp/out")
    printf 'kernel\t%s\tmir_build\t%s\tmir_compile\t%s\tmir_compile_cycles\t%s\tclif_translate\t%s\tclif_compile\t%s\tclif_compile_cycles\t%s\tmir_iter\t%s\tmir_iter_cycles\t%s\tclif_iter\t%s\tclif_iter_cycles\t%s\n' \
        "$k" "$mb" "$mc" "$mcc" "$ct" "$cc" "$ccc" "$mr" "$mrc" "$cr" "$crc"
    printf 'info\t%s\tmir_allocs\t%s\tclif_allocs\t%s\tmir\t%s\tclif\t%s\tmir_run\t%s\tclif_run\t%s\n' \
        "$k" "$mir_allocs" "$clif_allocs" "$mir_info" "$clif_info" "$mir_exit" "$clif_exit"
done
echo "# uptime end: $(uptime)" >&2
