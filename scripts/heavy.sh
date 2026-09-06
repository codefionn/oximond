#!/usr/bin/env bash
# Run a memory-heavy command (release/LTO build, conformance run, benchmark,
# integration workload) through a machine-wide gate shared by every agent
# session and worktree of this repository.
#
#   scripts/heavy.sh [-x] [-m MEMORY] [--] COMMAND [ARGS...]
#   scripts/heavy.sh --status
#
# At most OXIMOND_HEAVY_SLOTS (default 2) commands run at once; the rest wait.
# -x takes every slot, so benchmarks and timing runs have the gated machine
# to themselves; new jobs queue behind a waiting -x job instead of starving it.
# Each command runs under a hard memory cap (default 8G, -m to change) and is
# killed on its own when it exceeds it, instead of dragging the machine into
# swap. Cargo gets CARGO_BUILD_JOBS=8 unless set. Holders and waiters are
# listed on a shared board.
set -eu

common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || {
    echo "heavy.sh: run inside the oximond repository" >&2
    exit 2
}
gate="$common/oximond-heavy"
board="$gate/board"
mkdir -p "$gate"
touch "$board"

# Board edits from concurrent sessions serialize on their own lock.
prune() {
    while read -r pid rest; do
        kill -0 "$pid" 2>/dev/null && printf '%s %s\n' "$pid" "$rest"
    done < "$board" > "$board.$$" || true
    mv "$board.$$" "$board"
}
note() { (flock 8; prune; printf '%s %s\n' "$$" "$*" >> "$board") 8>"$gate/board.lock"; }
unnote() {
    (flock 8; grep -v "^$$ " "$board" > "$board.$$" || true; mv "$board.$$" "$board") 8>"$gate/board.lock"
}

if [ "${1:-}" = "--status" ]; then
    (flock 8; prune) 8>"$gate/board.lock"
    if [ -s "$board" ]; then cat "$board"; else echo "no heavy jobs"; fi
    free -h | sed -n '1,2p'
    exit 0
fi

memory=8G
exclusive=
while :; do
    case "${1:-}" in
        -m) memory=$2; shift 2 ;;
        -x) exclusive=1; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] || { echo "usage: scripts/heavy.sh [-x] [-m MEMORY] [--] COMMAND..." >&2; exit 2; }

slots=${OXIMOND_HEAVY_SLOTS:-2}
where=$(git rev-parse --show-toplevel)
describe() { printf '%s since %s mem=%s in %s: %s' "$1" "$(date +%H:%M:%S)" "$memory" "$where" "$job"; }
job=$*
mode=${exclusive:+exclusive }
cleanup() {
    unnote
    if [ -n "$exclusive" ] && [ "$(cat "$gate/exclusive" 2>/dev/null)" = "$$" ]; then
        rm -f "$gate/exclusive"
    fi
}
trap cleanup EXIT INT TERM
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"

run() {
    unnote
    note "$(describe "${mode}running")"
    # Child runners can reuse this scope instead of queuing inside our slot.
    export OXIMOND_HEAVY_ACTIVE=1
    if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
        systemd-run --user --scope -q -p MemoryMax="$memory" -p MemorySwapMax=0 "$@"
    else
        "$@"
    fi
}

# An -x job waiting for slots; other jobs hold back until it has run.
exclusive_pending() {
    [ -s "$gate/exclusive" ] && kill -0 "$(cat "$gate/exclusive")" 2>/dev/null
}

note "$(describe "${mode}waiting")"
if [ -n "$exclusive" ]; then
    while exclusive_pending; do sleep 5; done
    echo $$ > "$gate/exclusive"
    slot=0
    while [ "$slot" -lt "$slots" ]; do
        # Blocking acquisition in slot order cannot deadlock with another -x.
        eval "exec $((slot + 10))>\"\$gate/slot.\$slot\""
        flock $((slot + 10))
        slot=$((slot + 1))
    done
    run "$@"
    exit $?
fi
while :; do
    if ! exclusive_pending; then
        slot=0
        while [ "$slot" -lt "$slots" ]; do
            exec 9>"$gate/slot.$slot"
            if flock -n 9; then
                run "$@"
                exit $?
            fi
            exec 9>&-
            slot=$((slot + 1))
        done
    fi
    sleep 5
done
