#!/usr/bin/env bash
# Rewrites crates/oximond-cli/startup-order.txt: the functions a start of
# `oximond` runs, in the order it first runs them, and then the ones that
# warming up a trace runs (scripts/startup-order/warmup.lua) and a start does
# not. crates/oximond-cli/build.rs turns the list into a linker script that
# places those functions side by side, so a start faults in a few pages of
# code instead of dozens, and a trace's first compile a few dozen instead of a
# hundred (see the note there). A name that has several aliases (identical functions the linker
# folds) is listed under each. The `.llvm.<hash>` suffix of a promoted name may
# be stale: the script matches any suffix.
#
#   scripts/startup-order.sh [BINARY]
#
# BINARY defaults to target/release/oximond. Run it after a release build, and
# again when a toolchain update renames many functions; build, then rebuild so
# the new list takes effect. Needs cc, nm and python3, and a kernel that lets
# ptrace single-step a child.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
binary=${1:-$root/target/release/oximond}
work=$(mktemp -d "${TMPDIR:-/tmp}/startup-order.XXXXXX")
trap 'rm -rf "$work"' EXIT
cc -O2 -o "$work/trace" "$root/scripts/startup-order/trace.c"
: > "$work/empty.lua"
"$work/trace" "$work/addresses" 555555554000 "$binary" \
    --compat=lua51 --execution=tracing -E "$work/empty.lua"
# With one CPU the compiles run on the tracing thread, which the tracer sees.
taskset -c 0 "$work/trace" "$work/warm" 555555554000 "$binary" \
    --compat=lua51 --execution=tracing -E "$root/scripts/startup-order/warmup.lua"
python3 -I "$root/scripts/startup-order/functions.py" "$binary" "$work/addresses" "$work/warm" \
    > "$root/crates/oximond-cli/startup-order.txt"
wc -l "$root/crates/oximond-cli/startup-order.txt"
