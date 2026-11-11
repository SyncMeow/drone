#!/usr/bin/env bash
# Run the six regression scenarios and write their output (stdout + stderr) to a directory.
# Wall-clock log timestamps "[HH:MM:SS]" are masked so runs can be diffed byte for byte.
# usage: bash baseline/run_all.sh [out_dir]      (default: baseline/, run from lab6/sim)
#        diff -r baseline /tmp/new_run           to compare against the baseline
set -e
cd "$(dirname "$0")/.."
OUT=${1:-baseline}
PY=${PYTHON:-python}
mkdir -p "$OUT"
run() {
    name=$1; shift
    "$PY" "$@" 2>&1 | sed -E 's/^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\]/[HH:MM:SS]/' > "$OUT/$name.txt" &
}
run seg12        run_seg12.py
run seg12_noise  run_seg12.py 1.5 2
run seg345       run_seg345.py
run seg345_noise run_seg345.py 1.5 2
run follow35     run_follow.py 35
run seg5         run_seg5.py
wait
