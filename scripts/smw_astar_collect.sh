#!/usr/bin/env bash
# Teacher-only data with the A* teacher on the 112 level specs of runs/smw/v3_all (64 levels + 48 variants).
# Resumable: finished episodes are on disk (runs/smw/astar_all/data) and skipped; a stopped episode restarts.
#
#   scripts/smw_astar_collect.sh 2        # start / resume with 2 workers (low priority, nice 15)
#   scripts/smw_astar_collect.sh stop     # stop (unfinished episodes are lost, finished ones kept)
#   grep -v Warn runs/smw/astar_all/log.txt | tail   # progress
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == stop ]]; then
    systemctl --user stop astar-collect.scope
    exit 0
fi
workers=${1:-2}
levels=$(python3 -c "import json;print(json.load(open('runs/smw/v3_all/results.json'))['args']['levels'])")
mkdir -p runs/smw/astar_all
systemd-run --user --scope -p MemoryMax=$((2 * workers + 1))G --unit=astar-collect \
    nice -n 15 env OMP_NUM_THREADS=1 PYTHONPATH=scripts .venv/bin/python -u scripts/smw_collect.py \
    --levels "$levels" --runs 1 --teacher astar-div --config-from runs/smw/abc_v2/results.json \
    --workers "$workers" --out runs/smw/astar_all >> runs/smw/astar_all/log.txt 2>&1 &
echo "started astar-collect with $workers workers; log: runs/smw/astar_all/log.txt"
