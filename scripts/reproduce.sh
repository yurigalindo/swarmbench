#!/usr/bin/env bash
# Full SwarmBench pipeline on the LANL 2015 data.
# Usage: scripts/reproduce.sh <dir containing auth.txt.gz and flows.txt.gz> [output dir, default: out]
# Env: PY (python interpreter, default python3).
# Test days 6, 8 and 9 with seeds 0-4 (15 seed-days, the paper's main test). Day 7 was used to design SwarmScore
# and is left out. Each test day uses the previous day as its pre-swarm reference, so days 5-9 are prepared.
# Rough cost on a laptop: prepare ~30-60 min (reads the first 9 days of both files), build ~2 min, score ~1 h
# (the PMF baseline dominates; drop it with DETECTORS=convergence,first_time,swarmscore), evaluate ~5 min.
# Disk: ~4 GB for the per-day parquet files. Memory: ~16 GB peak.
set -euo pipefail
PY="${PY:-python3}"
LANL="${1:?usage: scripts/reproduce.sh <dir with auth.txt.gz and flows.txt.gz> [out_dir]}"
OUT="${2:-out}"
DETECTORS="${DETECTORS:-convergence,first_time,swarmscore,pmf}"
cd "$(dirname "$0")/.."
$PY -m swarmbench prepare  --auth "$LANL/auth.txt.gz" --flows "$LANL/flows.txt.gz" --days 5-9 --data "$OUT/data"
$PY -m swarmbench build    --data "$OUT/data" --bench "$OUT/bench" --days 6,8,9 --seeds 0-4
$PY -m swarmbench score    --data "$OUT/data" --bench "$OUT/bench" --scores "$OUT/scores" --detectors "$DETECTORS"
$PY -m swarmbench evaluate --bench "$OUT/bench" --scores "$OUT/scores" --out "$OUT/results"
echo "Results: $OUT/results/RESULTS.md and $OUT/results/results.json"
