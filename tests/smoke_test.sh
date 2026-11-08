#!/usr/bin/env bash
# End-to-end smoke test on small synthetic logs (no real data). Runs in a few minutes.
# Usage: tests/smoke_test.sh [python]   (default: python3)
set -euo pipefail
PY="${1:-python3}"
cd "$(dirname "$0")/.."
W="${SMOKE_DIR:-$PWD/out/smoke}"   # inside the repo (gitignored); removed at the end
rm -rf "$W"; mkdir -p "$W"
trap 'rm -rf "$W"' EXIT
$PY -m swarmbench synthetic --out "$W/raw" --hosts 1500 --days 1-3
$PY -m swarmbench prepare --auth "$W/raw/auth.txt.gz" --flows "$W/raw/flows.txt.gz" --days 1-3 --data "$W/data"
$PY -m swarmbench build --data "$W/data" --bench "$W/bench" --days 2,3 --seeds 0-1 --size 0.05
$PY -m swarmbench score --data "$W/data" --bench "$W/bench" --scores "$W/scores"
$PY -m swarmbench evaluate --bench "$W/bench" --scores "$W/scores" --out "$W/results" --boot 200 --draws 5
$PY - "$W" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1] + "/results/results.json"))
assert set(r["teams_caught"]) == {"convergence", "swarmscore", "pmf", "first_time"}, r["teams_caught"].keys()
for det, w in r["window"].items():
    assert w["n_pos"] > 0 and 0.0 <= w["auroc"] <= 1.0, (det, w)
print("SMOKE TEST PASSED")
PYEOF
