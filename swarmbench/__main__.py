"""SwarmBench command line.

    python -m swarmbench prepare   --auth auth.txt.gz --flows flows.txt.gz --days 5-9 --data data/lanl
    python -m swarmbench build     --data data/lanl --bench out/bench --days 6,8,9 --seeds 0-4
    python -m swarmbench score     --data data/lanl --bench out/bench --scores out/scores [--detectors convergence,first_time,swarmscore,pmf]
    python -m swarmbench evaluate  --bench out/bench --scores out/scores --out out/results
    python -m swarmbench synthetic --out data/synth_raw          (small LANL-format logs for the smoke test)
"""
import argparse
import json
import os
import time

import polars as pl

from . import build as BLD
from . import data as D
from . import detectors as DET
from . import evaluate as EV

NAMES = {"convergence": "Convergence rule", "swarmscore": "SwarmScore (convergence + first-time accounts)",
         "pmf": "Per-machine anomaly model (PMF)", "first_time": "First-time accounts alone"}


def _range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def cmd_prepare(a):
    D.prepare(a.auth, a.flows, _range(a.days), a.data)


def cmd_build(a):
    tpl = BLD.load_template(a.template)
    for d in _range(a.days):
        for s in _range(a.seeds):
            t0 = time.time(); sd = BLD.build_seed_day(a.data, a.bench, d, s, tpl, size=a.size)
            print("built", os.path.basename(sd), "%.0fs" % (time.time() - t0), flush=True)


def cmd_score(a):
    names = a.detectors.split(",")
    for sd, w in D.instance_ids(a.bench):
        todo = [n for n in names if not os.path.exists(os.path.join(a.scores, n, sd, w + ".parquet"))]
        if not todo:
            continue
        t0 = time.time(); hist, win, elig, _ = D.load_instance(a.data, a.bench, sd, w); el = set(elig)
        for n, sc in DET.score_all(hist, win, todo).items():
            os.makedirs(os.path.join(a.scores, n, sd), exist_ok=True)
            hs = sorted(sc)
            pl.DataFrame({"host": hs, "score": [sc[h] for h in hs], "elig": [h in el for h in hs]}).write_parquet(os.path.join(a.scores, n, sd, w + ".parquet"))
        print("scored", sd, w, ",".join(todo), "%.0fs" % (time.time() - t0), flush=True)


def _load_scores(a, det):
    by_sd = {}
    for sd, w in D.instance_ids(a.bench):
        p = os.path.join(a.scores, det, sd, w + ".parquet")
        if not os.path.exists(p):
            return None
        lab = json.load(open(os.path.join(a.bench, sd, "labels", w + ".json")))
        meta = json.load(open(os.path.join(a.bench, sd, "meta.json")))["windows"][w]
        by_sd.setdefault(sd, []).append(dict(scores=pl.read_parquet(p), carriers=set(lab["carrier_hosts"]),
                                             carrier_team={h: int(t) for h, t in lab["carrier_team"].items()}, start=meta["start_hour"], L=meta["L"]))
    return by_sd


def cmd_evaluate(a):
    os.makedirs(a.out, exist_ok=True); R = {"window": {}, "teams_caught": {}, "paired_at_10": {}}; tabs = {}
    for det in ("convergence", "swarmscore", "pmf", "first_time"):
        by_sd = _load_scores(a, det)
        if by_sd is None:
            continue
        R["window"][det] = EV.window_metrics([w for v in by_sd.values() for w in v])
        rows = []
        for sd, ws in sorted(by_sd.items()):
            ws = sorted(ws, key=lambda w: w["start"])
            for r in EV.day_teams(ws, ws[0]["L"], draws=a.draws, seed=EV.team_seed(sd)):
                r["seedday"] = sd; rows.append(r)
        t = pl.DataFrame(rows); tabs[det] = t
        R["teams_caught"][det] = {int(N): EV.hboot(t.filter(pl.col("budget") == N).select("seedday", "team", "caught"), B=a.boot) for N in EV.BUDGETS}
    for x, y in (("swarmscore", "convergence"), ("convergence", "pmf"), ("swarmscore", "pmf")):
        if x in tabs and y in tabs:
            f = lambda t: t.filter(pl.col("budget") == 10).select("seedday", "team", "caught")
            R["paired_at_10"]["%s - %s" % (x, y)] = EV.paired(f(tabs[x]), f(tabs[y]), B=a.boot)
    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    pc = lambda r: "%.0f%% [%.0f%%, %.0f%%]" % (100 * r["mean"], 100 * r["lo"], 100 * r["hi"])
    L = ["| Method | Teams caught, N=5 | N=10 | N=20 | N=50 | AUROC | Recall @ 1% FPR |", "|---|---|---|---|---|---|---|"]
    for det, tc in R["teams_caught"].items():
        w = R["window"][det]
        L.append("| %s | %s | %s | %s | %s | %.2f | %.2f |" % (NAMES[det], pc(tc[5]), pc(tc[10]), pc(tc[20]), pc(tc[50]), w["auroc"], w["recall_at_1pct_fpr"]))
    L += ["", "| Paired difference at N=10 | value [95% CI] |", "|---|---|"]
    for k, v in R["paired_at_10"].items():
        L.append("| %s | %+.2f [%+.2f, %+.2f] |" % (k, v["mean"], v["lo"], v["hi"]))
    n = next(iter(R["teams_caught"].values()))[10] if R["teams_caught"] else dict(n_teams=0, n_seeddays=0)
    txt = "Teams caught at N alerts/day per 10k hosts (%d teams, %d seed-days; 95%% hierarchical-bootstrap CI)\n\n" % (n["n_teams"], n["n_seeddays"]) + "\n".join(L) + "\n"
    open(os.path.join(a.out, "RESULTS.md"), "w").write(txt); print(txt)


def cmd_synthetic(a):
    from . import synthetic
    synthetic.write_raw(a.out, n_hosts=a.hosts, days=_range(a.days), seed=a.seed)


def main():
    ap = argparse.ArgumentParser(prog="swarmbench"); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare"); p.add_argument("--auth", required=True); p.add_argument("--flows", required=True); p.add_argument("--days", default="5-9"); p.add_argument("--data", required=True)
    p = sp.add_parser("build"); p.add_argument("--data", required=True); p.add_argument("--bench", required=True); p.add_argument("--days", default="6,8,9")
    p.add_argument("--seeds", default="0-4"); p.add_argument("--size", type=float, default=0.01); p.add_argument("--template", default=None)
    p = sp.add_parser("score"); p.add_argument("--data", required=True); p.add_argument("--bench", required=True); p.add_argument("--scores", required=True)
    p.add_argument("--detectors", default="convergence,first_time,swarmscore,pmf")
    p = sp.add_parser("evaluate"); p.add_argument("--bench", required=True); p.add_argument("--scores", required=True); p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=2000); p.add_argument("--draws", type=int, default=20)
    p = sp.add_parser("synthetic"); p.add_argument("--out", required=True); p.add_argument("--hosts", type=int, default=600); p.add_argument("--days", default="1-2"); p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dict(prepare=cmd_prepare, build=cmd_build, score=cmd_score, evaluate=cmd_evaluate, synthetic=cmd_synthetic)[a.cmd](a)


if __name__ == "__main__":
    main()
