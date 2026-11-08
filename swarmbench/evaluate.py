"""Evaluation of per-instance detector scores.

Window level (pooled over all instances): AUROC and recall at 1% false-positive rate. Population of a window = eligible
hosts plus swarm carriers. Scores are first turned into window-relative p-values, -log10 p, p = (1 + #greater + 0.5 #ties) / (1 + n).

Day level (the operational metric): for one seed-day, the window p-values of each host are merged with Fisher's method
(X = sum -2 ln p over the windows where the host was monitored, referred to chi2 with 2m degrees of freedom). Each day the
analysts get a budget of N alerts per 10,000 monitored hosts (B = max(1, round(N * H / 1e4 * covered_hours / 24))),
released in step with the windows; at each window the top not-yet-alerted hosts fill the quota. A swarm team is caught if at
least one of its machines is alerted by the end of the day. Ties are broken at random, averaged over 20 draws.

Confidence intervals: hierarchical bootstrap (seed-days resampled, then teams inside each), B = 2000; paired differences
between two methods use the same resamples.
"""
import math
import zlib

import numpy as np
import polars as pl
from scipy.stats import chi2, rankdata

BUDGETS = (5, 10, 20, 50)


# ------------------------------------------------------------------------------------------------ window level
def auroc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at_fpr(s, y, fpr=0.01):
    """expected recall when floor(fpr * n_neg) negatives may be flagged, exact over random tie-breaking."""
    y = np.asarray(y, bool); n0 = int((~y).sum()); n1 = int(y.sum())
    if n0 == 0 or n1 == 0:
        return float("nan")
    k = int(math.floor(fpr * n0)); u, inv = np.unique(s, return_inverse=True)
    pos = np.bincount(inv, weights=y.astype(float), minlength=len(u))[::-1]; neg = np.bincount(inv, weights=(~y).astype(float), minlength=len(u))[::-1]
    ci = np.cumsum(neg); before = ci - neg; full = ci <= k; tot = pos[full].sum()
    if (~full).any():
        g = int(np.argmax(~full)); tot += pos[g] * (k - before[g] + 1) / (neg[g] + 1)
    return float(tot / n1)


def calibrate(score):
    s = np.asarray(score, float); n = len(s); srt = np.sort(s)
    hi = np.searchsorted(srt, s, "right"); lo = np.searchsorted(srt, s, "left")
    return -np.log10((1.0 + (n - hi) + 0.5 * (hi - lo - 1)) / (1.0 + n))


def window_metrics(instances):
    """instances: list of dict(scores DataFrame[host, score, elig], carriers set). Pooled AUROC and recall@1% FPR."""
    S, Y = [], []
    for it in instances:
        df = it["scores"]; car = it["carriers"]
        df = df.filter(pl.col("elig") | pl.col("host").is_in(sorted(car)))
        S.append(calibrate(df["score"].to_numpy())); Y.append(np.isin(df["host"].to_numpy(), list(car)))
    s, y = np.concatenate(S), np.concatenate(Y)
    return dict(auroc=auroc(s, y), recall_at_1pct_fpr=recall_at_fpr(s, y, 0.01), n_pos=int(y.sum()), n_neg=int((~y).sum()))


# ------------------------------------------------------------------------------------------------ day level
def _pvals(v, U):
    r = np.sort(v); n = len(r)
    gt = n - np.searchsorted(r, v, "right"); eq = np.searchsorted(r, v, "right") - np.searchsorted(r, v, "left") - 1
    return (1 + gt + U * np.maximum(eq, 0)) / (1 + n)


def rhu(x):
    return int(np.floor(x + 0.5))


def _alert(score, P, A, rng):
    n = score.shape[1]; first = np.full(n, -1); al = np.zeros(n, bool)
    for k in range(score.shape[0]):
        a = A[k + 1] - A[k]
        if a <= 0:
            continue
        cand = np.nonzero(P[k] & ~al)[0]
        if not len(cand):
            continue
        o = np.lexsort((rng.random(len(cand)), -score[k, cand]))[:a]; sel = cand[o]; al[sel] = True; first[sel] = k
    return first


def day_teams(windows, L, budgets=BUDGETS, draws=20, seed=0):
    """windows: one seed-day's instances in time order, each dict(scores, carriers, carrier_team).
    Returns rows (team, budget, caught) with caught = P(team has >= 1 alerted member by day end)."""
    hosts = sorted(set().union(*[set(w["scores"]["host"].to_list()) for w in windows])); ix = {h: i for i, h in enumerate(hosts)}
    K, n = len(windows), len(hosts); S = np.full((K, n), np.nan); P = np.zeros((K, n), bool); team = {}
    for k, w in enumerate(windows):
        f = w["scores"]; j = np.array([ix[h] for h in f["host"].to_list()])
        S[k, j] = f["score"].to_numpy(); P[k, j] = f["elig"].to_numpy() | np.isin(f["host"].to_numpy(), list(w["carriers"]))
        team.update(w["carrier_team"])
    y = np.array([h in team for h in hosts]); tm = np.array([team.get(h, -1) for h in hosts])
    teams = {t: np.nonzero((tm == t) & y)[0] for t in sorted(set(team.values()))}
    H = int(P.any(0).sum()); cov = float(L * K); cum = [float(L * (k + 1)) for k in range(K)]
    res = {(t, N): [] for t in teams for N in budgets}
    for d in range(draws):
        rng = np.random.default_rng([seed, d])
        c = np.full(S.shape, np.nan)
        for k in range(K):
            idx = np.nonzero(P[k])[0]; c[k, idx] = -np.log10(_pvals(S[k, idx], rng.random(len(idx))))
        mon = ~np.isnan(c); X = np.cumsum(2 * np.where(mon, c * np.log(10), 0.0), 0); m = np.cumsum(mon, 0)
        F = np.where(m > 0, -np.log10(np.maximum(chi2.sf(X, 2 * np.maximum(m, 1)), 1e-300)), 0.0)
        for N in budgets:
            B = max(1, rhu(N * H / 1e4 * cov / 24.0)); A = [0] + [rhu(B * cc / cov) for cc in cum]
            first = _alert(F, P, A, rng)
            for t, mem in teams.items():
                res[(t, N)].append(float((first[mem] >= 0).any()))
    return [dict(team=t, budget=N, caught=float(np.mean(v))) for (t, N), v in res.items()]


# ------------------------------------------------------------------------------------------------ bootstrap
def hboot(table, col="caught", B=2000, seed=20261108):
    """table: DataFrame[seedday, team, col]. Mean and 95% CI by hierarchical bootstrap (seed-days, then teams)."""
    rng = np.random.default_rng(seed); sds = table["seedday"].unique().sort().to_list()
    by = {s: table.filter(pl.col("seedday") == s)[col].to_numpy() for s in sds}; res = []
    for _ in range(B):
        vals = [by[sds[i]][rng.integers(0, len(by[sds[i]]), len(by[sds[i]]))] for i in rng.choice(len(sds), len(sds), replace=True)]
        res.append(float(np.concatenate(vals).mean()))
    return dict(mean=float(table[col].mean()), lo=float(np.percentile(res, 2.5)), hi=float(np.percentile(res, 97.5)),
                n_teams=table.height, n_seeddays=len(sds))


def paired(ta, tb, B=2000):
    j = ta.join(tb, on=["seedday", "team"], suffix="_b").with_columns(diff=pl.col("caught") - pl.col("caught_b"))
    return hboot(j, "diff", B)


def team_seed(sd):
    return zlib.crc32(sd.encode())
