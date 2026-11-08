"""Detectors. Each one maps (history, window) event tables to a score per window host (higher = more suspicious).

Event table schema (polars DataFrame), one row per login or network-flow record:
    t (int, seconds), kind ("auth" | "flow"), src, dst, dport (str, "" for auth), account (str, "" for flow)

Detectors read only these columns. They never see labels.

    convergence  - convergence rule: for every destination, (distinct sources now + 1) / (distinct sources in the reference + 1);
                   a host scores the largest ratio among the destinations it contacts.
    first_time   - first-time account rule: number of distinct accounts a host uses now that it never used in the reference.
    swarmscore   - SwarmScore: equal-weight average of the two rules' log-rank scores, log10(N / rank).
                   (Evidence over a day's windows is merged with Fisher's method in evaluate.py.)
    pmf          - per-machine Poisson matrix factorisation baseline (Sanna Passino, Turcotte & Heard, 2022): surprise
                   -ln(1 - exp(-lambda)) of each edge that is new relative to the reference, max per host.
"""
import zlib

import numpy as np
import polars as pl
from scipy import sparse
from scipy.stats import rankdata

COLS = ["t", "kind", "src", "dst", "dport", "account"]


def window_hosts(win):
    return sorted(win["src"].unique().to_list())


def _as_dict(scores, hosts):
    return {h: float(scores.get(h, 0.0)) for h in hosts}


# --------------------------------------------------------------------------------------------- convergence rule
def _destinations(df):
    """distinct (src, destination) pairs from flow and auth rows; empty and self edges dropped."""
    d = df.filter(pl.col("kind").is_in(["flow", "auth"]) & (pl.col("src") != "") & (pl.col("dst") != "") & (pl.col("src") != pl.col("dst")))
    return d.select("src", "dst").unique()


def convergence(hist, win):
    hosts = window_hosts(win)
    H, W = _destinations(hist), _destinations(win)
    nH = H.group_by("dst").agg(pl.len().alias("nH"))
    nW = W.group_by("dst").agg(pl.len().alias("nW"))
    j = (W.join(nW, on="dst").join(nH, on="dst", how="left").with_columns(pl.col("nH").fill_null(0))
         .with_columns(r=(pl.col("nW") + 1) / (pl.col("nH") + 1)))
    m = j.group_by("src").agg(pl.col("r").max())
    return _as_dict(dict(zip(m["src"].to_list(), m["r"].to_list())), hosts)


# --------------------------------------------------------------------------------------------- first-time accounts
def _accounts(df):
    return (df.filter((pl.col("kind") == "auth") & (pl.col("account") != ""))
            .select(pl.col("src").cast(pl.Utf8), pl.col("account").cast(pl.Utf8)).unique())


def first_time(hist, win):
    hosts = window_hosts(win)
    new = _accounts(win).join(_accounts(hist), on=["src", "account"], how="anti")
    g = new.group_by("src").len()
    cnt = dict(zip(g["src"].to_list(), g["len"].to_list()))
    # deterministic tie-break far below one count
    return {h: float(cnt.get(h, 0)) + (zlib.crc32(str(h).encode()) / 2 ** 32) * 1e-6 for h in hosts}


# --------------------------------------------------------------------------------------------- SwarmScore
def swarmscore_from(conv, ft, weight=0.5):
    """fuse two per-host score dicts by log-rank: S = w log10(N / rank_conv) + (1 - w) log10(N / rank_ft)."""
    hosts = sorted(conv)
    a = np.array([conv[h] for h in hosts]); b = np.array([ft.get(h, 0.0) for h in hosts]); N = len(hosts)
    ra = rankdata(-a, method="average"); rb = rankdata(-b, method="average")
    s = weight * np.log10(N / ra) + (1 - weight) * np.log10(N / rb)
    return dict(zip(hosts, s.tolist()))


def swarmscore(hist, win):
    return swarmscore_from(convergence(hist, win), first_time(hist, win))


# --------------------------------------------------------------------------------------------- PMF baseline
def _pmf_fit(rows, cols, n_r, n_c, R, iters, seed):
    """KL-NMF (Poisson factorisation) by multiplicative updates on a binary edge list."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.1, 1.0, (n_r, R)); Hm = rng.uniform(0.1, 1.0, (n_c, R))
    for _ in range(iters):
        for A_r, A_c, X, Y in ((rows, cols, W, Hm), (cols, rows, Hm, W)):
            lam = np.einsum("ij,ij->i", X[A_r], Y[A_c]) + 1e-12
            M = sparse.csr_matrix((1.0 / lam, (A_r, A_c)), shape=(X.shape[0], Y.shape[0]))
            X *= (M @ Y) / (Y.sum(0)[None, :] + 1e-12)
    return W, Hm


def _pmf_edges(df, which):
    if which == "dst":   # login graph: source computer x destination computer
        d = df.filter((pl.col("kind") == "auth") & (pl.col("src") != "") & (pl.col("dst") != "")).select("src", pl.col("dst").alias("col"))
    else:                # service graph: source x (destination, port) from flows
        d = df.filter(pl.col("kind") == "flow").select("src", (pl.lit("f:") + pl.col("dst") + pl.lit("|") + pl.col("dport")).alias("col"))
    return d.unique(maintain_order=True)


def _pmf_surprises(hist, win, which, R, iters, seed):
    H = _pmf_edges(hist, which); W = _pmf_edges(win, which)
    if H.height == 0 or W.height == 0:
        return [], []
    if which == "dst":   # shared node space for sources and destinations
        nodes = pl.concat([H["src"], H["col"], W["src"], W["col"]]).unique(maintain_order=True).to_list()
        rmap = cmap = {n: i for i, n in enumerate(nodes)}; nr = nc = len(nodes)
    else:
        rn = pl.concat([H["src"], W["src"]]).unique(maintain_order=True).to_list()
        cn = pl.concat([H["col"], W["col"]]).unique(maintain_order=True).to_list()
        rmap = {n: i for i, n in enumerate(rn)}; cmap = {n: i for i, n in enumerate(cn)}; nr, nc = len(rn), len(cn)
    hr = np.array([rmap[x] for x in H["src"].to_list()]); hc = np.array([cmap[x] for x in H["col"].to_list()])
    seen = set(zip(hr.tolist(), hc.tolist()))
    U, V = _pmf_fit(hr, hc, nr, nc, R, iters, seed)
    keep = [(a, b) for a, b in zip(W["src"].to_list(), W["col"].to_list()) if (rmap[a], cmap[b]) not in seen]
    if not keep:
        return [], []
    a = np.array([rmap[x] for x, _ in keep]); b = np.array([cmap[y] for _, y in keep])
    Ua = U[a]
    hs = np.unique(hr); cold = ~np.isin(a, hs)
    if cold.any():       # cold-start sources get the mean factor row of the reference sources
        Ua = Ua.copy(); Ua[cold] = U[hs].mean(0)
    lam = np.maximum(np.einsum("ij,ij->i", Ua, V[b]), 1e-9)
    sur = -np.log(np.maximum(-np.expm1(-lam), 1e-9))
    return [s for s, _ in keep], sur.tolist()


def pmf(hist, win, R=10, iters=60, seed=0, matrices=("dst", "svc")):
    """final setting: rank 10, login + service matrices, max edge surprise per host, cold-start fallback = mean."""
    hosts = window_hosts(win); per = {}
    for m in matrices:
        srcs, sur = _pmf_surprises(hist, win, m, R, iters, seed)
        for s, v in zip(srcs, sur):
            per[s] = max(per.get(s, -np.inf), v)
    return _as_dict(per, hosts)


DETECTORS = {"convergence": convergence, "first_time": first_time, "swarmscore": swarmscore, "pmf": pmf}


def score_all(hist, win, names=("convergence", "first_time", "swarmscore", "pmf")):
    """score one window with several detectors; SwarmScore reuses the two rules' scores."""
    out = {}
    if "convergence" in names or "swarmscore" in names:
        out["convergence"] = convergence(hist, win)
    if "first_time" in names or "swarmscore" in names:
        out["first_time"] = first_time(hist, win)
    if "swarmscore" in names:
        out["swarmscore"] = swarmscore_from(out["convergence"], out["first_time"])
    if "pmf" in names:
        out["pmf"] = pmf(hist, win)
    return {k: v for k, v in out.items() if k in names}
