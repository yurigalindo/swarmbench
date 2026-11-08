"""SwarmBench builder: plant a measured AI-agent swarm into one real test day of LANL logs.

One swarm is planted per (test day, seed). Its shape is drawn from swarm_template.json (aggregate statistics only):

  1. teams     - about 1% of the day's active machines become swarm carriers, split into 2-4 teams (Dirichlet sizes, >= 3 each);
                 carriers are real machines picked across activity deciles.
  2. targets   - each agent gets destination and credential "tokens" with the measured per-agent counts; tokens are shared
                 inside a team with the measured fan-in (a Chinese-restaurant process whose concentration is set so that about 87% of agent-target slots fall on shared targets; measured: 86% and 90%).
  3. mapping   - each token maps onto an existing LANL service (destination, port) or user account: half to popular items,
                 half to rare ones (<= 2 distinct sources), the measured isolation of 0.49. No new identifiers are created.
  4. channel   - one coordination channel per day: a rare port on a popular destination; each carrier joins it with p = 0.6.
  5. growth    - join times follow the measured growth curve (11 / 76 / 210 agents by +1 / +6 / +24 h, scaled), assigned
                 to carriers in order of their first activity; activity spans follow the measured persistence distribution.
  6. traffic   - while active, a carrier adds swarm events equal to `share` (0.6) of its own events: flows to its mapped
                 services and the channel, and logins with its mapped credentials (real accounts, often ones it has never used).
                 Event times follow the measured team waves and agent sessions.

The swarm is added on top of the carrier's real traffic. The reference period (the day before) is left untouched,
so it is a clean, pre-swarm reference. Outputs per seed-day: injected.parquet (swarm rows only), labels/<window>.json, meta.json.
"""
import json
import os

import numpy as np
import polars as pl

from . import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = dict(size=0.01, share=0.6, iso=0.5, chan_part=0.6, window_h=2, window_starts=(8, 12, 16, 20), block_s=7200)


def load_template(path=None):
    return json.load(open(path or os.path.join(HERE, "swarm_template.json")))


def _qinv(grid, u):
    return float(np.interp(u, grid["p"], grid.get("x", grid.get("hours"))))


def _count_grid(stats):
    """quantile grid for a per-agent count from (frac_zero, p25, p50, p75, p90, max)."""
    p = [0.0, stats["frac_zero"], 0.25, 0.5, 0.75, 0.9, 1.0]; x = [0, 0, stats["p25"], stats["p50"], stats["p75"], stats["p90"], stats["max"]]
    keep = [0] + [i for i in range(1, len(p)) if p[i] > p[i - 1]]
    return dict(p=[p[i] for i in keep], x=[x[i] for i in keep])


# ------------------------------------------------------------------------------------------------ swarm shape
def pick_teams(rng, n_agents):
    for _ in range(10000):
        k = int(rng.integers(2, 5)); w = rng.dirichlet(np.ones(k) * 2.0)
        sz = np.maximum(3, np.round(w * n_agents).astype(int)); sz[-1] += n_agents - sz.sum()
        if sz.min() >= 3:
            return sz.tolist()
    raise RuntimeError("cannot split %d agents into teams" % n_agents)


def team_tokens(rng, sizes, tpl, kind, alpha):
    """per agent a list of team-local token ids for one kind ('destination' | 'credential')."""
    g = _count_grid(tpl["tokens_per_agent"][kind]); out = []
    for ti, n in enumerate(sizes):
        counts = {}   # token -> agents using it (Chinese restaurant process inside the team)
        for _ in range(n):
            k = int(round(_qinv(g, rng.random()))); mine = set()
            for _ in range(k):
                tot = sum(counts.values())
                if counts and rng.random() >= alpha / (tot + alpha):
                    toks = [t for t in counts if t not in mine]
                    if toks:
                        w = np.array([counts[t] for t in toks], float); t = toks[int(rng.choice(len(toks), p=w / w.sum()))]
                    else:
                        t = (ti, len(counts))
                else:
                    t = (ti, len(counts))
                mine.add(t); counts[t] = counts.get(t, 0) + 1
            out.append(sorted(mine))
    return out


def realised_sharing(tokens):
    from collections import Counter
    c = Counter(t for a in tokens for t in a)
    slots = sum(c.values())
    return dict(n_tokens=len(c), frac_shared=float(np.mean([v >= 2 for v in c.values()])) if c else 0.0,
                slot_share_on_shared=float(sum(v for v in c.values() if v >= 2) / slots) if slots else 0.0)


# ------------------------------------------------------------------------------------------------ timing
def _lnmix(rng, m):
    while True:
        j = int(rng.random() >= m["weights"][0]); g = float(np.exp(m["mu_log"][j] + m["sigma_log"][j] * rng.standard_normal()))
        if g < m["max_s"]:
            return int(round(g))


def _session_size(rng, pmf):
    u = rng.random(); c = 0.0
    for k in range(1, 11):
        c += pmf[str(k)]
        if u < c:
            return k
    return int(rng.integers(11, 30))


def team_waves(rng, T, W):
    mean_bw = float(np.mean(T["big_waves_per_active_day"]["x"]))
    n = 1 + int(rng.poisson(mean_bw * W / 86400.0)); out = []
    for _ in range(n):
        on = int(rng.integers(0, W)); du = int(round(_qinv(T["big_wave_duration_s"], rng.random())))
        out.append((on, min(W - 1, on + max(1, du))))
    return sorted(out)


def carrier_times(rng, T, waves, n, W):
    """n second offsets in [0, W): the carrier joins each team wave after a follow delay and acts in sessions."""
    pool = []
    for _ in range(60):
        for on, end in waves:
            t = on + int(round(_qinv(T["follow_delay_s"], rng.random())))
            while t <= end:
                for _k in range(_session_size(rng, T["events_per_session_pmf"])):
                    if t > end:
                        break
                    pool.append(t); t += _lnmix(rng, T["within_session_gap_lognormal_mix"])
                t += int(round(_qinv(T["between_session_gap_s"], rng.random())))
        if len(pool) >= n:
            break
    if len(pool) < n:
        pool += [int(rng.integers(0, W)) for _ in range(n - len(pool))]
    pick = rng.permutation(len(pool))[:n]
    return [int(min(W - 1, max(0, pool[i]))) for i in pick]


# ------------------------------------------------------------------------------------------------ item pools
def item_pools(hist, test, carriers):
    """popular and rare services / accounts of the test day; channel candidates."""
    both = pl.concat([hist, test])
    e = both.filter((pl.col("src") != "") & (pl.col("dst") != "") & (pl.col("src") != pl.col("dst")))
    dom = e.select("src", "dst").unique().group_by("dst").agg(pl.len().alias("fin"))
    fl = test.filter((pl.col("kind") == "flow") & (pl.col("dst") != "") & (pl.col("src") != pl.col("dst")) & ~pl.col("dst").is_in(sorted(carriers)))
    sv = fl.select("src", "dst", "dport").unique().group_by(["dst", "dport"]).agg(pl.len().alias("nsrc")).join(dom, on="dst", how="left").fill_null(0)
    sv2 = (both.filter(pl.col("kind") == "flow").select("src", "dst", "dport").unique().group_by(["dst", "dport"]).agg(pl.len().alias("nsrc2")))
    sv = sv.join(sv2, on=["dst", "dport"], how="left").sort(["nsrc", "dst", "dport"], descending=[True, False, False])
    svc_pop = list(zip(sv["dst"].to_list(), sv["dport"].to_list()))
    r = sv.filter(pl.col("fin") <= 2)
    svc_rare = list(zip(r["dst"].to_list(), r["dport"].to_list()))
    thr = float(np.percentile(dom["fin"].to_numpy(), 95))
    ch = sv.filter((pl.col("fin") >= thr) & (pl.col("nsrc2") <= 2)).sort(["dst", "dport"])
    chan = list(zip(ch["dst"].to_list(), ch["dport"].to_list()))
    au = test.filter((pl.col("kind") == "auth") & pl.col("account").str.starts_with("U") & (pl.col("dst") != ""))
    ac = au.select("src", "account").unique().group_by("account").agg(pl.len().alias("nsrc")).sort(["nsrc", "account"], descending=[True, False])
    acc_pop = ac["account"].to_list(); acc_rare = ac.filter(pl.col("nsrc") <= 2)["account"].to_list()
    # where each account logs in (most frequent destination on the test day): planted logins reuse it
    ad = au.group_by(["account", "dst"]).agg(pl.len().alias("n")).sort(["account", "n", "dst"], descending=[False, True, False]).group_by("account", maintain_order=True).first()
    acc_dst = dict(zip(ad["account"].to_list(), ad["dst"].to_list()))
    return dict(svc_pop=svc_pop, svc_rare=svc_rare, acc_pop=acc_pop, acc_rare=acc_rare, acc_dst=acc_dst, chan=chan)


def map_tokens(rng, tokens, pop, rare, iso):
    """token -> real item: rare with probability iso, otherwise popular by usage rank (most-used tokens -> most popular items)."""
    from collections import Counter
    c = Counter(t for a in tokens for t in a)
    order = sorted(c, key=lambda t: (-c[t], rng.random()))
    m = {}; npop = 0; rare_left = list(rng.permutation(len(rare))) if rare else []
    for t in order:
        if rng.random() < iso and rare_left:
            m[t] = rare[int(rare_left.pop())]
        else:
            m[t] = pop[min(npop, len(pop) - 1)]; npop += 1
    return m


# ------------------------------------------------------------------------------------------------ build
def build_seed_day(data_dir, out_dir, day, seed, tpl=None, **dials):
    st = dict(DEFAULT, **dials); tpl = tpl or load_template(); T = tpl["timing"]
    D0 = D.day_lo(day); D1 = D0 + D.DAY; H0 = D0 - D.DAY
    hist = D.load_day(data_dir, day - 1); test = D.load_day(data_dir, day)
    # carriers: day-level eligible hosts, stratified by activity decile
    elig = D.eligible_hosts(test, hist)
    act = test.filter(pl.col("src").is_in(elig)).group_by("src").agg(pl.len().alias("n")).sort(["n", "src"])
    el = act["src"].to_list(); n_active = test["src"].n_unique()
    rT = np.random.default_rng([7400, day, seed])
    n_agents = max(6, int(round(st["size"] * n_active))); sizes = pick_teams(rT, n_agents)
    nb = 10; dec = [el[i * len(el) // nb:(i + 1) * len(el) // nb] for i in range(nb)]
    used, carriers = set(), []
    for dd in rT.permutation(np.tile(np.arange(nb), n_agents // nb + 1))[:n_agents]:
        cand = [h for h in dec[int(dd)] if h not in used] or [h for h in el if h not in used]
        h = cand[int(rT.integers(len(cand)))]; used.add(h); carriers.append(h)
    team_of = [ti for ti, n in enumerate(sizes) for _ in range(n)]
    # tokens and their mapping onto real items
    rK = np.random.default_rng([7401, day, seed])
    dtok = team_tokens(rK, sizes, tpl, "destination", alpha=20.0); ptok = team_tokens(rK, sizes, tpl, "credential", alpha=12.0)
    P = item_pools(hist, test, set(carriers))
    smap = map_tokens(rK, dtok, P["svc_pop"], P["svc_rare"], st["iso"])
    amap = map_tokens(rK, ptok, [a for a in P["acc_pop"] if a in P["acc_dst"]], [a for a in P["acc_rare"] if a in P["acc_dst"]], st["iso"])
    rC = np.random.default_rng([7402, day, seed])
    chan = P["chan"][int(rC.integers(len(P["chan"])))] if P["chan"] else None
    part = (rC.random(n_agents) < st["chan_part"]).tolist() if chan else [False] * n_agents
    # growth and persistence
    rG = np.random.default_rng([7404, day, seed]); g = tpl["growth"]["curve"]; s = tpl["persistence"]["span_quantiles"]
    join = sorted(D0 + int(round(3600 * float(np.interp(u, g["share_joined"], g["hours"])))) for u in rG.random(n_agents))
    first = dict(test.filter(pl.col("src").is_in(carriers)).group_by("src").agg(pl.col("t").min()).iter_rows())
    order = sorted(range(n_agents), key=lambda i: (first.get(carriers[i], D1), carriers[i]))
    jn = [0] * n_agents
    for r, i in enumerate(order):
        jn[i] = join[r]
    span_h = [float(np.interp(u, s["p"], s["hours"])) for u in rG.random(n_agents)]
    lv = [min(D1, j + int(round(3600 * sp))) for j, sp in zip(jn, span_h)]
    # swarm traffic
    rng = np.random.default_rng([7405, day, seed]); rows = []; wave_cache = {}; stats = []
    CR = test.filter(pl.col("src").is_in(carriers)).select("src", "t")
    own_t = {h: np.sort(g["t"].to_numpy()) for (h,), g in CR.group_by(["src"])}
    for ai, h in enumerate(carriers):
        ti = team_of[ai]; t = own_t.get(h, np.array([], np.int64)); m = (t >= jn[ai]) & (t < lv[ai]); nd = int(m.sum())
        toks = [("d", x) for x in dtok[ai]] + [("p", x) for x in ptok[ai]] + ([("C", None)] if part[ai] else [])
        n_inj = int(round(st["share"] * nd)) if nd and toks else 0
        if n_inj:
            n_inj = max(n_inj, len(toks))
        seq = [toks[i] for i in rng.permutation(len(toks))]; seq = [seq[i % len(seq)] for i in range(n_inj)]
        times = np.sort(rng.choice(t[m], n_inj, replace=True)) if n_inj else np.array([], np.int64)
        # re-time inside the team's waves, block by block
        byb = {}
        for k, x in enumerate(times):
            byb.setdefault(int((x - H0) // st["block_s"]), []).append(k)
        for b, ks in byb.items():
            blo = H0 + b * st["block_s"]; a_, z_ = max(jn[ai], blo) - blo, min(lv[ai], blo + st["block_s"]) - blo
            wv = wave_cache.setdefault((ti, blo), team_waves(np.random.default_rng([7100, seed * 1000 + day, blo, ti]), T, st["block_s"]))
            wc = [(max(on, a_), min(en, z_ - 1)) for on, en in wv if min(en, z_ - 1) > max(on, a_)]
            if not wc:
                continue
            tt = carrier_times(np.random.default_rng([7101, seed * 1000 + day, blo, ti, ai]), T, wc, len(ks), st["block_s"])
            for k, x in zip(ks, tt):
                times[k] = blo + int(x)
        for (kind, tok), x in zip(seq, times):
            if kind == "p":
                a = amap[tok]; rows.append((int(x), "auth", h, P["acc_dst"][a], "", a))
            else:
                dst, dport = chan if kind == "C" else smap[tok]; rows.append((int(x), "flow", h, dst, dport, ""))
        stats.append(dict(host=h, team=ti, join=jn[ai], leave=lv[ai], own_events=nd, swarm_events=n_inj, channel=bool(part[ai])))
    inj = pl.DataFrame(rows, schema={"t": pl.Int64, "kind": pl.Utf8, "src": pl.Utf8, "dst": pl.Utf8, "dport": pl.Utf8, "account": pl.Utf8}, orient="row").sort(["t", "src"])
    # write
    sd = os.path.join(out_dir, "d%d_s%02d" % (day, seed)); os.makedirs(os.path.join(sd, "labels"), exist_ok=True)
    inj.write_parquet(os.path.join(sd, "injected.parquet"))
    wins = {}
    for s0 in st["window_starts"]:
        lo, hi = D0 + 3600 * s0, D0 + 3600 * (s0 + st["window_h"]); tag = "w%02dL%02d" % (s0, st["window_h"])
        wins[tag] = dict(lo=lo, hi=hi, start_hour=s0, L=st["window_h"])
        car = sorted(set(inj.filter((pl.col("t") >= lo) & (pl.col("t") < hi))["src"].to_list()))
        json.dump(dict(carrier_hosts=car, carrier_team={h: team_of[carriers.index(h)] for h in car}), open(os.path.join(sd, "labels", tag + ".json"), "w"))
    json.dump(dict(day=day, seed=seed, dials=st, team_sizes=sizes, n_active=n_active, n_agents=n_agents, windows=wins,
                   sharing=dict(destination=realised_sharing(dtok), credential=realised_sharing(ptok)), channel_found=chan is not None,
                   carriers=stats), open(os.path.join(sd, "meta.json"), "w"), indent=1)
    return sd
