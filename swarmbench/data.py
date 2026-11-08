"""Data handling: convert the raw LANL 2015 files to per-day parquet, and load a benchmark instance.

Raw input (from the official LANL page, see README):
    auth.txt.gz   time,source user@domain,destination user@domain,source computer,destination computer,
                  authentication type,logon type,authentication orientation,success/failure
    flows.txt.gz  time,duration,source computer,source port,destination computer,destination port,protocol,packet count,byte count

Per-day parquet (one file per day d, rows with t in [(d-1)*86400, d*86400)):
    t (Int64), kind ("auth" | "flow"), src, dst, dport ("" for auth), account (source user@domain, "" for flow)
"""
import gzip
import io
import json
import os

import polars as pl

DAY = 86400
COLS = ["t", "kind", "src", "dst", "dport", "account"]
AUTH_COLS = ["t", "account", "dst_account", "src", "dst", "auth_type", "logon_type", "orient", "res"]
FLOW_COLS = ["t", "dur", "src", "sport", "dst", "dport", "proto", "pkts", "bytes"]


def day_lo(d):
    return (d - 1) * DAY


def day_file(data_dir, d):
    return os.path.join(data_dir, "day%d.parquet" % d)


def _to_schema(df, kind):
    df = df.with_columns(pl.col("t").cast(pl.Int64))
    if kind == "auth":
        return df.select("t", pl.lit("auth").alias("kind"), pl.col("src").cast(pl.Utf8), pl.col("dst").cast(pl.Utf8),
                         pl.lit("").alias("dport"), pl.col("account").cast(pl.Utf8).fill_null(""))
    return df.select("t", pl.lit("flow").alias("kind"), pl.col("src").cast(pl.Utf8), pl.col("dst").cast(pl.Utf8),
                     pl.col("dport").cast(pl.Utf8).fill_null(""), pl.lit("").alias("account"))


def _stream(path, names, kind, t_hi, chunk_lines=4_000_000):
    """yield schema-converted chunks of a time-sorted gzip CSV until t >= t_hi."""
    with gzip.open(path, "rb") as f:
        while True:
            lines = f.readlines(chunk_lines * 60)
            if not lines:
                return
            df = pl.read_csv(io.BytesIO(b"".join(lines)), has_header=False, new_columns=names,
                             schema_overrides={n: pl.Utf8 for n in names if n != "t"}, infer_schema_length=0)
            df = _to_schema(df.with_columns(pl.col("t").cast(pl.Int64)), kind)
            stop = df["t"].max() >= t_hi
            yield df.filter(pl.col("t") < t_hi)
            if stop:
                return


def prepare(auth_path, flows_path, days, out_dir):
    """convert the raw LANL files into day<d>.parquet for every d in days (the files are read once, up to the last day)."""
    os.makedirs(out_dir, exist_ok=True)
    days = sorted(days); t_lo, t_hi = day_lo(days[0]), day_lo(days[-1] + 1)
    tmp = os.path.join(out_dir, "_parts"); os.makedirs(tmp, exist_ok=True)
    parts = {d: [] for d in days}   # chunks go to disk so memory stays at one chunk plus one day
    for path, names, kind in ((auth_path, AUTH_COLS, "auth"), (flows_path, FLOW_COLS, "flow")):
        for i, df in enumerate(_stream(path, names, kind, t_hi)):
            df = df.filter(pl.col("t") >= t_lo)
            for d in days:
                p = df.filter((pl.col("t") >= day_lo(d)) & (pl.col("t") < day_lo(d + 1)))
                if p.height:
                    f = os.path.join(tmp, "%s_d%d_%05d.parquet" % (kind, d, i)); p.write_parquet(f); parts[d].append(f)
        print("read", kind, flush=True)
    for d in days:
        df = pl.read_parquet(parts[d]).sort(["t", "kind", "src", "dst", "dport", "account"]) if parts[d] else pl.DataFrame(schema={c: pl.Utf8 for c in COLS})
        df.write_parquet(day_file(out_dir, d))
        for f in parts[d]:
            os.remove(f)
        print("day", d, df.height, "rows", flush=True)
    os.rmdir(tmp)


def load_day(data_dir, d, dedup=False):
    df = pl.read_parquet(day_file(data_dir, d))
    if dedup:  # detectors only use distinct (src, destination, port, account) facts of the reference period
        df = df.group_by([c for c in COLS if c != "t"]).agg(pl.col("t").max()).select(COLS).sort(COLS)
    return df


# ---------------------------------------------------------------------------------------------- instances
def eligible_hosts(win, hist):
    """monitored population: hosts with >= 1 row in the window, excluding busy servers (destination fan-in >= the 90th
    percentile of destination fan-in) and the top 1% of hosts by fan-out (scanners, infrastructure)."""
    rows = pl.concat([hist.select("src", "dst"), win.select("src", "dst")]).filter((pl.col("src") != "") & (pl.col("dst") != "") & (pl.col("src") != pl.col("dst"))).unique()
    fin = rows.group_by("dst").agg(pl.len().alias("fin")); fout = rows.group_by("src").agg(pl.len().alias("fout"))
    p90 = float(fin["fin"].quantile(0.9)) if fin.height else 0.0
    top = float(fout["fout"].quantile(0.99)) if fout.height else 0.0
    servers = set(fin.filter(pl.col("fin") >= p90)["dst"].to_list())
    scanners = set(fout.filter(pl.col("fout") > top)["src"].to_list())
    active = win["src"].unique().to_list()
    return sorted(h for h in active if h and h not in servers and h not in scanners)


def instance_ids(bench_dir):
    """[(seedday_dir, window_tag)] for every built instance."""
    out = []
    for sd in sorted(os.listdir(bench_dir)):
        p = os.path.join(bench_dir, sd, "meta.json")
        if os.path.exists(p):
            for w in json.load(open(p))["windows"]:
                out.append((sd, w))
    return out


def load_instance(data_dir, bench_dir, seedday, wtag, _cache={}):
    """-> (history, window, eligible hosts, labels). history = the raw day before the test day (a pre-swarm reference)."""
    meta = json.load(open(os.path.join(bench_dir, seedday, "meta.json")))
    d = meta["day"]; w = meta["windows"][wtag]
    key = (data_dir, d)
    if _cache.get("key") != key:
        _cache.clear(); _cache.update(key=key, hist=load_day(data_dir, d - 1, dedup=True), test=load_day(data_dir, d))
    test, hist = _cache["test"], _cache["hist"]
    inj = pl.read_parquet(os.path.join(bench_dir, seedday, "injected.parquet"))
    lo, hi = w["lo"], w["hi"]
    win = pl.concat([test.filter((pl.col("t") >= lo) & (pl.col("t") < hi)), inj.filter((pl.col("t") >= lo) & (pl.col("t") < hi)).select(COLS)]).sort(["t", "src"])
    lab = json.load(open(os.path.join(bench_dir, seedday, "labels", wtag + ".json")))
    return hist, win, eligible_hosts(win, hist), lab
