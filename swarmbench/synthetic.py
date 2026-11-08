"""Small synthetic logs in the raw LANL 2015 format, for the smoke test only (no real data involved).

Hosts C1..Cn: the first 5% are servers, the rest workstations. Each workstation has its own user account, a few habitual
servers and peers, and a diurnal activity profile; 5% of its flows go to random (destination, high port) pairs, which
creates the long tail of rare services the benchmark builder needs. Numbers are illustrative, not calibrated to LANL.
"""
import gzip
import os

import numpy as np

DAY = 86400
PORTS = ["445", "80", "443", "88", "389", "53", "135", "139"]


def _times(rng, n, day):
    hours = np.arange(24); w = 0.3 + np.exp(-0.5 * ((hours - 13) / 4.0) ** 2); w = w / w.sum()
    h = rng.choice(24, n, p=w)
    return (day - 1) * DAY + h * 3600 + rng.integers(0, 3600, n)


def write_raw(out_dir, n_hosts=600, days=(1, 2), seed=0):
    rng = np.random.default_rng(seed); os.makedirs(out_dir, exist_ok=True)
    n_srv = max(5, n_hosts // 20); hosts = ["C%d" % i for i in range(1, n_hosts + 1)]
    zipf = 1.0 / np.arange(1, n_srv + 1) ** 1.1; zipf /= zipf.sum()
    habit = {}
    for i in range(n_srv, n_hosts):
        srv = rng.choice(n_srv, 3, replace=False, p=zipf).tolist(); peers = rng.choice(np.arange(n_srv, n_hosts), 2, replace=False).tolist()
        habit[i] = (srv, peers, rng.choice(PORTS, 2, replace=False).tolist())
    auth, flow = [], []
    for d in days:
        for i in range(n_srv, n_hosts):
            srv, peers, ports = habit[i]; h = hosts[i]
            nf = int(rng.poisson(120)); na = int(rng.poisson(40))
            tf = _times(rng, nf, d); ta = _times(rng, na, d)
            for t in tf:
                u = rng.random()
                if u < 0.05:
                    dst = hosts[int(rng.integers(0, n_hosts))]; port = str(int(rng.integers(1024, 65535)))
                elif u < 0.80:
                    dst = hosts[srv[int(rng.integers(0, 3))]]; port = ports[int(rng.integers(0, 2))]
                else:
                    dst = hosts[peers[int(rng.integers(0, 2))]]; port = "445"
                flow.append((int(t), int(rng.integers(0, 60)), h, "N%d" % int(rng.integers(1, 9999)), dst, port, "6", int(rng.integers(1, 50)), int(rng.integers(60, 5000))))
            for t in ta:
                acct = ("U%d@DOM1" % i) if rng.random() < 0.85 else ("%s$@DOM1" % h)
                dst = hosts[srv[int(rng.integers(0, 3))]] if rng.random() < 0.9 else h
                lt = "Network" if dst != h else "Interactive"
                auth.append((int(t), acct, acct, h, dst, "Kerberos", lt, "LogOn", "Success"))
        for j in range(n_srv):   # servers: an admin account touching many hosts
            for t in _times(rng, 300, d):
                dst = hosts[int(rng.integers(n_srv, n_hosts))]
                auth.append((int(t), "U0@DOM1", "U0@DOM1", hosts[j], dst, "NTLM", "Network", "LogOn", "Success"))
    auth.sort(key=lambda r: r[0]); flow.sort(key=lambda r: r[0])
    for name, rows in (("auth.txt.gz", auth), ("flows.txt.gz", flow)):
        with gzip.open(os.path.join(out_dir, name), "wt") as f:
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
    print("synthetic logs:", len(auth), "auth rows,", len(flow), "flow rows ->", out_dir)
