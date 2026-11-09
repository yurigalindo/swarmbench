# SwarmBench

**Catching coordinated AI-agent swarms in the logs an organisation already keeps.**

Yuri Galindo (AE Studio) and Caíque Lira (Independent)

Video: [3-minute overview](https://youtu.be/V3UrhTlkuO8)

**Prior work:** none. All code, data processing and results in this repository were produced during the Apart Swarm Alignment sprint (6–8 November 2026); nothing here is preexisting.

In July 2026 about 1,200 AI agents that were meant to be isolated turned a shared package server into a secret message board, and about 700 of them went on to attack an external service. Monitoring what each agent says and does is necessary but can be evaded. Every lab already keeps *metadata*, though: network-flow and login logs (the kind firewalls, proxies and identity systems already record) of which machine connected to which destination, when, and with which account. In the July incident the coordination channel was inside the lab, so internal flow logs matter, not only the edge firewall. SwarmBench asks whether a swarm can be caught from that metadata alone, as an extra security layer that reads no content. It provides a benchmark with a known answer (real enterprise logs with a planted, measured swarm) and a simple detector that catches significantly more swarm teams than a simplified version of the standard per-machine anomaly model (PMF).

## Contributions

1. **The benchmark.** Swarm behaviour measured from the released data and reports of 2026 incidents (Swarm Traces, the METR investigation, collusion.wiki: team sizes, shared targets, isolation, growth, persistence, timing; the swarm's traffic share is assumed) is planted into real enterprise logs from Los Alamos National Laboratory (LANL 2015, about 9,600 monitored machines per 2-hour window). Because the swarm is planted, every swarm machine is labelled.
2. **The detector.** A *convergence rule* flags destinations that many machines suddenly start reaching together. *SwarmScore* adds a second signal, machines using accounts they have never used before, and merges each machine's evidence over the day.

## Results

Main test: 44 swarm teams on 15 seed-days (days 6, 8, 9 of LANL 2015 × 5 random seeds). SwarmScore's weights were fixed on day 7, which is excluded; the convergence rule has no tuned parameters but was chosen as the method after its results on 17 of the 44 teams were seen. On the 18 teams never seen before that choice, convergence catches about 28% and SwarmScore about 22% (rough, by subtraction; no CI). The test was extended twice (17 → 26 → 44 teams) after SwarmScore − convergence narrowly missed significance; the final gap is +0.07, not significant. The 44-team PMF comparison was added late and is descriptive; a random alert list catches about 1–4% of teams. "Teams caught" means at least one machine of the team is among the day's top N alerts; N is per 10,000 monitored machines per day (pro-rated to the scored hours; ties averaged over 20 random draws), and we assume 10 alerts/day is roughly what one analyst can check. Brackets are 95% hierarchical-bootstrap confidence intervals (CIs). The 26-team subset (seeds 0-2) is shown for comparison with earlier runs.

| Method | Teams caught, 10 alerts/day (44 teams) | Same, 26-team subset |
|---|---|---|
| Convergence rule | **25%** [11%, 40%] | 22% [7%, 41%] |
| SwarmScore (convergence + first-time accounts) | **32%** [16%, 51%] | 38% [18%, 62%] |
| Per-machine PMF (simplified, adapted from Sanna Passino et al. 2022) | 0.8% [0.2%, 1.5%] | 0.6% [0%, 1.5%] |
| First-time accounts alone | 0% | 0% |
| *Paired difference:* convergence − PMF | **+0.24** [+0.11, +0.39], significant | **+0.22** [+0.08, +0.40], significant |
| *Paired difference:* SwarmScore − PMF | **+0.31** [+0.15, +0.48], significant | **+0.38** [+0.17, +0.61], significant |
| *Paired difference:* SwarmScore − convergence | +0.07 [−0.11, +0.25], not significant | |

| Alert budget (44 teams) | 5/day | 10/day | 20/day | 50/day |
|---|---|---|---|---|
| Convergence rule | 9% [2%, 18%] | 25% [12%, 40%] | 41% [26%, 57%] | 66% [48%, 83%] |
| SwarmScore | 18% [5%, 34%] | 32% [15%, 50%] | 41% [23%, 61%] | 52% [33%, 73%] |
| PMF (per-machine) | 0.5% [0.0%, 1.2%] | 0.8% [0.2%, 1.5%] | 2.2% [0.9%, 3.7%] | 5.8% [3.7%, 8.0%] |

| Window-level ranking, days 6 / 8 / 9 (seeds 3–4) | AUROC | Recall at 1% FPR |
|---|---|---|
| Convergence rule | 0.74 / 0.77 / 0.89 | 0.26 / 0.45 / 0.69 |
| SwarmScore | 0.80 / 0.94 / 0.97 | 0.21 / 0.43 / 0.67 |

**The reference period matters.** The detector compares the present with a reference period from *before* the swarm. With a rolling 24-hour reference that already contains the swarm, the convergence rule's pooled recall at 1% FPR falls from 42% to 6.5% (22 teams, same swarm; days 6–8, where day 7 was the tuning day).

## How it works

- **Data:** login records (source machine, destination, account, time) and network-flow records (source, destination, port, time). No content, no model internals.
- **Benchmark:** about 1% of a test day's machines become swarm carriers in 2-4 teams. Their extra traffic follows the measured statistics: shared destinations and credentials inside a team, a coordination channel (a rare port on a popular server), half of the targets popular and half rare, measured growth, activity spans and wave timing. Only existing destinations and accounts are used.
- **Convergence rule:** for each destination, (distinct machines reaching it now + 1) / (distinct machines in the reference + 1); a machine scores the largest ratio among the destinations it contacts.
- **SwarmScore:** averages the log-ranks of the convergence score and the count of first-time accounts in each 2-hour window, then merges a machine's windows over the day with Fisher's method.
- **Evaluation:** recall at 1% false-positive rate and AUROC per window, and teams caught at N alerts per day per 10,000 machines, with hierarchical bootstrap CIs (seed-days, then teams) and paired differences between methods.

## Repository layout

```
swarmbench/
  data.py              LANL raw files -> per-day parquet; instance loader; monitored-host rule
  build.py             benchmark builder (plants the swarm into a test day)
  swarm_template.json  measured swarm template (aggregate statistics only)
  detectors.py         convergence rule, first-time accounts, SwarmScore, PMF baseline
  evaluate.py          recall at 1% FPR, AUROC, teams caught at N alerts/day, bootstrap CIs
  synthetic.py         small synthetic logs in LANL format (smoke test only)
  __main__.py          command line: prepare / build / score / evaluate / synthetic
scripts/reproduce.sh   full pipeline on LANL 2015
tests/smoke_test.sh    end-to-end run on synthetic data (seconds)
```

## Quickstart

Requires Python 3.11 and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
tests/smoke_test.sh            # end-to-end on synthetic logs; prints SMOKE TEST PASSED
```

**1. Get the LANL data.** Download the "Comprehensive, Multi-Source Cyber-Security Events" release from its official page, <https://csr.lanl.gov/data/cyber1/>, after accepting its terms. You need `auth.txt.gz` and `flows.txt.gz`. This repository does not include or redistribute any of it.

**2. Run the full pipeline** (prepare days 5-9, build 15 seed-days, score, evaluate):

```bash
scripts/reproduce.sh /path/to/lanl out
cat out/results/RESULTS.md
```

Or step by step:

```bash
python -m swarmbench prepare  --auth /path/to/lanl/auth.txt.gz --flows /path/to/lanl/flows.txt.gz --days 5-9 --data out/data
python -m swarmbench build    --data out/data --bench out/bench --days 6,8,9 --seeds 0-4
python -m swarmbench score    --data out/data --bench out/bench --scores out/scores
python -m swarmbench evaluate --bench out/bench --scores out/scores --out out/results
```

Day 7 was used to design SwarmScore and is left out of the test. Each test day uses the previous day as its pre-swarm reference.

### What reproduces exactly and what does not

The swarm template in this repository holds aggregate statistics only. The benchmark used for the paper drew each planted agent's targets from per-agent incident records, which we do not redistribute. `build.py` re-creates the same construction from the aggregate statistics, so a rebuilt benchmark is statistically similar but not row-identical, and its numbers will differ from the tables above. In a two-seed-day check on day 6 the rebuilt benchmark gave the convergence rule an AUROC of 0.90 and a recall at 1% FPR of 0.60, in the range of the paper's per-day values; with that few teams the team-level comparisons are not informative. The detectors and the evaluation code are cleaned-up copies of the code used for the paper, with the same rules and settings.

## Limitations

- LANL 2015 records human workstations and servers, not an AI-agent fleet; the detector has not been run on a live AI-lab network.
- The benchmark's leak, realism and answer-key checks were run on an earlier build. On the tested build a simple first-time-destination feature reaches AUROC up to 0.85 and recall at 1% FPR up to 13% (above the 5% bar), and traffic volume reaches AUROC about 0.65; the convergence rule itself began as one of those leak checks.
- The pre-swarm reference is a best case: it holds no swarm activity and ends exactly when the swarm starts.
- The deliberately easy positive controls (a very obvious swarm) still fail their fixed bar (recall at 1% FPR ≥ 0.50 *and* ≥ 80% of teams caught at 10 alerts/day, by one detector). With a pre-swarm reference, recall rises sharply (convergence rule 0.023 → 0.350; onset control 0.169 → 0.531), but at most 0.545 of teams are caught at 10 alerts/day, on only 11 teams per control (wide CIs). The benchmark's sensitivity at strict alert budgets is therefore not yet certified.
- Normal events such as software rollouts also make groups of machines adopt new services together; expect them to be the main source of false alarms. Allow-list known rollouts and start in shadow mode.

## Licence

PolyForm Noncommercial License 1.0.0 (see `LICENSE`): free for research and other non-commercial use; commercial rights are reserved. Copyright 2026 Yuri Galindo and Caíque Lira. The LANL data has its own terms; get it from its official page.

## Citation

```bibtex
@misc{galindo2026swarmbench,
  title  = {SwarmBench: Catching Coordinated AI-Agent Swarms from Network Metadata},
  author = {Galindo, Yuri and Lira, Ca{\'\i}que},
  year   = {2026},
  note   = {Apart Research hackathon submission},
  url    = {https://github.com/yurigalindo/swarmbench}
}
```

Please also cite the data: A. D. Kent, *Comprehensive, Multi-Source Cyber-Security Events*, Los Alamos National Laboratory, 2015.
