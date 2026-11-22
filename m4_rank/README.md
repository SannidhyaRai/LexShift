# M4: fusion ranking, evaluation and delivery

Branch: `m4-rank`. You consume everyone's functions through the contracts, so integration starts on day one against the
stubs and the real modules replace them one by one.

**Goal.** Combine all signals into the final ranking, prove it works, and package the submission.

## What you build

- [x] `rank()`: top-100 from `search()`, add continuity, health and authority, normalise, weighted sum, return top-K with the
      per-signal breakdown and an explanation. `rank.py`, `fusion.py`, `normalize.py`, `explain.py`, `weights.py`
- [x] Ablation configs: **B0** BM25 only, **B1** + statute continuity, **full** (+ health + authority). `common/config.yaml`
- [ ] **By hand, the team's job:** 30 queries (10 dev, 20 test), four types: **A** BNS query needing an IPC precedent, **B**
      changed or omitted provisions, **C** doctrines with overruled cases, **D** bare-number collisions. `eval/queries.jsonl`.
      Ready for it: 30 candidate queries and the writing rules in [eval/JUDGING_GUIDE.md](../eval/JUDGING_GUIDE.md)
- [x] Pooling tool: the top-20 of every system into a blind judge sheet, incrementally. `eval/pool.py`
- [x] Two-judge qrels tooling: kappa, disagreements, adjudication, `qrels.tsv`. `eval/make_qrels.py`, `eval/agreement.py`,
      `eval/check_data.py`
- [ ] **By hand, the team's job:** the grades (2 = relevant and good law, 1 = relevant but law materially changed or precedent
      criticised, 0 = irrelevant or overruled on the queried point) and `eval/gold_overrulings.csv`. Never generated
- [x] Metrics: P@5, Recall@10, MAP, nDCG@10, harmful@10 (overruled cases in the top 10), plus P@10 and judged@10. Tune weights
      on **dev only**. `eval/metrics.py`, `eval/tuning.py`, `eval/run_ablation.py`
- [x] Demo (CLI) with score breakdowns and evidence sentences. `app/cli.py`. A Streamlit page is optional and not started
- [x] README (kept current), report skeleton, video script, submission checklist: drafts in `docs/`. The numbers in them are
      placeholders until the real run exists; nothing in the drafts is a result

## How the code fits together

```
common.providers.load_providers()  -> real functions or labelled stubs, per config.yaml
        |
m4_rank.rank.collect()             -> top-100 hits + the signals a config needs, raw, contract-checked
        |
m4_rank.fusion.fuse()              -> normalise per signal, weighted sum, heap top-K, Result + explanation
        |
eval.run_ablation / app.cli        -> metrics tables, chart, demo
```

* Weights per config live in `common/config.yaml` (`ranking.configs`) and are placeholders until tuned. A tuned set is saved
  to `common/weights_tuned.yaml` by `--tune` and then used automatically.
* Normalisation is configured per signal (`ranking.normalize`): min-max for `rel` and `auth`, identity for `cont` and
  `health`. The reason is DECISIONS.md D-006.
* `collect()` and `fuse_collected()` are split so evaluation and tuning never repeat provider calls (D-013).

## Commands

```bash
python -m app.cli "BNS 103 murder" --offence-date 2025-01-10 --verbose
python -m eval.pool --round round1                   # blind judge sheet from the systems' top-20
python -m eval.make_qrels --round round1             # two judges -> qrels.tsv (+ kappa, disagreements)
python -m eval.check_data --strict                   # hand-made data against the plan
python -m eval.run_ablation --tune --split test      # tune on dev, report test
python -m eval.run_ablation --split test
python -m app.docmeta                                # optional: titles for the demo, from judgments.jsonl
```

## Order of work from here

1. Write the queries (the guide has examples), then switch the real modules on and run `python eval/smoke.py`.
2. `eval.pool`, grade independently, `eval.make_qrels`, adjudicate, `eval.make_qrels` again, write the gold list, `eval.check_data`.
3. `eval.run_ablation --tune`, then pool a second round for the tuned weights, grade it, and report on test.
4. Fill the report and the video from the generated files: [report](../docs/REPORT_SKELETON.md),
   [video](../docs/VIDEO_SCRIPT.md), [checklist](../docs/SUBMISSION_CHECKLIST.md).

## You hand over

`rank()`, the `eval/` scripts, **one command that prints the full metrics table**, the demo app, README, and the report and
video drafts.

## IR concepts you explain in the video

Net score, evaluation metrics (P@k, recall, MAP, nDCG), pooling, ablation.

## Done when

One command reproduces the metrics table, the demo runs live, and the report draft has all required sections.

## Watch out for

Never tune on test queries. Keep stubs working so integration starts on day one. The judged queries, the qrels and the gold
overruling list are made by hand: do not let a model generate the labels the system is scored against. Never write "dead
law" or "bad law" in the UI: say "treatment signals", show evidence and confidence, and include the not-legal-advice note.

## Phase 2: flat clustering, inexact top-K, new ablations, evaluation extras

| Piece | Where | Status |
|---|---|---|
| K-means (ours): k-means++, restarts, empty-cluster refill; K from the RSS elbow, AIC reported | `cluster.py` | done, K = 12 on the full corpus |
| Cluster evaluation: purity, NMI, Rand index, F1/F5 against the dominant offence (automatic labelling) | `cluster.py` | done, scikit-learn cross-check agrees |
| Cluster labels (differential centroid terms as surface words), `cluster_of()` | `cluster.py` | done |
| Inexact top-K: champion lists, tiered index, index elimination, cluster pruning; speed vs nDCG@10 | `inexact.py` | done, measured on dev |
| `full_hits` (HITS authority) and `full_tiered` (tiered candidates) configs | `rank.py`, `weights.py`, `common/config.yaml` | done, in `run_ablation` |
| PR curves, 11-point interpolated AP, R-precision, MRR | `eval/metrics.py`, `eval/plots.py` | done |
| Demo: did-you-mean, wildcard expansions, cluster chip and facet, HITS hubs, cluster browser, inexact table | `app/` | done |

```bash
python -m m4_rank.cluster build          # vectors, RSS curve, K, clusters, evaluation, labels (about 2 minutes, offline)
python -m m4_rank.inexact build          # champion lists, tier-1 postings, cluster-pruning leaders (seconds, offline)
python -m m4_rank.inexact report         # speed vs quality on the judged dev queries
python -m m4_rank.inexact query "dowry death" --method champion
python -m eval.run_ablation --split dev  # b0, b1, full, full_hits, full_tiered (+ PR curves)
```

The builds write git-ignored files under `data/processed/` (`clusters.jsonl`, `clusters_meta.json`, `inexact.pkl.gz`) and the
committed reports in [reports/](reports/): [clusters.md](reports/clusters.md) and [inexact.md](reports/inexact.md). Both are
seeded and reproducible. Choices and the measured numbers are in DECISIONS.md D-041.

What the numbers say, plainly. The clusters line up only partly with the main offence (NMI 0.26): they are topical (sentencing,
dowry and cruelty, sexual offences, quashing under inherent powers) and one groups judgments by author. Index elimination and
champion lists keep most of exact BM25's top 10 at about a fifth of the time. The tiered index and cluster pruning lose much
more, so `full_tiered` is reported as an ablation with its cost, never used as the default. HITS authority did not beat PageRank
on the dev queries with the untuned weights.

IR concepts for the video: flat clustering, K-means and RSS, choosing K, purity / NMI / RI / F, the cluster hypothesis;
champion lists, static quality g(d), tiered indexes, index elimination, cluster pruning; PR curves, interpolation, R-precision,
MRR.

## Status

See the status table in the top-level [README](../README.md#status).
