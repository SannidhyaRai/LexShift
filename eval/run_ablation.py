"""One command that prints and writes the metrics table: B0 (BM25) vs B1 (+continuity) vs full (+health +authority).

    python -m eval.run_ablation                      # evaluate on the test split with the current weights
    python -m eval.run_ablation --tune               # tune on DEV, write common/weights_tuned.yaml, report dev
    python -m eval.run_ablation --tune --split test  # tune on DEV, then report on TEST (the headline numbers)

Guard rails: tuning only ever sees dev queries; results computed with any stub provider are refused unless --allow-stubs, and
then they are written as stub_* files and no tuned weights are saved; queries without judgements are skipped and counted.
Everything printed comes from real calls to rank()'s building blocks on the hand-made queries and qrels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import load_config, resolve_path  # noqa: E402
from common.io import write_delimited  # noqa: E402
from common.providers import Providers, load_providers  # noqa: E402
from common.schema import Query  # noqa: E402
from eval.loaders import EvalDataError, load_overruled, load_qrels, load_queries  # noqa: E402
from eval.metrics import (  # noqa: E402
    METRIC_NAMES, OBJECTIVES, RECALL_LEVELS, aggregate, evaluate_query, interpolated_precision, paired_bootstrap,
)
from eval.tuning import TuningError, save_tuned, tune_config  # noqa: E402
from m4_rank.rank import (  # noqa: E402
    ArtefactError, Collected, ContractViolation, collect, fuse_collected, load_checked, stubbed_groups,
)
from m4_rank.weights import (  # noqa: E402
    SIGNALS, active_signals, canonical_config, format_weights, load_weights, signals_used, variant, weights_source,
)

DEFAULT_CONFIGS = "b0,b1,full,full_hits,full_tiered"
VariantKey = tuple[str, str]  # (candidate_source, auth_source)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test"), help="split to report (default: test, or dev with --tune)")
    ap.add_argument("--configs", default=DEFAULT_CONFIGS, help=f"comma list (default {DEFAULT_CONFIGS})")
    ap.add_argument("--tune", action="store_true", help="tune weights on the DEV split first")
    ap.add_argument("--objective", default="nDCG@10", choices=OBJECTIVES, help="tuning objective, higher is better (default nDCG@10)")
    ap.add_argument("--step", type=float, default=0.1, help="weight grid step (default 0.1)")
    ap.add_argument("--min-rel", type=float, default=0.3, help="floor on the relevance weight while tuning (default 0.3)")
    ap.add_argument("--allow-stubs", action="store_true", help="run with stub providers; output is named stub_* and is not a result")
    ap.add_argument("--out", help="output directory (default: paths.results_dir)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples for the b0 comparison (default 10000)")
    return ap.parse_args(argv)


def usable_queries(queries: list[Query], qrels: Mapping[str, Mapping[str, int]], split: str) -> tuple[list[Query], list[str]]:
    """Queries of `split` that have at least one judgement, and the ids of those skipped for having none."""
    chosen = [q for q in queries if q.split == split]
    keep = [q for q in chosen if qrels.get(q.qid)]
    return keep, [q.qid for q in chosen if not qrels.get(q.qid)]


def collect_all(
    queries: list[Query], signals: tuple[str, ...], providers: Providers, cfg: dict[str, Any],
    key: tuple[str, str] = ("exact", "pagerank"),
) -> dict[str, Collected]:
    return {q.qid: collect(q.text, q.offence_date, signals, providers=providers, cfg=cfg,
                           candidate_source=key[0], auth_source=key[1]) for q in queries}


def variant_key(name: str, cfg: dict[str, Any]) -> tuple[str, str]:
    v = variant(name, cfg)
    return v["candidate_source"], v["auth_source"]


def phase2_unavailable(name: str, providers: Providers, cfg: dict[str, Any]) -> str:
    """Why a Phase 2 variant cannot run on these providers ('' if it can). The inexact candidates read M1's real index and HITS
    reads M3's real citation graph, so they only make sense next to the real search() / authority() (or their flagged stubs)."""
    cand, auth = variant_key(name, cfg)
    real_search = "search" not in providers.stubbed
    if cand != "exact" and real_search and getattr(providers.search, "__module__", "") != "m1_index.searcher":
        return f"its candidates come from M1's index ({cand}), but search() here is not M1's"
    if cand != "exact" and real_search:
        from m4_rank.inexact import _path

        if not _path("artefact", cfg).exists():
            return "the inexact-retrieval artefact is missing (python -m m4_rank.inexact build)"
    if auth == "hits" and "authority" not in providers.stubbed \
            and getattr(providers.authority, "__module__", "") != "m3_treatment.scores":
        return "HITS reads M3's citation graph, but authority() here is not M3's"
    return ""


def by_variant(names: list[str], cfg: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = {}
    for n in names:
        groups.setdefault(variant_key(n, cfg), []).append(n)
    return groups


def collect_groups(
    groups: Mapping[tuple[str, str], list[str]], weights: Mapping[str, Mapping[str, float]], queries: list[Query],
    providers: Providers, cfg: dict[str, Any], reuse: Mapping[tuple[str, str], Mapping[str, Collected]] | None = None,
) -> dict[tuple[str, str], dict[str, Collected]]:
    """Signals collected once per (variant, query), reusing what an earlier step (tuning) already collected."""
    out: dict[tuple[str, str], dict[str, Collected]] = {}
    for key, members in groups.items():
        signals = signals_used(weights, members)
        old = (reuse or {}).get(key, {})
        keep = {qid: c for qid, c in old.items() if set(signals) <= set(c.signals)}
        missing = [q for q in queries if q.qid not in keep]
        out[key] = {**{q.qid: keep[q.qid] for q in queries if q.qid in keep},
                    **collect_all(missing, signals, providers, cfg, key)}
    return out


def evaluate_configs(
    weights_by_config: Mapping[str, Mapping[str, float]],
    collected: Mapping[str, Mapping[str, Collected]],
    queries: list[Query],
    qrels: Mapping[str, Mapping[str, int]],
    overruled: set[str],
    cfg: dict[str, Any],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """config -> qid -> metrics. `collected` is config -> qid -> Collected, shared by the configs of one variant, so each
    config only re-fuses signals that were collected once."""
    ev = cfg["evaluation"]
    depth = int(ev["depth"])
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for name, weights in weights_by_config.items():
        out[name] = {}
        for q in queries:
            ranked = [r.doc_id for r in fuse_collected(collected[name][q.qid], weights, depth, cfg)]
            out[name][q.qid] = evaluate_query(
                ranked, qrels[q.qid], overruled or None,
                depth=depth, threshold=int(ev["relevant_threshold"]), gain=str(ev["gain"]),
            )
    return out


def pr_curves(
    weights_by_config: Mapping[str, Mapping[str, float]],
    collected: Mapping[str, Mapping[str, Collected]],
    queries: list[Query],
    qrels: Mapping[str, Mapping[str, int]],
    cfg: dict[str, Any],
) -> dict[str, list[float]]:
    """config -> mean 11-point interpolated precision over the queries that have a relevant document (IIR 8.4)."""
    ev = cfg["evaluation"]
    depth, threshold = int(ev["depth"]), int(ev["relevant_threshold"])
    out: dict[str, list[float]] = {}
    for name, weights in weights_by_config.items():
        rows = []
        for q in queries:
            ranked = [r.doc_id for r in fuse_collected(collected[name][q.qid], weights, depth, cfg)]
            ip = interpolated_precision(ranked, qrels[q.qid], threshold)
            if ip is not None:
                rows.append(ip)
        out[name] = [sum(col) / len(rows) for col in zip(*rows)] if rows else [0.0] * len(RECALL_LEVELS)
    return out


def pr_markdown(curves: Mapping[str, list[float]]) -> str:
    head = ["config", *(f"{lv:.1f}" for lv in RECALL_LEVELS)]
    lines = ["", "## 11-point interpolated precision (mean over queries)", "",
             "| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join([name, *(f"{v:.3f}" for v in ys)]) + " |" for name, ys in curves.items()]
    lines += ["", "Recall is relative to the judged relevant documents, inside the evaluation depth."]
    return "\n".join(lines) + "\n"


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def summary_rows(
    results: Mapping[str, Mapping[str, Mapping[str, float | None]]], weights: Mapping[str, Mapping[str, float]]
) -> list[dict[str, Any]]:
    rows = []
    for name, per_query in results.items():
        agg = aggregate(per_query)
        row: dict[str, Any] = {"config": name, "n_queries": len(per_query)}
        row.update({f"w_{s}": round(weights[name][s], 4) for s in SIGNALS})
        row.update({m: ("" if agg[m][0] is None else round(agg[m][0], 4)) for m in METRIC_NAMES})
        rows.append(row)
    return rows


def render_markdown(
    split: str,
    queries: list[Query],
    skipped: list[str],
    results: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    weights: Mapping[str, Mapping[str, float]],
    sources: Mapping[str, str],
    providers: Providers,
    cfg: dict[str, Any],
    n_overruled: int,
    tuned_now: bool,
    n_boot: int,
) -> str:
    depth = cfg["evaluation"]["depth"]
    lines = [
        f"# Ablation results: {split} split, {len(queries)} queries, depth {depth}",
        "",
        f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Providers: {providers.describe()}.",
        "",
    ]
    if providers.stubbed:
        lines += ["> **STUB RUN: these numbers are not results.** At least one signal came from a fixed-value stub.", ""]
    if split == "dev" and tuned_now:
        lines += ["> Weights were tuned on this split, so these dev numbers are optimistic. Report the test split.", ""]
    header = ["config", "weights rel/cont/health/auth", *METRIC_NAMES]
    lines += ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for name, per_query in results.items():
        agg = aggregate(per_query)
        cells = [name, format_weights(weights[name]), *(_fmt(agg[m][0]) for m in METRIC_NAMES)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Weights source: " + "; ".join(f"{n}: {s}" for n, s in sources.items()) + "."]
    lines += [
        f"Relevant means grade >= {cfg['evaluation']['relevant_threshold']}; nDCG gain `{cfg['evaluation']['gain']}`. "
        "Documents without a judgement count as grade 0; `judged@10` is the share of the top 10 that has one.",
        f"Gold overruling list: {n_overruled} rows" + ("" if n_overruled else " (so harmful@10 is n/a)") + ".",
    ]
    if skipped:
        lines.append(f"Skipped {len(skipped)} {split} queries with no judgements: {', '.join(skipped)}.")

    if "b0" in results and len(results) > 1:
        lines += ["", f"## Difference from b0 (paired bootstrap over queries, 95% interval, {n_boot} resamples)", ""]
        lines += ["| config | metric | mean difference | 95% interval | queries |", "|---|---|---|---|---|"]
        for name in results:
            if name == "b0":
                continue
            for metric in ("P@5", "nDCG@10", "harmful@10"):
                a = {q: m[metric] for q, m in results[name].items()}
                b = {q: m[metric] for q, m in results["b0"].items()}
                res = paired_bootstrap(a, b, n_boot=n_boot, seed=int(cfg["evaluation"]["seed"]))
                if res is None:
                    lines.append(f"| {name} | {metric} | n/a | n/a | 0 |")
                else:
                    mean, lo, hi, n = res
                    lines.append(f"| {name} | {metric} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {n} |")
        lines.append("")
        lines.append("With this few queries the interval is wide; read it as that reminder, not as a significance test.")

    types = sorted({q.type for q in queries})
    if len(types) > 1:
        by_qid = {q.qid: q.type for q in queries}
        lines += ["", "## nDCG@10 by query type", "", "| type | queries | " + " | ".join(results) + " |",
                  "|---|---|" + "|".join("---" for _ in results) + "|"]
        for t in types:
            members = [q for q, ty in by_qid.items() if ty == t]
            cells = [_fmt(aggregate({q: results[name][q] for q in members})["nDCG@10"][0]) for name in results]
            lines.append(f"| {t} | {len(members)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path, prefix: str, split: str, queries: list[Query], results: Mapping[str, Any],
    weights: Mapping[str, Mapping[str, float]], markdown: str, plots: bool,
    curves: Mapping[str, list[float]] | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rows = summary_rows(results, weights)
    path = out_dir / f"{prefix}ablation_{split}.csv"
    write_delimited(path, list(rows[0]), rows)
    written.append(path)

    qtype = {q.qid: q.type for q in queries}
    per_rows = [
        {"qid": qid, "type": qtype[qid], "config": name, **{m: ("" if v is None else round(v, 4)) for m, v in metrics.items()}}
        for name, per_query in results.items() for qid, metrics in per_query.items()
    ]
    path = out_dir / f"{prefix}per_query_{split}.csv"
    write_delimited(path, ["qid", "type", "config", *METRIC_NAMES], per_rows)
    written.append(path)

    path = out_dir / f"{prefix}ablation_{split}.md"
    path.write_text(markdown, encoding="utf-8", newline="\n")
    written.append(path)

    if plots:
        from eval.plots import bar_chart

        png = out_dir / f"{prefix}ablation_{split}.png"
        if bar_chart(rows, png, f"{split} split ({len(queries)} queries)"):
            written.append(png)
            if curves:
                from eval.plots import pr_curve

                pr_png = out_dir / f"{prefix}pr_{split}.png"
                if pr_curve(dict(curves), pr_png, f"{split} split ({len(queries)} queries)"):
                    written.append(pr_png)
        else:
            print("matplotlib is not installed: skipping the chart (pip install matplotlib).")
    return written


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except ArtefactError as exc:
        print(f"A module could not find its data: {exc}")
        return 2
    except ContractViolation as exc:
        print(f"A module returned data that breaks its contract: {exc}")
        return 3


def _main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    if args.bootstrap < 1:
        print("--bootstrap must be at least 1.")
        return 2
    try:
        names = list(dict.fromkeys(canonical_config(n, cfg) for n in args.configs.split(",") if n.strip()))
        if not names:
            print("--configs must name at least one system (b0, b1, full).")
            return 2
        queries = load_queries()
        if not queries:
            print("No queries in eval/queries.jsonl. The judged query set is written by hand (see eval/README.md); "
                  "there is nothing to evaluate yet.")
            return 2
        qrels = load_qrels(known_qids={q.qid for q in queries})
        if not qrels:
            print("eval/qrels.tsv has no judgements. Judge the pooled top-20 of every system first (see eval/README.md).")
            return 2
        overruled = load_overruled()
    except EvalDataError as exc:
        print(f"Evaluation data problem: {exc}")
        return 2

    split = args.split or ("dev" if args.tune else "test")
    providers = load_checked(load_providers, cfg)
    explicit = args.configs != DEFAULT_CONFIGS
    for n in list(names):
        why = phase2_unavailable(n, providers, cfg)
        if why:
            if explicit:
                print(f"Cannot run {n}: {why}.")
                return 2
            print(f"Skipping {n}: {why}.")
            names.remove(n)
    weights = {n: load_weights(n, cfg) for n in names}
    sources = {n: weights_source(n, cfg) for n in names}
    signals = signals_used(weights, names)

    stubbed = stubbed_groups(providers, signals)
    if stubbed and not args.allow_stubs:
        print("Refusing to run: these providers are fixed-value stubs, so the numbers would not be results: "
              + ", ".join(stubbed) + ".\nFinish the real functions and set their `stubs:` switches to false in "
              "common/config.yaml (or LEXSHIFT_STUBS=none), or pass --allow-stubs to exercise the pipeline "
              "(output is then named stub_* and nothing is saved to common/).")
        return 2
    prefix = "stub_" if stubbed else ""
    if stubbed:
        print(f"*** STUB RUN ({', '.join(stubbed)}): exercising the pipeline only; these numbers are not results. ***")

    tuned_now = False
    dev_collected: dict[tuple[str, str], dict[str, Collected]] = {}  # kept so a dev evaluation after tuning does not call the providers again
    if args.tune:
        dev, dev_skipped = usable_queries(queries, qrels, "dev")
        if dev_skipped:
            print(f"Dev queries without judgements are skipped for tuning: {', '.join(dev_skipped)}")
        try:
            tunable = [n for n in names if len(active_signals(weights[n])) > 1]
            dev_collected = collect_groups(by_variant(tunable, cfg), weights, dev, providers, cfg)
            new: dict[str, dict[str, float]] = {}
            for n in tunable:
                res = tune_config(n, dev_collected[variant_key(n, cfg)], dev, qrels, overruled or None, objective=args.objective,
                                  step=args.step, floors={"rel": args.min_rel}, cfg=cfg)
                new[n] = res.best.weights
                print(f"\ntuned {n} on {res.n_queries} dev queries ({res.grid_size} weight vectors), objective {res.objective}:")
                for t in res.trials[:5]:
                    print(f"  {format_weights(t.weights)}  {args.objective}={t.score:.4f}")
        except TuningError as exc:
            print(f"Tuning problem: {exc}")
            return 2
        if new:
            weights.update(new)
            sources.update({n: "tuned on dev (this run)" for n in new})
            tuned_now = True
            if stubbed:
                print("Stub run: tuned weights are NOT saved.")
            else:
                print(f"Saved tuned weights to {save_tuned(new, cfg)}")

    queries_in_split, skipped = usable_queries(queries, qrels, split)
    if not queries_in_split:
        print(f"No {split} queries with judgements: nothing to evaluate.")
        return 2
    by_key = collect_groups(by_variant(names, cfg), weights, queries_in_split, providers, cfg, reuse=dev_collected)
    collected = {n: by_key[variant_key(n, cfg)] for n in names}
    results = evaluate_configs(weights, collected, queries_in_split, qrels, overruled, cfg)
    markdown = render_markdown(split, queries_in_split, skipped, results, weights, sources, providers, cfg,
                               len(overruled), tuned_now, args.bootstrap)
    curves = pr_curves(weights, collected, queries_in_split, qrels, cfg)
    markdown += pr_markdown(curves)
    print("\n" + markdown)
    out_dir = Path(args.out) if args.out else resolve_path("results_dir", cfg)
    for path in write_outputs(out_dir, prefix, split, queries_in_split, results, weights, markdown, not args.no_plots, curves):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
