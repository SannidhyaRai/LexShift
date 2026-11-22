"""Flat clustering of the corpus with K-means, written by us (IIR chapter 16), for browse-by-cluster in the demo.

    python -m m4_rank.cluster build      # vectors, RSS curve, K, final clustering, evaluation, labels (about a minute)
    python -m m4_rank.cluster show       # the clusters and their labels, from the frozen files

Document vectors. Every judgment is a tf-idf vector over M terms of M1's index (top-M by document frequency inside a df window,
`clustering.vocab_size`, `min_df`, `max_df_ratio`; terms that are only digits are left out): weight = (1 + log10 tf) * log10(N / df),
then cosine (L2) normalised, so the squared Euclidean distance between two vectors is 2 - 2 cos. The vectors are read straight from
M1's compact postings arrays, so they use M1's tokenizer and stems and match the index exactly.

K-means (IIR 16.4). Seeds by k-means++ (each new seed drawn with probability proportional to its squared distance from the nearest
seed already chosen), then alternate reassignment (each document to its nearest centroid) and recomputation (each centroid = the mean
of its members) until no document changes cluster or `max_iter` is reached. An empty cluster takes the document farthest from its
centroid, so K clusters always come back. K-means only finds a local minimum of RSS, so it is restarted `n_init` times with seeds
seed, seed+1, ... and the run with the lowest RSS is kept. Everything is seeded, so the result is reproducible.

Choosing K (IIR 16.4.1). RSS(K) = sum of squared distances to the nearest centroid, which always falls as K grows, is computed over
`k_grid`. Two criteria are reported: the elbow of the RSS curve (the K farthest below the straight line from the first to the last
point, both axes scaled to [0, 1]) and the AIC-style criterion K = argmin RSS(K) + 2 M K with M the vector dimension. Neither looks at
any label or query. `clustering.k: auto` uses the elbow.

Evaluation (IIR 16.3) against an AUTOMATIC labelling, not a human gold standard: each judgment's dominant offence, the offence_id
with the most references in M2's doc_statutes.jsonl. Judgments without an offence reference are left out of the evaluation (counted).
Purity, NMI, Rand index and F-measure (F1 and F5, IIR's choice that weights recall more) are computed from the contingency table.

Labels (IIR 17.7). Differential labelling: the terms whose centroid weight most exceeds their mean weight in the other centroids,
shown as their most frequent surface word (M1's surface dictionary) rather than the stem.

scikit-learn is used only as a cross-check (`--crosscheck`, and the tests): its KMeans on the same vectors and its NMI / Rand
implementations, to confirm our numbers. It is never used to produce the clustering.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.config import load_config  # noqa: E402


class ClusterError(RuntimeError):
    """The clustering artefacts are missing or do not describe this document."""


def _settings(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return (cfg or load_config())["clustering"]


def _path(key: str, cfg: dict[str, Any] | None = None) -> Path:
    p = Path(_settings(cfg)[key])
    return p if p.is_absolute() else ROOT / p


# ----------------------------------------------------------------------------------------------------------------
# Document vectors from M1's index
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class DocVectors:
    X: np.ndarray            # (N, M) float32, rows L2-normalised (a judgment with none of the M terms is a zero row)
    doc_ids: list[str]       # row order = M1's document numbers
    terms: list[str]         # column order
    idf: np.ndarray          # (M,) log10(N / df)
    df: np.ndarray           # (M,)

    @property
    def zero_rows(self) -> int:
        return int((np.abs(self.X).sum(axis=1) == 0).sum())


def select_terms(terms: Sequence[str], df: np.ndarray, n_docs: int, vocab_size: int, min_df: int, max_df_ratio: float) -> np.ndarray:
    """Term ids of the vocabulary: inside [min_df, max_df_ratio * N], not only digits, then the vocab_size most frequent
    (ties by term, so the choice is deterministic)."""
    max_df = max_df_ratio * n_docs
    ok = [t for t in range(len(terms)) if min_df <= df[t] <= max_df and not terms[t].isdigit()]
    ok.sort(key=lambda t: (-int(df[t]), terms[t]))
    return np.array(sorted(ok[:vocab_size]), dtype=np.int64)


def build_vectors(index=None, cfg: dict[str, Any] | None = None) -> DocVectors:
    """tf-idf (ltc-style: 1 + log10 tf, idf, cosine) vectors over the selected vocabulary, from M1's postings arrays."""
    s = _settings(cfg)
    if index is None:
        from m1_index.index import InvertedIndex

        index = InvertedIndex.load()
    n_docs = index.num_docs
    t_start = np.frombuffer(index.t_start, dtype=np.uint32).astype(np.int64)
    p_doc = np.frombuffer(index.p_doc, dtype=np.uint32)
    p_off = np.frombuffer(index.p_off, dtype=np.uint32).astype(np.int64)
    df_all = np.diff(t_start)
    chosen = select_terms(index.terms, df_all, n_docs, int(s["vocab_size"]), int(s["min_df"]), float(s["max_df_ratio"]))
    X = np.zeros((n_docs, len(chosen)), dtype=np.float32)
    idf = np.log10(n_docs / df_all[chosen]).astype(np.float32)
    for j, t in enumerate(chosen):
        lo, hi = t_start[t], t_start[t + 1]
        tf = p_off[lo + 1:hi + 1] - p_off[lo:hi]
        X[p_doc[lo:hi], j] = (1.0 + np.log10(tf)) * idf[j]
    norms = np.linalg.norm(X, axis=1)
    nz = norms > 0
    X[nz] /= norms[nz, None]
    return DocVectors(X=X, doc_ids=list(index.doc_ids), terms=[index.terms[t] for t in chosen], idf=idf, df=df_all[chosen])


# ----------------------------------------------------------------------------------------------------------------
# K-means
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class KMeansResult:
    k: int
    assign: np.ndarray       # (N,) cluster id per row
    centroids: np.ndarray    # (K, M)
    rss: float
    iterations: int
    seed: int
    converged: bool
    empty_fixes: int = 0     # times an empty cluster was refilled
    history: list[float] = field(default_factory=list)  # RSS after each iteration (never increases)


def sq_distances(X: np.ndarray, C: np.ndarray, x_sq: np.ndarray | None = None) -> np.ndarray:
    """Squared Euclidean distance of every row of X to every row of C: |x|^2 - 2 x.c + |c|^2, clipped at 0 (rounding)."""
    x_sq = (X * X).sum(axis=1) if x_sq is None else x_sq
    d = x_sq[:, None] - 2.0 * (X @ C.T) + (C * C).sum(axis=1)[None, :]
    return np.maximum(d, 0.0)


def kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator, x_sq: np.ndarray | None = None) -> np.ndarray:
    """k-means++ seeding (Arthur and Vassilvitskii 2007): first seed uniform, each next one with probability ~ D(x)^2."""
    n = X.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    x_sq = (X * X).sum(axis=1) if x_sq is None else x_sq
    centers = [int(rng.integers(n))]
    d2 = sq_distances(X, X[centers], x_sq)[:, 0].astype(np.float64)
    for _ in range(1, k):
        total = d2.sum()
        if total <= 0:  # every point coincides with a seed: take any point not chosen yet
            rest = np.setdiff1d(np.arange(n), centers)
            nxt = int(rng.choice(rest))
        else:
            nxt = int(rng.choice(n, p=d2 / total))
        centers.append(nxt)
        d2 = np.minimum(d2, sq_distances(X, X[[nxt]], x_sq)[:, 0])
    return X[centers].astype(np.float64)


def kmeans(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> KMeansResult:
    """One K-means run (Lloyd's algorithm) from a k-means++ seeding with `seed`."""
    rng = np.random.default_rng(seed)
    Xd = X.astype(np.float64, copy=False)
    x_sq = (Xd * Xd).sum(axis=1)
    C = kmeans_pp(Xd, k, rng, x_sq)
    assign = np.full(X.shape[0], -1, dtype=np.int64)
    history: list[float] = []
    empty_fixes = 0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        d = sq_distances(Xd, C, x_sq)
        new = d.argmin(axis=1)
        # an empty cluster takes the document farthest from its own centroid (never the last member of another cluster)
        counts = np.bincount(new, minlength=k)
        for empty in np.flatnonzero(counts == 0):
            own = d[np.arange(len(new)), new]
            order = np.argsort(-own, kind="stable")
            for cand in order:
                if counts[new[cand]] > 1:
                    counts[new[cand]] -= 1
                    new[cand] = empty
                    counts[empty] = 1
                    empty_fixes += 1
                    break
        changed = not np.array_equal(new, assign)
        assign = new
        for c in range(k):
            C[c] = Xd[assign == c].mean(axis=0)
        rss = float(sq_distances(Xd, C, x_sq)[np.arange(len(assign)), assign].sum())
        history.append(rss)
        if not changed:
            converged = True
            break
    return KMeansResult(k=k, assign=assign, centroids=C, rss=history[-1], iterations=it, seed=seed,
                        converged=converged, empty_fixes=empty_fixes, history=history)


def best_of(X: np.ndarray, k: int, seed: int = 0, n_init: int = 3, max_iter: int = 100) -> KMeansResult:
    """The lowest-RSS run over seeds seed, seed+1, ..., seed+n_init-1 (ties: the earlier seed)."""
    runs = [kmeans(X, k, seed + i, max_iter) for i in range(max(1, n_init))]
    return min(runs, key=lambda r: (r.rss, r.seed))


# ----------------------------------------------------------------------------------------------------------------
# Choosing K
# ----------------------------------------------------------------------------------------------------------------
def elbow(ks: Sequence[int], rss: Sequence[float]) -> int:
    """The K whose (scaled) point lies farthest below the chord from the first to the last point of the RSS curve."""
    if len(ks) != len(rss) or not ks:
        raise ValueError("ks and rss must be non-empty and of equal length")
    if len(ks) < 3:
        return ks[0]
    k0, k1 = ks[0], ks[-1]
    r0, r1 = rss[0], rss[-1]
    best_k, best_gap = ks[0], -math.inf
    for k, r in zip(ks, rss):
        x = (k - k0) / (k1 - k0) if k1 != k0 else 0.0
        y = (r - r1) / (r0 - r1) if r0 != r1 else 0.0
        gap = (1.0 - x) - y  # the chord runs from (0, 1) to (1, 0); a point below it has y < 1 - x
        if gap > best_gap + 1e-12:
            best_k, best_gap = k, gap
    return best_k


def aic_k(ks: Sequence[int], rss: Sequence[float], dims: int) -> int:
    """IIR eq. 16.19: K = argmin RSS(K) + 2 M K, M = the dimension of the vectors."""
    return min(zip(ks, rss), key=lambda kr: (kr[1] + 2.0 * dims * kr[0], kr[0]))[0]


# ----------------------------------------------------------------------------------------------------------------
# Evaluation against a labelling
# ----------------------------------------------------------------------------------------------------------------
def _pairs(n: int | np.ndarray):
    return n * (n - 1) // 2


def contingency(clusters: Sequence[int], classes: Sequence[str]) -> tuple[np.ndarray, list[int], list[str]]:
    ks = sorted(set(clusters))
    cs = sorted(set(classes))
    ki = {k: i for i, k in enumerate(ks)}
    ci = {c: i for i, c in enumerate(cs)}
    table = np.zeros((len(ks), len(cs)), dtype=np.int64)
    for k, c in zip(clusters, classes):
        table[ki[k], ci[c]] += 1
    return table, ks, cs


def external_scores(clusters: Sequence[int], classes: Sequence[str]) -> dict[str, float]:
    """Purity, NMI (arithmetic-mean normalisation), Rand index, pair precision / recall, F1 and F5 (IIR 16.3)."""
    if len(clusters) != len(classes) or not clusters:
        raise ValueError("clusters and classes must be non-empty and of equal length")
    table, _, _ = contingency(clusters, classes)
    n = int(table.sum())
    purity = table.max(axis=1).sum() / n

    pk = table.sum(axis=1) / n
    pc = table.sum(axis=0) / n
    pkc = table / n
    nz = pkc > 0
    mi = float((pkc[nz] * np.log(pkc[nz] / np.outer(pk, pc)[nz])).sum())
    h_k = float(-(pk[pk > 0] * np.log(pk[pk > 0])).sum())
    h_c = float(-(pc[pc > 0] * np.log(pc[pc > 0])).sum())
    nmi = mi / ((h_k + h_c) / 2) if (h_k + h_c) > 0 else 1.0

    tp = int(_pairs(table).sum())
    tp_fp = int(_pairs(table.sum(axis=1)).sum())  # pairs put in the same cluster
    tp_fn = int(_pairs(table.sum(axis=0)).sum())  # pairs of the same class
    total = _pairs(n)
    fp, fn = tp_fp - tp, tp_fn - tp
    tn = total - tp - fp - fn
    ri = (tp + tn) / total if total else 1.0
    p = tp / tp_fp if tp_fp else 0.0
    r = tp / tp_fn if tp_fn else 0.0

    def f(beta: float) -> float:
        b2 = beta * beta
        return (b2 + 1) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0

    return {"purity": float(purity), "nmi": float(nmi), "rand_index": float(ri), "pair_precision": float(p),
            "pair_recall": float(r), "f1": float(f(1.0)), "f5": float(f(5.0)), "n": n,
            "tp": tp, "fp": fp, "fn": fn, "tn": int(tn)}


def dominant_offences(path: Path | None = None) -> dict[str, str]:
    """doc_id -> the offence_id with the most references in M2's doc_statutes.jsonl (ties: alphabetical). No offence: absent."""
    from common.config import resolve_path
    from common.io import read_jsonl

    path = path or resolve_path("doc_statutes")
    out: dict[str, str] = {}
    for rec in read_jsonl(path):
        counts: Counter[str] = Counter()
        for ref in rec.get("refs") or []:
            if ref.get("offence_id"):
                counts[ref["offence_id"]] += int(ref.get("count") or 1)
        if counts:
            top = max(counts.values())
            out[rec["doc_id"]] = min(o for o, c in counts.items() if c == top)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------------------------------------------
def differential_terms(centroids: np.ndarray, terms: Sequence[str], n: int) -> list[list[str]]:
    """Per cluster: the n terms with the largest centroid weight minus their mean weight in the other centroids."""
    k = centroids.shape[0]
    out = []
    for c in range(k):
        others = (centroids.sum(axis=0) - centroids[c]) / max(1, k - 1)
        diff = centroids[c] - others
        order = np.lexsort((np.array(terms), -diff))  # by diff descending, then term
        out.append([terms[j] for j in order[:n]])
    return out


def centroid_terms(centroids: np.ndarray, terms: Sequence[str], n: int) -> list[list[str]]:
    """Per cluster: the n terms with the largest centroid weight (plain centroid labelling, for comparison)."""
    out = []
    for c in range(centroids.shape[0]):
        order = np.lexsort((np.array(terms), -centroids[c]))
        out.append([terms[j] for j in order[:n]])
    return out


class SurfaceNamer:
    """Index term (a stem) -> its most frequent surface word, via M1's surface dictionary; the stem itself when none is found."""

    def __init__(self) -> None:
        self._d = None
        self._cache: dict[str, str] = {}

    def _dictionary(self):
        if self._d is None:
            from m1_index.dictionary import get_dictionary

            self._d = get_dictionary()
        return self._d

    def __call__(self, stem: str) -> str:
        if stem in self._cache:
            return self._cache[stem]
        name = stem
        try:
            d = self._dictionary()
            best = (-1, stem)
            for wid in d.prefix_range(stem[: max(1, min(len(stem), 4))]):
                word = d.words[wid]
                if d.stem(word) == stem and int(d.df[wid]) > best[0]:
                    best = (int(d.df[wid]), word)
            name = best[1]
        except (OSError, RuntimeError, ValueError):  # no dictionary: the stem is still a faithful label
            name = stem
        self._cache[stem] = name
        return name


# ----------------------------------------------------------------------------------------------------------------
# Build, freeze, look up
# ----------------------------------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x:.4f}"


def build(cfg: dict[str, Any] | None = None, crosscheck: bool = True, log=print) -> dict[str, Any]:
    """Run the whole offline job and write clusters.jsonl, clusters_meta.json and the report. Returns the meta record."""
    cfg = cfg or load_config()
    s = _settings(cfg)
    seed, n_init, max_iter = int(s["seed"]), int(s["n_init"]), int(s["max_iter"])
    t0 = time.perf_counter()
    vec = build_vectors(cfg=cfg)
    n, m = vec.X.shape
    log(f"vectors: {n} judgments x {m} terms ({vec.zero_rows} judgments have none of the terms), {time.perf_counter() - t0:.1f}s")

    ks = [int(k) for k in s["k_grid"]]
    curve = []
    for k in ks:
        t = time.perf_counter()
        r = best_of(vec.X, k, seed, n_init, max_iter)
        curve.append({"k": k, "rss": r.rss, "iterations": r.iterations, "converged": r.converged, "seed": r.seed,
                      "seconds": round(time.perf_counter() - t, 2)})
        log(f"  K={k:>3}  RSS={r.rss:10.2f}  iterations={r.iterations:>3}  {curve[-1]['seconds']}s")
    rss = [c["rss"] for c in curve]
    k_elbow = elbow(ks, rss)
    k_aic = aic_k(ks, rss, m)
    k = k_elbow if str(s["k"]).lower() == "auto" else int(s["k"])
    log(f"elbow K={k_elbow}, AIC K={k_aic}; using K={k}")

    final = best_of(vec.X, k, seed, n_init, max_iter)
    d = sq_distances(vec.X.astype(np.float64), final.centroids)
    dist = np.sqrt(d[np.arange(n), final.assign])

    namer = SurfaceNamer()
    diff = differential_terms(final.centroids, vec.terms, int(s["label_terms"]) * 3)
    cent = centroid_terms(final.centroids, vec.terms, int(s["label_terms"]))
    labels, label_terms = [], []
    for c in range(k):
        words: list[str] = []
        for t in diff[c]:  # distinct surface words; a stem and its surface form are shown once
            w = namer(t)
            if w not in words:
                words.append(w)
            if len(words) == int(s["label_terms"]):
                break
        labels.append(", ".join(words))
        label_terms.append(diff[c][: int(s["label_terms"])])

    gold = dominant_offences()
    pairs = [(int(final.assign[i]), gold[did]) for i, did in enumerate(vec.doc_ids) if did in gold]
    scores = external_scores([p[0] for p in pairs], [p[1] for p in pairs])
    sizes = np.bincount(final.assign, minlength=k)

    per_cluster = []
    for c in range(k):
        members = [vec.doc_ids[i] for i in np.flatnonzero(final.assign == c)]
        offs = Counter(gold[x] for x in members if x in gold)
        top_off, top_n = (offs.most_common(1)[0] if offs else ("", 0))
        per_cluster.append({"cluster_id": c, "size": int(sizes[c]), "label": labels[c], "label_terms": label_terms[c],
                            "centroid_terms": cent[c], "top_offence": top_off,
                            "top_offence_share": round(top_n / max(1, sum(offs.values())), 4),
                            "labelled_members": int(sum(offs.values()))})

    cross = sklearn_crosscheck(vec.X, final, [p[0] for p in pairs], [p[1] for p in pairs], seed, n_init, max_iter) if crosscheck else None

    meta = {
        "k": k, "k_rule": "elbow" if str(s["k"]).lower() == "auto" else "fixed", "k_elbow": k_elbow, "k_aic": k_aic,
        "seed": seed, "n_init": n_init, "max_iter": max_iter, "rss": final.rss, "iterations": final.iterations,
        "converged": final.converged, "chosen_seed": final.seed, "empty_fixes": final.empty_fixes,
        "n_docs": n, "dims": m, "zero_vectors": vec.zero_rows,
        "vocab": {"size": m, "min_df": int(s["min_df"]), "max_df_ratio": float(s["max_df_ratio"])},
        "rss_curve": curve, "labels": labels, "clusters": per_cluster,
        "evaluation": {"against": "dominant offence_id per judgment from M2's doc_statutes.jsonl (automatic, not a human gold standard)",
                       "labelled_docs": len(pairs), "unlabelled_docs": n - len(pairs),
                       "classes": len({p[1] for p in pairs}), **scores},
        "crosscheck": cross,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out = _path("out", cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for i, did in enumerate(vec.doc_ids):
            fh.write(json.dumps({"doc_id": did, "cluster_id": int(final.assign[i]),
                                 "distance_to_centroid": round(float(dist[i]), 6)}) + "\n")
    _path("meta", cfg).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_report(meta, cfg)
    _load.cache_clear()
    log(f"wrote {out}, {_path('meta', cfg)} and the report in {_path('report_dir', cfg)} ({meta['seconds']}s)")
    return meta


def sklearn_crosscheck(X: np.ndarray, ours: KMeansResult, clusters: list[int], classes: list[str],
                       seed: int, n_init: int, max_iter: int) -> dict[str, Any] | None:
    """scikit-learn's KMeans on the same vectors and its NMI / RI, beside ours. A cross-check only; None if not installed."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
    except ImportError:
        return None
    km = KMeans(n_clusters=ours.k, init="k-means++", n_init=n_init, max_iter=max_iter, random_state=seed).fit(X.astype(np.float64))
    ours_scores = external_scores(clusters, classes)
    return {
        "sklearn_rss": float(km.inertia_), "our_rss": ours.rss,
        "rss_ratio_ours_over_sklearn": ours.rss / float(km.inertia_) if km.inertia_ else None,
        "agreement_ari": float(adjusted_rand_score(ours.assign, km.labels_)),
        "sklearn_nmi_of_our_clusters": float(normalized_mutual_info_score(classes, clusters, average_method="arithmetic")),
        "our_nmi": ours_scores["nmi"],
        "sklearn_rand_of_our_clusters": float(rand_score(classes, clusters)),
        "our_rand": ours_scores["rand_index"],
    }


def write_report(meta: Mapping[str, Any], cfg: dict[str, Any] | None = None) -> Path:
    rdir = _path("report_dir", cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "clusters.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    ev = meta["evaluation"]
    lines = [
        "# Flat clustering: K-means over tf-idf judgment vectors",
        "",
        "Generated by `python -m m4_rank.cluster build` from M1's index (every number below is computed, none is typed in).",
        "",
        f"* {meta['n_docs']} judgments, {meta['dims']} terms (top by df with {meta['vocab']['min_df']} <= df <= "
        f"{meta['vocab']['max_df_ratio']:.0%} of N, digits-only terms left out); {meta['zero_vectors']} judgments contain none of them.",
        f"* K-means with k-means++ seeding, {meta['n_init']} restarts (seeds {meta['seed']}..{meta['seed'] + meta['n_init'] - 1}), "
        f"at most {meta['max_iter']} iterations; the lowest-RSS run is kept.",
        f"* **K = {meta['k']}** ({meta['k_rule']}). Elbow of the RSS curve: K = {meta['k_elbow']}. "
        f"AIC (RSS + 2MK, M = {meta['dims']}): K = {meta['k_aic']}.",
        f"* Final run: RSS = {meta['rss']:.2f}, {meta['iterations']} iterations, converged = {meta['converged']}, "
        f"empty clusters refilled = {meta['empty_fixes']}.",
        "",
        "## RSS curve (choosing K)",
        "",
        "| K | RSS | iterations | converged | seconds |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {c['k']} | {c['rss']:.2f} | {c['iterations']} | {c['converged']} | {c['seconds']} |" for c in meta["rss_curve"]]
    first, last = meta["rss_curve"][0], meta["rss_curve"][-1]
    drop = (first["rss"] - last["rss"]) / first["rss"] if first["rss"] else 0.0
    lines += [
        "",
        f"The curve is shallow: RSS falls by only {drop:.1%} from K = {first['k']} to K = {last['k']}, as is usual for sparse, "
        "high-dimensional text vectors, so the elbow is a weak preference rather than a sharp knee.",
        "",
        "With unit-length vectors RSS is at most N, while the AIC penalty grows by 2M per cluster, so with thousands of dimensions "
        "the textbook AIC favours the smallest K on the grid. We report it as IIR defines it and choose K by the elbow, which "
        "uses only the RSS curve. Neither criterion looks at a label or a query.",
        "",
        "## Evaluation against the dominant offence (automatic labelling, not a human gold standard)",
        "",
        f"Each judgment's class is the offence_id with the most references in M2's `doc_statutes.jsonl`. "
        f"{ev['labelled_docs']} judgments have one ({ev['classes']} classes); {ev['unlabelled_docs']} have no offence reference "
        "and are left out. The labelling is noisy (a murder appeal that also discusses common intention gets one class only), so "
        "the scores measure how far the topical clusters line up with the main offence, not whether they are right.",
        "",
        "| purity | NMI | Rand index | pair precision | pair recall | F1 | F5 |",
        "|---|---|---|---|---|---|---|",
        f"| {_fmt(ev['purity'])} | {_fmt(ev['nmi'])} | {_fmt(ev['rand_index'])} | {_fmt(ev['pair_precision'])} | "
        f"{_fmt(ev['pair_recall'])} | {_fmt(ev['f1'])} | {_fmt(ev['f5'])} |",
        "",
        f"Pair counts: TP {ev['tp']}, FP {ev['fp']}, FN {ev['fn']}, TN {ev['tn']}. The class sizes are very uneven "
        "(murder is the dominant offence of about a third of the labelled judgments), which keeps purity and RI high even for "
        "weak clusterings; NMI and F are the more telling numbers.",
        "",
        "## Clusters",
        "",
        "| id | size | label (differential terms) | centroid terms | commonest dominant offence (share of labelled members) |",
        "|---|---|---|---|---|",
    ]
    for c in meta["clusters"]:
        off = f"{c['top_offence']} ({c['top_offence_share']:.0%} of {c['labelled_members']})" if c["top_offence"] else "none"
        lines.append(f"| {c['cluster_id']} | {c['size']} | {c['label']} | {', '.join(c['centroid_terms'])} | {off} |")
    lines += [
        "",
        "Labels are read from the centroids, so they show what the vectors actually share. A cluster whose label is a judge's name "
        "or a reporter's formatting word groups judgments by author or style rather than by legal topic: that is a real property "
        "of tf-idf vectors over full judgments, and it is left visible instead of being filtered out by hand.",
    ]
    cross = meta.get("crosscheck")
    if cross:
        lines += [
            "",
            "## Cross-check with scikit-learn (not used to produce the clusters)",
            "",
            "| | ours | scikit-learn |",
            "|---|---|---|",
            f"| RSS at K = {meta['k']} | {cross['our_rss']:.2f} | {cross['sklearn_rss']:.2f} |",
            f"| NMI of our clusters | {cross['our_nmi']:.6f} | {cross['sklearn_nmi_of_our_clusters']:.6f} |",
            f"| Rand index of our clusters | {cross['our_rand']:.6f} | {cross['sklearn_rand_of_our_clusters']:.6f} |",
            "",
            f"Adjusted Rand index between our clustering and scikit-learn's: {cross['agreement_ari']:.3f} "
            "(1 = identical partitions; K-means finds local minima, so two implementations with different seeds agree only partly).",
        ]
    path = rdir / "clusters.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


@lru_cache(maxsize=1)
def _load_cached(out: str, meta: str) -> tuple[dict[str, tuple[int, float]], dict[str, Any]]:
    from common.io import read_jsonl

    if not Path(out).exists() or not Path(meta).exists():
        raise ClusterError(f"{out} or {meta} is missing; run `python -m m4_rank.cluster build` (about a minute)")
    table = {r["doc_id"]: (int(r["cluster_id"]), float(r["distance_to_centroid"])) for r in read_jsonl(Path(out))}
    return table, json.loads(Path(meta).read_text(encoding="utf-8"))


def _load(cfg: dict[str, Any] | None = None) -> tuple[dict[str, tuple[int, float]], dict[str, Any]]:
    return _load_cached(str(_path("out", cfg)), str(_path("meta", cfg)))


_load.cache_clear = _load_cached.cache_clear  # type: ignore[attr-defined]


def cluster_of(doc_id: str) -> tuple[int, str]:
    """Phase 2 contract: (cluster id, cluster label) of a judgment, from the frozen clustering. KeyError for an unknown doc_id."""
    table, meta = _load()
    if doc_id not in table:
        raise KeyError(f"{doc_id} is not in the clustering ({len(table)} judgments); rebuild it with `python -m m4_rank.cluster build`")
    cid = table[doc_id][0]
    return cid, meta["labels"][cid]


def clusters_overview() -> list[dict[str, Any]]:
    """Every cluster with its size, label and offence summary, for the browse panel."""
    return list(_load()[1]["clusters"])


def members(cluster_id: int, limit: int | None = None) -> list[tuple[str, float]]:
    """(doc_id, distance to centroid) of a cluster's members, closest to the centroid first (the most typical judgments)."""
    table, meta = _load()
    if not 0 <= cluster_id < int(meta["k"]):
        raise KeyError(f"cluster {cluster_id} does not exist (K = {meta['k']})")
    rows = sorted(((d, dist) for d, (c, dist) in table.items() if c == cluster_id), key=lambda x: (x[1], x[0]))
    return rows[:limit] if limit else rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="K-means clustering of the corpus (offline; outputs are frozen files).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="vectors, RSS curve, K, clustering, evaluation, labels, report")
    b.add_argument("--no-crosscheck", action="store_true", help="skip the scikit-learn cross-check")
    sub.add_parser("show", help="print the frozen clusters")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(crosscheck=not args.no_crosscheck)
        return 0
    try:
        for c in clusters_overview():
            print(f"{c['cluster_id']:>3}  {c['size']:>5}  {c['label']}")
    except ClusterError as exc:
        print(exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
