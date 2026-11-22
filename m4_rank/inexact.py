"""Inexact top-K retrieval (IIR 7.1 and 7.2): find a small set A of contenders, score only A, return the best K of A.

    python -m m4_rank.inexact build     # champion lists, tiers, cluster-pruning leaders -> data/processed/inexact.pkl.gz (offline)
    python -m m4_rank.inexact report    # speed vs quality on the DEV queries -> m4_rank/reports/inexact.{md,json}

Exact top-K (M1's search()) scores every judgment that contains any query term: about 4,700 of the 4,819 judgments for a typical
query here, because legal queries carry words such as "section" that occur almost everywhere. The methods below shrink that set:

* champion lists (IIR 7.1.3, with the static quality of 7.1.4): for each term, the r postings with the highest
  g(d) + tf_part(t, d), built offline. A = the union of the query terms' champion lists.
* tiered index (IIR 7.2.1): tier 1 holds the judgments of high static quality (authority g(d) in the top `1 - tier_cut`, or a bench
  of at least `tier1_min_bench` judges), with its own posting lists. A = the tier-1 judgments containing a query term; only when that
  gives fewer than K does the search fall through to tier 2 (everything else).
* index elimination (IIR 7.1.2): leave out query terms whose df exceeds `elim_max_df_ratio * N` (low idf, long posting lists) and
  keep only judgments containing at least `elim_min_share` of the remaining terms (relaxed one term at a time when fewer than K
  qualify). Only the remaining terms are scored, which is where the saving comes from.
* cluster pruning (IIR 7.1.6): round(sqrt(N)) leaders drawn at random (seeded); every judgment follows its nearest leader by cosine
  in the tf-idf space of m4_rank.cluster. A query visits leaders from the most similar down and takes their followers until A
  holds at least K judgments.

g(d) is M3's authority() (PageRank over the citation graph, in [0, 1]); tf_part is BM25's saturating, length-normalised term
frequency tf / (tf + k1 (1 - b + b dl / avgdl)), also in [0, 1), so the two parts weigh alike. Neither is tuned.

Scoring A uses the same zone-weighted BM25 as M1's search() (M1's constants, idf, zone lengths and order of additions), so on the
exact candidate set the scores are identical to search()'s (a test checks it) and every difference in the table comes from the
choice of A alone. Work is counted as documents scored and postings scored, which do not depend on the machine, next to time.

Free-text queries only: a query with Boolean, phrase, proximity or wildcard operators, or with filters, is answered by M1's exact
search(), and `last_stats.fallback` says so. Nothing here is in the default config; `full_tiered` is an ablation with its cost shown.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.config import load_config, resolve_path  # noqa: E402
from common.schema import ZONES, Hit  # noqa: E402

METHODS = ("exact", "champion", "tiered", "elimination", "cluster")
FORMAT = 1
_LANE = 16
_LANE_MAX = (1 << _LANE) - 1


class InexactError(RuntimeError):
    """The inexact-retrieval artefact is missing, or was built from another index."""


def _settings(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return (cfg or load_config())["inexact"]


def _path(key: str, cfg: dict[str, Any] | None = None) -> Path:
    p = Path(_settings(cfg)[key])
    return p if p.is_absolute() else ROOT / p


@dataclass
class Stats:
    """What one inexact search did."""

    method: str
    terms: list[str] = field(default_factory=list)       # query terms scored
    candidates: int = 0                                  # |A|, judgments scored
    postings_scored: int = 0                             # (term, judgment) BM25 evaluations
    postings_read: int = 0                               # posting entries read to build A and to score it
    tier: int = 0                                        # tiered: the deepest tier used
    fallback: str = ""                                   # why the exact search() answered instead, if it did


# ----------------------------------------------------------------------------------------------------------------
# Zone-weighted BM25 over a candidate set, identical to M1's search()
# ----------------------------------------------------------------------------------------------------------------
class Scorer:
    """M1's zone-weighted BM25, vectorised over a candidate set with numpy. Reads M1's index through its public arrays."""

    def __init__(self, index) -> None:
        from m1_index.searcher import B, K1, ZONE_WEIGHTS

        self.ix = index
        self.k1, self.b = K1, B
        self.zone_w = [ZONE_WEIGHTS.get(z, 1.0) for z in ZONES]
        self.n = index.num_docs
        self.doc_count = len(index.doc_meta)  # M1's N in the idf
        self.t_start = np.frombuffer(index.t_start, dtype=np.uint32).astype(np.int64)
        self.p_doc = np.frombuffer(index.p_doc, dtype=np.uint32).astype(np.int64)
        self.p_zc = np.frombuffer(index.p_zc, dtype=np.uint64)
        lens = [[index.doc_lengths.get(d, {}).get(z, 0) for d in index.doc_ids] for z in ZONES]
        self.zone_len = np.array(lens, dtype=np.float64)                       # (zones, N)
        all_ids = list(index.doc_meta.keys())
        self.avg = [(sum(index.doc_lengths.get(d, {}).get(z, 0) for d in all_ids) / len(all_ids)) if all_ids else 1.0 for z in ZONES]
        self.total_len = np.frombuffer(index.doc_total_length, dtype=np.uint32).astype(np.float64)
        self.avgdl = float(self.total_len.mean()) if self.n else 1.0

    def term_id(self, term: str) -> int:
        from bisect import bisect_left

        t = bisect_left(self.ix.terms, term)
        return t if t < len(self.ix.terms) and self.ix.terms[t] == term else -1

    def df(self, t: int) -> int:
        return int(self.t_start[t + 1] - self.t_start[t])

    def idf(self, t: int) -> float:
        df = self.df(t)
        return 0.0 if df == 0 else math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def postings_of(self, t: int, cand: np.ndarray | None, stats: Stats | None = None) -> np.ndarray:
        """Posting indices of term t whose judgment is in `cand` (sorted doc numbers; None = all): probe each candidate by binary
        search when that reads less than scanning the list (M1's rule), else scan."""
        lo, hi = int(self.t_start[t]), int(self.t_start[t + 1])
        df = hi - lo
        if cand is None:
            if stats:
                stats.postings_read += df
            return np.arange(lo, hi)
        if len(cand) * max(1, df.bit_length()) < df:
            pos = np.searchsorted(self.p_doc[lo:hi], cand) + lo
            ok = (pos < hi)
            pos = pos[ok]
            pos = pos[self.p_doc[pos] == cand[ok]]
            if stats:
                stats.postings_read += len(cand) * max(1, df.bit_length())
            return pos
        if stats:
            stats.postings_read += df
        docs = self.p_doc[lo:hi]
        return np.flatnonzero(np.isin(docs, cand, assume_unique=True)) + lo

    def zone_tfs(self, idx: np.ndarray) -> np.ndarray:
        """(zones, len(idx)) term frequencies per zone; saturated lanes are read exactly from M1's overflow table."""
        zc = self.p_zc[idx]
        lanes = np.stack([((zc >> np.uint64(_LANE * z)) & np.uint64(_LANE_MAX)).astype(np.int64) for z in range(len(ZONES))])
        sat = np.flatnonzero((lanes == _LANE_MAX).any(axis=0))
        for j in sat:
            lanes[:, j] = self.ix.zc_overflow[int(idx[j])]
        return lanes

    def score(self, terms: Sequence[str], cand: np.ndarray | None, stats: Stats | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(doc numbers, totals, per-zone rows) for the judgments of `cand` (None = every judgment containing a term) that contain at
        least one of `terms`. Same arithmetic, in the same order, as M1's `_zone_scores` and `_total`."""
        rows = np.zeros((len(ZONES), self.n), dtype=np.float64)
        touched = np.zeros(self.n, dtype=bool)
        k1p1 = self.k1 + 1.0
        for term in terms:
            t = self.term_id(term)
            if t < 0:
                continue
            idx = self.postings_of(t, cand, stats)
            if not len(idx):
                continue
            nums = self.p_doc[idx]
            tfs = self.zone_tfs(idx)
            idf = self.idf(t)
            for z in range(len(ZONES)):
                tf = tfs[z]
                has = tf > 0
                if not has.any():
                    continue
                n_z, tf_z = nums[has], tf[has].astype(np.float64)
                rows[z, n_z] += idf * ((tf_z * k1p1) / (tf_z + self.k1 * (1.0 - self.b + self.b * (self.zone_len[z, n_z] / self.avg[z]))))
            touched[nums] = True
            if stats:
                stats.postings_scored += len(idx)
        docs = np.flatnonzero(touched)
        total = np.zeros(len(docs), dtype=np.float64)
        for z in range(len(ZONES)):
            total = total + self.zone_w[z] * rows[z, docs]
        return docs, total, rows[:, docs]

    def top_hits(self, docs: np.ndarray, total: np.ndarray, rows: np.ndarray, k: int) -> list[Hit]:
        """The k best as Hits, ties by doc_id (M1's top_k order)."""
        ids = self.ix.doc_ids
        order = sorted(range(len(docs)), key=lambda i: (-total[i], ids[docs[i]]))[:k]
        return [Hit(doc_id=ids[docs[i]], rel=float(total[i]), zone_scores={z: float(rows[zi, i]) for zi, z in enumerate(ZONES)})
                for i in order]

    def tf_part(self, idx: np.ndarray) -> np.ndarray:
        """BM25's saturating, length-normalised tf in [0, 1) for posting indices `idx` (all zones together)."""
        tf = np.diff(np.frombuffer(self.ix.p_off, dtype=np.uint32).astype(np.int64))[idx].astype(np.float64)
        dl = self.total_len[self.p_doc[idx]]
        return tf / (tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl))


# ----------------------------------------------------------------------------------------------------------------
# The offline artefact
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Artefact:
    fingerprint: dict[str, int]
    settings: dict[str, Any]
    g: np.ndarray                 # (N,) static quality g(d) = authority
    tier1: np.ndarray             # (N,) bool
    champ_start: np.ndarray       # (T+1,) CSR over terms
    champ_doc: np.ndarray         # champion doc numbers, increasing within a term
    tier_start: np.ndarray        # (T+1,)
    tier_doc: np.ndarray          # tier-1 doc numbers per term, increasing
    vocab: list[str]              # cluster-pruning term space (m4_rank.cluster vocabulary)
    vocab_idf: np.ndarray
    leaders: np.ndarray           # (L,) doc numbers
    leader_vecs: np.ndarray       # (L, M) float32, unit length
    follower_of: np.ndarray       # (N,) leader position of each judgment
    seconds: float = 0.0


def fingerprint(index) -> dict[str, int]:
    return {"docs": int(index.num_docs), "terms": len(index.terms), "postings": len(index.p_doc)}


def _csr(lists_lengths: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(lists_lengths)]).astype(np.int64)


def build(cfg: dict[str, Any] | None = None, log=print) -> Artefact:
    """Champion lists, tier-1 postings and cluster-pruning leaders from M1's index, M3's authority and M4's doc_meta."""
    cfg = cfg or load_config()
    s = _settings(cfg)
    t0 = time.perf_counter()
    from m1_index.searcher import RankedSearchEngine

    index = RankedSearchEngine.get_instance().index
    sc = Scorer(index)
    n, n_terms = index.num_docs, len(index.terms)

    from common.providers import load_providers

    providers = load_providers(cfg)
    if "authority" in providers.stubbed:
        raise InexactError("authority() is a fixed-value stub: g(d) would not be real. Set stubs.authority to false first.")
    g = np.array([float(providers.authority(d)) for d in index.doc_ids], dtype=np.float64)

    bench = {}
    meta_path = resolve_path("doc_meta", cfg)
    if meta_path.exists():
        from common.io import read_jsonl

        bench = {r["doc_id"]: r.get("bench_size") for r in read_jsonl(meta_path)}
    bench_arr = np.array([bench.get(d) or 0 for d in index.doc_ids], dtype=np.int64)
    cut = float(np.quantile(g, float(s["tier_cut"])))
    tier1 = (g >= cut) | (bench_arr >= int(s["tier1_min_bench"]))
    log(f"g(d) = authority; tier 1: authority >= {cut:.4f} (quantile {s['tier_cut']}) or bench >= {s['tier1_min_bench']}: "
        f"{int(tier1.sum())} of {n} judgments")

    r = int(s["champion_r"])
    key = g[sc.p_doc] + sc.tf_part(np.arange(len(sc.p_doc)))
    champ_lists, champ_len, tier_len = [], np.zeros(n_terms, dtype=np.int64), np.zeros(n_terms, dtype=np.int64)
    in_tier1 = tier1[sc.p_doc]
    tier_doc = sc.p_doc[in_tier1]
    seg = np.add.reduceat(in_tier1.astype(np.int64), sc.t_start[:-1]) if len(sc.p_doc) else np.zeros(n_terms, dtype=np.int64)
    empty = sc.t_start[:-1] == sc.t_start[1:]
    tier_len[:] = np.where(empty, 0, seg)
    for t in range(n_terms):
        lo, hi = int(sc.t_start[t]), int(sc.t_start[t + 1])
        if hi - lo <= r:
            docs = sc.p_doc[lo:hi]
        else:
            part = np.argpartition(-key[lo:hi], r - 1)[:r]
            docs = np.sort(sc.p_doc[lo:hi][part])
        champ_lists.append(docs)
        champ_len[t] = len(docs)
    champ_doc = np.concatenate(champ_lists) if champ_lists else np.zeros(0, dtype=np.int64)
    log(f"champion lists: r = {r}, {len(champ_doc)} entries (the index has {len(sc.p_doc)} postings); "
        f"tier-1 postings: {len(tier_doc)}")

    from m4_rank.cluster import build_vectors

    vec = build_vectors(index, cfg)
    n_lead = int(s["leaders"]) or int(round(math.sqrt(n)))
    rng = np.random.default_rng(int(s["seed"]))
    leaders = np.sort(rng.choice(n, size=n_lead, replace=False))
    leader_vecs = vec.X[leaders].astype(np.float32)
    follower_of = (vec.X @ leader_vecs.T).argmax(axis=1).astype(np.int64)
    follower_of[leaders] = np.arange(n_lead)  # a leader follows itself
    sizes = np.bincount(follower_of, minlength=n_lead)
    log(f"cluster pruning: {n_lead} leaders, followers per leader min {sizes.min()} / median {int(np.median(sizes))} / max {sizes.max()}")

    art = Artefact(
        fingerprint=fingerprint(index),
        settings={k: s[k] for k in ("champion_r", "tier_cut", "tier1_min_bench", "leaders", "seed")} | {"tier_cut_value": cut},
        g=g, tier1=tier1, champ_start=_csr(champ_len), champ_doc=champ_doc.astype(np.int64),
        tier_start=_csr(tier_len), tier_doc=tier_doc.astype(np.int64), vocab=vec.terms, vocab_idf=vec.idf.astype(np.float64),
        leaders=leaders.astype(np.int64), leader_vecs=leader_vecs, follower_of=follower_of,
        seconds=round(time.perf_counter() - t0, 1),
    )
    out = _path("artefact", cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    try:
        with gzip.open(tmp, "wb", compresslevel=3) as fh:
            pickle.dump({"format": FORMAT, **art.__dict__}, fh, protocol=5)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {art.seconds}s")
    _ENGINE.reset()
    return art


def load_artefact(index, cfg: dict[str, Any] | None = None) -> Artefact:
    path = _path("artefact", cfg)
    if not path.exists():
        raise InexactError(f"{path} is missing; run `python -m m4_rank.inexact build` (under a minute)")
    with gzip.open(path, "rb") as fh:
        payload = pickle.load(fh)
    if payload.pop("format", None) != FORMAT:
        raise InexactError(f"{path} is in another format; rebuild it with `python -m m4_rank.inexact build`")
    art = Artefact(**payload)
    if art.fingerprint != fingerprint(index):
        raise InexactError(f"{path} was built from another index ({art.fingerprint} vs {fingerprint(index)}); "
                           "rebuild it with `python -m m4_rank.inexact build`")
    return art


# ----------------------------------------------------------------------------------------------------------------
# Query time
# ----------------------------------------------------------------------------------------------------------------
def free_text_terms(query: str) -> list[str] | None:
    """The distinct index terms of a free-text query (as M1 tokenizes it), or None when the query uses operators."""
    from m1_index.query_parser import TermNode, parse_query
    from m1_index.tokenizer import tokenize

    try:
        ast = parse_query(query)
    except (ValueError, RecursionError):
        ast = None
    if ast is not None and not isinstance(ast, TermNode):
        return None
    return list(dict.fromkeys(tokenize(query)))


class Engine:
    def __init__(self) -> None:
        self._lock = threading.RLock()  # artefact() calls scorer() while holding it
        self.reset()

    def reset(self) -> None:
        self._scorer: Scorer | None = None
        self._art: Artefact | None = None
        self.last_stats = Stats("none")

    def scorer(self) -> Scorer:
        if self._scorer is None:
            with self._lock:
                if self._scorer is None:
                    from m1_index.searcher import RankedSearchEngine

                    self._scorer = Scorer(RankedSearchEngine.get_instance().index)
        return self._scorer

    def artefact(self) -> Artefact:
        if self._art is None:
            with self._lock:
                if self._art is None:
                    self._art = load_artefact(self.scorer().ix)
        return self._art

    # --- candidate sets ---
    def _union(self, start: np.ndarray, docs: np.ndarray, tids: list[int], stats: Stats) -> np.ndarray:
        parts = [docs[start[t]:start[t + 1]] for t in tids]
        stats.postings_read += sum(len(p) for p in parts)
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def candidates(self, method: str, terms: list[str], k: int, stats: Stats) -> tuple[np.ndarray | None, list[str]]:
        """(A as sorted doc numbers, the terms to score). A = None means every judgment containing a term (exact)."""
        sc = self.scorer()
        tids = [t for t in (sc.term_id(x) for x in terms) if t >= 0]
        if method == "exact":
            return None, terms
        art = self.artefact()
        if method == "champion":
            return self._union(art.champ_start, art.champ_doc, tids, stats), terms
        if method == "tiered":
            stats.tier = 1
            a = self._union(art.tier_start, art.tier_doc, tids, stats)
            if len(a) >= k:
                return a, terms
            stats.tier = 2
            return None, terms
        if method == "elimination":
            max_df = float(_settings()["elim_max_df_ratio"]) * sc.n
            kept = [t for t in tids if sc.df(t) <= max_df]
            if not kept and tids:
                kept = [min(tids, key=lambda t: (sc.df(t), t))]  # the rarest term survives
            counts = np.zeros(sc.n, dtype=np.int64)
            for t in kept:
                lo, hi = int(sc.t_start[t]), int(sc.t_start[t + 1])
                counts[sc.p_doc[lo:hi]] += 1
                stats.postings_read += hi - lo
            need = max(1, math.ceil(float(_settings()["elim_min_share"]) * len(kept)))
            a = np.flatnonzero(counts >= need)
            while len(a) < k and need > 1:  # relax one term at a time until K judgments qualify
                need -= 1
                a = np.flatnonzero(counts >= need)
            return a, [sc.ix.terms[t] for t in kept]
        if method == "cluster":
            idx = {w: i for i, w in enumerate(art.vocab)}
            q = np.zeros(len(art.vocab), dtype=np.float64)
            for x in terms:
                if x in idx:
                    q[idx[x]] = art.vocab_idf[idx[x]]  # ltc query weight with tf = 1 (terms are distinct)
            if not q.any():
                stats.fallback = "no query term is in the cluster-pruning vocabulary"
                return None, terms
            sims = art.leader_vecs.astype(np.float64) @ (q / np.linalg.norm(q))
            order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
            chosen, size = [], 0
            sizes = np.bincount(art.follower_of, minlength=len(art.leaders))
            for li in order:
                chosen.append(li)
                size += int(sizes[li])
                if size >= k:
                    break
            stats.postings_read += len(art.leaders)  # one similarity per leader
            return np.flatnonzero(np.isin(art.follower_of, chosen)), terms
        raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")

    def search(self, query: str, k: int = 100, method: str = "tiered", filters: dict | None = None) -> list[Hit]:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")
        stats = Stats(method)
        if k <= 0 or not query.strip():
            self.last_stats = stats
            return []
        terms = free_text_terms(query)
        if terms is None or filters:
            from m1_index import search as exact_search

            stats.fallback = "filters" if filters else "query operators (Boolean, phrase, proximity or wildcard)"
            self.last_stats = stats
            return exact_search(query, k=k, filters=filters)
        cand, score_terms = self.candidates(method, terms, k, stats)
        stats.terms = list(score_terms)
        docs, total, rows = self.scorer().score(score_terms, cand, stats)
        stats.candidates = len(docs)
        self.last_stats = stats
        return self.scorer().top_hits(docs, total, rows, k)


_ENGINE = Engine()


def inexact_search(query: str, k: int = 100, method: str = "tiered", filters: dict | None = None) -> list[Hit]:
    """Top-k Hits like M1's search(), from the reduced candidate set of `method` (exact | champion | tiered | elimination | cluster)."""
    return _ENGINE.search(query, k=k, method=method, filters=filters)


def last_stats() -> Stats:
    return _ENGINE.last_stats


def search_fn(method: str) -> Callable[..., list[Hit]]:
    """A callable with search()'s signature that answers with `method`: what rank() uses as the candidate source."""
    if method not in METHODS:
        raise ValueError(f"unknown candidate source {method!r}; choose one of {METHODS}")

    def _search(query: str, k: int = 100, filters: dict | None = None) -> list[Hit]:
        return inexact_search(query, k=k, method=method, filters=filters)

    _search.__name__ = f"search_{method}"
    return _search


# ----------------------------------------------------------------------------------------------------------------
# Speed vs quality on the DEV queries
# ----------------------------------------------------------------------------------------------------------------
def _timed(fn: Callable[[], Any], repeat: int) -> tuple[Any, float]:
    best, out = math.inf, None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


def report(repeat: int = 5, k: int = 20, cfg: dict[str, Any] | None = None, log=print) -> dict[str, Any]:
    """Every method on every judged DEV query: time, work, overlap with exact, and nDCG@10 of the BM25 ranking it returns."""
    from eval.loaders import load_qrels, load_queries
    from eval.metrics import judged_at_k, ndcg_at_k
    from m1_index import search as m1_search

    cfg = cfg or load_config()
    queries = [q for q in load_queries() if q.split == "dev"]
    qrels = load_qrels(known_qids={q.qid for q in queries})
    queries = [q for q in queries if qrels.get(q.qid)]
    if not queries:
        raise InexactError("no judged dev queries: the speed vs quality table needs eval/queries.jsonl and eval/qrels.tsv")
    gain = str(cfg["evaluation"]["gain"])
    _ENGINE.artefact()
    m1_search("warm up", k=1)

    per_query: list[dict[str, Any]] = []
    for q in queries:
        _, m1_t = _timed(lambda: m1_search(q.text, k=k), repeat)
        exact_ids = None
        for method in METHODS:
            hits, secs = _timed(lambda: inexact_search(q.text, k=k, method=method), repeat)
            st = last_stats()
            ids = [h.doc_id for h in hits]
            if method == "exact":
                exact_ids = ids
            per_query.append({
                "qid": q.qid, "method": method, "ms": secs * 1000, "m1_search_ms": m1_t * 1000,
                "candidates": st.candidates, "postings_scored": st.postings_scored, "postings_read": st.postings_read,
                "tier": st.tier, "fallback": st.fallback,
                "overlap@10": len(set(ids[:10]) & set((exact_ids or [])[:10])) / 10,
                "nDCG@10": ndcg_at_k(ids, qrels[q.qid], 10, gain), "judged@10": judged_at_k(ids, qrels[q.qid], 10),
            })
        log(f"  {q.qid}: done")

    summary = []
    for method in METHODS:
        rows = [r for r in per_query if r["method"] == method]
        nd = [r["nDCG@10"] for r in rows if r["nDCG@10"] is not None]
        summary.append({
            "method": method, "queries": len(rows),
            "median_ms": statistics.median(r["ms"] for r in rows),
            "mean_candidates": statistics.mean(r["candidates"] for r in rows),
            "mean_postings_scored": statistics.mean(r["postings_scored"] for r in rows),
            "mean_postings_read": statistics.mean(r["postings_read"] for r in rows),
            "mean_overlap@10": statistics.mean(r["overlap@10"] for r in rows),
            "mean_nDCG@10": statistics.mean(nd) if nd else None,
            "mean_judged@10": statistics.mean(r["judged@10"] for r in rows),
            "fallbacks": sum(1 for r in rows if r["fallback"]),
            "tier2": sum(1 for r in rows if r["tier"] == 2),
        })
    exact = summary[0]
    for row in summary:
        row["speedup"] = exact["median_ms"] / row["median_ms"] if row["median_ms"] else None
        row["work_ratio"] = row["mean_postings_scored"] / exact["mean_postings_scored"] if exact["mean_postings_scored"] else None
        row["nDCG_change"] = (row["mean_nDCG@10"] - exact["mean_nDCG@10"]) if (row["mean_nDCG@10"] is not None and exact["mean_nDCG@10"] is not None) else None
    out = {"k": k, "repeat": repeat, "queries": [q.qid for q in queries], "settings": _ENGINE.artefact().settings,
           "m1_search_median_ms": statistics.median(r["m1_search_ms"] for r in per_query if r["method"] == "exact"),
           "summary": summary, "per_query": per_query}
    write_report(out, cfg)
    return out


def write_report(out: dict[str, Any], cfg: dict[str, Any] | None = None) -> Path:
    rdir = _path("report_dir", cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "inexact.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    s = out["settings"]

    def f(x, spec=".3f"):
        return "n/a" if x is None else format(x, spec)

    lines = [
        "# Inexact top-K: speed vs quality",
        "",
        f"Generated by `python -m m4_rank.inexact report` on the {len(out['queries'])} judged DEV queries "
        f"({', '.join(out['queries'])}); top-{out['k']} retrieval, BM25 ranking only (no fusion), best of {out['repeat']} timings per "
        "query, median over queries. Nothing here is tuned: the parameters are fixed in `common/config.yaml` (`inexact:`).",
        "",
        f"Settings: champion r = {s['champion_r']}; tier 1 = authority >= {s['tier_cut_value']:.4f} (quantile {s['tier_cut']}) or "
        f"bench >= {s['tier1_min_bench']}; cluster pruning with {('sqrt(N)' if not s['leaders'] else s['leaders'])} leaders, seed {s['seed']}.",
        "",
        "| method | median ms | speed-up | judgments scored | postings scored | work vs exact | overlap@10 with exact | nDCG@10 | change | judged@10 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out["summary"]:
        lines.append(f"| {r['method']} | {r['median_ms']:.1f} | {f(r['speedup'], '.2f')}x | {r['mean_candidates']:.0f} | "
                     f"{r['mean_postings_scored']:.0f} | {f(r['work_ratio'], '.0%')} | {r['mean_overlap@10']:.2f} | "
                     f"{f(r['mean_nDCG@10'])} | {f(r['nDCG_change'], '+.3f')} | {r['mean_judged@10']:.2f} |")
    notes = [r for r in out["summary"] if r["fallbacks"] or r["tier2"]]
    lines += [
        "",
        f"For reference, M1's own `search()` (the live exact path, pure Python) takes {out['m1_search_median_ms']:.1f} ms median on "
        "the same queries; the `exact` row is the same zone-weighted BM25 over the same candidates, vectorised, so the "
        "speed-ups compare like with like.",
        "",
        "How to read it. *Postings scored* and *judgments scored* are the machine-independent work; the time also includes "
        "building A. *overlap@10* is the share of exact BM25's top 10 that the method keeps. nDCG@10 uses the dev qrels, "
        "which were pooled from exact systems: a method that surfaces judgments nobody judged is scored as if they were "
        "irrelevant (see judged@10), so its nDCG change is a pessimistic estimate.",
    ]
    for r in notes:
        if r["tier2"]:
            lines.append(f"* tiered fell through to tier 2 on {r['tier2']} of {r['queries']} queries (fewer than K tier-1 matches).")
        if r["fallbacks"]:
            lines.append(f"* {r['method']} answered {r['fallbacks']} queries with the exact search (see `inexact.json`, `fallback`).")
    lines += ["", "## Per query", "", "| qid | method | ms | judgments scored | overlap@10 | nDCG@10 |", "|---|---|---|---|---|---|"]
    for r in out["per_query"]:
        lines.append(f"| {r['qid']} | {r['method']} | {r['ms']:.1f} | {r['candidates']} | {r['overlap@10']:.1f} | {f(r['nDCG@10'])} |")
    path = rdir / "inexact.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inexact top-K: build the artefact, or measure speed vs quality on dev queries.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="champion lists, tiers and leaders (offline, frozen file)")
    rp = sub.add_parser("report", help="speed vs quality table on the judged dev queries")
    rp.add_argument("--repeat", type=int, default=5)
    rp.add_argument("--k", type=int, default=20)
    q = sub.add_parser("query", help="run one query with a method and print the hits and the work")
    q.add_argument("text")
    q.add_argument("--method", default="tiered", choices=METHODS)
    q.add_argument("-k", type=int, default=10)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            build()
        elif args.cmd == "report":
            out = report(repeat=args.repeat, k=args.k)
            print(Path(_path("report_dir")) / "inexact.md")
            for r in out["summary"]:
                print(f"{r['method']:>12}  {r['median_ms']:7.1f} ms  scored {r['mean_candidates']:6.0f}  overlap@10 {r['mean_overlap@10']:.2f}  "
                      f"nDCG@10 {r['mean_nDCG@10'] if r['mean_nDCG@10'] is None else round(r['mean_nDCG@10'], 3)}")
        else:
            for h in inexact_search(args.text, k=args.k, method=args.method):
                print(f"{h.rel:9.3f}  {h.doc_id}")
            print(last_stats())
    except InexactError as exc:
        print(exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
