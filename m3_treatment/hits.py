"""HITS (Kleinberg's hubs and authorities) on the citation graph, written by us (IIR 21.3), next to PageRank (graph.py).

On a citation graph, a good AUTHORITY is a judgment cited by many good hubs, and a good HUB is a judgment that cites many good
authorities: a hub is a judgment that surveys a doctrine (it walks through the leading cases), an authority is a leading case.

Query-dependent, as Kleinberg defined it:
  1. root set  = the top-N search() results for the query (M1's BM25, `m3_treatment.hits.root_n`);
  2. base set  = root set + every judgment a root judgment cites + up to d judgments that cite each root judgment
                 (d = `max_in_per_root`, Kleinberg's cap: one landmark cited by hundreds would otherwise pull the base set
                 away from the query's topic, the "topic drift" problem);
  3. on the subgraph induced by the base set, iterate until both vectors stop changing (or `iterations` is reached):
        a = A^T h      (authority of d = sum of the hub scores of the judgments that cite d)
        h = A a        (hub of d       = sum of the authority scores of the judgments d cites)
     normalising each vector to unit length after every step. a and h converge to the principal eigenvectors of A^T A and
     A A^T.

Edges: the same ones PageRank uses (`m3_treatment.authority.edge_labels`: followed and neutral citations, resolved, not appeal
history, no self-citations), each counted once per (citing, cited) pair and unweighted, as in Kleinberg's formulation. A
negative treatment is not an endorsement, so it is not a link here either.

The returned scores are divided by their maximum, so the top hub and the top authority score 1.0 and every score is in
[0, 1] (the same range as authority()); the ORDER is what the eigenvectors give. A base set with no edge has every score 0.

    python -m m3_treatment.hits compare     -> m3_treatment/reports/hits_vs_pagerank.{md,json} (dev queries only)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable


@dataclass
class Graph:
    out: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # citing -> cited
    inn: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # cited -> citing

    def add(self, src: str, dst: str) -> None:
        if src and dst and src != dst:
            self.out[src].add(dst)
            self.inn[dst].add(src)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.out.values())

    def edges(self) -> Iterable[tuple[str, str]]:
        for s, ds in self.out.items():
            for d in ds:
                yield s, d


def _m3() -> dict:
    from common.config import load_config

    return load_config()["m3_treatment"]


def graph_from_citations(records: Iterable[dict], edge_labels: Iterable[str]) -> Graph:
    labels = set(edge_labels)
    g = Graph()
    for r in records:
        if r.get("cited_doc") and not r.get("is_appeal_history") and r.get("label") in labels:
            g.add(r["citing_doc"], r["cited_doc"])
    return g


@lru_cache(maxsize=1)
def load_graph() -> Graph:
    """The citation graph from data/processed/citations.jsonl, read once per process (under a second on the full corpus)."""
    from common.config import resolve_path
    from common.io import read_jsonl

    path = resolve_path("citations")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m app.setup` (restores it) or `make m3`")
    return graph_from_citations(read_jsonl(path), _m3()["authority"]["edge_labels"])


def base_set(root: Iterable[str], g: Graph, max_in_per_root: int = 50) -> list[str]:
    """Root set + everything a root judgment cites + at most `max_in_per_root` citing judgments per root judgment.

    Which citing judgments are kept when there are more than d is fixed (sorted doc_id), so the result is reproducible.
    """
    root = list(dict.fromkeys(root))
    nodes = dict.fromkeys(root)
    for d in root:
        for c in sorted(g.out.get(d, ())):
            nodes.setdefault(c)
        for c in sorted(g.inn.get(d, ()))[:max_in_per_root]:
            nodes.setdefault(c)
    return list(nodes)


def _unit(v: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(x * x for x in v.values()))
    return {k: x / norm for k, x in v.items()} if norm else {k: 0.0 for k in v}


def _by_max(v: dict[str, float]) -> dict[str, float]:
    top = max(v.values(), default=0.0)
    return {k: (x / top if top > 0 else 0.0) for k, x in v.items()}


def hits_on(nodes: Iterable[str], g: Graph, iterations: int = 50, tol: float = 1e-8) -> tuple[dict[str, float], dict[str, float], int]:
    """HITS on the subgraph of `g` induced by `nodes`: (hubs, authorities, iterations run), each scaled to max 1."""
    nodes = list(dict.fromkeys(nodes))
    inside = set(nodes)
    out = {d: [c for c in g.out.get(d, ()) if c in inside] for d in nodes}
    inn: dict[str, list[str]] = {d: [] for d in nodes}
    for s, ds in out.items():
        for d in ds:
            inn[d].append(s)
    h = {d: 1.0 for d in nodes}
    a = {d: 1.0 for d in nodes}
    run = 0
    for run in range(1, iterations + 1):
        a_new = _unit({d: sum(h[s] for s in inn[d]) for d in nodes})
        h_new = _unit({d: sum(a_new[c] for c in out[d]) for d in nodes})
        delta = sum(abs(a_new[d] - a[d]) + abs(h_new[d] - h[d]) for d in nodes)
        a, h = a_new, h_new
        if delta < tol:
            break
    return _by_max(h), _by_max(a), run


def hits(root_doc_ids: list[str], iterations: int = 50) -> tuple[dict[str, float], dict[str, float]]:
    """(hubs, authorities) over the base set of `root_doc_ids`, each doc_id -> score in [0, 1] (Phase 2 contract).

    Every doc_id of the base set is a key of both dicts (root judgments with no citation edge score 0).
    """
    c = _m3()["hits"]
    g = load_graph()
    hubs, auths, _ = hits_on(base_set(root_doc_ids, g, c["max_in_per_root"]), g, iterations, c["tol"])
    return hubs, auths


# ----------------------------------------------------------------------------------------------------------------
# Comparison with PageRank
# ----------------------------------------------------------------------------------------------------------------
def top(scores: dict[str, float], k: int) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def overlap_at(a: dict[str, float], b: dict[str, float], k: int) -> float:
    """|top-k(a) & top-k(b)| / k."""
    return len(set(top(a, k)) & set(top(b, k))) / k if k else 0.0


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    """Kendall's tau-b between two score lists over the same items (ties allowed): (C - D) / sqrt((n0 - n1)(n0 - n2)).

    C and D count concordant and discordant pairs, n1 and n2 the pairs tied in x and in y. O(n^2); fine for base sets
    (hundreds of judgments) and for the judgments with at least one in-link (a few thousand).
    """
    n = len(x)
    if n != len(y):
        raise ValueError("lists differ in length")
    conc = disc = tie_x = tie_y = 0
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n):
            dx, dy = xi - x[j], yi - y[j]
            if dx == 0 and dy == 0:
                tie_x += 1
                tie_y += 1
            elif dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - tie_x) * (n0 - tie_y))
    return (conc - disc) / denom if denom else 0.0


def _titles() -> dict[str, str]:
    from common.config import resolve_path
    from common.io import read_jsonl

    path = resolve_path("doc_meta")
    return {r["doc_id"]: r.get("title", "") for r in read_jsonl(path)} if path.exists() else {}


def run_compare(top_k: int = 10) -> dict:
    """HITS against PageRank on the real graph: globally, and per DEV query (never the test queries)."""
    from common.config import ROOT, load_config, resolve_path
    from common.io import read_jsonl
    from common.providers import load_providers
    from m3_treatment.graph import pagerank_raw
    from m3_treatment.scores import authority

    m3 = _m3()
    hc = m3["hits"]
    t0 = time.perf_counter()
    load_graph.cache_clear()
    g = load_graph()
    t_graph = time.perf_counter() - t0
    nodes_all = [r["doc_id"] for r in read_jsonl(resolve_path("doc_meta"))]
    pr = pagerank_raw(((s, d, 1.0) for s, d in g.edges()), nodes=nodes_all, damping=m3["pagerank"]["damping"], tol=m3["pagerank"]["tol"], max_iter=m3["pagerank"]["max_iter"])
    shipped = {d: authority(d) for d in nodes_all}  # what the ranking fuses today: bench-weighted log PageRank
    titles = _titles()

    # Global: HITS on the whole graph (the textbook query-independent variant) against PageRank.
    t0 = time.perf_counter()
    g_hubs, g_auth, g_iter = hits_on(nodes_all, g, iterations=500, tol=hc["tol"])
    t_global = time.perf_counter() - t0
    cited = [d for d in nodes_all if g.inn.get(d)]
    glob = {
        "nodes": len(nodes_all),
        "edges": g.n_edges,
        "with_in_links": len(cited),
        "iterations": g_iter,
        "seconds": round(t_global, 3),
        "overlap": {k: overlap_at(g_auth, pr, k) for k in (10, 50, 100)},
        "overlap_vs_shipped_authority": {k: overlap_at(g_auth, shipped, k) for k in (10, 50, 100)},
        "kendall_tau_b_cited": kendall_tau_b([g_auth[d] for d in cited], [pr[d] for d in cited]),
        "top_authorities": [{"doc_id": d, "title": titles.get(d, ""), "hits": round(g_auth[d], 4), "pagerank_rank": top(pr, len(pr)).index(d) + 1} for d in top(g_auth, top_k)],
        "top_hubs": [{"doc_id": d, "title": titles.get(d, ""), "hub": round(g_hubs[d], 4), "cites": len(g.out.get(d, ()))} for d in top(g_hubs, top_k)],
    }

    # Per dev query: root set from the real search(), base set, HITS, compared with PageRank restricted to the base set.
    prov = load_providers()
    if "search" in prov.stubbed:
        raise RuntimeError("search() is served by a stub; HITS needs real search results for its root set")
    queries = [q for q in read_jsonl(ROOT / load_config()["paths"]["queries"]) if q["split"] == "dev"]
    per_query = []
    for q in queries:
        root = [h.doc_id for h in prov.search(q["text"], k=hc["root_n"])]
        t0 = time.perf_counter()
        base = base_set(root, g, hc["max_in_per_root"])
        hubs, auths, it = hits_on(base, g, hc["iterations"], hc["tol"])
        ms = (time.perf_counter() - t0) * 1000
        _h_conv, a_conv, _ = hits_on(base, g, 1000, hc["tol"])  # run to convergence, to check what the cap changes
        pr_base = {d: pr.get(d, 0.0) for d in base}
        sub_edges = sum(1 for d in base for c in g.out.get(d, ()) if c in auths)
        per_query.append(
            {
                "qid": q["qid"],
                "query": q["text"],
                "root": len(root),
                "base": len(base),
                "edges": sub_edges,
                "iterations": it,
                f"top{top_k}_same_as_converged": top(auths, top_k) == top(a_conv, top_k),
                "ms": round(ms, 1),
                f"overlap@{top_k}": overlap_at(auths, pr_base, top_k),
                "kendall_tau_b": kendall_tau_b([auths[d] for d in base], [pr_base[d] for d in base]),
                "top_authorities": [{"doc_id": d, "title": titles.get(d, ""), "score": round(auths[d], 4), "in_root": d in root} for d in top(auths, 5) if auths[d] > 0],
                "top_hubs": [{"doc_id": d, "title": titles.get(d, ""), "score": round(hubs[d], 4), "in_root": d in root} for d in top(hubs, 3) if hubs[d] > 0],
            }
        )
    out = {"graph_load_seconds": round(t_graph, 3), "config": hc, "global": glob, "per_query": per_query}
    rep = ROOT / m3["reports_dir"]
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "hits_vs_pagerank.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = compare_markdown(out, top_k)
    (rep / "hits_vs_pagerank.md").write_text(md, encoding="utf-8")
    out["markdown"] = md
    return out


def _cut(s: str, n: int = 70) -> str:
    s = " ".join(s.split()).replace("|", "/")
    return s if len(s) <= n else s[: n - 3] + "..."


def compare_markdown(out: dict, top_k: int) -> str:
    g = out["global"]
    pq = out["per_query"]
    lines = [
        "# HITS against PageRank on the LexShift citation graph",
        "",
        "Generated by `python -m m3_treatment.hits compare`. Graph: resolved, non-appeal citations labelled followed or neutral "
        f"(the PageRank edges), one unweighted edge per (citing, cited) pair: {g['nodes']} judgments, {g['edges']} edges, "
        f"{g['with_in_links']} judgments cited at least once. Loaded in {out['graph_load_seconds']} s. PageRank here is the same power "
        "iteration as `graph.py` on the same unweighted edges; `authority()` is the shipped bench-weighted score.",
        "",
        "## Whole graph (query-independent)",
        "",
        f"HITS converged in {g['iterations']} iterations ({g['seconds']} s).",
        "",
        "| Measure | k = 10 | k = 50 | k = 100 |",
        "|---|---|---|---|",
        "| Overlap of top-k authorities with PageRank | " + " | ".join(f"{g['overlap'][k]:.2f}" for k in (10, 50, 100)) + " |",
        "| Overlap with the shipped authority() | " + " | ".join(f"{g['overlap_vs_shipped_authority'][k]:.2f}" for k in (10, 50, 100)) + " |",
        "",
        f"Kendall tau-b between HITS authority and PageRank over the {g['with_in_links']} cited judgments: {g['kendall_tau_b_cited']:.3f}.",
        "",
        f"Top {len(g['top_authorities'])} HITS authorities:",
        "",
        "| # | Judgment | HITS | PageRank rank |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(g["top_authorities"], 1):
        lines.append(f"| {i} | {_cut(r['title']) or r['doc_id']} (`{r['doc_id']}`) | {r['hits']:.3f} | {r['pagerank_rank']} |")
    lines += ["", f"Top {len(g['top_hubs'])} HITS hubs (judgments that survey the leading cases):", "", "| # | Judgment | Hub | Corpus judgments cited |", "|---|---|---|---|"]
    for i, r in enumerate(g["top_hubs"], 1):
        lines.append(f"| {i} | {_cut(r['title']) or r['doc_id']} (`{r['doc_id']}`) | {r['hub']:.3f} | {r['cites']} |")
    lines += [
        "",
        f"## Per query (dev queries only; root set = top {out['config']['root_n']} of search(), at most {out['config']['max_in_per_root']} citing judgments per root)",
        "",
        f"| Query | Root | Base | Edges | Iter. | ms | Overlap@{top_k} with PageRank | Kendall tau-b |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in pq:
        lines.append(f"| {r['qid']}: {_cut(r['query'], 50)} | {r['root']} | {r['base']} | {r['edges']} | {r['iterations']} | {r['ms']} | {r[f'overlap@{top_k}']:.2f} | {r['kendall_tau_b']:.3f} |")
    if pq:
        ms = sorted(r["ms"] for r in pq)
        lines += [
            "",
            f"HITS per query (base set + iterations, graph already in memory): median {ms[len(ms) // 2]} ms, max {ms[-1]} ms. "
            f"Mean overlap@{top_k} {sum(r[f'overlap@{top_k}'] for r in pq) / len(pq):.2f}, mean Kendall tau-b {sum(r['kendall_tau_b'] for r in pq) / len(pq):.3f}. "
            f"Iter. = power iterations run; {out['config']['iterations']} (the cap) means the L1 change was still above "
            f"{out['config']['tol']:g} when it stopped. The top-{top_k} authorities, in order, equal those of a run to convergence "
            f"for {sum(r[f'top{top_k}_same_as_converged'] for r in pq)} of {len(pq)} queries.",
            "",
            "Top hubs per query (candidates for the demo's \"judgments that survey this doctrine\" panel):",
            "",
        ]
        for r in pq:
            hubs = "; ".join(f"{_cut(h['title'], 60) or h['doc_id']} ({h['score']:.2f}{', in root' if h['in_root'] else ''})" for h in r["top_hubs"]) or "none (no citation edge in the base set)"
            lines.append(f"- **{r['qid']}** {hubs}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m m3_treatment.hits", description="HITS hubs and authorities on the citation graph")
    ap.add_argument("cmd", choices=("compare",))
    ap.parse_args(argv)
    try:
        md = run_compare().pop("markdown")
        enc = sys.stdout.encoding or "utf-8"
        print(md.encode(enc, errors="replace").decode(enc))  # stdout may not be UTF-8 on Windows; the report file is
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
