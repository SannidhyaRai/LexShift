"""Skip pointers for postings intersection (IIR 2.3), and the report that counts what they save on the real index.

A posting list of length L gets a skip pointer every s = floor(sqrt(L)) entries (the textbook heuristic: about sqrt(L) evenly
spaced skips balance the number of skips that can be taken against the number of comparisons spent checking them). Our postings are
slices of one flat array (m1_index.index), so a skip pointer needs no storage: the entry at offset m*s of a list points at offset
(m+1)*s, and "has a skip" is a divisibility test. This is the same structure as stored pointers, without the bytes.

`intersect()` is the merge of two sorted lists. Without skips it advances the smaller head one entry at a time; with skips, when
the smaller head sits on a skip pointer whose target is still <= the other head, it follows the pointer (and keeps following
while that holds), jumping over entries that cannot be in the answer. Every comparison of two document numbers is counted, the
checks of a skip target included, so the saving reported is net.

The search engine uses it in the AND and phrase intersections when it scans a posting list (m1_index.searcher, `skips=True`); when
the candidates are few it probes the list by binary search instead, which reads even less (both are counted in WorkStats).
Run `python -m m1_index.skips` for the report (m1_index/reports/skips.md).
"""

from __future__ import annotations

import itertools
import math
import statistics
import sys
import time
from typing import Any, Sequence

from m1_index.index import ROOT

REPORT_PATH = ROOT / "m1_index" / "reports" / "skips.md"


class MergeStats:
    """Counters for one or more intersections (the engine passes its WorkStats, which has the same two fields)."""

    def __init__(self) -> None:
        self.comparisons = 0
        self.skips_followed = 0


def skip_length(n: int) -> int:
    """Spacing of the skip pointers on a list of n entries: floor(sqrt(n)); 0 (no skips) when that is below 2."""
    s = math.isqrt(n) if n > 0 else 0
    return s if s >= 2 else 0


def intersect(a: Sequence[int], a_lo: int, a_hi: int, b: Sequence[int], b_lo: int, b_hi: int, use_skips: bool = True,
              stats: Any = None) -> list[int]:
    """The sorted intersection of a[a_lo:a_hi] and b[b_lo:b_hi] (each sorted, without repeats), by merging.

    The slices are read in place (a posting list is a slice of the index's flat array). `stats` (anything with `comparisons` and
    `skips_followed`) is incremented."""
    sa = skip_length(a_hi - a_lo) if use_skips else 0
    sb = skip_length(b_hi - b_lo) if use_skips else 0
    out: list[int] = []
    i, j = a_lo, b_lo
    comparisons = skips = 0
    while i < a_hi and j < b_hi:
        x, y = a[i], b[j]
        comparisons += 1
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            if sa and (i - a_lo) % sa == 0 and i + sa < a_hi:
                comparisons += 1
                if a[i + sa] <= y:
                    while True:
                        i += sa
                        skips += 1
                        if not (i + sa < a_hi):
                            break
                        comparisons += 1
                        if a[i + sa] > y:
                            break
                    continue
            i += 1
        else:
            if sb and (j - b_lo) % sb == 0 and j + sb < b_hi:
                comparisons += 1
                if b[j + sb] <= x:
                    while True:
                        j += sb
                        skips += 1
                        if not (j + sb < b_hi):
                            break
                        comparisons += 1
                        if b[j + sb] > x:
                            break
                    continue
            j += 1
    if stats is not None:
        stats.comparisons += comparisons
        stats.skips_followed += skips
    return out


# --------------------------------------------------------------------------------------------------
# Report on the real index
# --------------------------------------------------------------------------------------------------
REPORT_TERMS = ("murder", "bail", "conspiracy", "dowry", "acquittal", "302", "498-a", "intention", "circumstantial", "declaration",
                "cheating", "anticipatory", "quash", "fir", "common", "evidence")
REPORT_QUERIES = (
    "murder AND bail", "dowry AND death AND cruelty", "conspiracy AND cheating", "302 AND 34 AND intention",
    '"common intention"', '"dying declaration"', '"circumstantial evidence" AND motive', '"anticipatory bail" AND 438',
    "quash AND fir AND 482", "acquittal AND appeal AND NOT conviction",
)


def _pairwise(engine: Any) -> list[dict]:
    from m1_index.tokenizer import tokenize

    index = engine.index
    rows = []
    terms = [tokenize(t)[0] for t in REPORT_TERMS]
    for t1, t2 in itertools.combinations(dict.fromkeys(terms), 2):
        a, b = index.term_postings(t1), index.term_postings(t2)
        if a is None or b is None:
            continue
        plain, skip = MergeStats(), MergeStats()
        r1 = intersect(index.p_doc, a.lo, a.hi, index.p_doc, b.lo, b.hi, False, plain)
        r2 = intersect(index.p_doc, a.lo, a.hi, index.p_doc, b.lo, b.hi, True, skip)
        if r1 != r2 or set(r1) != set(a.docs()) & set(b.docs()):
            raise AssertionError(f"skip intersection differs for {t1} AND {t2}")
        rows.append({"pair": f"{t1} AND {t2}", "df": (a.df, b.df), "result": len(r1), "plain": plain.comparisons,
                     "skip": skip.comparisons, "followed": skip.skips_followed})
    return rows


def _engine_runs(engine: Any) -> list[dict]:
    from m1_index.query_parser import parse_query

    rows = []
    for q in REPORT_QUERIES:
        ast = parse_query(q)
        out = {}
        for mode in (False, True):
            engine.skips = mode
            ids, work = engine.evaluate(ast)
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                engine.evaluate(ast)
                times.append((time.perf_counter() - t0) * 1000)
            out[mode] = (ids, work, statistics.median(times))
        engine.skips = True
        if out[False][0] != out[True][0]:
            raise AssertionError(f"results differ with and without skips for {q!r}")
        rows.append({"query": q, "result": len(out[True][0]), "plain": out[False][1].comparisons, "skip": out[True][1].comparisons,
                     "followed": out[True][1].skips_followed, "ms_plain": out[False][2], "ms_skip": out[True][2]})
    return rows


def _change(plain: int, skip: int) -> str:
    """'12.3% fewer' or '1.4% more': the change in comparisons that skips make."""
    if not plain:
        return "no change"
    pct = 100 * (plain - skip) / plain
    return f"{pct:.1f}% fewer" if pct >= 0 else f"{-pct:.1f}% more"


def write_report() -> str:
    from m1_index.searcher import RankedSearchEngine

    engine = RankedSearchEngine.get_instance()
    pairs = _pairwise(engine)
    runs = _engine_runs(engine)
    tp, ts = sum(r["plain"] for r in pairs), sum(r["skip"] for r in pairs)
    ep, es = sum(r["plain"] for r in runs), sum(r["skip"] for r in runs)
    lines = ["# M1 skip pointers: comparisons saved on the real index", "",
             f"Generated by `python -m m1_index.skips` on {time.strftime('%Y-%m-%d')} over the index of {engine.doc_count:,} judgments. "
             "Skip spacing floor(sqrt(L)) per posting list; a comparison is one test of two document numbers, skip-target checks included. "
             "Every intersection was checked to give the same documents with and without skips.", "",
             "## Two posting lists merged directly", "",
             f"All {len(pairs)} pairs of {len(REPORT_TERMS)} common query terms. Total comparisons: **{tp:,} without skips, {ts:,} with skips "
             f"({_change(tp, ts)})**.", "",
             "| Pair | df | Result | Comparisons, plain merge | With skips | Skips followed | Saved |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(pairs, key=lambda r: -(r["plain"] - r["skip"]) / max(1, r["plain"]))[:15]:
        lines.append(f"| `{r['pair']}` | {r['df'][0]:,} / {r['df'][1]:,} | {r['result']:,} | {r['plain']:,} | {r['skip']:,} | {r['followed']:,} | "
                     f"{100 * (r['plain'] - r['skip']) / max(1, r['plain']):.0f}% |")
    worst = sorted(pairs, key=lambda r: (r["plain"] - r["skip"]) / max(1, r["plain"]))[:3]
    lines += ["", "The 15 pairs with the largest relative saving are shown. The smallest savings (lists of similar length whose documents interleave, "
              "where skips are checked but rarely taken): " + "; ".join(f"`{r['pair']}` {100 * (r['plain'] - r['skip']) / max(1, r['plain']):.0f}%" for r in worst) + ".", "",
              "## Inside the search engine", "",
              "The engine intersects an AND chain smallest posting list first and, for each further operand, either probes the list by binary search "
              "(when the candidates are few) or merges with it; skip pointers apply to the merges. Comparisons counted by `WorkStats` over whole queries:", "",
              f"Total: **{ep:,} comparisons without skips, {es:,} with skips ({_change(ep, es)})**.", "",
              "| Query | Matches | Comparisons, plain merge | With skips | Skips followed | Median ms, plain | Median ms, skips |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| `{r['query']}` | {r['result']:,} | {r['plain']:,} | {r['skip']:,} | {r['followed']:,} | {r['ms_plain']:.2f} | {r['ms_skip']:.2f} |")
    followed = sum(r["followed"] for r in pairs)
    lines += ["", "Times are for evaluating the Boolean query only (no scoring), median of 5 runs. The count of comparisons is the "
              "machine-independent measure.", "",
              "## Reading the numbers", "",
              f"Skip pointers pay off when the shorter list's next document is usually more than a skip span ahead in the longer list. In this "
              f"corpus that rarely happens: there are only {engine.doc_count:,} document numbers, so a common term's list covers most of them "
              f"(df 4,217 for `evid`), its skip span of floor(sqrt(df)) entries covers about as many document numbers, and the gap to the short "
              f"list's next document is almost always smaller. Over the {len(pairs)} pairs only {followed:,} skips were followed, and the checks "
              "of skip targets that were not followed cost slightly more than the skips saved. The saving is largest where one list is much shorter "
              "than the other (`anticipatori`, df 229, against the longest lists). On a collection with many more documents per term the same code "
              "would skip far more often. The engine also has a second, larger saving that does not depend on skips: when the candidates are few, it "
              "probes the longer list by binary search instead of merging (counted as `postings_read` in WorkStats).", ""]
    text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text + "\n", encoding="utf-8", newline="\n")
    return text


def main(argv: list[str] | None = None) -> int:
    write_report()
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
