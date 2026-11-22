"""Postings compression (IIR 5.3): gap encoding, variable-byte codes and Elias gamma codes, measured on the real index.

* **Gap encoding.** A posting list stores increasing document numbers, so it is stored as the first number followed by the
  differences ("gaps"), which are small for frequent terms. Document numbers are shifted by one (document 0 becomes docID 1) so
  every gap is at least 1, as gamma needs. Positions are gap-encoded inside each zone: the index numbers positions from 0 in each
  zone (m1_index.index), so a posting's position list is one increasing run per zone; the run lengths are the per-zone term
  frequencies the index keeps anyway, which is how a decoder knows where a run restarts. Positions are shifted by one as well.
* **Variable-byte (VB) code.** 7 payload bits per byte, the high bit set on the last byte of a number. Byte aligned, so fast to
  decode.
* **Elias gamma code.** A number n >= 1 is its length in unary (len(offset) ones, then a zero) followed by the offset (n in binary
  without its leading 1): 2 floor(log2 n) + 1 bits. Bit aligned: smaller for small gaps, slower to decode.

Every encoder here is our own code and is checked by decoding back to the original arrays. `python -m m1_index.compress` writes the
size and decode-time table for the whole index to m1_index/reports/compression.md.

The live in-memory format stays the uncompressed arrays (D-040): decoding VB would cost a few milliseconds per query, which the
one-second budget allows, but nothing needs the memory it saves and other code reads the arrays directly.
"""

from __future__ import annotations

import gzip
import statistics
import sys
import time
from array import array
from typing import Any, Iterable, Sequence

from common.schema import ZONES
from m1_index.index import ROOT, ZONE_INDEX, InvertedIndex

REPORT_PATH = ROOT / "m1_index" / "reports" / "compression.md"


# --------------------------------------------------------------------------------------------------
# Gaps
# --------------------------------------------------------------------------------------------------
def to_gaps(values: Sequence[int]) -> list[int]:
    """[v0, v1 - v0, v2 - v1, ...] for an increasing sequence."""
    out, prev = [], 0
    for v in values:
        out.append(v - prev)
        prev = v
    return out


def from_gaps(gaps: Iterable[int]) -> list[int]:
    out, total = [], 0
    for g in gaps:
        total += g
        out.append(total)
    return out


# --------------------------------------------------------------------------------------------------
# Variable-byte code
# --------------------------------------------------------------------------------------------------
def vb_encode_number(n: int) -> bytes:
    """IIR figure 5.8: 7-bit groups, most significant first, the last byte marked by its high bit."""
    if n < 0:
        raise ValueError("VB encodes non-negative integers")
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(n & 0x7F)
        n >>= 7
    out.reverse()
    out[-1] |= 0x80
    return bytes(out)


def vb_encode(numbers: Iterable[int]) -> bytes:
    buf = bytearray()
    for n in numbers:
        if n < 128:  # the common case for gaps, written without building a list
            if n < 0:
                raise ValueError("VB encodes non-negative integers")
            buf.append(n | 0x80)
        else:
            buf += vb_encode_number(n)
    return bytes(buf)


def vb_decode(data: bytes) -> list[int]:
    out, n = [], 0
    for byte in data:
        if byte < 128:
            n = (n << 7) | byte
        else:
            out.append((n << 7) | (byte & 0x7F))
            n = 0
    if n:
        raise ValueError("VB stream ends inside a number")
    return out


# --------------------------------------------------------------------------------------------------
# Elias gamma code
# --------------------------------------------------------------------------------------------------
def gamma_code(n: int) -> str:
    """The gamma code of n >= 1 as a string of bits: 13 -> offset 101, length 1110 -> '1110101'."""
    if n < 1:
        raise ValueError("gamma encodes integers >= 1")
    offset = bin(n)[3:]  # binary without '0b' and without the leading 1
    return "1" * len(offset) + "0" + offset


def gamma_encode(numbers: Iterable[int]) -> tuple[bytes, int]:
    """(bytes, number of bits). The bit string is padded with zeros to whole bytes."""
    bits = "".join(map(gamma_code, numbers))
    nbits = len(bits)
    if not nbits:
        return b"", 0
    nbytes = (nbits + 7) // 8
    padded = bits + "0" * (nbytes * 8 - nbits)
    return int(padded, 2).to_bytes(nbytes, "big"), nbits


def gamma_decode(data: bytes, nbits: int) -> list[int]:
    """Decode the first `nbits` bits of a gamma stream."""
    if not nbits:
        return []
    bits = bin(int.from_bytes(data, "big"))[2:].zfill(len(data) * 8)
    out, i = [], 0
    find = bits.find
    while i < nbits:
        j = find("0", i)  # end of the unary length
        if j < 0 or j >= nbits:
            raise ValueError("gamma stream ends inside a length")
        length = j - i
        if length:
            out.append(int("1" + bits[j + 1:j + 1 + length], 2))
        else:
            out.append(1)
        i = j + 1 + length
    if i != nbits:
        raise ValueError("gamma stream ends inside an offset")
    return out


def gamma_bits(n: int) -> int:
    """Length of the gamma code of n in bits: 2 floor(log2 n) + 1."""
    return 2 * (n.bit_length() - 1) + 1


# --------------------------------------------------------------------------------------------------
# The index as gap streams
# --------------------------------------------------------------------------------------------------
def term_streams(index: InvertedIndex, t: int) -> tuple[list[int], list[int]]:
    """The two gap streams of term number t: document gaps (docID = number + 1) and position gaps (position + 1, restarting in
    every zone of every posting). Every value is >= 1."""
    lo, hi = index.t_start[t], index.t_start[t + 1]
    p_doc, p_off, pos = index.p_doc, index.p_off, index.pos
    doc_gaps, prev = [], 0
    for i in range(lo, hi):
        d = p_doc[i] + 1
        doc_gaps.append(d - prev)
        prev = d
    pos_gaps: list[int] = []
    for i, run_lengths in zip(range(lo, hi), _zone_runs(index, lo, hi)):
        k = p_off[i]
        for n in run_lengths:
            prev = 0
            for p in pos[k:k + n]:
                pos_gaps.append(p + 1 - prev)
                prev = p + 1
            k += n
    return doc_gaps, pos_gaps


def _zone_runs(index: InvertedIndex, lo: int, hi: int):
    """For each posting lo..hi-1, the lengths of its increasing position runs: its term frequency in each zone, in the order the
    document's zones were indexed."""
    from m1_index.index import TermPostings

    tp = TermPostings(index, lo, hi)
    ids, p_doc, lengths = index.doc_ids, index.p_doc, index.doc_lengths
    for i in range(lo, hi):
        tfs = tp.zone_tfs(i)
        yield [tfs[ZONE_INDEX[z]] for z in lengths[ids[p_doc[i]]] if tfs[ZONE_INDEX[z]]]


def decode_term(index: InvertedIndex, t: int, doc_gaps: list[int], pos_gaps: list[int]) -> tuple[list[int], list[int]]:
    """Rebuild a term's document numbers and positions from its gap streams (needs the per-zone counts, kept uncompressed)."""
    docs = [d - 1 for d in from_gaps(doc_gaps)]
    lo, hi = index.t_start[t], index.t_start[t + 1]
    positions: list[int] = []
    k = 0
    for run_lengths in _zone_runs(index, lo, hi):
        for n in run_lengths:
            acc = 0
            for g in pos_gaps[k:k + n]:
                acc += g
                positions.append(acc - 1)
            k += n
    return docs, positions


# --------------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------------
def measure(index: InvertedIndex, progress: bool = True) -> dict[str, Any]:
    """Encode every posting list of the index with VB and gamma, decode it back, check it, and time the decoding."""
    n_terms = len(index.terms)
    raw_doc = index.p_doc.itemsize * len(index.p_doc)
    raw_pos = index.pos.itemsize * len(index.pos)

    t0 = time.perf_counter()
    gz_doc = gzip.compress(index.p_doc.tobytes(), 6)
    gz_pos = gzip.compress(index.pos.tobytes(), 6)
    gz_encode = time.perf_counter() - t0

    vb_doc, vb_pos, g_doc, g_pos = [], [], [], []
    vb_size = [0, 0]
    g_bits = [0, 0]
    max_gap = [0, 0]
    t0 = time.perf_counter()
    for t in range(n_terms):
        dg, pg = term_streams(index, t)
        if dg:
            max_gap[0] = max(max_gap[0], max(dg))
        if pg:
            max_gap[1] = max(max_gap[1], max(pg))
        a, b = vb_encode(dg), vb_encode(pg)
        vb_doc.append(a)
        vb_pos.append(b)
        vb_size[0] += len(a)
        vb_size[1] += len(b)
        ga, gb = gamma_encode(dg), gamma_encode(pg)
        g_doc.append(ga)
        g_pos.append(gb)
        g_bits[0] += ga[1]
        g_bits[1] += gb[1]
        if progress and t and t % 50000 == 0:
            print(f"  encoded {t:,} of {n_terms:,} terms", file=sys.stderr)
    encode_s = time.perf_counter() - t0
    vbgz_doc = len(gzip.compress(b"".join(vb_doc), 6))
    vbgz_pos = len(gzip.compress(b"".join(vb_pos), 6))

    # --- decode timings (every posting list to Python ints) ---
    t0 = time.perf_counter()
    for t in range(n_terms):
        lo, hi = index.t_start[t], index.t_start[t + 1]
        index.p_doc[lo:hi].tolist()
        index.pos[index.p_off[lo]:index.p_off[hi]].tolist()
    raw_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    a = array("I")
    a.frombytes(gzip.decompress(gz_doc))
    b = array("I")
    b.frombytes(gzip.decompress(gz_pos))
    a.tolist()
    b.tolist()
    gz_s = time.perf_counter() - t0
    if a != index.p_doc or b != index.pos:
        raise AssertionError("gzip round trip differs")

    t0 = time.perf_counter()
    for t in range(n_terms):
        from_gaps(vb_decode(vb_doc[t]))
        vb_decode(vb_pos[t])
    vb_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for t in range(n_terms):
        from_gaps(gamma_decode(*g_doc[t]))
        gamma_decode(*g_pos[t])
    g_s = time.perf_counter() - t0

    # --- correctness: every term decodes back to the index's own arrays ---
    t0 = time.perf_counter()
    for t in range(n_terms):
        lo, hi = index.t_start[t], index.t_start[t + 1]
        want_docs = index.p_doc[lo:hi].tolist()
        want_pos = index.pos[index.p_off[lo]:index.p_off[hi]].tolist()
        for docs_gaps, pos_gaps in ((vb_decode(vb_doc[t]), vb_decode(vb_pos[t])), (gamma_decode(*g_doc[t]), gamma_decode(*g_pos[t]))):
            docs, positions = decode_term(index, t, docs_gaps, pos_gaps)
            if docs != want_docs or positions != want_pos:
                raise AssertionError(f"round trip differs for term {index.terms[t]!r}")
    verify_s = time.perf_counter() - t0

    return {
        "terms": n_terms, "postings": len(index.p_doc), "positions": len(index.pos), "max_gap": max_gap,
        "raw": (raw_doc, raw_pos), "gzip": (len(gz_doc), len(gz_pos)), "vb": tuple(vb_size), "vb_gzip": (vbgz_doc, vbgz_pos),
        "gamma": ((g_bits[0] + 7) // 8, (g_bits[1] + 7) // 8), "gamma_bits": tuple(g_bits),
        "decode_s": {"raw": raw_s, "gzip": gz_s, "vb": vb_s, "gamma": g_s}, "encode_s": encode_s, "gzip_encode_s": gz_encode,
        "verify_s": verify_s, "vb_streams": (vb_doc, vb_pos),
    }


def per_query_overhead(index: InvertedIndex, vb_streams: tuple[list[bytes], list[bytes]]) -> list[tuple[str, int, float, float]]:
    """For each benchmark query: (query, postings of its terms, ms to slice them from the arrays, ms to decode them from VB).
    This is what the search path would add per query if it read VB-coded postings."""
    from bisect import bisect_left

    from m1_index.benchmark import QUERIES
    from m1_index.tokenizer import tokenize

    vb_doc, vb_pos = vb_streams
    rows = []
    for q in QUERIES:
        ts = []
        for term in dict.fromkeys(tokenize(q)):
            t = bisect_left(index.terms, term)
            if t < len(index.terms) and index.terms[t] == term:
                ts.append(t)
        n_post = sum(index.t_start[t + 1] - index.t_start[t] for t in ts)
        raw, vb = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            for t in ts:
                lo, hi = index.t_start[t], index.t_start[t + 1]
                index.p_doc[lo:hi].tolist()
                index.pos[index.p_off[lo]:index.p_off[hi]].tolist()
            raw.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            for t in ts:
                from_gaps(vb_decode(vb_doc[t]))
                vb_decode(vb_pos[t])
            vb.append((time.perf_counter() - t0) * 1000)
        rows.append((q, n_post, statistics.median(raw), statistics.median(vb)))
    return rows


def write_report(index: InvertedIndex | None = None) -> str:
    from m1_index.index import INDEX_DIR, INDEX_FILE

    idx = index if index is not None else InvertedIndex.load()
    m = measure(idx)
    rows = per_query_overhead(idx, m["vb_streams"])
    mb = lambda n: f"{n / (1024 * 1024):.1f} MB"  # noqa: E731
    total = {k: sum(m[k]) for k in ("raw", "gzip", "vb", "vb_gzip", "gamma")}
    disk = (INDEX_DIR / INDEX_FILE).stat().st_size if (INDEX_DIR / INDEX_FILE).exists() else None
    lines = [
        "# M1 postings compression: sizes and decode times on the real index", "",
        f"Generated by `python -m m1_index.compress` on {time.strftime('%Y-%m-%d')} over the index of {idx.num_docs:,} judgments: "
        f"{m['terms']:,} terms, {m['postings']:,} postings (document numbers) and {m['positions']:,} positions. Every encoded list was decoded "
        f"back and compared with the index's arrays ({m['verify_s']:.0f} s); all of them match.", "",
        "## Sizes", "",
        "| Encoding | Document numbers | Positions | Total | vs raw |", "|---|---:|---:|---:|---:|",
        f"| Raw arrays (4-byte integers, the live in-memory format) | {mb(m['raw'][0])} | {mb(m['raw'][1])} | {mb(total['raw'])} | 100% |",
        f"| gzip of the raw arrays (level 6) | {mb(m['gzip'][0])} | {mb(m['gzip'][1])} | {mb(total['gzip'])} | {100 * total['gzip'] / total['raw']:.0f}% |",
        f"| Gaps + variable-byte | {mb(m['vb'][0])} | {mb(m['vb'][1])} | {mb(total['vb'])} | {100 * total['vb'] / total['raw']:.0f}% |",
        f"| Gaps + Elias gamma | {mb(m['gamma'][0])} | {mb(m['gamma'][1])} | {mb(total['gamma'])} | {100 * total['gamma'] / total['raw']:.0f}% |",
        f"| Gaps + variable-byte, then gzip | {mb(m['vb_gzip'][0])} | {mb(m['vb_gzip'][1])} | {mb(total['vb_gzip'])} | {100 * total['vb_gzip'] / total['raw']:.0f}% |", "",
        f"Bits per entry: document gaps {8 * m['vb'][0] / m['postings']:.2f} (VB) and {m['gamma_bits'][0] / m['postings']:.2f} (gamma); "
        f"position gaps {8 * m['vb'][1] / m['positions']:.2f} (VB) and {m['gamma_bits'][1] / m['positions']:.2f} (gamma). "
        f"Largest gap: {m['max_gap'][0]:,} between documents, {m['max_gap'][1]:,} between positions.", "",
        "Not in the table: the per-zone term counts (`p_zc`, 8 bytes a posting) and the posting offsets (`p_off`, derivable from the counts), "
        "which the decoder needs to know where each zone's run of positions restarts, and the vocabulary itself."
        + (f" The whole index file on disk (`{INDEX_FILE}`, all arrays and tables, gzip) is {mb(disk)}." if disk else ""), "",
        "## Decode time (every posting list of the index to Python integers)", "",
        "| Encoding | Decode all postings | Encode all postings |", "|---|---:|---:|",
        f"| Raw arrays (slice to a list) | {m['decode_s']['raw']:.2f} s | none |",
        f"| gzip (decompress, then to a list) | {m['decode_s']['gzip']:.2f} s | {m['gzip_encode_s']:.1f} s |",
        f"| Variable-byte | {m['decode_s']['vb']:.2f} s | {m['encode_s']:.1f} s (VB and gamma together, including building the gap streams) |",
        f"| Elias gamma | {m['decode_s']['gamma']:.2f} s | (with VB) |", "",
        "## What a query would pay", "",
        "The benchmark queries of `m1_index.benchmark`: the postings of their terms read from the arrays versus decoded from VB (median of 5).", "",
        "| Query | Postings read | Arrays (ms) | VB decode (ms) |", "|---|---:|---:|---:|",
    ]
    for q, n, raw_ms, vb_ms in rows:
        lines.append(f"| `{q}` | {n:,} | {raw_ms:.2f} | {vb_ms:.2f} |")
    worst = max(rows, key=lambda r: r[3])
    lines += ["", "## Decision", "",
              f"The live index keeps the raw arrays in memory. VB would cut the postings to {100 * total['vb'] / total['raw']:.0f}% of their size and "
              f"make reading a query's postings at most {worst[3] - worst[2]:.1f} ms slower (worst case `{worst[0]}`: {worst[3]:.1f} ms instead of "
              f"{worst[2]:.1f} ms), "
              f"{'so search would still stay well under one second and the budget would allow the change' if worst[3] < 1000 else 'which would break the one-second budget'}. "
              "It is not made because nothing needs it: the memory saved "
              f"({mb(total['raw'] - total['vb'])}) does not matter on a laptop, the file on disk is already gzip-compressed, and other code reads the "
              "arrays directly (the Index page in `app/m1_view.py` uses `index.p_doc`), so a new live format would mean changing code outside M1 "
              "for no gain the demo can show. The encoders stay here, tested, for the report and the video. See DECISIONS.md D-040.", ""]
    text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text + "\n", encoding="utf-8", newline="\n")
    return text


def main(argv: list[str] | None = None) -> int:
    t0 = time.perf_counter()
    write_report()
    print(f"Wrote {REPORT_PATH} in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
