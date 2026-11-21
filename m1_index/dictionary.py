"""Tolerant retrieval: the dictionary of surface forms, k-gram and permuterm indexes, edit distance, spelling suggestions and Soundex.

IR concepts (IIR chapter 3), all implemented here:

* **Dictionary as a sorted array.** The vocabulary is a sorted Python list searched by binary search (`bisect`), the array
  counterpart of a B-tree: it answers exact lookups and prefix ranges (`murd*` is the range [murd, mure)). A hash table would
  answer exact lookups in O(1) but cannot answer a range, which is why it is not used for wildcards.
* **Stemming versus wildcards.** The inverted index stores Porter stems (`homicide` is indexed as `homicid`), so a wildcard run
  against the index's own vocabulary would miss `*cide` entirely and would match stems nobody typed. Wildcards therefore run on a
  second dictionary of *surface* words (the lower-cased words before stemming, exactly the words the tokenizer keeps), and every
  matching surface word is mapped to its index term with the same tokenizer. See DECISIONS.md D-040.
* **k-gram index** (k = 3, `$` marks the word boundary): gram -> sorted list of surface-word ids. A wildcard query is split at
  `*`, the k-grams of the literal pieces are intersected, and because k-grams over-generate (`red*` asks for `$re` and `red`,
  which `retired` also contains), every candidate is post-filtered against the pattern.
* **Permuterm index**: every rotation of `word$`, sorted; `X*Y` becomes the prefix query `Y$X`. Built only for the size
  comparison in the report: it holds one entry per character of the vocabulary, several times the k-gram index.
* **Spelling correction** (isolated-term): candidates are the words sharing enough k-grams (Jaccard coefficient of the gram sets),
  ranked by Levenshtein distance (dynamic programming, our own) and then by document frequency. Statute numbers, citation tokens
  and the legal abbreviations the tokenizer protects are never corrected.
* **Soundex** (the American variant, with the H/W rule): a phonetic code for party names in case titles (Mohammed, Muhammad and
  Mohd all give M530), used by `search(..., filters={"party": name})`.

Public functions (Phase 2 contract): `expand_wildcard(pattern, limit=50)`, `suggest(query)`, `soundex(name)`.

The dictionary is built from data/processed/judgments.jsonl together with the index (`python -m m1_index.index build` runs it) and is
stored beside it as data/processed/index/dictionary.pkl.gz. Run `python -m m1_index.dictionary report` for the measurements.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import re
import sys
import threading
import time
from array import array
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from m1_index import tokenizer as T
from m1_index.index import INDEX_DIR, JUDGMENTS_PATH, ROOT

DICT_FILE = "dictionary.pkl.gz"
FORMAT_VERSION = 1
REPORT_PATH = ROOT / "m1_index" / "reports" / "dictionary.md"
BOUNDARY = "$"

# Defaults for the `tolerant:` section of common/config.yaml (the file wins when it has the key).
DEFAULTS: dict[str, Any] = {
    "kgram_k": 3,              # k of the k-gram index
    "spell_max_edit": 2,       # largest Levenshtein distance a suggestion may have (1 for words of 4 letters or fewer)
    "spell_jaccard_min": 0.2,  # smallest k-gram Jaccard overlap for a word to be a spelling candidate at all
    "spell_transpositions": True,  # an adjacent swap ("culpabel") counts as one edit (Damerau), not two
    "suggest_min_df": 3,       # a suggested word must occur in at least this many judgments
    "suggest_rare_df": 3,      # a word in at most this many judgments is "rare" and may be corrected ...
    "suggest_df_ratio": 20,    # ... when a candidate occurs in at least this many times more judgments
    "suggest_min_hits": 3,     # a query matching fewer judgments than this lets every word be checked against the ratio rule
    "wildcard_limit": 50,      # surface words a wildcard in a query expands to (the most frequent ones)
    "wildcard_phrase_limit": 64,  # phrases a phrase with wildcards expands to
}


def settings() -> dict[str, Any]:
    """The `tolerant:` settings: DEFAULTS overlaid with common/config.yaml."""
    out = dict(DEFAULTS)
    try:
        from common.config import load_config

        out.update(load_config().get("tolerant") or {})
    except Exception:  # pragma: no cover - a broken config is reported by the config's own checks, not here
        pass
    return out


# --------------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------------
def kgrams(word: str, k: int = 3) -> set[str]:
    """The k-grams of `$word$` (IIR 3.2.2). A word shorter than k - 2 still gives its one padded gram."""
    s = f"{BOUNDARY}{word}{BOUNDARY}"
    if len(s) <= k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def pattern_kgrams(pattern: str, k: int = 3) -> set[str]:
    """The k-grams every word matching the wildcard pattern must contain: those of the literal pieces of `$pattern$` between the
    stars. `murd*` gives {$mu, mur, urd}; `*cide` gives {cid, ide, de$}; a piece shorter than k gives none."""
    out: set[str] = set()
    for piece in f"{BOUNDARY}{pattern}{BOUNDARY}".split("*"):
        for i in range(len(piece) - k + 1):
            out.add(piece[i:i + k])
    return out


def pattern_regex(pattern: str) -> re.Pattern:
    """The post-filter: the pattern as an anchored regular expression, `*` meaning any run of characters."""
    return re.compile("".join(".*" if part == "*" else re.escape(part) for part in re.split(r"(\*)", pattern)) + r"\Z")


def jaccard(a: set, b: set) -> float:
    """|A & B| / |A | B| (1.0 for two empty sets)."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def levenshtein(a: str, b: str, max_dist: int | None = None, transpositions: bool = False) -> int:
    """Edit distance (insert, delete, substitute, each cost 1) by dynamic programming (IIR 3.3.3).

    `transpositions=True` also counts swapping two adjacent letters as one edit (Damerau's fourth operation, in its "optimal
    string alignment" form: no substring is edited twice). "culpabel" -> "culpable" is then 1 edit, not 2. Swapped letters are
    among the commonest typing errors, which is why the spelling suggester turns it on (D-040).

    With `max_dist`, the computation stops as soon as every cell of a row exceeds it and returns max_dist + 1, which is all a caller
    that only wants distances up to max_dist needs to know."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if max_dist is not None and len(a) - len(b) > max_dist:
        return max_dist + 1
    before: list[int] | None = None  # row i - 2, for transpositions
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            best = min(prev[j] + 1,                # delete ca
                       cur[j - 1] + 1,             # insert cb
                       prev[j - 1] + (ca != cb))   # substitute (free when equal)
            if transpositions and before is not None and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                best = min(best, before[j - 2] + 1)  # swap the last two letters
            cur[j] = best
        if max_dist is not None and min(cur) > max_dist and (not transpositions or min(prev) > max_dist):
            return max_dist + 1
        before, prev = prev, cur
    d = prev[-1]
    return d if max_dist is None or d <= max_dist else max_dist + 1


# ---------------------------------------------------------------------------------------------- Soundex
_SOUNDEX_CODES = {**dict.fromkeys("BFPV", "1"), **dict.fromkeys("CGJKQSXZ", "2"), **dict.fromkeys("DT", "3"), "L": "4",
                  **dict.fromkeys("MN", "5"), "R": "6"}
# Abbreviations of Indian given names that Soundex cannot see through (Mohd has no M-vowel-M), expanded before coding. A short,
# fixed list of spellings courts use in titles; see DECISIONS.md D-040.
NAME_ABBREVIATIONS = {"mohd": "mohammad", "mohd.": "mohammad", "md": "mohammad", "md.": "mohammad"}


def _soundex_word(word: str) -> str:
    letters = "".join(ch for ch in word.upper() if "A" <= ch <= "Z")
    if not letters:
        return ""
    first = letters[0]
    digits: list[str] = []
    last = _SOUNDEX_CODES.get(first, "")
    for ch in letters[1:]:
        code = _SOUNDEX_CODES.get(ch, "")
        if code and code != last:
            digits.append(code)
            if len(digits) == 3:
                break
        if ch not in "HW":  # H and W do not separate two letters with the same code; vowels (and Y) do
            last = code
    return (first + "".join(digits) + "000")[:4]


def soundex(name: str) -> str:
    """American Soundex of a name: 'Mohammed' -> 'M530'. A name of several words gives one code per word, space separated
    ('Shivaji Rao' -> 'S120 R000'); the abbreviations in NAME_ABBREVIATIONS are expanded first ('Mohd' -> 'M530')."""
    codes = []
    for w in re.findall(r"[A-Za-z]+\.?", name or ""):
        w = NAME_ABBREVIATIONS.get(w.lower(), w)
        code = _soundex_word(w)
        if code:
            codes.append(code)
    return " ".join(codes)


# Words of a case title that are not party names and are never used for a phonetic match.
TITLE_STOPWORDS = {"v", "vs", "versus", "and", "or", "ors", "anr", "another", "others", "the", "of", "in", "re", "through", "thr",
                   "by", "etc", "state", "union", "india", "govt", "government"}


class SoundexIndex:
    """Soundex code -> numbers of the documents whose title has a word with that code (the party names)."""

    def __init__(self, titles: Iterable[tuple[str, str]]) -> None:
        self.codes: dict[str, set[str]] = {}
        for doc_id, title in titles:
            for word in re.findall(r"[A-Za-z]+\.?", title or ""):
                bare = word.rstrip(".").lower()
                if len(bare) < 2 or bare in TITLE_STOPWORDS:
                    continue
                code = soundex(word)
                if code:
                    self.codes.setdefault(code, set()).add(doc_id)

    def match(self, name: str) -> set[str]:
        """Documents whose title sounds like every word of `name` (each word of the name matches some word of the title)."""
        found: set[str] | None = None
        for code in soundex(name).split():
            docs = self.codes.get(code, set())
            found = set(docs) if found is None else found & docs
            if not found:
                return set()
        return found or set()


# --------------------------------------------------------------------------------------------------
# The dictionary of surface forms
# --------------------------------------------------------------------------------------------------
class SurfaceDictionary:
    """Sorted surface words with their document frequency, and a k-gram index over them in a compressed-row layout:

        words[w]                     the w-th surface word (sorted, found by binary search)
        df[w]                        the number of judgments that contain it
        grams[g]                     the g-th k-gram (sorted)
        g_start[g] .. g_start[g+1]   its postings: a slice of g_terms, word ids in increasing order
    """

    def __init__(self, words: list[str], df: array, k: int = 3, grams: list[str] | None = None, g_start: array | None = None,
                 g_terms: array | None = None) -> None:
        self.words, self.df, self.k = words, df, k
        if grams is None or g_start is None or g_terms is None:
            grams, g_start, g_terms = self._build_kgrams(words, k)
        self.grams, self.g_start, self.g_terms = grams, g_start, g_terms
        self._stem_cache: dict[str, str] = {}

    # ---------------------------------------------------------------- building
    @staticmethod
    def _build_kgrams(words: list[str], k: int) -> tuple[list[str], array, array]:
        lists: dict[str, array] = {}
        for wid, word in enumerate(words):
            for g in kgrams(word, k):
                a = lists.get(g)
                if a is None:
                    a = lists[g] = array("I")
                a.append(wid)  # word ids arrive in increasing order, so every list is sorted
        grams = sorted(lists)
        g_start, g_terms = array("I", [0]), array("I")
        for g in grams:
            g_terms.extend(lists.pop(g))
            g_start.append(len(g_terms))
        return grams, g_start, g_terms

    @classmethod
    def from_counts(cls, counts: dict[str, int], k: int = 3) -> "SurfaceDictionary":
        words = sorted(counts)
        return cls(words, array("I", (counts[w] for w in words)), k)

    # ---------------------------------------------------------------- saving and loading
    def save(self, directory: Path | str = INDEX_DIR) -> Path:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        out, tmp = p / DICT_FILE, p / f"{DICT_FILE}.tmp{os.getpid()}"
        payload = {"format": FORMAT_VERSION, "k": self.k, "words": self.words, "df": self.df, "grams": self.grams,
                   "g_start": self.g_start, "g_terms": self.g_terms}
        try:
            with gzip.open(tmp, "wb", compresslevel=3) as f:
                pickle.dump(payload, f, protocol=5)
            os.replace(tmp, out)
        finally:
            if tmp.exists():
                tmp.unlink()
        return out

    @classmethod
    def load_or_build(cls, directory: Path | str = INDEX_DIR) -> "SurfaceDictionary":
        """Load the dictionary; when it is missing but the corpus is there (an index built before the dictionary existed), build
        it from the corpus first (about 15 s, once) and say so."""
        if not (Path(directory) / DICT_FILE).exists() and JUDGMENTS_PATH.exists():
            print(f"{Path(directory) / DICT_FILE} is missing: building it from {JUDGMENTS_PATH.name} (once, about 15 s)", file=sys.stderr)
            return build_from_processed(JUDGMENTS_PATH, directory)
        return cls.load(directory)

    @classmethod
    def load(cls, directory: Path | str = INDEX_DIR) -> "SurfaceDictionary":
        path = Path(directory) / DICT_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. It is built with the index: run 'python -m m1_index.index build' "
                                    "(or 'python -m m1_index.dictionary build' for the dictionary alone).")
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("format") != FORMAT_VERSION:
            raise RuntimeError(f"{path} is in another format. Rebuild it with 'python -m m1_index.dictionary build'.")
        return cls(payload["words"], payload["df"], payload["k"], payload["grams"], payload["g_start"], payload["g_terms"])

    # ---------------------------------------------------------------- lookups
    def __len__(self) -> int:
        return len(self.words)

    def word_id(self, word: str) -> int:
        i = bisect_left(self.words, word)
        return i if i < len(self.words) and self.words[i] == word else -1

    def df_of(self, word: str) -> int:
        i = self.word_id(word)
        return 0 if i < 0 else self.df[i]

    def gram_postings(self, gram: str) -> array:
        g = bisect_left(self.grams, gram)
        if g == len(self.grams) or self.grams[g] != gram:
            return array("I")
        return self.g_terms[self.g_start[g]:self.g_start[g + 1]]

    def prefix_range(self, prefix: str) -> range:
        """Ids of the words that start with `prefix`: one contiguous range of the sorted array, found with two binary searches."""
        lo = bisect_left(self.words, prefix)
        hi = bisect_left(self.words, prefix + "\U0010ffff")
        return range(lo, hi)

    def stem(self, word: str) -> str:
        """The index term of a surface word (what the tokenizer emits for it)."""
        s = self._stem_cache.get(word)
        if s is None:
            toks = T.tokenize(word)
            s = self._stem_cache[word] = toks[0] if toks else word
        return s

    # ---------------------------------------------------------------- wildcards
    def wildcard_ids(self, pattern: str, stats: dict | None = None) -> list[int]:
        """Ids of the surface words matching the pattern (`*` = any run of characters), in id (alphabetical) order.

        Candidates come from intersecting the posting lists of the pattern's k-grams, rarest first; a pattern with no k-gram but a
        literal prefix (`ab*`) uses the prefix range of the sorted array. Every candidate is then checked against the pattern,
        because a word can contain all the k-grams without matching (k-grams over-generate)."""
        pattern = pattern.lower()
        if "*" not in pattern:
            i = self.word_id(pattern)
            return [i] if i >= 0 else []
        grams = pattern_kgrams(pattern, self.k)
        if grams:
            lists = sorted((self.gram_postings(g) for g in grams), key=len)
            cand: Iterable[int] = lists[0]
            for other in lists[1:]:
                if not cand:
                    break
                cand = sorted(set(cand).intersection(other))
        else:
            prefix = pattern.split("*", 1)[0]
            if not prefix:
                raise ValueError(f"wildcard {pattern!r} is too broad: give at least {self.k - 1} letters next to a '*' or a prefix")
            cand = self.prefix_range(prefix)
        rx = pattern_regex(pattern)
        cand = list(cand)
        out = [w for w in cand if rx.match(self.words[w])]
        if stats is not None:
            stats.update(kgrams=sorted(grams), candidates=len(cand), matches=len(out), false_positives=len(cand) - len(out))
        return out

    def expand_wildcard(self, pattern: str, limit: int = 50) -> list[str]:
        """The surface words matching the pattern, most frequent first (ties alphabetical), at most `limit`."""
        ids = self.wildcard_ids(pattern)
        ids.sort(key=lambda w: (-self.df[w], self.words[w]))
        return [self.words[w] for w in ids[:max(0, limit)]]

    def wildcard_terms(self, pattern: str, limit: int = 50) -> list[str]:
        """The index terms a wildcard stands for: the stems of its `limit` most frequent surface words, without repeats."""
        return list(dict.fromkeys(self.stem(w) for w in self.expand_wildcard(pattern, limit)))

    # ---------------------------------------------------------------- spelling correction
    def candidates(self, word: str, max_edit: int = 2, jaccard_min: float = 0.2,
                   transpositions: bool = True) -> list[tuple[str, int, int, float]]:
        """Spelling candidates for `word`: (candidate, edit distance, df, k-gram Jaccard), best first.

        Step 1, k-gram overlap: count, for every dictionary word, how many k-grams it shares with `word` (one pass over the posting
        lists of the word's grams), and keep the words whose Jaccard coefficient is at least `jaccard_min` and whose length is
        within `max_edit` (a longer gap cannot be bridged by max_edit edits). Step 2, edit distance: keep those within `max_edit`
        Levenshtein edits. Order: distance, then document frequency (higher first), then the word."""
        word = word.lower()
        grams = kgrams(word, self.k)
        overlap: Counter = Counter()
        for g in grams:
            overlap.update(self.gram_postings(g))
        n = len(grams)
        out = []
        for wid, shared in overlap.items():
            cand = self.words[wid]
            if cand == word or abs(len(cand) - len(word)) > max_edit:
                continue
            # |grams(cand)| is len(cand) for k = 3 when no gram repeats; computing the set exactly costs little and is always right
            j = shared / (n + len(kgrams(cand, self.k)) - shared)
            if j < jaccard_min:
                continue
            d = levenshtein(word, cand, max_edit, transpositions)
            if d <= max_edit:
                out.append((cand, d, self.df[wid], j))
        out.sort(key=lambda c: (c[1], -c[2], c[0]))
        return out

    def correct(self, word: str, *, few_hits: bool = False, cfg: dict | None = None) -> str | None:
        """The suggested spelling of one surface word, or None. Rules (D-040): a word in no judgment is replaced by its best
        candidate occurring in at least `suggest_min_df` judgments; a word in at most `suggest_rare_df` judgments (or any word when
        the query found fewer than `suggest_min_hits` judgments) is replaced only by a candidate at least `suggest_df_ratio` times
        as frequent."""
        cfg = cfg or settings()
        w = word.lower()
        df = self.df_of(w)
        max_edit = 1 if len(w) <= 4 else int(cfg["spell_max_edit"])
        if df > 0 and not few_hits and df > int(cfg["suggest_rare_df"]):
            return None
        need = int(cfg["suggest_min_df"]) if df == 0 else max(int(cfg["suggest_min_df"]), int(cfg["suggest_df_ratio"]) * df)
        for cand, _d, cdf, _j in self.candidates(w, max_edit, float(cfg["spell_jaccard_min"]), bool(cfg["spell_transpositions"])):
            if cdf >= need:
                return cand
        return None

    def suggest(self, query: str, hit_count: Callable[[str], int] | None = None, cfg: dict | None = None) -> str | None:
        """'Did you mean' text for a query, or None when nothing would change (see `suggest`)."""
        cfg = cfg or settings()
        pieces: list[tuple[int, int, str]] = []
        few_hits: bool | None = None
        for m in _QUERY_WORD.finditer(query or ""):
            word = m.group()
            if not _correctable(query, m):
                continue
            w = word.lower()
            df = self.df_of(w)
            if df > int(cfg["suggest_rare_df"]) and few_hits is None and hit_count is not None:
                few_hits = hit_count(query) < int(cfg["suggest_min_hits"])
            fix = self.correct(w, few_hits=bool(few_hits), cfg=cfg)
            if fix and fix != w:
                pieces.append((m.start(), m.end(), _match_case(word, fix)))
        if not pieces:
            return None
        out, last = [], 0
        for start, end, text in pieces:
            out.append(query[last:start])
            out.append(text)
            last = end
        out.append(query[last:])
        return "".join(out)


_QUERY_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_OPERATORS = {"and", "or", "not"}


def _correctable(query: str, m: re.Match) -> bool:
    """False for the words a suggestion must never touch: operators, proximity operators (/s, /p), wildcards, words glued to digits
    or underscores (IPC_302, 120-B, 2025INSC), the tokenizer's legal abbreviations (IPC, BNS, CrPC, SCC, ...), stop words and words
    of fewer than three letters."""
    word, start, end = m.group(), m.start(), m.end()
    before = query[start - 1] if start > 0 else " "
    after = query[end] if end < len(query) else " "
    if before in "/*_.-" or before.isdigit() or after in "*_" or after.isdigit():
        return False
    w = word.lower()
    if len(w) < 3 or w in _OPERATORS or w in T.STOPWORDS or T.is_legal_token(w) or w in T.LEGAL_ABBREVIATIONS:
        return False
    return True


def _match_case(original: str, fix: str) -> str:
    if original.isupper() and len(original) > 1:
        return fix.upper()
    if original[:1].isupper():
        return fix[:1].upper() + fix[1:]
    return fix


# --------------------------------------------------------------------------------------------------
# Permuterm index (for the size comparison)
# --------------------------------------------------------------------------------------------------
class PermutermIndex:
    """Every rotation of `word$`, sorted, each pointing at its word id (IIR 3.2.1)."""

    def __init__(self, words: list[str]) -> None:
        entries = []
        for wid, word in enumerate(words):
            s = word + BOUNDARY
            for i in range(len(s)):
                entries.append((s[i:] + s[:i], wid))
        entries.sort()
        self.rotations = [r for r, _ in entries]
        self.word_ids = array("I", (w for _, w in entries))
        self.words = words

    def __len__(self) -> int:
        return len(self.rotations)

    def lookup(self, pattern: str) -> list[int]:
        """Word ids matching the pattern: rotate `pattern$` so the last `*` is at the end, look up the part before the first `*`
        as a prefix range, then post-filter (needed only when the pattern has more than one `*`)."""
        pattern = pattern.lower()
        s = pattern + BOUNDARY
        if "*" not in pattern:
            prefix = s
        else:
            last = s.rindex("*")
            rotated = s[last + 1:] + s[:last + 1]
            prefix = rotated.split("*", 1)[0]
        lo = bisect_left(self.rotations, prefix)
        hi = bisect_left(self.rotations, prefix + "\U0010ffff")
        ids = sorted(set(self.word_ids[lo:hi]))
        if pattern.count("*") > 1:
            rx = pattern_regex(pattern)
            ids = [w for w in ids if rx.match(self.words[w])]
        elif "*" not in pattern:
            ids = [w for w in ids if self.words[w] == pattern]
        return ids


def size_comparison(d: SurfaceDictionary, perm: PermutermIndex) -> dict[str, Any]:
    """Entries and bytes of the two wildcard indexes over the same vocabulary. Bytes count what the structures hold as data: the
    characters of every key plus a 4-byte word id per posting or rotation (the Python object overhead is left out of both)."""
    kgram_postings = len(d.g_terms)
    kgram_bytes = sum(len(g.encode("utf-8")) for g in d.grams) + 4 * kgram_postings + 4 * len(d.g_start)
    perm_bytes = sum(len(r.encode("utf-8")) for r in perm.rotations) + 4 * len(perm)
    return {"words": len(d), "kgram_keys": len(d.grams), "kgram_postings": kgram_postings, "kgram_bytes": kgram_bytes,
            "permuterm_rotations": len(perm), "permuterm_bytes": perm_bytes, "ratio": perm_bytes / kgram_bytes}


# --------------------------------------------------------------------------------------------------
# Building from the corpus
# --------------------------------------------------------------------------------------------------
class SurfaceCounter:
    """Document frequency of every surface word, one judgment at a time."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.docs = 0

    def add(self, zone_texts: Iterable[str]) -> None:
        seen: set[str] = set()
        for text in zone_texts:
            for surface, _term in T.surface_tokens(text):
                seen.add(surface)
        self.counts.update(seen)
        self.docs += 1


def _iter_zone_texts(path: Path) -> Iterator[list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                yield list((rec.get("zones") or {}).values())


def build_from_processed(judgments_path: Path | str | None = None, out_dir: Path | str | None = None, k: int | None = None) -> SurfaceDictionary:
    """Count surface words over judgments.jsonl (the zones, as the index does), build the k-gram index and save both."""
    from m1_index.index import memoised_tokenizer

    jpath = Path(judgments_path) if judgments_path else JUDGMENTS_PATH
    k = int(k or settings()["kgram_k"])
    started = time.perf_counter()
    counter = SurfaceCounter()
    with memoised_tokenizer():
        for zones in _iter_zone_texts(jpath):
            counter.add(zones)
    d = SurfaceDictionary.from_counts(counter.counts, k)
    path = d.save(out_dir or INDEX_DIR)
    print(f"Saved surface dictionary ({len(d)} words over {counter.docs} judgments, {len(d.grams)} {k}-grams, "
          f"{len(d.g_terms)} k-gram postings, {path.stat().st_size / (1024 * 1024):.2f} MB) to {path} "
          f"in {time.perf_counter() - started:.0f} s", file=sys.stderr)
    return d


# --------------------------------------------------------------------------------------------------
# Public API (one dictionary per process, loaded on first use)
# --------------------------------------------------------------------------------------------------
_LOCK = threading.Lock()
_DICT: SurfaceDictionary | None = None


def get_dictionary() -> SurfaceDictionary:
    global _DICT
    if _DICT is None:
        with _LOCK:
            if _DICT is None:
                _DICT = SurfaceDictionary.load_or_build(INDEX_DIR)
    return _DICT


def expand_wildcard(pattern: str, limit: int = 50) -> list[str]:
    """Phase 2 contract: the surface words of the corpus that match a wildcard pattern, most frequent first.
    'murd*' -> ['murder', 'murdered', ...]. A pattern too broad to look up (only stars, or one letter between stars) gives []."""
    try:
        return get_dictionary().expand_wildcard(pattern, limit)
    except ValueError:
        return []


def _hit_count(query: str) -> int:
    from m1_index.searcher import RankedSearchEngine, WorkStats

    engine = RankedSearchEngine.get_instance()
    nums, _terms = engine._candidates(query, None, WorkStats())
    return len(nums)


def suggest(query: str) -> str | None:
    """Phase 2 contract: 'did you mean' text for the query, or None if no word would change.
    'culpabel homicide' -> 'culpable homicide'. Statute and citation tokens are never changed."""
    return get_dictionary().suggest(query, hit_count=_hit_count)


# --------------------------------------------------------------------------------------------------
# Report and command line
# --------------------------------------------------------------------------------------------------
REPORT_WILDCARDS = ("murd*", "*cide", "cons*acy", "acquit*", "intent*", "*bail", "dow*y", "498*")
REPORT_MISSPELLINGS = ("culpabel homicide", "anticipatry bail", "circumstancial evidence", "dying declaraton", "murdr", "dowry deth", "bale",
                       "procecution witness", "section 302 IPC", "common intension")
REPORT_NAMES = ("Mohammed", "Muhammad", "Mohd", "Shivaji", "Sivaji")
REPORT_QUERY = '"common intent*" /s murd*'


def _ms(fn: Callable[[], Any], repeat: int = 5) -> tuple[Any, float]:
    best, value = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        value = fn()
        best = min(best, (time.perf_counter() - t0) * 1000)
    return value, best


def write_report(path: Path = REPORT_PATH) -> str:
    from m1_index.searcher import RankedSearchEngine

    d = get_dictionary()
    lines = ["# M1 dictionary and tolerant retrieval: measurements", "",
             f"Generated by `python -m m1_index.dictionary report` on {time.strftime('%Y-%m-%d')} from the index and dictionary built from "
             "`data/processed/judgments.jsonl`. Every number below comes from that run; times are the best of 5 on this machine.", ""]

    t0 = time.perf_counter()
    perm = PermutermIndex(d.words)
    perm_build = time.perf_counter() - t0
    sc = size_comparison(d, perm)
    lines += ["## k-gram index vs permuterm index", "",
              f"Vocabulary: **{sc['words']:,} surface words** (lower-cased words the tokenizer keeps, before stemming) over the corpus.", "",
              "| Index | Keys | Postings / rotations | Bytes of keys + 4-byte ids |", "|---|---:|---:|---:|",
              f"| k-gram (k = {d.k}) | {sc['kgram_keys']:,} grams | {sc['kgram_postings']:,} | {sc['kgram_bytes'] / 1e6:.1f} MB |",
              f"| permuterm | {sc['permuterm_rotations']:,} rotations | {sc['permuterm_rotations']:,} | {sc['permuterm_bytes'] / 1e6:.1f} MB |", "",
              f"The permuterm index is **{sc['ratio']:.1f} times** the k-gram index (built in {perm_build:.1f} s; it is not kept in the live search path).", ""]

    lines += ["## Wildcards", "",
              "k-gram candidates are intersected rarest gram first and post-filtered against the pattern. The permuterm column is the same query "
              "on the permuterm index; the two must agree.", "",
              "| Pattern | k-grams used | Candidates | Matches | False positives removed | Permuterm matches | Time (ms) | Most frequent matches (df) |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for pat in REPORT_WILDCARDS:
        stats: dict = {}
        ids, ms = _ms(lambda: d.wildcard_ids(pat, stats))
        pids = perm.lookup(pat)
        top = sorted(ids, key=lambda w: (-d.df[w], d.words[w]))[:6]
        lines.append(f"| `{pat}` | {' '.join('`' + g + '`' for g in stats['kgrams'])} | {stats['candidates']:,} | {stats['matches']:,} | "
                     f"{stats['false_positives']:,} | {len(pids):,}{'' if pids == ids else ' (DIFFERS)'} | {ms:.2f} | "
                     + ", ".join(f"{d.words[w]} ({d.df[w]})" for w in top) + " |")
    lines.append("")

    engine = RankedSearchEngine.get_instance()
    engine.search(REPORT_QUERY, k=10)  # warm the dictionary and the caches once
    hits, ms = _ms(lambda: engine.search(REPORT_QUERY, k=10))
    lines += [f"### `{REPORT_QUERY}` through `search()`", "",
              f"{len(hits)} ranked judgments in **{ms:.0f} ms** (the index and dictionary already in memory). Top 5:", "",
              "| # | doc_id | Title | BM25 |", "|---:|---|---|---:|"]
    for i, h in enumerate(hits[:5], 1):
        lines.append(f"| {i} | `{h.doc_id}` | {engine.doc_meta.get(h.doc_id, {}).get('title', '')} | {h.rel:.2f} |")
    lines.append("")

    cfg = settings()
    lines += ["## Spelling suggestions", "",
              f"Rules (D-040): k-gram Jaccard at least {cfg['spell_jaccard_min']}, Levenshtein at most {cfg['spell_max_edit']} (1 for words of 4 "
              f"letters or fewer; an adjacent swap counts as one edit: {bool(cfg['spell_transpositions'])}), ties by df. "
              f"A word in no judgment is replaced by its best candidate in at least {cfg['suggest_min_df']} judgments; "
              f"a word in at most {cfg['suggest_rare_df']} judgments, or any word of a query that finds fewer than {cfg['suggest_min_hits']}, only by "
              f"a candidate at least {cfg['suggest_df_ratio']} times as frequent. Statute numbers, citation tokens and legal abbreviations are never changed.", "",
              "| Query | Word df | Suggestion | Time (ms) | Top candidates (edit, df) |", "|---|---|---|---:|---|"]
    for q in REPORT_MISSPELLINGS:
        sug, ms = _ms(lambda: d.suggest(q, hit_count=_hit_count), repeat=3)
        words = [m.group().lower() for m in _QUERY_WORD.finditer(q) if _correctable(q, m)]
        dfs = ", ".join(f"{w} {d.df_of(w)}" for w in words) or "(nothing correctable)"
        cands = []
        for w in words:
            cs = d.candidates(w, 1 if len(w) <= 4 else int(cfg["spell_max_edit"]), float(cfg["spell_jaccard_min"]),
                              bool(cfg["spell_transpositions"]))[:3]
            cands.append(f"{w}: " + ", ".join(f"{c} ({e}, {df})" for c, e, df, _ in cs))
        lines.append(f"| `{q}` | {dfs} | {('`' + sug + '`') if sug else 'none'} | {ms:.0f} | {'; '.join(cands)} |")
    lines.append("")

    sx = SoundexIndex((doc_id, m.get("title") or "") for doc_id, m in engine.doc_meta.items())
    lines += ["## Soundex on party names", "",
              f"{len(sx.codes):,} distinct codes over the words of {len(engine.doc_meta):,} case titles. `search(q, filters={{'party': name}})` keeps the "
              "judgments whose title has a word with the same code as each word of the name.", "",
              "| Name | Code | Judgments whose title matches | Examples |", "|---|---|---:|---|"]
    for name in REPORT_NAMES:
        docs = sorted(sx.match(name))
        ex = "; ".join(engine.doc_meta[x]["title"] for x in docs[:2])
        lines.append(f"| {name} | {soundex(name)} | {len(docs)} | {ex} |")
    lines.append("")
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return text


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "help"
    if cmd == "build":
        rest = args[1:]
        jpath = rest[rest.index("--judgments") + 1] if "--judgments" in rest else None
        out = rest[rest.index("--out") + 1] if "--out" in rest else None
        build_from_processed(jpath, out)
        return 0
    if cmd == "report":
        write_report()
        print(f"Wrote {REPORT_PATH}")
        return 0
    if cmd == "wildcard" and len(args) > 1:
        print(expand_wildcard(args[1], int(args[2]) if len(args) > 2 else 50))
        return 0
    if cmd == "suggest" and len(args) > 1:
        print(suggest(" ".join(args[1:])))
        return 0
    if cmd == "soundex" and len(args) > 1:
        print(soundex(" ".join(args[1:])))
        return 0
    print("usage: python -m m1_index.dictionary build [--judgments FILE] [--out DIR] | report | wildcard PATTERN [LIMIT] | "
          "suggest QUERY | soundex NAME")
    return 2


if __name__ == "__main__":
    sys.exit(main())
