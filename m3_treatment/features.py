"""Features shared by the course classifiers (Naive Bayes, Rocchio, kNN): tokens and tf-idf vectors of a citation window.

Tokens come from M1's tokenizer (m1_index.tokenizer.tokenize: case folding, stop words, Porter stemming, statute tokens kept
whole), so a window's vector lives in the same term space as the search index.

One window often cites several cases, and the label is about the TARGET only (the mention inside [[ ]]). Two things make the
features target-aware without any keyword rule:
  * the target mention is replaced by one placeholder token, so party names (noise for treatment) never become features and
    the placeholder anchors the position of the target;
  * the text near the target (LOCAL characters either side) is tokenized a second time with the prefix "n:", so the
    classifier can learn that "overrul" right after the target matters more than "overrul" two sentences away.

Known limitation (DECISIONS.md D-P2-03): M1's stop list removes "not" and "no", so "not good law" and "good law" share
features. We keep M1's tokenizer, as the Phase 2 guide asks, and report the cost instead of forking it.

tf-idf here is the "ltc" weighting of the SMART notation: (1 + log10 tf) * log10(N / df), cosine-normalised. idf is
computed on the TRAINING windows only, so nothing about a test window leaks into its own vector.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

from m3_treatment.classifier import local_context

TARGET_TOKEN = "zztarget"  # survives M1's tokenizer unchanged (not a stop word, Porter leaves it alone)
_TARGET = re.compile(r"\[\[(.+?)\]\]", re.DOTALL)
NEAR = "n:"


def _tokenize(text: str) -> list[str]:
    from m1_index.tokenizer import tokenize

    return tokenize(text)


def window_tokens(marked_window: str) -> list[str]:
    """Tokens of the whole window plus "n:"-prefixed tokens of the text near the target (both with the target replaced)."""
    near = local_context(marked_window)
    whole = _TARGET.sub(f" {TARGET_TOKEN} ", marked_window)
    near = _TARGET.sub(f" {TARGET_TOKEN} ", near)
    return _tokenize(whole) + [NEAR + t for t in _tokenize(near)]


# ----------------------------------------------------------------------------------------------------------------
# tf-idf (ltc) with our own vectoriser
# ----------------------------------------------------------------------------------------------------------------
SparseVec = dict[str, float]


class TfIdf:
    """Fit idf on training token lists; transform token lists into cosine-normalised (1 + log tf) * idf vectors."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.n_docs = 0

    def fit(self, docs: Sequence[Sequence[str]]) -> "TfIdf":
        df: Counter = Counter()
        for toks in docs:
            df.update(set(toks))
        self.n_docs = len(docs)
        self.idf = {t: math.log10(self.n_docs / d) for t, d in df.items()}
        return self

    def transform_one(self, toks: Iterable[str]) -> SparseVec:
        tf = Counter(t for t in toks if t in self.idf)
        vec = {t: (1.0 + math.log10(c)) * self.idf[t] for t, c in tf.items()}
        vec = {t: w for t, w in vec.items() if w > 0}  # a term in every training window has idf 0 and carries nothing
        norm = math.sqrt(sum(w * w for w in vec.values()))
        return {t: w / norm for t, w in vec.items()} if norm else {}

    def transform(self, docs: Sequence[Sequence[str]]) -> list[SparseVec]:
        return [self.transform_one(d) for d in docs]


def cosine(a: SparseVec, b: SparseVec) -> float:
    """Dot product of two vectors; equals the cosine when both are unit length (as TfIdf produces them)."""
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


# ----------------------------------------------------------------------------------------------------------------
# Folds and scoring for model selection (training data only)
# ----------------------------------------------------------------------------------------------------------------
def stratified_folds(labels: Sequence[str], k: int, seed: int = 0) -> list[list[int]]:
    """k lists of indices; every class is dealt round-robin over the folds after a seeded shuffle, so each fold keeps the
    class proportions (a rare class such as overruled appears in every fold when it has at least k examples)."""
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    offset = 0
    for lab in sorted(by_class):
        idx = by_class[lab]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[(j + offset) % k].append(i)
        offset += len(idx)  # start the next class where this one stopped, so fold sizes stay even
    return [sorted(f) for f in folds]


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Unweighted mean of per-class F1 over the classes present in `gold` (or `labels`)."""
    classes = list(labels) if labels is not None else sorted(set(gold))
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0
