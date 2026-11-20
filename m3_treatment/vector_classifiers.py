"""Vector space classification of citation windows (IIR chapter 14): Rocchio and k nearest neighbours, written by us.

Both work on the tf-idf (ltc) vectors of m3_treatment.features, built with M1's tokenizer, so a window is a point in the same
term space as the search index, and similarity is the cosine.

Rocchio (IIR 14.2): one centroid per class, mu(c) = (1 / |D_c|) * sum of the unit vectors of the training windows of c. A
window goes to the class whose centroid has the largest cosine with it. It is a LINEAR classifier: the boundary between two
classes is the hyperplane of points equidistant from their centroids. It assumes each class is one roughly spherical blob,
which a rare and varied class such as "overruled" ("is hereby overruled", "no longer good law", "per incuriam") is not.

kNN (IIR 14.3): the k training windows with the largest cosine vote, each vote weighted by its cosine. It is NON-LINEAR (the
boundary follows the training points), has low bias and high variance: k = 1 copies the nearest window's label, noise and
all; a larger k smooths. k is chosen from {1, 3, 5, 7} by cross-validation on the training data only. The neighbours are
found through an inverted index over the training vectors (term -> postings of (window, weight)), accumulating dot
products term by term, as a search engine scores documents, then a heap keeps the top k.

Confidence: Rocchio, the share of the best centroid's cosine in the sum over all centroids; kNN, the share of the winning
class in the weighted vote. Both are in [0, 1]; neither is a calibrated probability.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Sequence

from m3_treatment.features import SparseVec, TfIdf, cosine

Tokens = Sequence[str]


class Rocchio:
    def __init__(self) -> None:
        self.tfidf = TfIdf()
        self.centroids: dict[str, SparseVec] = {}

    def fit(self, docs: Sequence[Tokens], labels: Sequence[str]) -> "Rocchio":
        if not docs:
            raise ValueError("no training windows")
        vecs = self.tfidf.fit(docs).transform(docs)
        sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        sizes = Counter(labels)
        for v, lab in zip(vecs, labels):
            for t, w in v.items():
                sums[lab][t] += w
        for c in sorted(sizes):
            mu = {t: w / sizes[c] for t, w in sums[c].items()}
            norm = math.sqrt(sum(w * w for w in mu.values()))
            # Comparing cosines needs unit centroids; the centroid itself is the mean (IIR eq. 14.1).
            self.centroids[c] = {t: w / norm for t, w in mu.items()} if norm else {}
        return self

    def predict_vec(self, vec: SparseVec) -> tuple[str, float]:
        sims = {c: cosine(vec, mu) for c, mu in self.centroids.items()}
        best = min(sims, key=lambda c: (-sims[c], c))
        total = sum(max(s, 0.0) for s in sims.values())
        return best, (sims[best] / total if total > 0 else 0.0)

    def predict_one(self, toks: Tokens) -> tuple[str, float]:
        return self.predict_vec(self.tfidf.transform_one(toks))

    def predict(self, docs: Sequence[Tokens]) -> list[tuple[str, float]]:
        return [self.predict_one(d) for d in docs]


class KNN:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.tfidf = TfIdf()
        self.labels: list[str] = []
        self.postings: dict[str, list[tuple[int, float]]] = {}
        self.majority = ""

    def fit(self, docs: Sequence[Tokens], labels: Sequence[str]) -> "KNN":
        if not docs:
            raise ValueError("no training windows")
        vecs = self.tfidf.fit(docs).transform(docs)
        self.labels = list(labels)
        post: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, v in enumerate(vecs):
            for t, w in v.items():
                post[t].append((i, w))
        self.postings = dict(post)
        counts = Counter(labels)
        self.majority = min(counts, key=lambda c: (-counts[c], c))
        return self

    def neighbours(self, vec: SparseVec, k: int) -> list[tuple[float, int]]:
        """The k training windows with the largest cosine, best first: [(cosine, index)]. Term-at-a-time accumulation."""
        acc: dict[int, float] = defaultdict(float)
        for t, w in vec.items():
            for i, wi in self.postings.get(t, ()):
                acc[i] += w * wi
        return heapq.nlargest(k, ((s, -i) for i, s in acc.items() if s > 0))  # -i: ties go to the earlier window

    def vote(self, nbrs: list[tuple[float, int]], k: int) -> tuple[str, float]:
        nbrs = nbrs[:k]
        if not nbrs:
            return self.majority, 0.0  # shares no term with any training window: the prior class, with no confidence
        votes: dict[str, float] = defaultdict(float)
        for s, neg_i in nbrs:
            votes[self.labels[-neg_i]] += s
        best = min(votes, key=lambda c: (-votes[c], c))
        return best, votes[best] / sum(votes.values())

    def predict_one(self, toks: Tokens, k: int | None = None) -> tuple[str, float]:
        k = k or self.k
        return self.vote(self.neighbours(self.tfidf.transform_one(toks), k), k)

    def predict(self, docs: Sequence[Tokens], k: int | None = None) -> list[tuple[str, float]]:
        return [self.predict_one(d, k) for d in docs]


def cross_validate_knn(docs: Sequence[Tokens], labels: Sequence[str], ks: Sequence[int], folds: int, seed: int = 0) -> list[dict]:
    """Macro-F1 of kNN for every k over stratified folds of the TRAINING data (the neighbours are found once per window, for
    the largest k, and every smaller k reuses the head of that list)."""
    from m3_treatment.features import macro_f1, stratified_folds

    parts = stratified_folds(labels, folds, seed)
    kmax = max(ks)
    preds = {k: [""] * len(labels) for k in ks}
    for held in parts:
        held_set = set(held)
        tr = [i for i in range(len(labels)) if i not in held_set]
        model = KNN(kmax).fit([docs[i] for i in tr], [labels[i] for i in tr])
        for i in held:
            nbrs = model.neighbours(model.tfidf.transform_one(docs[i]), kmax)
            for k in ks:
                preds[k][i] = model.vote(nbrs, k)[0]
    present = sorted(set(labels))
    return [{"k": k, "macro_f1": macro_f1(labels, preds[k], present)} for k in ks]


def cross_validate_rocchio(docs: Sequence[Tokens], labels: Sequence[str], folds: int, seed: int = 0) -> dict:
    """Macro-F1 of Rocchio over the same folds (Rocchio has no hyperparameter; reported for comparison)."""
    from m3_treatment.features import macro_f1, stratified_folds

    pred = [""] * len(labels)
    for held in stratified_folds(labels, folds, seed):
        held_set = set(held)
        tr = [i for i in range(len(labels)) if i not in held_set]
        model = Rocchio().fit([docs[i] for i in tr], [labels[i] for i in tr])
        for i in held:
            pred[i] = model.predict_one(docs[i])[0]
    return {"macro_f1": macro_f1(labels, pred, sorted(set(labels)))}


# ----------------------------------------------------------------------------------------------------------------
# Public API (Phase 2 contract)
# ----------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _rocchio() -> Rocchio:
    from m3_treatment.compare import training_set

    return Rocchio().fit(*training_set())


@lru_cache(maxsize=1)
def _knn() -> KNN:
    from m3_treatment.compare import training_set

    return KNN().fit(*training_set())


def classify_rocchio(window: str) -> tuple[str, float]:
    """(label, confidence) for a marked citation window, from the class centroids of the silver training set."""
    from m3_treatment.features import window_tokens

    return _rocchio().predict_one(window_tokens(window))


def classify_knn(window: str, k: int = 5) -> tuple[str, float]:
    """(label, confidence) by a cosine-weighted vote of the k nearest silver training windows. The default k = 5 is fixed by
    the contract; the k chosen by cross-validation (reports/model_selection.json) is what the F1 table and
    treatment_preds.jsonl use, and callers who want it pass it explicitly."""
    from m3_treatment.features import window_tokens

    return _knn().predict_one(window_tokens(window), k)
