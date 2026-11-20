"""Naive Bayes treatment classifier, written by us (IIR chapter 13). Labels: followed / distinguished / doubted / overruled / neutral.

Multinomial NB (IIR fig. 13.2):
    c* = argmax_c  log P(c) + sum over token occurrences t in the window of log P(t | c)
    P(c)     = N_c / N                                          (share of training windows in class c)
    P(t | c) = (T_ct + 1) / (sum_t' T_ct' + |V|)                (add-one / Laplace smoothing over the vocabulary V)
Scores are sums of logs, never products of probabilities, so a 300-token window does not underflow to 0.

Bernoulli NB (IIR fig. 13.3), the optional variant: one binary event per vocabulary term, present or absent,
    P(t | c) = (N_ct + 1) / (N_c + 2),   score = log P(c) + sum_{t in V} log [P(t|c) if t in window else 1 - P(t|c)].
It ignores how often a term occurs and counts every ABSENT term as evidence, which is why it does worse on long windows: a
long window is penalised by thousands of absences that say nothing about treatment (IIR 13.3).

Feature selection (IIR 13.5): mutual information or chi-square between "term occurs in the window" and "window is in
class c", computed from the 2x2 document counts; a term's utility is its maximum over the classes, and the top-k terms form
V. The method and k are chosen by cross-validation on the TRAINING data only (compare.py select).

Confidence = the posterior of the chosen class, normalised over the classes (log-sum-exp). NB posteriors are known to be
over-confident (the independence assumption multiplies correlated evidence), so treat it as a ranking, not a probability.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Sequence

Tokens = Sequence[str]


# ----------------------------------------------------------------------------------------------------------------
# Feature selection
# ----------------------------------------------------------------------------------------------------------------
def _doc_counts(docs: Sequence[Tokens], labels: Sequence[str]) -> tuple[dict[str, Counter], Counter, int]:
    """(term -> class -> number of training windows of that class containing the term, class sizes, N)."""
    df: dict[str, Counter] = defaultdict(Counter)
    for toks, lab in zip(docs, labels):
        for t in set(toks):
            df[t][lab] += 1
    return df, Counter(labels), len(labels)


def _mi(n11: int, n10: int, n01: int, n00: int) -> float:
    """Expected mutual information I(U; C) in bits from the 2x2 table (IIR eq. 13.17); 0 * log 0 is taken as 0."""
    n = n11 + n10 + n01 + n00
    total = 0.0
    for nij, ni, nj in ((n11, n11 + n10, n11 + n01), (n01, n01 + n00, n11 + n01), (n10, n11 + n10, n10 + n00), (n00, n01 + n00, n10 + n00)):
        if nij:
            total += nij / n * math.log2(n * nij / (ni * nj))
    return total


def _chi2(n11: int, n10: int, n01: int, n00: int) -> float:
    """Chi-square statistic of the 2x2 table (IIR eq. 13.19)."""
    denom = (n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00)
    return (n11 + n10 + n01 + n00) * (n11 * n00 - n10 * n01) ** 2 / denom if denom else 0.0


UTILITIES = {"mi": _mi, "chi2": _chi2}


def term_utilities(docs: Sequence[Tokens], labels: Sequence[str], method: str) -> dict[str, float]:
    """term -> max over classes of the utility (MI or chi-square) of the term for that class."""
    fn = UTILITIES[method]
    df, sizes, n = _doc_counts(docs, labels)
    out = {}
    for t, by_class in df.items():
        n_t = sum(by_class.values())
        best = 0.0
        for c, n_c in sizes.items():
            n11 = by_class.get(c, 0)
            best = max(best, fn(n11, n_t - n11, n_c - n11, n - n_t - n_c + n11))
        out[t] = best
    return out


def top_k(utilities: dict[str, float], k: int) -> set[str] | None:
    """The k terms of highest utility (ties broken alphabetically, so the choice is reproducible); None for k = 0 (all terms)."""
    if not k or k >= len(utilities):
        return None
    return {t for t, _ in sorted(utilities.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}


# ----------------------------------------------------------------------------------------------------------------
# Classifiers
# ----------------------------------------------------------------------------------------------------------------
def _posterior(scores: dict[str, float]) -> tuple[str, float]:
    """(argmax class, its posterior) from log scores; ties go to the alphabetically first class."""
    best = min(scores, key=lambda c: (-scores[c], c))
    top = scores[best]
    z = sum(math.exp(s - top) for s in scores.values())
    return best, 1.0 / z


class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes: list[str] = []
        self.log_prior: dict[str, float] = {}
        self.log_cond: dict[str, dict[str, float]] = {}
        self.log_unseen: dict[str, float] = {}
        self.vocab: set[str] = set()

    def fit(self, docs: Sequence[Tokens], labels: Sequence[str], vocab: set[str] | None = None) -> "MultinomialNB":
        if not docs:
            raise ValueError("no training windows")
        self.vocab = set(vocab) if vocab is not None else {t for d in docs for t in d}
        sizes = Counter(labels)
        self.classes = sorted(sizes)
        n = len(labels)
        counts: dict[str, Counter] = {c: Counter() for c in self.classes}
        for toks, lab in zip(docs, labels):
            counts[lab].update(t for t in toks if t in self.vocab)
        v = len(self.vocab)
        for c in self.classes:
            self.log_prior[c] = math.log(sizes[c] / n)
            denom = sum(counts[c].values()) + self.alpha * v
            self.log_cond[c] = {t: math.log((k + self.alpha) / denom) for t, k in counts[c].items()}
            self.log_unseen[c] = math.log(self.alpha / denom)  # a vocabulary term never seen with class c
        return self

    def scores(self, toks: Tokens) -> dict[str, float]:
        kept = Counter(t for t in toks if t in self.vocab)  # terms outside V carry no evidence (IIR: ignored at test time)
        return {
            c: self.log_prior[c] + sum(k * self.log_cond[c].get(t, self.log_unseen[c]) for t, k in kept.items())
            for c in self.classes
        }

    def predict_one(self, toks: Tokens) -> tuple[str, float]:
        return _posterior(self.scores(toks))

    def predict(self, docs: Sequence[Tokens]) -> list[tuple[str, float]]:
        return [self.predict_one(d) for d in docs]


class BernoulliNB:
    def __init__(self) -> None:
        self.classes: list[str] = []
        self.log_prior: dict[str, float] = {}
        self.log_p: dict[str, dict[str, float]] = {}  # log P(t | c)
        self.gain: dict[str, dict[str, float]] = {}  # log P(t|c) - log (1 - P(t|c)): what presence adds over absence
        self.base: dict[str, float] = {}  # sum over V of log (1 - P(t | c)): the score of a window with no vocabulary term
        self.vocab: set[str] = set()

    def fit(self, docs: Sequence[Tokens], labels: Sequence[str], vocab: set[str] | None = None) -> "BernoulliNB":
        if not docs:
            raise ValueError("no training windows")
        self.vocab = set(vocab) if vocab is not None else {t for d in docs for t in d}
        sizes = Counter(labels)
        self.classes = sorted(sizes)
        df: dict[str, Counter] = {c: Counter() for c in self.classes}
        for toks, lab in zip(docs, labels):
            df[lab].update(set(toks) & self.vocab)
        n = len(labels)
        for c in self.classes:
            self.log_prior[c] = math.log(sizes[c] / n)
            gain, base = {}, 0.0
            for t in self.vocab:
                p = (df[c].get(t, 0) + 1) / (sizes[c] + 2)
                base += math.log(1 - p)
                gain[t] = math.log(p) - math.log(1 - p)
            self.gain[c], self.base[c] = gain, base
        return self

    def scores(self, toks: Tokens) -> dict[str, float]:
        present = set(toks) & self.vocab
        return {c: self.log_prior[c] + self.base[c] + sum(self.gain[c][t] for t in present) for c in self.classes}

    def predict_one(self, toks: Tokens) -> tuple[str, float]:
        return _posterior(self.scores(toks))

    def predict(self, docs: Sequence[Tokens]) -> list[tuple[str, float]]:
        return [self.predict_one(d) for d in docs]


VARIANTS = {"multinomial": MultinomialNB, "bernoulli": BernoulliNB}


def train(docs: Sequence[Tokens], labels: Sequence[str], variant: str = "multinomial", selection: str = "chi2", k: int = 0):
    """Feature selection on these training windows, then fit. k = 0 keeps every term."""
    vocab = top_k(term_utilities(docs, labels, selection), k) if k else None
    return VARIANTS[variant]().fit(docs, labels, vocab)


def cross_validate(
    docs: Sequence[Tokens],
    labels: Sequence[str],
    variants: Sequence[str],
    methods: Sequence[str],
    ks: Sequence[int],
    folds: int,
    seed: int = 0,
) -> list[dict]:
    """Macro-F1 of every (variant, selection method, k) over stratified folds of the TRAINING data. One row per setting."""
    from m3_treatment.features import macro_f1, stratified_folds

    parts = stratified_folds(labels, folds, seed)
    preds: dict[tuple, list[str]] = defaultdict(lambda: [""] * len(labels))
    for held in parts:
        held_set = set(held)
        tr = [i for i in range(len(labels)) if i not in held_set]
        tr_docs, tr_labels = [docs[i] for i in tr], [labels[i] for i in tr]
        for method in methods:
            util = term_utilities(tr_docs, tr_labels, method)  # computed once per fold, reused for every k
            for k in ks:
                vocab = top_k(util, k)
                for variant in variants:
                    if method != methods[0] and not k:
                        continue  # k = 0 (every term) is the same model whatever the utility
                    model = VARIANTS[variant]().fit(tr_docs, tr_labels, vocab)
                    key = (variant, method, k)
                    for i in held:
                        preds[key][i] = model.predict_one(docs[i])[0]
    labels_present = sorted(set(labels))
    return [
        {"variant": v, "selection": m if k else "none", "k": k, "macro_f1": macro_f1(labels, p, labels_present)}
        for (v, m, k), p in sorted(preds.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    ]


# ----------------------------------------------------------------------------------------------------------------
# Public API (Phase 2 contract): classify_nb(window) -> (label, confidence)
# ----------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _model():
    from m3_treatment.compare import load_selection, training_set

    sel = load_selection()["nb"]
    docs, labels = training_set()
    return train(docs, labels, sel["variant"], sel["selection"], sel["k"])


def classify_nb(window: str) -> tuple[str, float]:
    """(label, confidence) for a marked citation window ([[ ]] around the cited case), from the NB variant, feature selection
    and k chosen by cross-validation on the silver training set (reports/model_selection.json)."""
    from m3_treatment.features import window_tokens

    return _model().predict_one(window_tokens(window))
