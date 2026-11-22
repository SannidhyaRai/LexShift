"""M4 Phase 2: our K-means, the choice of K, and the external cluster-evaluation measures (m4_rank/cluster.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from m4_rank import cluster as C


def blobs(seed: int = 0, per: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(per, 2)) for c in centres])
    y = np.repeat(np.arange(3), per)
    return X, y


def test_kmeans_recovers_well_separated_blobs():
    X, y = blobs()
    r = C.best_of(X, 3, seed=0, n_init=3)
    assert r.converged
    scores = C.external_scores(r.assign.tolist(), [str(v) for v in y])
    assert scores["purity"] == 1.0 and scores["nmi"] == pytest.approx(1.0) and scores["rand_index"] == 1.0


def test_rss_never_increases_between_iterations():
    X, _ = blobs(1)
    r = C.kmeans(X, 5, seed=3)
    assert all(b <= a + 1e-9 for a, b in zip(r.history, r.history[1:]))


def test_kmeans_is_deterministic_for_a_seed():
    X, _ = blobs(2)
    a, b = C.kmeans(X, 4, seed=7), C.kmeans(X, 4, seed=7)
    assert np.array_equal(a.assign, b.assign) and a.rss == b.rss


def test_every_cluster_is_non_empty_even_with_duplicate_points():
    X = np.vstack([np.zeros((10, 2)), np.ones((2, 2))])  # far fewer distinct points than clusters
    r = C.kmeans(X, 5, seed=0)
    assert set(np.bincount(r.assign, minlength=5)) - {0} and (np.bincount(r.assign, minlength=5) > 0).all()


def test_kmeans_pp_rejects_bad_k():
    X, _ = blobs()
    with pytest.raises(ValueError):
        C.kmeans_pp(X, 0, np.random.default_rng(0))


def test_elbow_and_aic():
    ks = [1, 2, 3, 4, 5, 6]
    rss = [100.0, 40.0, 20.0, 17.0, 15.0, 14.0]
    assert C.elbow(ks, rss) == 3
    assert C.aic_k(ks, rss, dims=1) == 4   # RSS + 2K: 102, 44, 26, 25, 25, 26 -> the first minimum
    assert C.aic_k(ks, rss, dims=1000) == 1  # a large M makes the penalty dominate


def test_external_scores_match_iir_figure_16_4():
    # IIR figure 16.4: three clusters of x / o / diamond. Purity 0.71, RI 0.68, NMI 0.36, F5 0.46 (rounded in the book).
    clusters = [0] * 6 + [1] * 6 + [2] * 5
    classes = list("xxxxxo") + list("xooood") + list("xxddd")
    s = C.external_scores(clusters, classes)
    assert s["purity"] == pytest.approx(12 / 17)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (20, 20, 24, 72)
    assert s["rand_index"] == pytest.approx(92 / 136)
    assert s["nmi"] == pytest.approx(0.36, abs=0.01)
    assert s["f5"] == pytest.approx(0.456, abs=0.001)


def test_external_scores_agree_with_scikit_learn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    clusters = rng.integers(0, 6, size=400).tolist()
    classes = [f"c{v}" for v in rng.integers(0, 4, size=400)]
    s = C.external_scores(clusters, classes)
    assert s["nmi"] == pytest.approx(sk.normalized_mutual_info_score(classes, clusters, average_method="arithmetic"))
    assert s["rand_index"] == pytest.approx(sk.rand_score(classes, clusters))


def test_select_terms_window_digits_and_ties():
    terms = ["a", "b", "c", "123", "d", "e"]
    df = np.array([5, 50, 20, 30, 20, 99])
    chosen = C.select_terms(terms, df, n_docs=100, vocab_size=2, min_df=10, max_df_ratio=0.6)
    assert [terms[t] for t in chosen] == ["b", "c"]  # "e" too common, "a" too rare, "123" digits, "c" beats "d" on the tie


def test_differential_labels_prefer_distinctive_terms():
    cents = np.array([[0.9, 0.8, 0.0], [0.9, 0.0, 0.7]])
    terms = ["court", "murder", "bail"]
    assert C.differential_terms(cents, terms, 1) == [["murder"], ["bail"]]
    assert C.centroid_terms(cents, terms, 1) == [["court"], ["court"]]


def test_cluster_of_reads_frozen_files(tmp_path, monkeypatch):
    out, meta = tmp_path / "clusters.jsonl", tmp_path / "meta.json"
    out.write_text(json.dumps({"doc_id": "D1", "cluster_id": 1, "distance_to_centroid": 0.5}) + "\n"
                   + json.dumps({"doc_id": "D2", "cluster_id": 1, "distance_to_centroid": 0.2}) + "\n", encoding="utf-8")
    meta.write_text(json.dumps({"k": 2, "labels": ["zero", "one"], "clusters": []}), encoding="utf-8")
    cfg = {"clustering": {"out": str(out), "meta": str(meta)}}
    monkeypatch.setattr(C, "load_config", lambda: cfg)
    C._load.cache_clear()
    try:
        assert C.cluster_of("D1") == (1, "one")
        assert C.members(1) == [("D2", 0.2), ("D1", 0.5)]
        with pytest.raises(KeyError):
            C.cluster_of("NOPE")
        with pytest.raises(KeyError):
            C.members(5)
    finally:
        C._load.cache_clear()


def test_missing_artefacts_give_a_clear_error(tmp_path, monkeypatch):
    cfg = {"clustering": {"out": str(tmp_path / "x.jsonl"), "meta": str(tmp_path / "x.json")}}
    monkeypatch.setattr(C, "load_config", lambda: cfg)
    C._load.cache_clear()
    try:
        with pytest.raises(C.ClusterError, match="m4_rank.cluster build"):
            C.cluster_of("D1")
    finally:
        C._load.cache_clear()
