"""M3 Phase 2: Naive Bayes, Rocchio, kNN, HITS and the classifier comparison.

Where the textbook gives a worked example (IIR examples 13.1 to 13.4) the test reproduces its numbers; scikit-learn and SciPy
are used only as cross-checks of our own implementations, never inside them.
"""

import json
import math
import random

import pytest
import yaml

from common import config as common_config
from common.config import DEFAULT_CONFIG
from common.io import write_jsonl
from m3_treatment.features import TARGET_TOKEN, TfIdf, cosine, macro_f1, stratified_folds, window_tokens
from m3_treatment.hits import Graph, base_set, graph_from_citations, hits_on, kendall_tau_b, overlap_at
from m3_treatment.nb import BernoulliNB, MultinomialNB, _chi2, _mi, cross_validate, term_utilities, top_k, train
from m3_treatment.vector_classifiers import KNN, Rocchio, cross_validate_knn

# IIR table 13.1: is the document about China?
IIR_TRAIN = [
    ("Chinese Beijing Chinese".split(), "c"),
    ("Chinese Chinese Shanghai".split(), "c"),
    ("Chinese Macao".split(), "c"),
    ("Tokyo Japan Chinese".split(), "j"),
]
IIR_TEST = "Chinese Chinese Chinese Tokyo Japan".split()


# ----------------------------------------------------------------------------------------------------------------
# Naive Bayes
# ----------------------------------------------------------------------------------------------------------------
def test_multinomial_nb_reproduces_iir_example_13_1():
    nb = MultinomialNB().fit([d for d, _ in IIR_TRAIN], [c for _, c in IIR_TRAIN])
    s = nb.scores(IIR_TEST)
    # P(c|d) ~ 3/4 * (3/7)^3 * 1/14 * 1/14 ;  P(j|d) ~ 1/4 * (2/9)^3 * 2/9 * 2/9
    assert math.exp(s["c"]) == pytest.approx(3 / 4 * (3 / 7) ** 3 * (1 / 14) ** 2)
    assert math.exp(s["j"]) == pytest.approx(1 / 4 * (2 / 9) ** 3 * (2 / 9) ** 2)
    label, conf = nb.predict_one(IIR_TEST)
    assert label == "c" and 0.5 < conf < 1


def test_bernoulli_nb_reproduces_iir_example_13_2():
    nb = BernoulliNB().fit([d for d, _ in IIR_TRAIN], [c for _, c in IIR_TRAIN])
    s = nb.scores(IIR_TEST)
    # IIR: P(c|d) ~ 3/4 * 4/5 * 1/5 * 1/5 * (1-2/5)(1-2/5)(1-2/5) ≈ 0.005 ; P(j|d) ~ 1/4 * 2/3 * 2/3 * 2/3 * (1-1/3)^3 ≈ 0.022
    assert math.exp(s["c"]) == pytest.approx(3 / 4 * 4 / 5 * 1 / 5 * 1 / 5 * (3 / 5) ** 3)
    assert math.exp(s["j"]) == pytest.approx(1 / 4 * (2 / 3) ** 3 * (1 - 1 / 3) ** 3)
    assert nb.predict_one(IIR_TEST)[0] == "j"  # the Bernoulli model gets this one "wrong", as the book shows


def test_mutual_information_and_chi2_reproduce_iir_examples_13_3_and_13_4():
    # class poultry, term export (Reuters-RCV1)
    assert _mi(49, 27652, 141, 774106) == pytest.approx(0.0001105, rel=1e-3)
    assert _chi2(49, 27652, 141, 774106) == pytest.approx(284, rel=1e-2)


def test_feature_selection_keeps_the_class_indicative_terms():
    docs = [["overrul", "x"], ["overrul", "y"], ["follow", "x"], ["follow", "y"]]
    labels = ["overruled", "overruled", "followed", "followed"]
    for method in ("mi", "chi2"):
        util = term_utilities(docs, labels, method)
        assert top_k(util, 2) == {"overrul", "follow"}
        assert util["x"] == pytest.approx(0.0)
    assert top_k({"a": 1.0}, 0) is None  # k = 0 means every term


def test_nb_matches_sklearn_on_random_counts():
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.naive_bayes import BernoulliNB as SkB
    from sklearn.naive_bayes import MultinomialNB as SkM

    rng = random.Random(3)
    vocab = [f"t{i}" for i in range(30)]
    labels = [rng.choice("abc") for _ in range(80)]
    docs = [[rng.choice(vocab[: 10 + 10 * "abc".index(lab)]) for _ in range(rng.randint(3, 25))] for lab in labels]
    tests = [[rng.choice(vocab) for _ in range(rng.randint(3, 25))] for _ in range(40)]
    dv = DictVectorizer()
    count = lambda d: {t: d.count(t) for t in set(d)}  # noqa: E731
    X = dv.fit_transform([count(d) for d in docs])
    Xt = dv.transform([count(d) for d in tests])

    ours = MultinomialNB().fit(docs, labels)
    sk = SkM(alpha=1.0).fit(X, labels)
    sk_post = sk.predict_proba(Xt)
    for i, d in enumerate(tests):
        lab, conf = ours.predict_one(d)
        assert lab == sk.classes_[sk_post[i].argmax()]
        assert conf == pytest.approx(sk_post[i].max(), rel=1e-6)

    ours_b = BernoulliNB().fit(docs, labels)
    skb = SkB(alpha=1.0, binarize=0.5).fit(X, labels)
    skb_post = skb.predict_proba(Xt)
    for i, d in enumerate(tests):
        lab, conf = ours_b.predict_one(d)
        assert lab == skb.classes_[skb_post[i].argmax()]
        assert conf == pytest.approx(skb_post[i].max(), rel=1e-6)


def test_nb_cross_validation_grid_shape():
    rng = random.Random(0)
    labels = ["a"] * 20 + ["b"] * 20
    docs = [["x", rng.choice("pq")] if lab == "a" else ["y", rng.choice("pq")] for lab in labels]
    rows = cross_validate(docs, labels, ("multinomial", "bernoulli"), ("mi", "chi2"), (2, 0), folds=4)
    # k = 0 (all terms) appears once per variant, not once per selection method
    assert sorted((r["variant"], r["selection"], r["k"]) for r in rows) == sorted(
        [(v, m, 2) for v in ("multinomial", "bernoulli") for m in ("mi", "chi2")] + [("multinomial", "none", 0), ("bernoulli", "none", 0)]
    )
    assert all(r["macro_f1"] == pytest.approx(1.0) for r in rows)  # the classes are separable by x / y
    assert train(docs, labels, "multinomial", "mi", 2).vocab == {"x", "y"}  # p and q carry no information
    # With k = 1 only "x" is kept: a class-b window has no vocabulary term, so multinomial NB falls back to the prior,
    # while Bernoulli NB still uses the ABSENCE of "x" as evidence.
    assert train(docs, labels, "bernoulli", "mi", 1).predict_one(["y", "p"])[0] == "b"


# ----------------------------------------------------------------------------------------------------------------
# Features, Rocchio, kNN
# ----------------------------------------------------------------------------------------------------------------
def test_window_tokens_replace_the_target_and_add_near_features():
    toks = window_tokens("Many cases were cited. For these reasons [[Suresh Kumar Koushal (supra)]] is hereby overruled.")
    assert TARGET_TOKEN in toks and "n:" + TARGET_TOKEN in toks
    assert "koushal" not in toks and "n:koushal" not in toks  # party names never become features
    assert "n:overrul" in toks and "overrul" in toks


def test_tfidf_vectors_are_unit_length_and_drop_terms_in_every_window():
    tf = TfIdf().fit([["a", "b"], ["a", "c"], ["a", "c", "c"]])
    v = tf.transform_one(["a", "c", "c", "zzz"])
    assert "a" not in v and "zzz" not in v  # idf(a) = 0; unseen terms are ignored
    assert sum(w * w for w in v.values()) == pytest.approx(1.0)
    assert cosine(v, v) == pytest.approx(1.0)


def test_stratified_folds_partition_and_keep_every_class_in_every_fold():
    labels = ["neutral"] * 50 + ["followed"] * 30 + ["overruled"] * 5
    folds = stratified_folds(labels, 5, seed=1)
    assert sorted(i for f in folds for i in f) == list(range(len(labels)))
    assert all(sum(labels[i] == "overruled" for i in f) == 1 for f in folds)
    assert max(map(len, folds)) - min(map(len, folds)) <= 1


def test_macro_f1_averages_classes_equally():
    assert macro_f1(["a", "a", "a", "b"], ["a", "a", "a", "a"]) == pytest.approx((1.0 * 2 * 3 / (6 + 1) + 0.0) / 2)


def test_rocchio_assigns_to_the_nearest_centroid():
    docs = [["over", "rule"], ["over", "rule", "bench"], ["follow", "rely"], ["follow", "apply"], ["cite"]]
    labels = ["overruled", "overruled", "followed", "followed", "neutral"]
    r = Rocchio().fit(docs, labels)
    assert r.predict_one(["rule", "over"])[0] == "overruled"
    assert r.predict_one(["rely", "follow"])[0] == "followed"
    for mu in r.centroids.values():
        assert sum(w * w for w in mu.values()) == pytest.approx(1.0)
    lab, conf = r.predict_one(["unknown"])
    assert conf == 0.0


def test_knn_neighbours_match_sklearn_cosine():
    from sklearn.neighbors import NearestNeighbors

    rng = random.Random(5)
    vocab = [f"w{i}" for i in range(25)]
    docs = [[rng.choice(vocab) for _ in range(rng.randint(4, 15))] for _ in range(60)]
    labels = [rng.choice("xyz") for _ in docs]
    knn = KNN(5).fit(docs, labels)
    terms = sorted(knn.tfidf.idf)
    dense = lambda v: [v.get(t, 0.0) for t in terms]  # noqa: E731
    train_vecs = [dense(v) for v in knn.tfidf.transform(docs)]
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(train_vecs)
    for _ in range(15):
        q = [rng.choice(vocab) for _ in range(8)]
        qv = knn.tfidf.transform_one(q)
        ours = knn.neighbours(qv, 5)
        dist, idx = nn.kneighbors([dense(qv)])
        assert [round(s, 9) for s, _ in ours] == [round(1 - d, 9) for d in dist[0]]


def test_knn_vote_is_similarity_weighted_and_cv_picks_from_the_grid():
    knn = KNN(3).fit([["a"], ["a", "b"], ["b"], ["b", "c"]], ["x", "x", "y", "y"])
    lab, conf = knn.predict_one(["a"])
    assert lab == "x" and 0.5 < conf <= 1.0
    assert knn.predict_one(["nothing"]) == (knn.majority, 0.0)
    labels = ["x"] * 10 + ["y"] * 10
    docs = [["a", str(i)] for i in range(10)] + [["b", str(i)] for i in range(10)]
    rows = cross_validate_knn(docs, labels, (1, 3, 5, 7), folds=5)
    assert [r["k"] for r in rows] == [1, 3, 5, 7] and all(0 <= r["macro_f1"] <= 1 for r in rows)


# ----------------------------------------------------------------------------------------------------------------
# HITS
# ----------------------------------------------------------------------------------------------------------------
def _graph(edges):
    g = Graph()
    for s, d in edges:
        g.add(s, d)
    return g


def test_hits_matches_the_principal_eigenvectors():
    # h1 -> a1, a2 ; h2 -> a1.   A^T A = [[2, 1], [1, 1]] on (a1, a2): principal eigenvector (1, 0.618...), and h = A a.
    g = _graph([("h1", "a1"), ("h1", "a2"), ("h2", "a1")])
    hubs, auths, _ = hits_on(["h1", "h2", "a1", "a2"], g, iterations=200, tol=1e-12)
    golden = (math.sqrt(5) - 1) / 2
    assert auths["a1"] == pytest.approx(1.0) and auths["a2"] == pytest.approx(golden)
    assert hubs["h1"] == pytest.approx(1.0) and hubs["h2"] == pytest.approx(golden)
    assert auths["h1"] == 0.0 and hubs["a1"] == 0.0


def test_hits_complete_bipartite_graph_is_uniform_and_empty_graph_is_zero():
    g = _graph([(h, a) for h in ("h1", "h2") for a in ("a1", "a2", "a3")])
    hubs, auths, _ = hits_on(["h1", "h2", "a1", "a2", "a3"], g)
    assert all(auths[a] == pytest.approx(1.0) for a in ("a1", "a2", "a3"))
    assert all(hubs[h] == pytest.approx(1.0) for h in ("h1", "h2"))
    hubs, auths, _ = hits_on(["x", "y"], Graph())
    assert hubs == {"x": 0.0, "y": 0.0} and auths == {"x": 0.0, "y": 0.0}


def test_base_set_adds_cited_and_at_most_d_citing_judgments():
    g = _graph([("r", "c1"), ("r", "c2")] + [(f"in{i}", "r") for i in range(5)] + [("far", "c1")])
    base = base_set(["r"], g, max_in_per_root=2)
    assert base[0] == "r" and {"c1", "c2"} <= set(base)
    assert sum(b.startswith("in") for b in base) == 2  # Kleinberg's cap d
    assert "far" not in base  # neighbours of neighbours are not added


def test_graph_uses_only_resolved_non_appeal_edges_with_the_pagerank_labels():
    recs = [
        {"citing_doc": "a", "cited_doc": "b", "is_appeal_history": False, "label": "followed"},
        {"citing_doc": "a", "cited_doc": "b", "is_appeal_history": False, "label": "neutral"},  # parallel: one edge
        {"citing_doc": "a", "cited_doc": "c", "is_appeal_history": False, "label": "overruled"},  # not an endorsement
        {"citing_doc": "a", "cited_doc": "d", "is_appeal_history": True, "label": "neutral"},
        {"citing_doc": "a", "cited_doc": None, "is_appeal_history": False, "label": "neutral"},
        {"citing_doc": "a", "cited_doc": "a", "is_appeal_history": False, "label": "neutral"},
    ]
    g = graph_from_citations(recs, ["followed", "neutral"])
    assert sorted(g.edges()) == [("a", "b")]


def test_kendall_tau_b_matches_scipy_with_ties():
    from scipy.stats import kendalltau

    rng = random.Random(2)
    for _ in range(5):
        x = [rng.choice([0, 0, 1, 2, 3]) for _ in range(40)]
        y = [rng.choice([0, 1, 1, 2]) for _ in range(40)]
        assert kendall_tau_b(x, y) == pytest.approx(kendalltau(x, y).statistic)
    assert overlap_at({"a": 3, "b": 2, "c": 1}, {"a": 1, "c": 5, "b": 0}, 2) == pytest.approx(0.5)


# ----------------------------------------------------------------------------------------------------------------
# compare.py end to end, on a tiny project (silver -> select -> evaluate -> preds), with gold windows held out of training
# ----------------------------------------------------------------------------------------------------------------
def _w(text):
    return text.replace("TARGET", "[[Alpha v. Beta]]")


SILVER_TEXTS = {
    "overruled": ["For these reasons TARGET is hereby overruled.", "TARGET stands overruled and no longer holds the field.", "We overrule TARGET today."],
    "followed": ["We follow TARGET and apply it here.", "Relying on TARGET we follow the ratio.", "The ratio of TARGET is followed and applied."],
    "neutral": ["Counsel cited TARGET in passing.", "Reference was made to TARGET by counsel.", "TARGET was cited at the bar by counsel."],
}


@pytest.fixture()
def tiny(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open(DEFAULT_CONFIG, encoding="utf-8"))
    m3 = cfg["m3_treatment"]
    cfg["paths"]["treatment_gold"] = str(tmp_path / "gold.csv")
    m3["silver"] = str(tmp_path / "silver.jsonl")
    m3["preds"] = str(tmp_path / "preds.jsonl")
    m3["reports_dir"] = str(tmp_path / "reports")
    m3["labelling_dir"] = str(tmp_path / "labelling")
    m3["llm"]["cache"] = str(tmp_path / "llm.jsonl")
    m3["llm"]["few_shot_per_class"] = 0
    m3["classifiers"].update(folds=3, nb_k_grid=[5, 0], model_selection=str(tmp_path / "reports" / "model_selection.json"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("LEXSHIFT_CONFIG", str(path))
    common_config._load.cache_clear()
    from m3_treatment import compare

    compare._training.cache_clear()
    rows = []
    for lab, texts in SILVER_TEXTS.items():
        for i, t in enumerate(texts):
            rows.append({"window_id": f"{lab}{i}", "citing_doc": "d", "cited_doc": "e", "window": _w(t), "label": lab, "confidence": 0.9, "model": "m", "model_version": "m", "shots": "zero-shot"})
    write_jsonl(m3["silver"], rows)
    yield tmp_path
    common_config._load.cache_clear()
    compare._training.cache_clear()


def test_compare_select_without_gold_and_evaluate_refuses(tiny):
    from m3_treatment import compare

    sel = compare.run_select()
    assert sel["training"]["windows"] == 9 and sel["training"]["excluded_because_in_gold"] == 0
    assert sel["knn"]["k"] in (1, 3, 5, 7)
    assert json.loads((tiny / "reports" / "model_selection.json").read_text(encoding="utf-8"))["nb"] == sel["nb"]
    with pytest.raises(RuntimeError, match="no settled hand labels"):
        compare.run_evaluate()
    rep = compare.run_preds()
    assert rep["by_split"] == {"silver": 9, "silver_oof": 36}  # four classifiers x nine out-of-fold predictions


def test_compare_evaluate_holds_gold_windows_out_of_training(tiny):
    from common.io import write_delimited
    from common.schema import TREATMENT_GOLD_COLUMNS
    from m3_treatment import compare

    gold_window = _w(SILVER_TEXTS["overruled"][0])  # also in the silver set: must be removed from training
    gold = [
        {"window_id": "g1", "window": gold_window, "gold_label": "overruled", "labeller": "L1"},
        {"window_id": "g2", "window": _w("We respectfully follow TARGET."), "gold_label": "followed", "labeller": "L1"},
        {"window_id": "g3", "window": _w("TARGET was cited by counsel for the State."), "gold_label": "neutral", "labeller": "L1"},
    ]
    write_delimited(tiny / "gold.csv", TREATMENT_GOLD_COLUMNS, gold)
    compare.run_select()
    windows, _labels, _docs, excluded = compare._training()
    assert excluded == 1 and gold_window not in windows
    out = compare.run_evaluate()
    assert out["n_eval"] == 3
    assert "skipped" in out["classifiers"]["llm"]  # no cached LLM label: said so, never invented
    for name in ("lr", "nb", "rocchio", "knn"):
        r = out["classifiers"][name]
        assert r["n"] == 3 and r["micro_f1"] == pytest.approx(r["accuracy"])
        assert set(r["per_class"]) == {"followed", "distinguished", "doubted", "overruled", "neutral"}
    md = (tiny / "reports" / "classifier_compare.md").read_text(encoding="utf-8")
    assert "Naive Bayes (ours)" in md and "n=1" in md
    rep = compare.run_preds()
    assert rep["by_split"]["gold"] == 12  # four trained classifiers x three gold windows (no LLM label cached)


def test_public_classify_functions_return_a_label_and_a_confidence(tiny):
    from common.schema import TREATMENT_LABELS
    from m3_treatment import compare, nb, vector_classifiers

    compare.run_select()
    for fn in (nb._model, vector_classifiers._rocchio, vector_classifiers._knn):
        fn.cache_clear()
    try:
        w = _w("In the result TARGET is overruled.")
        for lab, conf in (nb.classify_nb(w), vector_classifiers.classify_rocchio(w), vector_classifiers.classify_knn(w, k=3)):
            assert lab in TREATMENT_LABELS and 0.0 <= conf <= 1.0
    finally:
        for fn in (nb._model, vector_classifiers._rocchio, vector_classifiers._knn):
            fn.cache_clear()


def test_phase2_stubs_have_the_contract_shapes():
    from stubs import fixed

    assert fixed.classify_nb("x") == ("neutral", 0.0)
    assert fixed.classify_knn("x", k=3) == ("neutral", 0.0)
    hubs, auths = fixed.hits(["STUB-001", "nope"])
    assert set(hubs) == set(auths) == {"STUB-001", "nope"} and auths["nope"] == 0.0
