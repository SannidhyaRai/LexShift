"""The five treatment classifiers side by side: LLM (Gemini, cached), LR (tf-idf + logistic regression), NB, Rocchio, kNN.

    python -m m3_treatment.compare silver     m3_mentions.jsonl + LLM cache -> data/llm_labels/m3_silver.jsonl (training data)
    python -m m3_treatment.compare select     cross-validation on the training data -> reports/model_selection.{json,md}
    python -m m3_treatment.compare evaluate   the F1 table on the HAND-LABELLED gold set -> reports/classifier_compare.{json,md}
    python -m m3_treatment.compare preds      data/processed/treatment_preds.jsonl
    python -m m3_treatment.compare all        silver (if missing), select, evaluate (if there is a gold set), preds

The rules (Phase 2 guide, M3; CLAUDE.md rule 5):
  * TRAINING data is the silver set: windows whose label is a cached Gemini answer. It is never a test standard.
  * The only TEST standard is data/treatment_gold.csv, written by people. Every gold window, including the LLM's few-shot
    pool, is removed from the training data, so no classifier is ever tested on a window it was trained on.
  * Hyperparameters (NB variant, feature selection and k; kNN k) are chosen by cross-validation on the training data only.
  * Every classifier sees only the marked window around one cited case. No keyword rule assigns a label.

The silver set covers the `cued` LLM scope only (resolved windows with an overrule or doubt cue near the citation), so its
class mix is not the gold set's: the classifiers learn from a skewed sample. This is a stated limitation (D-P2-02).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from common.config import ROOT, load_config
from common.io import read_jsonl, write_jsonl
from common.schema import TREATMENT_LABELS

CLASSIFIERS = ("llm", "lr", "nb", "rocchio", "knn")


def cfg() -> dict:
    return load_config()["m3_treatment"]


def _p(key: str) -> Path:
    return ROOT / cfg()[key]


# ----------------------------------------------------------------------------------------------------------------
# Silver training set
# ----------------------------------------------------------------------------------------------------------------
def build_silver(mentions: list[dict] | None = None) -> dict:
    """One row per distinct window that carries a zero-shot label from the configured model in the LLM cache.

    The window_id is the one the gold sampler would give the same window (first mention in file order), so the two sets can
    be joined on it as well as on the window text.
    """
    from m3_treatment.classifier import ZERO_SHOT, LLMCache, cache_key
    from m3_treatment.gold import window_id
    from m3_treatment.pipeline import load_mentions

    c = cfg()
    model = c["llm"]["model"]
    cache = LLMCache(ROOT / c["llm"]["cache"])
    rows, seen = [], set()
    for m in mentions if mentions is not None else load_mentions():
        if m["is_self"] or m["is_appeal_history"]:
            continue
        w = m["marked_window"]
        if w in seen:
            continue
        seen.add(w)
        hit = cache.get(cache_key(model, w, ZERO_SHOT))
        if hit is None:
            continue
        rows.append(
            {
                "window_id": window_id(m["citing_doc"], m["start"], m["end"]),
                "citing_doc": m["citing_doc"],
                "cited_doc": m["cited_doc"],
                "window": w,
                "label": hit["label"],
                "confidence": float(hit["confidence"]),
                "model": model,
                "model_version": hit.get("model_version"),
                "shots": ZERO_SHOT,
            }
        )
    n = write_jsonl(_p("silver"), rows)
    return {"silver_windows": n, "labels": dict(Counter(r["label"] for r in rows)), "model": model, "path": str(_p("silver"))}


def load_silver() -> list[dict]:
    path = _p("silver")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m m3_treatment.compare silver` (needs m3_mentions.jsonl: `make m3-extract`)")
    return list(read_jsonl(path))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------------------
# Gold (test) and training sets
# ----------------------------------------------------------------------------------------------------------------
def gold_all_windows() -> set[str]:
    """Every window that appears in the gold file, whatever its label state: none of them may be trained on."""
    from m3_treatment.gold import load_gold

    return {r.window for r in load_gold()}


def gold_eval() -> list[tuple[str, str, str]]:
    """(window_id, marked window, gold label) for the evaluation windows: settled gold labels minus the LLM's few-shot pool
    (held out of every classifier's evaluation, so all five are scored on exactly the same windows)."""
    from m3_treatment.gold import final_labels, load_gold
    from m3_treatment.pipeline import fewshot_pool

    rows = load_gold()
    ids = {r.window: r.window_id for r in rows}
    pool = {w for w, _ in fewshot_pool()}
    return [(ids[w], w, lab) for w, lab in final_labels(rows) if w not in pool]


@lru_cache(maxsize=1)
def _training() -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, ...], ...], int]:
    from m3_treatment.features import window_tokens

    held = gold_all_windows()
    silver = load_silver()
    kept = [r for r in silver if r["window"] not in held]
    windows = tuple(r["window"] for r in kept)
    return windows, tuple(r["label"] for r in kept), tuple(tuple(window_tokens(w)) for w in windows), len(silver) - len(kept)


def training_set() -> tuple[list[tuple[str, ...]], list[str]]:
    """(token lists, labels) of the silver windows that are not in the gold file."""
    _w, labels, docs, _ = _training()
    return list(docs), list(labels)


def load_selection() -> dict:
    path = _p_sel()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m m3_treatment.compare select` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _p_sel() -> Path:
    return ROOT / cfg()["classifiers"]["model_selection"]


def _reports() -> Path:
    d = _p("reports_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------------------------
# select
# ----------------------------------------------------------------------------------------------------------------
def run_select() -> dict:
    from m3_treatment.nb import cross_validate
    from m3_treatment.vector_classifiers import cross_validate_knn, cross_validate_rocchio

    cc = cfg()["classifiers"]
    windows, labels, docs, excluded = _training()
    docs, labels = list(docs), list(labels)
    t0 = time.time()
    nb_rows = cross_validate(docs, labels, ("multinomial", "bernoulli"), cc["nb_selection"], cc["nb_k_grid"], cc["folds"], cc["seed"])
    knn_rows = cross_validate_knn(docs, labels, cc["knn_k_grid"], cc["folds"], cc["seed"])
    rocchio = cross_validate_rocchio(docs, labels, cc["folds"], cc["seed"])
    # Best macro-F1; ties go to the simpler model (multinomial, fewer terms, larger k = smoother), so the choice is stable.
    best_nb = max(nb_rows, key=lambda r: (round(r["macro_f1"], 6), r["variant"] == "multinomial", -(r["k"] or 10**9)))
    best_knn = max(knn_rows, key=lambda r: (round(r["macro_f1"], 6), r["k"]))
    best_bern = max((r for r in nb_rows if r["variant"] == "bernoulli"), key=lambda r: round(r["macro_f1"], 6))
    out = {
        "note": "Model selection by stratified cross-validation on the TRAINING data (Gemini silver labels). The scores say how well "
        "each setting reproduces the silver labels; they are not an evaluation. The evaluation is against the hand-labelled gold set.",
        "training": {
            "windows": len(labels),
            "labels": dict(Counter(labels)),
            "excluded_because_in_gold": excluded,
            "silver_sha256_16": _sha(_p("silver")),
            "folds": cc["folds"],
            "seed": cc["seed"],
        },
        "nb": {"variant": best_nb["variant"], "selection": best_nb["selection"], "k": best_nb["k"], "cv_macro_f1": best_nb["macro_f1"]},
        "best_bernoulli": best_bern,
        "knn": {"k": best_knn["k"], "cv_macro_f1": best_knn["macro_f1"]},
        "rocchio": {"cv_macro_f1": rocchio["macro_f1"]},
        "grid": {"nb": nb_rows, "knn": knn_rows},
        "seconds": round(time.time() - t0, 1),
    }
    _p_sel().parent.mkdir(parents=True, exist_ok=True)
    _p_sel().write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (_reports() / "model_selection.md").write_text(selection_markdown(out), encoding="utf-8")
    return out


def selection_markdown(out: dict) -> str:
    tr = out["training"]
    lines = [
        "# M3 classifier model selection (training data only)",
        "",
        "Generated by `python -m m3_treatment.compare select`.",
        "",
        f"> {out['note']}",
        "",
        f"Training windows: {tr['windows']} (silver, {tr['excluded_because_in_gold']} removed because they are in the gold file); "
        f"labels {tr['labels']}; {tr['folds']}-fold stratified CV, seed {tr['seed']}; silver file sha256 {tr['silver_sha256_16']}.",
        "",
        "## Chosen",
        "",
        f"- Naive Bayes: {out['nb']['variant']}, feature selection {out['nb']['selection']}, k = {out['nb']['k'] or 'all terms'} "
        f"(CV macro-F1 {out['nb']['cv_macro_f1']:.3f})",
        f"- best Bernoulli setting for comparison: {out['best_bernoulli']['selection']}, k = {out['best_bernoulli']['k'] or 'all terms'} "
        f"(CV macro-F1 {out['best_bernoulli']['macro_f1']:.3f})",
        f"- kNN: k = {out['knn']['k']} (CV macro-F1 {out['knn']['cv_macro_f1']:.3f})",
        f"- Rocchio: no hyperparameter (CV macro-F1 {out['rocchio']['cv_macro_f1']:.3f})",
        "",
        "## Naive Bayes grid (CV macro-F1)",
        "",
        "| Variant | Selection | k | Macro-F1 |",
        "|---|---|---|---|",
    ]
    for r in out["grid"]["nb"]:
        lines.append(f"| {r['variant']} | {r['selection']} | {r['k'] or 'all'} | {r['macro_f1']:.3f} |")
    lines += ["", "## kNN grid (CV macro-F1)", "", "| k | Macro-F1 |", "|---|---|"]
    for r in out["grid"]["knn"]:
        lines.append(f"| {r['k']} | {r['macro_f1']:.3f} |")
    lines += ["", f"Grid search took {out['seconds']} s on a laptop CPU.", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# Fitted models (shared by evaluate and preds)
# ----------------------------------------------------------------------------------------------------------------
def fit_all(windows: list[str], docs: list, labels: list[str], sel: dict) -> dict:
    """name -> function(list of (marked window, tokens)) -> [(label, confidence)], each trained on the given data."""
    from m3_treatment.classifier import BaselineClassifier
    from m3_treatment.nb import train
    from m3_treatment.vector_classifiers import KNN, Rocchio

    nb = train(docs, labels, sel["nb"]["variant"], sel["nb"]["selection"], sel["nb"]["k"])
    roc = Rocchio().fit(docs, labels)
    knn = KNN(sel["knn"]["k"]).fit(docs, labels)
    lr = BaselineClassifier(seed=cfg()["classifiers"]["seed"]).fit(windows, labels)
    return {
        "lr": lambda items: lr.predict([w for w, _ in items]),
        "nb": lambda items: nb.predict([t for _, t in items]),
        "rocchio": lambda items: roc.predict([t for _, t in items]),
        "knn": lambda items: knn.predict([t for _, t in items]),
    }


def _llm_labels(windows: list[str]) -> list[tuple[str, float] | None]:
    """The cached LLM label of each window under the CURRENT prompt (few-shot pool fingerprint), or None."""
    from m3_treatment.classifier import LLMCache, cache_key, shots_fingerprint
    from m3_treatment.pipeline import few_shot_examples

    c = cfg()["llm"]
    cache = LLMCache(ROOT / c["cache"])
    shots = shots_fingerprint(few_shot_examples())
    out = []
    for w in windows:
        r = cache.get(cache_key(c["model"], w, shots))
        out.append((r["label"], float(r["confidence"])) if r else None)
    return out


# ----------------------------------------------------------------------------------------------------------------
# evaluate: the F1 table on the hand-labelled gold set
# ----------------------------------------------------------------------------------------------------------------
def report(gold: list[str], pred: list[str]) -> dict:
    """Per-class P/R/F1 with support, macro-F1, micro-F1 and the confusion matrix."""
    from m3_treatment.classifier import per_class_report

    r = per_class_report(gold, pred)
    tp = sum(1 for g, p in zip(gold, pred) if g == p)
    # Single-label multi-class: every error is one FP (for the predicted class) and one FN (for the gold class), so
    # micro-P = micro-R = micro-F1 = accuracy. Reported under its own name because the course asks for it.
    r["micro_f1"] = tp / len(gold) if gold else 0.0
    return r


def run_evaluate() -> dict:
    from m3_treatment.features import window_tokens

    ev = gold_eval()
    if not ev:
        raise RuntimeError("data/treatment_gold.csv has no settled hand labels yet: the F1 table cannot be computed (label the sheets from `python -m m3_treatment.gold sample`, then `gold merge`)")
    sel = load_selection()
    windows_tr, labels_tr, docs_tr, excluded = _training()
    if sel["training"]["windows"] != len(labels_tr):
        raise RuntimeError("reports/model_selection.json was made on a different training set (the gold file changed?); run `compare select` again")
    models = fit_all(list(windows_tr), list(docs_tr), list(labels_tr), sel)
    items = [(w, window_tokens(w)) for _, w, _ in ev]
    gold = [lab for _, _, lab in ev]
    out: dict = {
        "n_eval": len(ev),
        "gold_label_counts": dict(Counter(gold)),
        "training": {"windows": len(labels_tr), "labels": dict(Counter(labels_tr)), "excluded_because_in_gold": excluded},
        "selection": {"nb": sel["nb"], "knn": sel["knn"]},
        "classifiers": {},
    }
    llm = _llm_labels([w for w, _ in items])
    missing = sum(x is None for x in llm)
    out["classifiers"]["llm"] = (
        {"skipped": f"{missing} of {len(llm)} evaluation windows have no LLM label under the current prompt; run `python -m m3_treatment.pipeline label-llm`"}
        if missing
        else report(gold, [x[0] for x in llm])
    )
    for name in ("lr", "nb", "rocchio", "knn"):
        out["classifiers"][name] = report(gold, [lab for lab, _ in models[name](items)])
    md = compare_markdown(out)
    (_reports() / "classifier_compare.md").write_text(md, encoding="utf-8")
    (_reports() / "classifier_compare.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    out["markdown"] = md
    return out


TITLES = {
    "llm": "LLM (Gemini, zero/few-shot, cached)",
    "lr": "LR: tf-idf + logistic regression",
    "nb": "Naive Bayes (ours)",
    "rocchio": "Rocchio (ours)",
    "knn": "kNN (ours)",
}


def compare_markdown(out: dict) -> str:
    sel = out["selection"]
    lines = [
        "# M3 treatment classifiers on the hand-labelled gold set",
        "",
        "Generated by `python -m m3_treatment.compare evaluate`. Test windows: "
        f"{out['n_eval']} hand-labelled (gold label counts {out['gold_label_counts']}); LR, NB, Rocchio and kNN are trained on "
        f"{out['training']['windows']} silver windows (Gemini labels, {out['training']['excluded_because_in_gold']} removed because they are in the gold file). "
        f"NB: {sel['nb']['variant']}, {sel['nb']['selection']}, k = {sel['nb']['k'] or 'all'}; kNN: k = {sel['knn']['k']} (chosen by CV on the training data).",
        "",
        "## Summary",
        "",
        "| Classifier | Macro-F1 | Micro-F1 | " + " | ".join(f"F1 {c}" for c in TREATMENT_LABELS) + " |",
        "|---" * (3 + len(TREATMENT_LABELS)) + "|",
    ]
    for name in CLASSIFIERS:
        r = out["classifiers"][name]
        if "skipped" in r:
            lines.append(f"| {TITLES[name]} | not run | | " + " | ".join("" for _ in TREATMENT_LABELS) + " |")
            continue
        lines.append(
            f"| {TITLES[name]} | {r['macro_f1']:.3f} | {r['micro_f1']:.3f} | "
            + " | ".join(f"{r['per_class'][c]['f1']:.3f} (n={r['per_class'][c]['support']})" for c in TREATMENT_LABELS)
            + " |"
        )
    lines += ["", "Macro-F1 averages the classes present in the gold set; micro-F1 equals accuracy for single-label classification.", ""]
    for name in CLASSIFIERS:
        r = out["classifiers"][name]
        lines += [f"## {TITLES[name]}", ""]
        if "skipped" in r:
            lines += [f"Not run: {r['skipped']}.", ""]
            continue
        lines += ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for c in TREATMENT_LABELS:
            p = r["per_class"][c]
            lines.append(f"| {c} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['support']} |")
        lines += ["", "Confusion (rows gold, columns predicted):", "", "| | " + " | ".join(TREATMENT_LABELS) + " |", "|---" * (len(TREATMENT_LABELS) + 1) + "|"]
        for g in TREATMENT_LABELS:
            lines.append(f"| {g} | " + " | ".join(str(r["confusion"][g][p]) for p in TREATMENT_LABELS) + " |")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# preds: data/processed/treatment_preds.jsonl
# ----------------------------------------------------------------------------------------------------------------
def run_preds() -> dict:
    """Every classifier's label for every silver window (OUT-OF-FOLD: from a model that did not see that window) and for
    every gold evaluation window (from models trained on the whole training set)."""
    from m3_treatment.features import stratified_folds, window_tokens

    sel = load_selection()
    cc = cfg()["classifiers"]
    windows, labels, docs, _ = _training()
    windows, labels, docs = list(windows), list(labels), list(docs)
    silver_ids = {r["window"]: r["window_id"] for r in load_silver()}
    silver_conf = {r["window"]: r["confidence"] for r in load_silver()}
    rows = []
    for i, w in enumerate(windows):
        rows.append({"window_id": silver_ids[w], "classifier": "llm", "label": labels[i], "confidence": silver_conf[w], "split": "silver"})
    for held in stratified_folds(labels, cc["folds"], cc["seed"]):
        held_set = set(held)
        tr = [i for i in range(len(labels)) if i not in held_set]
        models = fit_all([windows[i] for i in tr], [docs[i] for i in tr], [labels[i] for i in tr], sel)
        items = [(windows[i], docs[i]) for i in held]
        for name in ("lr", "nb", "rocchio", "knn"):
            for i, (lab, conf) in zip(held, models[name](items)):
                rows.append({"window_id": silver_ids[windows[i]], "classifier": name, "label": lab, "confidence": round(conf, 4), "split": "silver_oof"})
    ev = gold_eval()
    if ev:
        models = fit_all(windows, docs, labels, sel)
        items = [(w, window_tokens(w)) for _, w, _ in ev]
        for name in ("lr", "nb", "rocchio", "knn"):
            for (wid, _, _), (lab, conf) in zip(ev, models[name](items)):
                rows.append({"window_id": wid, "classifier": name, "label": lab, "confidence": round(conf, 4), "split": "gold"})
        for (wid, _, _), x in zip(ev, _llm_labels([w for w, _ in items])):
            if x:
                rows.append({"window_id": wid, "classifier": "llm", "label": x[0], "confidence": x[1], "split": "gold"})
    n = write_jsonl(_p("preds"), rows)
    return {"records": n, "by_split": dict(Counter(r["split"] for r in rows)), "path": str(_p("preds"))}


# ----------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m m3_treatment.compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("silver", "select", "evaluate", "preds", "all"))
    args = ap.parse_args(argv)
    try:
        if args.cmd == "silver" or (args.cmd == "all" and not _p("silver").exists()):
            print(json.dumps(build_silver(), indent=2))
        if args.cmd in ("select", "all"):
            out = run_select()
            print(selection_markdown(out))
        if args.cmd == "evaluate" or (args.cmd == "all" and gold_eval()):
            print(run_evaluate().pop("markdown"))
        elif args.cmd == "all":
            print("gold set empty: the F1 table is not computed (it needs hand labels in data/treatment_gold.csv)")
        if args.cmd in ("preds", "all"):
            print(json.dumps(run_preds(), indent=2))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
