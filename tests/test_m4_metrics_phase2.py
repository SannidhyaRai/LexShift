"""M4 Phase 2: R-precision, MRR, the precision-recall curve and 11-point interpolated AP (eval/metrics.py), by hand."""

import pytest

from eval import metrics as M

# relevant: a, c, f (grade >= 1); b, d, e not relevant
GRADES = {"a": 2, "b": 0, "c": 1, "d": 0, "f": 2}
RANKED = ["a", "b", "c", "d", "e", "f"]


def test_pr_points():
    assert M.pr_points(RANKED, GRADES) == [
        (1 / 3, 1.0), (1 / 3, 0.5), (2 / 3, 2 / 3), (2 / 3, 0.5), (2 / 3, 0.4), (1.0, 0.5)]


def test_interpolated_precision_is_the_max_to_the_right():
    ip = M.interpolated_precision(RANKED, GRADES)
    assert len(ip) == 11
    assert ip[:4] == [1.0] * 4                         # recall 0 .. 0.3: max over every point (1.0 at recall 1/3)
    assert ip[4:7] == pytest.approx([2 / 3] * 3)        # recall 0.4 .. 0.6: best at or beyond 2/3 is 2/3
    assert ip[7:] == pytest.approx([0.5] * 4)           # recall 0.7 .. 1.0: only the last point qualifies
    assert all(x >= y for x, y in zip(ip, ip[1:]))      # never increases
    assert M.eleven_point_ap(RANKED, GRADES) == pytest.approx((4 + 3 * 2 / 3 + 4 * 0.5) / 11)


def test_unreached_recall_scores_zero():
    ip = M.interpolated_precision(["a", "b"], GRADES)  # only recall 1/3 is reached
    assert ip[:4] == [1.0] * 4 and ip[4:] == [0.0] * 7


def test_r_precision_and_reciprocal_rank():
    assert M.r_precision(RANKED, GRADES) == pytest.approx(2 / 3)  # R = 3: a, b, c -> 2 relevant
    assert M.reciprocal_rank(RANKED, GRADES) == 1.0
    assert M.reciprocal_rank(["b", "d", "c"], GRADES) == pytest.approx(1 / 3)
    assert M.reciprocal_rank(["b", "d"], GRADES) == 0.0


def test_undefined_without_relevant_documents():
    none = {"x": 0}
    assert M.r_precision(RANKED, none) is None
    assert M.reciprocal_rank(RANKED, none) is None
    assert M.interpolated_precision(RANKED, none) is None
    assert M.eleven_point_ap(RANKED, none) is None
    assert M.pr_points(RANKED, none) == []


def test_threshold_two_counts_only_grade_two():
    assert M.r_precision(RANKED, GRADES, threshold=2) == 0.5  # relevant a, f; R = 2: a, b
    assert M.reciprocal_rank(["c", "f"], GRADES, threshold=2) == 0.5


def test_evaluate_query_reports_the_new_metrics():
    out = M.evaluate_query(RANKED, GRADES, None)
    assert set(M.METRIC_NAMES) == set(out)
    assert out["MRR"] == 1.0 and out["R-prec"] == pytest.approx(2 / 3)
    assert out["AP11"] == pytest.approx(M.eleven_point_ap(RANKED, GRADES))
