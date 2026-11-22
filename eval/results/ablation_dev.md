# Ablation results: dev split, 10 queries, depth 20

Generated 2026-11-22 16:06 UTC. Providers: search=real, statute=real, health=real, authority=real.

| config | weights rel/cont/health/auth | P@5 | P@10 | R@10 | MAP | AP11 | R-prec | MRR | nDCG@10 | harmful@10 | judged@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b0 | 1.00/0.00/0.00/0.00 | 0.740 | 0.440 | 0.810 | 0.709 | 0.733 | 0.650 | 0.792 | 0.727 | n/a | 0.580 |
| b1 | 0.70/0.30/0.00/0.00 | 0.760 | 0.470 | 0.866 | 0.712 | 0.742 | 0.662 | 0.792 | 0.736 | n/a | 0.590 |
| full | 0.50/0.20/0.20/0.10 | 0.800 | 0.480 | 0.883 | 0.729 | 0.764 | 0.730 | 0.792 | 0.732 | n/a | 0.600 |
| full_hits | 0.50/0.20/0.20/0.10 | 0.680 | 0.450 | 0.838 | 0.668 | 0.709 | 0.645 | 0.792 | 0.703 | n/a | 0.570 |
| full_tiered | 0.50/0.20/0.20/0.10 | 0.480 | 0.240 | 0.432 | 0.378 | 0.405 | 0.432 | 0.850 | 0.489 | n/a | 0.280 |

Weights source: b0: starting weights (untuned placeholders); b1: starting weights (untuned placeholders); full: starting weights (untuned placeholders); full_hits: starting weights (untuned placeholders); full_tiered: starting weights (untuned placeholders).
Relevant means grade >= 1; nDCG gain `exp`. Documents without a judgement count as grade 0; `judged@10` is the share of the top 10 that has one.
Gold overruling list: 0 rows (so harmful@10 is n/a).

## Difference from b0 (paired bootstrap over queries, 95% interval, 10000 resamples)

| config | metric | mean difference | 95% interval | queries |
|---|---|---|---|---|
| b1 | P@5 | +0.020 | [-0.040, +0.080] | 10 |
| b1 | nDCG@10 | +0.009 | [-0.097, +0.106] | 10 |
| b1 | harmful@10 | n/a | n/a | 0 |
| full | P@5 | +0.060 | [-0.020, +0.140] | 10 |
| full | nDCG@10 | +0.004 | [-0.114, +0.109] | 10 |
| full | harmful@10 | n/a | n/a | 0 |
| full_hits | P@5 | -0.060 | [-0.140, +0.020] | 10 |
| full_hits | nDCG@10 | -0.025 | [-0.140, +0.080] | 10 |
| full_hits | harmful@10 | n/a | n/a | 0 |
| full_tiered | P@5 | -0.260 | [-0.360, -0.160] | 10 |
| full_tiered | nDCG@10 | -0.238 | [-0.369, -0.072] | 10 |
| full_tiered | harmful@10 | n/a | n/a | 0 |

With this few queries the interval is wide; read it as that reminder, not as a significance test.

## nDCG@10 by query type

| type | queries | b0 | b1 | full | full_hits | full_tiered |
|---|---|---|---|---|---|---|
| A | 3 | 0.860 | 0.732 | 0.756 | 0.728 | 0.472 |
| B | 2 | 0.729 | 0.718 | 0.716 | 0.681 | 0.455 |
| C | 3 | 0.794 | 0.794 | 0.755 | 0.721 | 0.529 |
| D | 2 | 0.427 | 0.674 | 0.674 | 0.659 | 0.488 |

## 11-point interpolated precision (mean over queries)

| config | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b0 | 0.850 | 0.850 | 0.850 | 0.833 | 0.825 | 0.805 | 0.732 | 0.673 | 0.615 | 0.513 | 0.513 |
| b1 | 0.863 | 0.863 | 0.843 | 0.826 | 0.826 | 0.796 | 0.760 | 0.672 | 0.654 | 0.533 | 0.533 |
| full | 0.873 | 0.873 | 0.853 | 0.833 | 0.833 | 0.825 | 0.767 | 0.717 | 0.703 | 0.562 | 0.562 |
| full_hits | 0.859 | 0.859 | 0.839 | 0.822 | 0.782 | 0.714 | 0.675 | 0.608 | 0.605 | 0.518 | 0.518 |
| full_tiered | 0.875 | 0.875 | 0.875 | 0.650 | 0.475 | 0.335 | 0.315 | 0.029 | 0.029 | 0.000 | 0.000 |

Recall is relative to the judged relevant documents, inside the evaluation depth.
