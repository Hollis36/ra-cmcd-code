# Anti-UAV-RGBT bootstraps on the validator's own statistics (2026-09-17 11:31)

B = 10000; image-level seed 20260515 + split index (validation 20260515); sequence-level the same seeds over sequence ids.
Point = today's re-validation (four-seed means); record = the stored val_fixed values behind the tables.

## image-level, night78 (78 images)

| contrast | family | point (pp) | record (pp) | 95% CI (pp) | p one-sided | p Holm (family) | positive |
|---|---|---:|---:|---|---:|---:|---:|
| reliability_vs_plain | ablation | +3.41 | +3.40 | [+1.47, +4.96] | 0.0002 | -- | 1.000 |
| reliability_vs_single_best | ablation | +4.36 | +4.36 | [+2.59, +5.77] | 0.0001 | -- | 1.000 |
| plain_vs_single_best | ablation | +0.95 | +0.96 | [-0.47, +2.42] | 0.0954 | -- | 0.905 |
| reliability_vs_conf_only | subterm | +1.27 | +1.24 | [+0.03, +2.56] | 0.0217 | 0.0651 | 0.978 |
| reliability_vs_drop_support | subterm | +1.37 | +1.33 | [-0.01, +2.84] | 0.0265 | 0.0651 | 0.974 |
| reliability_vs_drop_prior | subterm | +1.39 | +1.38 | [+0.22, +2.50] | 0.0103 | 0.0412 | 0.990 |
| reliability_vs_drop_iou | subterm | +0.34 | +0.32 | [-0.82, +1.66] | 0.3204 | 0.3204 | 0.680 |
| reliability_vs_soft_v2 | baselines | +1.04 | +1.04 | [-0.17, +2.41] | 0.0530 | 0.0872 | 0.947 |
| reliability_vs_abid_soft | baselines | +1.17 | +1.17 | [-0.18, +2.44] | 0.0436 | 0.0872 | 0.957 |
| reliability_vs_baseline | transfer | +0.96 | +0.94 | [-0.73, +2.49] | 0.1467 | -- | 0.853 |

## image-level, night160 (160 images)

| contrast | family | point (pp) | record (pp) | 95% CI (pp) | p one-sided | p Holm (family) | positive |
|---|---|---:|---:|---|---:|---:|---:|
| reliability_vs_plain | ablation | +2.53 | +2.51 | [+1.28, +3.42] | 0.0002 | -- | 1.000 |
| reliability_vs_single_best | ablation | +2.35 | +2.37 | [+1.24, +3.29] | 0.0001 | -- | 1.000 |
| plain_vs_single_best | ablation | -0.18 | -0.15 | [-0.93, +0.72] | 0.6096 | -- | 0.390 |
| reliability_vs_conf_only | subterm | +0.48 | +0.46 | [-0.25, +1.21] | 0.1013 | 0.3039 | 0.899 |
| reliability_vs_drop_support | subterm | +0.84 | +0.80 | [+0.06, +1.54] | 0.0167 | 0.0668 | 0.983 |
| reliability_vs_drop_prior | subterm | +0.19 | +0.20 | [-0.50, +0.83] | 0.3148 | 0.6295 | 0.685 |
| reliability_vs_drop_iou | subterm | +0.02 | +0.06 | [-0.73, +0.73] | 0.5005 | 0.6295 | 0.499 |
| reliability_vs_soft_v2 | baselines | +0.07 | +0.07 | [-0.75, +0.82] | 0.4481 | 0.5347 | 0.552 |
| reliability_vs_abid_soft | baselines | +0.24 | +0.24 | [-0.55, +1.03] | 0.2674 | 0.5347 | 0.733 |
| reliability_vs_baseline | transfer | +0.73 | +0.77 | [-0.71, +2.12] | 0.1751 | -- | 0.825 |

## image-level, val (586 images)

| contrast | family | point (pp) | record (pp) | 95% CI (pp) | p one-sided | p Holm (family) | positive |
|---|---|---:|---:|---|---:|---:|---:|
| reliability_vs_baseline | transfer | +1.40 | +1.40 | [+0.42, +2.32] | 0.0027 | -- | 0.997 |

## sequence-level, night78 (78 images, 16 sequences)

| contrast | family | point (pp) | record (pp) | 95% CI (pp) | p one-sided | p Holm (family) | positive |
|---|---|---:|---:|---|---:|---:|---:|
| reliability_vs_baseline | sequence | +0.96 | +0.94 | [-0.88, +2.43] | 0.1446 | -- | 0.856 |
| reliability_vs_plain | sequence | +3.41 | +3.40 | [+1.18, +4.65] | 0.0007 | -- | 0.999 |

## sequence-level, night160 (160 images, 16 sequences)

| contrast | family | point (pp) | record (pp) | 95% CI (pp) | p one-sided | p Holm (family) | positive |
|---|---|---:|---:|---|---:|---:|---:|
| reliability_vs_baseline | sequence | +0.73 | +0.77 | [-1.35, +2.55] | 0.2698 | -- | 0.730 |
| reliability_vs_plain | sequence | +2.53 | +2.51 | [+0.99, +3.30] | 0.0006 | -- | 1.000 |

