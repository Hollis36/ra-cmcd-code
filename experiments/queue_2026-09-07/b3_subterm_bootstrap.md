# B3 paired bootstrap, full RA-CMCD minus each variant (four-seed means)

B = 10000, seed 20260515, shared image indices, one-sided p for H0: full <= variant, Holm within family (subterm: conf_only, drop_support, drop_prior, drop_iou; baselines: soft_v2, abid_soft).

## night78 (78 images)

| contrast | metric | point diff (pp) | 95% CI (pp) | p one-sided | p Holm |
|---|---|---|---|---|---|
| full_vs_conf_only | mAP50 | +0.16 | [-0.65, +1.43] | 0.4510 | 0.4510 |
| full_vs_conf_only | mAP50_95 | +1.06 | [-0.22, +2.38] | 0.0533 | 0.1824 |
| full_vs_drop_support | mAP50 | +0.65 | [+0.00, +1.81] | 0.1258 | 0.3774 |
| full_vs_drop_support | mAP50_95 | +0.80 | [-0.46, +2.24] | 0.1199 | 0.2398 |
| full_vs_drop_prior | mAP50 | +0.49 | [+0.00, +1.31] | 0.0445 | 0.1780 |
| full_vs_drop_prior | mAP50_95 | +0.96 | [-0.15, +2.02] | 0.0456 | 0.1824 |
| full_vs_drop_iou | mAP50 | +0.81 | [+0.00, +2.30] | 0.1258 | 0.3774 |
| full_vs_drop_iou | mAP50_95 | +0.43 | [-0.71, +1.71] | 0.2697 | 0.2697 |
| full_vs_soft_v2 | mAP50 | +0.97 | [+0.00, +2.56] | 0.0430 | 0.0860 |
| full_vs_soft_v2 | mAP50_95 | +0.80 | [-0.37, +2.16] | 0.1019 | 0.2038 |
| full_vs_abid_soft | mAP50 | +0.49 | [-0.33, +1.66] | 0.2510 | 0.2510 |
| full_vs_abid_soft | mAP50_95 | +0.76 | [-0.63, +2.02] | 0.1585 | 0.2038 |

## night160 (160 images)

| contrast | metric | point diff (pp) | 95% CI (pp) | p one-sided | p Holm |
|---|---|---|---|---|---|
| full_vs_conf_only | mAP50 | +0.10 | [-0.03, +0.41] | 0.1519 | 0.1882 |
| full_vs_conf_only | mAP50_95 | +0.64 | [-0.11, +1.40] | 0.0493 | 0.1479 |
| full_vs_drop_support | mAP50 | +0.20 | [-0.01, +0.62] | 0.0941 | 0.1882 |
| full_vs_drop_support | mAP50_95 | +0.81 | [+0.07, +1.49] | 0.0161 | 0.0644 |
| full_vs_drop_prior | mAP50 | +0.34 | [+0.00, +0.93] | 0.0548 | 0.1816 |
| full_vs_drop_prior | mAP50_95 | +0.40 | [-0.31, +1.05] | 0.1448 | 0.2896 |
| full_vs_drop_iou | mAP50 | +0.29 | [+0.00, +0.80] | 0.0454 | 0.1816 |
| full_vs_drop_iou | mAP50_95 | +0.12 | [-0.62, +0.85] | 0.3888 | 0.3888 |
| full_vs_soft_v2 | mAP50 | +0.47 | [+0.00, +1.21] | 0.0871 | 0.1742 |
| full_vs_soft_v2 | mAP50_95 | +0.14 | [-0.69, +0.89] | 0.3907 | 0.6507 |
| full_vs_abid_soft | mAP50 | +0.21 | [-0.12, +0.88] | 0.1660 | 0.1742 |
| full_vs_abid_soft | mAP50_95 | +0.17 | [-0.58, +0.96] | 0.3254 | 0.6507 |
