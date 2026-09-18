# DroneVehicle students, bootstrap on the validator's own statistics (2026-09-17 18:53)

B = 10000, seed 20260913, 8980 official test images; arms are four-seed means; point estimates are the table values.

Four-seed means (mAP@0.5:0.95): gt_only 0.3467, plain 0.4026, weighted 0.4249, eq3 0.4270

| contrast | metric | point (pp) | 95% CI (pp) | p one-sided | positive |
|---|---|---:|---|---:|---:|
| weighted_vs_plain | mAP50 | +3.07 | [+2.64, +3.46] | 0.0001 | 1.000 |
| weighted_vs_plain | mAP50_95 | +2.23 | [+1.94, +2.49] | 0.0001 | 1.000 |
| weighted_vs_gt_only | mAP50 | +7.83 | [+7.29, +8.33] | 0.0001 | 1.000 |
| weighted_vs_gt_only | mAP50_95 | +7.82 | [+7.42, +8.19] | 0.0001 | 1.000 |
| plain_vs_gt_only | mAP50 | +4.76 | [+4.23, +5.32] | 0.0001 | 1.000 |
| plain_vs_gt_only | mAP50_95 | +5.59 | [+5.21, +5.99] | 0.0001 | 1.000 |
| eq3_vs_plain | mAP50 | +3.27 | [+2.85, +3.64] | 0.0001 | 1.000 |
| eq3_vs_plain | mAP50_95 | +2.44 | [+2.16, +2.70] | 0.0001 | 1.000 |
| eq3_vs_weighted | mAP50 | +0.20 | [+0.01, +0.39] | 0.0203 | 0.980 |
| eq3_vs_weighted | mAP50_95 | +0.22 | [+0.07, +0.35] | 0.0014 | 0.999 |
