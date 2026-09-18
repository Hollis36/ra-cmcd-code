# DroneVehicle student contrasts, paired image-level bootstrap on global AP (2026-09-17 14:08)

B = 2000, seed 20260913, shared test-image indices, each arm summarized by its four-seed mean global AP@0.5:0.95.
Official test split, 8980 images. True-positive flags are precomputed per image (greedy matching is
within-image, so resampling changes only the pooled ranking and the ground-truth count).

Four-seed mean global AP@0.5:0.95: plain 0.3803, weighted 0.3984, gt_only 0.3216, eq3 0.3996

| contrast | diff (pp) | 95% CI (pp) | p one-sided | positive draws |
|---|---:|---|---:|---:|
| weighted vs plain | +1.81 | [+1.51, +2.09] | 0.0000 | 100.0% |
| weighted vs gt_only | +7.68 | [+7.27, +8.08] | 0.0000 | 100.0% |
| plain vs gt_only | +5.87 | [+5.49, +6.27] | 0.0000 | 100.0% |
| eq3 vs plain | +1.93 | [+1.65, +2.19] | 0.0000 | 100.0% |
| eq3 vs weighted | +0.12 | [-0.03, +0.28] | 0.0610 | 93.9% |
