# DroneVehicle rescoring: paired image-level bootstrap on global AP (2026-09-17 11:08)

B = 10000, seed 20260910, 1469 validation images resampled with replacement; plain and Eq. 3 scorings share every draw.
Same evaluator as the table (rescore_eq3.py), so the point estimates below are the table's values exactly.

| metric | plain | Eq. 3 | difference (pp) | 95% CI (pp) | one-sided p | positive draws |
|---|---:|---:|---:|---|---:|---:|
| mAP@0.5 | 0.7184 | 0.7372 | +1.88 | [+1.06, +3.05] | < 1e-4 | 100.0% |
| mAP@0.5:0.95 | 0.4860 | 0.4979 | +1.18 | [+0.63, +1.98] | < 1e-4 | 100.0% |
