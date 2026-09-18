# DroneVehicle decision-level rescoring: Eq. 3 versus the June equal-weight variant

Identical fused boxes (class-wise WBF, IoU 0.55) on the 1,469-image validation split; only the ranking score changes.
No training, no GPU: the cached June teacher predictions are reused.

| priors | scoring | boxes | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|
| pilot | plain | 65667 | 0.7184 | 0.4860 |
| pilot | pilot_additive | 65667 | 0.7332 | 0.4958 |
| pilot | eq3 | 65667 | 0.7372 | 0.4979 |
| damped | plain | 65667 | 0.7182 | 0.4860 |
| damped | pilot_additive | 65667 | 0.7327 | 0.4956 |
| damped | eq3 | 65667 | 0.7370 | 0.4979 |

Deltas against plain WBF at the pilot priors:

- pilot/pilot_additive: +1.48 pp mAP@0.5, +0.97 pp mAP@0.5:0.95
- pilot/eq3: +1.88 pp mAP@0.5, +1.18 pp mAP@0.5:0.95
- damped/pilot_additive: +1.44 pp mAP@0.5, +0.96 pp mAP@0.5:0.95
- damped/eq3: +1.86 pp mAP@0.5, +1.18 pp mAP@0.5:0.95
