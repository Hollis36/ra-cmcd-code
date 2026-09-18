# C3 phase 2 result: DroneVehicle student experiment (2026-09-15 08:16)

Pre-registered in PREREGISTRATION.md before training. Settings frozen on the official validation split,
reported on the official 8,980-image test split.

| arm | seeds | val mAP@0.5:0.95 | test mAP@0.5 | test mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| gt_only | 4 | 0.3647 ± 0.0050 | 0.5488 ± 0.0069 | 0.3467 ± 0.0034 |
| plain | 4 | 0.4142 ± 0.0037 | 0.5965 ± 0.0028 | 0.4026 ± 0.0016 |
| weighted | 4 | 0.4362 ± 0.0031 | 0.6271 ± 0.0025 | 0.4249 ± 0.0018 |
| eq3 | 4 | 0.4385 ± 0.0036 | 0.6291 ± 0.0021 | 0.4270 ± 0.0016 |

## Pre-registered contrast

- weighted minus plain on test: **+2.23 pp** mAP@0.5:0.95 (Anti-UAV reference: +2.47 pp at 5.5% pseudo-label noise; here 39.4%)
- prediction P1 (gain materially larger than +2.47 pp): **not supported**
- stop rule (|difference| < 0.5 pp means do not add seeds): not triggered
- plain minus GT-only: +5.59 pp; weighted minus GT-only: +7.82 pp (prediction P2: pseudo labels help at all)
