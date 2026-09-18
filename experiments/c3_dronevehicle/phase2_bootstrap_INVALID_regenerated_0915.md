> **作废（2026-09-17 隔离）**：本文件于 2026-09-15 由评估链误调用已作废的逐图 AP 代理脚本重新生成，内容无效。有效结果只看 phase2_bootstrap_global.md。

# DroneVehicle student contrasts, paired image-level bootstrap (2026-09-15 08:16)

B = 10000, seed 20260913, shared test-image indices, arms summarized by their {'plain': 4, 'weighted': 4, 'gt_only': 4}-seed mean per-image AP.
Official test split, 8980 images. Per-image AP is an AP proxy computed image by image, so absolute values
differ from the Ultralytics mAP in PHASE2_RESULT.md; only the paired differences are interpreted.

| contrast | diff (pp) | 95% CI (pp) | p one-sided | positive draws |
|---|---:|---|---:|---:|
| weighted vs plain | +5.73 | [+5.55, +5.92] | 0.0000 | 100.0% |
| weighted vs gt_only | +4.94 | [+4.73, +5.14] | 0.0000 | 100.0% |
| plain vs gt_only | -0.80 | [-0.99, -0.60] | 1.0000 | 0.0% |
