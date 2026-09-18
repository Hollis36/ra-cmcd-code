> **作废**：本文件用的是逐图 AP 代理，其精确率在单图内计算，惩罚检测数量而非质量。在 DroneVehicle（每图约 15 个目标、置信度下限 0.001）上它与 Ultralytics 指标矛盾，甚至把「纯共识对下限」的符号算反。正确结果见 phase2_bootstrap_global.md。

# DroneVehicle student contrasts, paired image-level bootstrap (2026-09-13 12:07)

B = 10000, seed 20260913, shared test-image indices, arms summarized by their {'plain': 4, 'weighted': 4, 'gt_only': 4}-seed mean per-image AP.
Official test split, 8980 images. Per-image AP is an AP proxy computed image by image, so absolute values
differ from the Ultralytics mAP in PHASE2_RESULT.md; only the paired differences are interpreted.

| contrast | diff (pp) | 95% CI (pp) | p one-sided | positive draws |
|---|---:|---|---:|---:|
| weighted vs plain | +5.73 | [+5.55, +5.92] | 0.0000 | 100.0% |
| weighted vs gt_only | +4.94 | [+4.73, +5.14] | 0.0000 | 100.0% |
| plain vs gt_only | -0.80 | [-0.99, -0.60] | 1.0000 | 0.0% |
