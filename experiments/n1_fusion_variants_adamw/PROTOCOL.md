# N1: Fusion-variant retrain with the student recipe

Written 2026-09-07 before any result was produced. Retraining of the fusion variants with the student recipe.

## Why
The submitted Table VII compares the reliability-weighted late-fusion student (AdamW, lr 0.001, 3 warmup epochs) with early-fusion and mid-to-late-fusion YOLO11n-RGBT variants that were trained with SGD, lr 0.005, 5 warmup epochs (seeds 42-45), while calling this "the same protocol". The retrain removes the optimizer confound.

## What is fixed in advance
- Models: `yolo11n-RGBT-earlyfusion.yaml`, `yolo11n-RGBT-mid-to-late-fusion.yaml` from the fork at commit 9cc2e20 (unpatched upstream; human labels, reliability irrelevant).
- Data: `anti_uav_rgbt_full_dir.yaml` (dir-copy keyframe split, 1,452 train / 586 val), the same YAML the original variant runs and the dir-data human-label baseline used.
- Recipe: identical to `run_cmcd_student.py` and the late-fusion baseline: AdamW, lr0 0.001, warmup 3, 100 epochs, batch 8, 640 px, close_mosaic 10, AMP, deterministic, pretrained flag true, workers 4.
- Seeds: 42, 43, 44, 45 (the seeds of the original variant runs).
- Selection: per seed, on the validation split, reduced 6-setting grid (640/704 × 0.05/0.08/0.10 × NMS-IoU 0.40), frozen on night-78 and night-160. Same schema as `val_fixed_*.json`.
- Reporting: four-seed mean ± std of night-160 mAP@0.5:0.95 replaces the two Table VII rows and the numbers in Sec. IV-E; the old SGD values move to the all-runs appendix with their optimizer stated.
- No stopping rule or success criterion: the result is descriptive and is reported whatever it is.

## Environment difference to disclose
The retrain runs under Python 3.12, PyTorch 2.11.0+cu128; the submitted runs used PyTorch 2.1 / CUDA 12.1. Both are stated in the revised Implementation Details.

## Outputs
`<YOLO11_ROOT>\outputs\sota_runs\rgbt_variants_adamw\{tag}\` (weights, results.csv, args.yaml), `val_fixed_{tag}.json`, `n1_summary.json`, `n1_progress.log`.
