# Experiment queue: A1, B2, D1, B3, E2 (written before any result)

Authorised by the authors on 2026-09-07. Queue items: A1 (formula-matched Abid et al. baseline), B2 (threshold students), D1 (pool-size and composition students), B3 (paired p-values for the leave-one-out contrasts), E2 (student recall by target size). GPU: one RTX 4090, shared; stages run sequentially from `run_queue.py`, each stage resumable and logged to `queue_progress.log`.

## Fixed in advance

- Student recipe for every new run: identical to the headline students (AdamW, lr 0.001, warmup 3, 100 epochs, batch 8, 640 px, close_mosaic 10, AMP, deterministic), via `run_cmcd_student.py`; seeds 0-3 for every new arm.
- Selection: per seed on the full validation split over the reduced 6-setting grid (640/704 x 0.05/0.08/0.10 x NMS-IoU 0.40), frozen on night-78 and night-160 (`val_fixed_eval.py`). Same schema as every existing `val_fixed_*.json`.
- Consensus boxes: unless a stage changes the pool, every arm uses the identical 1,506-box training label set (`train_train_teacher_global`); only the per-box weight or the retained subset changes.
- Environment: PyTorch 2.11 / CUDA 12.8 (same as N1); disclosed in the paper's Implementation Details.
- Execution layout: the scripts run from the shim `<YOLO11_ROOT>\scripts\sota` (see README_SHIM.md there), so all outputs land under `<YOLO11_ROOT>\outputs\sota_runs\`.
- No stopping rule on results; every arm is reported whatever it shows. The only abort conditions are pipeline failures.

## Stage E2: student recall by target size (about 15 min)

Checkpoints: RA-CMCD all-consensus s0-3, plain consensus s42-45, single-best s42-45, Soft-Teacher s0-7. Evaluate each at its frozen tuple on the three validation strata (`images/val_small`, `val_medium`, `val_large`; max box side < 28 / 28-39 / >= 39 px) and report recall and mAP@0.5:0.95 per stratum, four-seed mean. Output: `e2_strata_recall.tsv`.

## Stage B3: paired bootstrap for the leave-one-out contrasts (about 1 h)

Regimes: full RA-CMCD (s0-3), confidence-only, drop-support, drop-prior, drop-localization (s0-3 each), Soft-Teacher-style (s0-7). Same four-seed mean-over-seeds paired image-level bootstrap as the paper (B = 10,000, seed 20260515, shared image indices, one-sided H0: delta <= 0), on night-78 and night-160, contrasts full minus each variant. Output: `b3_subterm_bootstrap.json/.md`. Multiplicity: five contrasts per split are reported with Holm-adjusted p-values alongside the raw ones.

## Stage A1: formula-matched Abid et al. (BMVC 2025) score (4 + 4 runs, about 5 GPU-h)

Score recomputed on the identical clusters from the cached teacher predictions: spread = 1 - C_loc; cf = exp(-5 * spread) * [1 + 0.1 * (models - 2)]; s* = (max member confidence) * cf; weight = min(1, (s*)^2) (their p = 2). Two arms:
- `abid_soft`: all 1,506 boxes, weight = (s*)^2 (same-box comparison with RA-CMCD and confidence-only; primary).
- `abid_soft_t04`: their retention rule, boxes with s* < 0.4 removed, same weights (secondary; not same-box).
Their clustering threshold (0.65) and pre-filter (0.35) are not reproduced because the comparison is fixed to the same boxes; this is stated in the paper.

## Stage B2: threshold students (15 runs, about 10 GPU-h)

Filtered datasets at tau = 0.2, 0.3, 0.5 (built with `build_cmcd_dataset.py --min-reliability`), 4 seeds each; tau = 0.4 gains seeds 1-3 next to the existing seed 0. Together with the label-level curve this gives the student-level threshold curve.

## Stage D1: pool-size and composition students (about 15 GPU-h)

Pools, each with its own validation-selected WBF tuple over the 210-configuration grid, applied unchanged to the training split, reliability recomputed with M = pool size:
- two-teacher pools: {YOLO11-RGBT, C2Former}, {YOLO11-RGBT, CFT}, {C2Former, CFT};
- four-teacher pool: the three sources plus ICAFusion (train-split predictions exported once);
- homogeneous pool: three YOLO11-RGBT late-fusion seeds (seed 0 exists; seeds 1 and 2 trained first with the baseline recipe, about 1.3 GPU-h).
Four student seeds per pool. Reported with the existing three-teacher headline as M = 3.

## Reporting

Every result row traces to a `val_fixed_*.json` or a bootstrap JSON under the queue folder; numbers enter the manuscript only through a `tools/fill_*.py` script that logs to CHANGELOG.md.

## Incident 2026-09-08 10:41
- 2026-09-08 10:41: queue incident. At 10:34 the B2 training process (rel02 s1, epoch 47/100) died from a console window-close event (stage processes had visible console windows; someone closed one). The driver moved on to D1, whose four heterogeneous pools failed because the June C2Former/CFT checkpoints live under the nested paper tree, not the shim root. Fixes: all stage/training launches now use CREATE_NO_WINDOW; d1_pools.py remaps teacher checkpoints to the nested root (cached June predictions reused, same tags); student_runs.py treats a run as trained only when results.csv logs all 100 epochs and moves partial runs aside (the killed 47-epoch rel02_s1 had been selected and its val_fixed JSON written; both quarantined in cmcd_student/_partial_killed_2026-09-08 together with the 3-epoch late_fusion_s1 stub). Queue relaunched: B2 then D1.

## Incident 2026-09-08 12:28
- 2026-09-08 12:28: second queue incident. B2 re-used a rel03 s0 student directory from 2026-05-12 (complete, same recipe) whose dataset image folder no longer existed before today's rebuild, so its labels cannot be verified; the selection file written on it is invalid. Quarantined in cmcd_student/_unverified_may_rel03_2026-09-08; queue killed at 12:27 and relaunched (rel03 s0 retrains). student_runs.py now also refuses to re-use any run directory older than 2026-09-06; June arms remain reusable only through their existing val_fixed selection files.
