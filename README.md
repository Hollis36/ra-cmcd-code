# RA-CMCD -- reliability-aware cross-model consensus distillation

Code and result records for *Reliability-Aware Cross-Model Consensus Distillation for Small-Target Detection in
RGB-Thermal Anti-UAV Surveillance* (Peifu Zhang, Han Xu, Ruichan Lv, Xidian University). The manuscript is under
review; this repository is the reproducibility artifact referenced by its data-availability statement.

## Contents

| Folder | What is in it |
|---|---|
| `experiments/exact_bootstrap/` | the paired image-level bootstrap on the validator's own per-image statistics: capture, weighted-mAP engine, the Anti-UAV-RGBT and DroneVehicle runs, and a second implementation that checks the engine against Ultralytics `ap_per_class` on explicitly duplicated statistics |
| `experiments/c3_dronevehicle/` | the DroneVehicle label-efficiency study: split construction, teacher training and class-wise WBF, the four student arms, decision-level rescoring, and its pre-registration |
| `experiments/queue_2026-09-07/` | the student arms of the main benchmark: Soft-Teacher-style and ensemble-consensus baselines, reliability-threshold sweep, teacher-pool composition, sub-term bootstrap, small-target recall |
| `experiments/f1_teacher_complementarity/`, `w1_weight_fit/`, `d3_disagreement_strata/`, `n1_fusion_variants_adamw/`, `n2_soft_teacher_highiou/`, `fig5_efficiency/` | teacher complementarity, the weight-simplex fit, disagreement strata, the fusion-variant retrain, the high-IoU evaluation, and the accuracy-vs-efficiency figure |
| `patches/` | the reliability-aware dataloader and loss as a patch against the YOLOv11-RGBT fork at commit 9cc2e20 (Ultralytics 8.3.75) |

Each result record sits beside the script that produced it, so a number in the paper can be traced to a file and the
code that wrote it.

## What is not here

- **Datasets.** Anti-UAV-RGBT and DroneVehicle are third-party benchmarks and must be obtained from their authors; no
  images or labels are redistributed. The scripts document how the splits, pseudo labels and manifests are built.
- **Checkpoints and training logs.** Not needed to re-check a number, and hundreds of megabytes.
- **Bootstrap draw matrices and captured validator statistics.** Regenerated deterministically from the seeds in the
  scripts (`capture_stats.py`, then `antiuav_bootstrap.py` / `dronevehicle_bootstrap.py`).

## Running the code

The scripts were run on Windows with Python 3.12, PyTorch 2.11+cu128 and the YOLOv11-RGBT fork (Ultralytics 8.3.75).
Local absolute paths have been replaced by the placeholders `<PAPER_ROOT>`, `<YOLO11_ROOT>`, `<FORK_ROOT>` and
`<HOME>`; set them to your own locations before running a script.

## Licence

- **`patches/`** derives from the YOLOv11-RGBT fork and from Ultralytics, both AGPL-3.0, and inherits that licence;
  see [`patches/LICENSE`](patches/LICENSE).
- **Everything else** (scripts, configs and result records in `experiments/`) is released under the MIT License; see
  [`LICENSE`](LICENSE).

Running the scripts requires the AGPL-3.0 Ultralytics / YOLOv11-RGBT code, which is not redistributed here and stays
under its own licence. If you use this code or its results, please cite the paper.
