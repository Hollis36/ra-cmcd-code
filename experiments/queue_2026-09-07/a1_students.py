"""A1 stage: train the Abid-score students (all-box primary arm, then the retention-threshold arm)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_runs import ensure_student, log  # noqa: E402

ROOT = Path(r"<YOLO11_ROOT>/outputs/sota_runs")

if __name__ == "__main__":
    for variant in ("abid_soft", "abid_soft_t04"):
        yaml = ROOT / f"cmcd_dataset_{variant}" / "anti_uav_cmcd_student.yaml"
        if not yaml.exists():
            log("A1", f"dataset missing for {variant}: {yaml}")
            continue
        for seed in range(4):
            ensure_student("A1", variant, seed, yaml)
    log("A1", "stage complete")
