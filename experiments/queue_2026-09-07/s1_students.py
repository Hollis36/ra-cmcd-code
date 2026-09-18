"""S1 stage: corrected Soft-Teacher-style students (real files), four seeds."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_runs import ensure_student, log  # noqa: E402

YAML = Path(r"<YOLO11_ROOT>/outputs/sota_runs/cmcd_dataset_soft_v2/anti_uav_cmcd_student.yaml")

if __name__ == "__main__":
    if not YAML.exists():
        log("S1", f"dataset missing: {YAML}")
        sys.exit(1)
    for seed in range(4):
        ensure_student("S1", "soft_v2", seed, YAML)
    log("S1", "stage complete")
