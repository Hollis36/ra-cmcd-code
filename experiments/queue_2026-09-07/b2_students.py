"""B2 stage: reliability-filtered students at tau = 0.2, 0.3, 0.4, 0.5 (four seeds each).

Datasets are built with the paper's build_cmcd_dataset.py (--min-reliability) from the identical
train_train_teacher_global label set; tau = 0.4 already has seed 0 (submitted) and tau = 0.5 already
has a dataset folder; both are reused.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_runs import CWD, PY, SHIM, ensure_student, log  # noqa: E402

ROOT = Path(r"<YOLO11_ROOT>/outputs/sota_runs")
FUSED = ROOT / "cross_model_wbf" / "fused"
TAUS = {"rel02": 0.2, "rel03": 0.3, "rel04": 0.4, "rel05": 0.5}


def ensure_dataset(name: str, tau: float) -> Path:
    d = ROOT / f"cmcd_dataset_{name}"
    yaml = d / "anti_uav_cmcd_student.yaml"
    if yaml.exists() and (d / "labels" / "train").exists() and any((d / "labels" / "train").iterdir()):
        log("B2", f"dataset {name} exists")
        return yaml
    log("B2", f"building dataset {name} (min reliability {tau})")
    logfile = Path(__file__).resolve().parent / "logs" / f"build_{name}.log"; logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as f:
        rc = subprocess.run([PY, "-u", str(SHIM / "build_cmcd_dataset.py"), "--train-teacher", str(FUSED / "train_train_teacher_global"),
                             "--val-teacher", str(FUSED / "val_val_teacher_global"), "--out", str(d), "--min-reliability", str(tau), "--force"],
                            cwd=CWD, stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000).returncode
    if rc != 0 or not yaml.exists():
        raise RuntimeError(f"dataset build failed for {name} rc={rc}")
    return yaml


if __name__ == "__main__":
    for name, tau in TAUS.items():
        try:
            yaml = ensure_dataset(name, tau)
        except Exception as exc:  # noqa: BLE001
            log("B2", f"{name} FAILED: {exc!r}"); continue
        for seed in range(4):
            ensure_student("B2", name, seed, yaml)
    log("B2", "stage complete")
