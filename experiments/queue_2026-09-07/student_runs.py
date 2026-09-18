"""Shared helper: train one student with the headline recipe and run its validation-fixed selection.

Everything goes through the execution shim <YOLO11_ROOT>/scripts/sota so outputs land under
<YOLO11_ROOT>/outputs/sota_runs/cmcd_student exactly like the existing runs. Resumable: a run whose
val_fixed JSON exists is skipped; a run whose best.pt exists is only evaluated.
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

PY = sys.executable
SHIM = Path(r"<YOLO11_ROOT>/scripts/sota")
CWD = r"<YOLO11_ROOT>"
RUNS = Path(r"<YOLO11_ROOT>/outputs/sota_runs/cmcd_student")
QUEUE = Path(__file__).resolve().parent
LOG = QUEUE / "queue_progress.log"
GRID = ["--imgsz", "640,704", "--confs", "0.05,0.08,0.1", "--ious", "0.4"]


def log(stage: str, msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {stage} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], stage: str, logfile: Path) -> int:
    with logfile.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(map(str, cmd))}\n")
        f.flush()
        proc = subprocess.run([str(c) for c in cmd], cwd=CWD, stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000)
    return proc.returncode


EPOCHS = 100  # every student and late-fusion arm trains 100 epochs; fewer logged epochs means a killed run


QUEUE_ERA = 1788739200.0  # 2026-09-06 00:00 UTC: run directories older than the revision effort are never re-used


def training_complete(run_dir: Path, epochs: int = EPOCHS) -> bool:
    """A run counts as trained only if best.pt exists, results.csv logs all epochs (guards against killed runs), and the
    run was produced during this revision effort (older, unverified May/June directories are set aside instead)."""
    ckpt = run_dir / "weights" / "best.pt"
    csv_ = run_dir / "results.csv"
    if not (ckpt.exists() and csv_.exists()):
        return False
    if csv_.stat().st_mtime < QUEUE_ERA:
        return False
    with csv_.open(encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1
    return n >= epochs


def set_aside_partial(run_dir: Path, stage: str) -> None:
    """Move an incomplete run directory out of the way so the retrain starts clean and the evidence is kept."""
    if run_dir.exists():
        import datetime as _dt
        dst = run_dir.with_name(f"{run_dir.name}_partial_{_dt.datetime.now():%Y%m%d_%H%M%S}")
        run_dir.rename(dst)
        log(stage, f"[partial] {run_dir.name}: incomplete run moved to {dst.name}")


def ensure_student(stage: str, variant: str, seed: int, data_yaml: Path) -> dict | None:
    tag = f"late_cmcd_pseudo_{variant}_s{seed}"
    ckpt = RUNS / "runs" / tag / "weights" / "best.pt"
    vf = RUNS / f"val_fixed_{tag}.json"
    logfile = QUEUE / "logs" / f"{tag}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    if vf.exists():
        log(stage, f"[skip] {tag}: val_fixed exists")
        return read_fixed(vf)
    if not training_complete(RUNS / "runs" / tag):
        set_aside_partial(RUNS / "runs" / tag, stage)
        log(stage, f"[train] {tag} start ({data_yaml.name})")
        rc = run([PY, "-u", SHIM / "run_cmcd_student.py", "--data", data_yaml, "--variant", variant, "--seed", str(seed)], stage, logfile)
        if rc != 0 or not ckpt.exists():
            log(stage, f"[train] {tag} FAILED rc={rc}")
            return None
        log(stage, f"[train] {tag} done")
    log(stage, f"[select] {tag} start")
    rc = run([PY, "-u", SHIM / "val_fixed_eval.py", "--ckpt", ckpt, *GRID, "--out", vf], stage, logfile)
    if rc != 0 or not vf.exists():
        log(stage, f"[select] {tag} FAILED rc={rc}")
        return None
    fx = read_fixed(vf)
    log(stage, f"[select] {tag} done {fx}")
    return fx


def read_fixed(vf: Path) -> dict:
    import json
    d = json.loads(vf.read_text(encoding="utf-8"))
    return {e["split"]: round(e["metrics"]["mAP50_95"], 4) for e in d.get("fixed_evals", [])}
