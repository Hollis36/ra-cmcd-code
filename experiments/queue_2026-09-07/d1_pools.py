"""D1 stage: pool-size and composition students.

Pools (each: validation-selected WBF tuple over the 210-configuration grid, applied unchanged to the
training split with --save-best, reliability recomputed with M = pool size; then four student seeds):
  pool_y_c2 : YOLO11-RGBT + C2Former        pool_y_cft : YOLO11-RGBT + CFT      pool_c2_cft : C2Former + CFT
  pool_4t   : YOLO11-RGBT + C2Former + CFT + ICAFusion (ica_full_s42, prior 0.40 as in the June with-ICA run)
  pool_homo : three YOLO11-RGBT late-fusion seeds (seed 0 = the dir-data baseline; seeds 1, 2 trained here)
The WBF script is driven in-process with its source table extended, mirroring wbf_with_ica.py.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"<YOLO11_ROOT>/scripts/sota")
from student_runs import CWD, PY, SHIM, ensure_student, log, set_aside_partial, training_complete  # noqa: E402
import cross_model_wbf as W  # noqa: E402

ROOT = Path(r"<YOLO11_ROOT>/outputs/sota_runs")
FUSED = ROOT / "cross_model_wbf" / "fused"
YOLO_RUNS = ROOT / "yolo11_rgbt_repro" / "runs"
GRID_VAL = ["--wbf-ious", "0.25,0.3,0.35,0.4,0.45", "--score-modes", "avg,max,weighted", "--min-models", "1,2",
            "--post-confs", "0.01,0.02,0.03,0.04,0.05,0.06,0.07", "--nms-ious", "0.5"]
POOLS = {
    "pool_y_c2": ["yolo11_dirdata", "c2former"],
    "pool_y_cft": ["yolo11_dirdata", "cft"],
    "pool_c2_cft": ["c2former", "cft"],
    "pool_4t": ["yolo11_dirdata", "c2former", "cft", "ica"],
    "pool_homo": ["yolo11_dirdata", "yolo11_s1_dirdata", "yolo11_s2_dirdata"],
}

_ORIG = W.default_sources
# The June teacher checkpoints live under the nested paper tree (see the checkpoint-path note in the protocol), not under the shim root the
# helper derives from its own location; remap by relative path. The prediction cache is keyed by checkpoint
# stem and inference settings, so the cached June predictions are reused unchanged.
ALT_ROOTS = [Path(r"<YOLO11_ROOT>/outputs/outputs/sota_runs"), Path(r"<YOLO11_ROOT>/outputs/sota_runs")]


def _resolve(spec):
    if spec.ckpt.exists():
        return spec
    try:
        rel = spec.ckpt.relative_to(ROOT)
    except ValueError:
        return spec
    for r in ALT_ROOTS:
        if (r / rel).exists():
            return dataclasses.replace(spec, ckpt=r / rel)
    return spec


def extended_sources(split: str):
    specs = [_resolve(s) for s in _ORIG(split)]
    specs.append(W.SourceSpec("ica", "icafusion", ROOT / "ica_full_s42" / "yolov5_runs" / "run" / "weights" / "best.pt", imgsz=640, conf=0.001, iou=0.4, weight=0.40))
    for s in (1, 2):
        specs.append(W.SourceSpec(f"yolo11_s{s}_dirdata", "yolo11", YOLO_RUNS / f"late_fusion_s{s}_full_dirdata" / "weights" / "best.pt", imgsz=704, conf=0.001, iou=0.6, weight=1.0))
    return specs


W.default_sources = extended_sources


def ensure_late_fusion_seed(seed: int) -> Path:
    run_dir = YOLO_RUNS / f"late_fusion_s{seed}_full_dirdata"
    ckpt = run_dir / "weights" / "best.pt"
    if training_complete(run_dir):
        log("D1", f"late-fusion seed {seed} exists"); return ckpt
    set_aside_partial(run_dir, "D1")
    log("D1", f"training late-fusion seed {seed} (baseline recipe, dir data)")
    logfile = HERE / "logs" / f"late_fusion_s{seed}_full_dirdata.log"; logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as f:
        rc = subprocess.run([PY, "-u", str(SHIM / "run_yolo11_rgbt_repro.py"), "--model", "late", "--data-mode", "dir", "--regime", "full", "--seed", str(seed)],
                            cwd=CWD, stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000).returncode
    if rc != 0 or not ckpt.exists():
        raise RuntimeError(f"late-fusion seed {seed} training failed rc={rc}")
    return ckpt


def wbf(split: str, tag: str, sources: list[str], extra: list[str]) -> Path:
    summary = ROOT / "cross_model_wbf" / f"summary_{split}_{tag}.json"
    if summary.exists():
        log("D1", f"WBF {split}/{tag} exists"); return summary
    log("D1", f"WBF {split}/{tag} start (sources {','.join(sources)})")
    sys.argv = ["cross_model_wbf.py", "--split", split, "--sources", ",".join(sources), "--tag", tag, *extra]
    W.main()
    if not summary.exists():
        raise RuntimeError(f"WBF summary missing: {summary}")
    return summary


def best_tuple(summary: Path) -> dict:
    b = json.loads(summary.read_text(encoding="utf-8"))["best"]
    assert b.get("kind") == "wbf", f"best configuration is not WBF: {b}"
    return {k: b[k] for k in ("iou", "score_mode", "min_models", "post_conf")}


def ensure_dataset(tag: str) -> Path:
    d = ROOT / f"cmcd_dataset_{tag}"; yaml = d / "anti_uav_cmcd_student.yaml"
    if yaml.exists():
        log("D1", f"dataset {tag} exists"); return yaml
    logfile = HERE / "logs" / f"build_{tag}.log"; logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as f:
        rc = subprocess.run([PY, "-u", str(SHIM / "build_cmcd_dataset.py"), "--train-teacher", str(FUSED / f"train_{tag}"), "--val-teacher", str(FUSED / f"val_{tag}"),
                             "--out", str(d), "--force"], cwd=CWD, stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000).returncode
    if rc != 0 or not yaml.exists():
        raise RuntimeError(f"dataset build failed for {tag} rc={rc}")
    return yaml


def run_pool(tag: str, sources: list[str]) -> None:
    val_summary = wbf("val", tag, sources, [*GRID_VAL, "--save-best"])
    t = best_tuple(val_summary)
    log("D1", f"{tag}: validation-selected tuple {t}")
    wbf("train", tag, sources, ["--wbf-ious", str(t["iou"]), "--score-modes", str(t["score_mode"]), "--min-models", str(t["min_models"]),
                               "--post-confs", str(t["post_conf"]), "--nms-ious", "0.5", "--save-best"])
    manifest = FUSED / f"train_{tag}" / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        log("D1", f"{tag}: training labels {m.get('num_boxes')} boxes on {m.get('num_images')} images")
    yaml = ensure_dataset(tag)
    for seed in range(4):
        ensure_student("D1", tag, seed, yaml)


if __name__ == "__main__":
    os.chdir(CWD)
    for tag, sources in POOLS.items():
        try:
            if tag == "pool_homo":
                ensure_late_fusion_seed(1); ensure_late_fusion_seed(2)
            run_pool(tag, sources)
        except Exception as exc:  # noqa: BLE001
            log("D1", f"{tag} FAILED: {exc!r}")
    log("D1", "stage complete")
