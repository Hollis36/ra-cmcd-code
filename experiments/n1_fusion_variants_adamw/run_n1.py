"""N1: retrain the early-fusion and mid-to-late-fusion YOLO11n-RGBT variants with the student recipe.

Recipe (identical to the headline students and the late-fusion human-label baseline,
see the run manifest): AdamW, lr0 0.001, warmup 3 epochs, 100 epochs,
batch 8, 640 px, close_mosaic 10, AMP, deterministic, seeds 42-45, dir-copy keyframe
dataset. After training, each checkpoint is selected on the validation split over the
reduced 6-setting grid and frozen on night-78 / night-160, in the same JSON schema as
outputs/sota_runs/cmcd_student/val_fixed_*.json.

Resumable: a run whose val_fixed JSON already exists is skipped; a run whose best.pt
exists but has no JSON is only evaluated.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

FORK = Path(r"<FORK_ROOT>")
OUT = Path(r"<YOLO11_ROOT>/outputs/sota_runs/rgbt_variants_adamw")
CFG = Path(r"<YOLO11_ROOT>/outputs/sota_runs/yolo11_rgbt_repro/cfg")
DATA_TRAIN = CFG / "anti_uav_rgbt_full_dir.yaml"
EVAL_YAML = {
    "val": CFG / "anti_uav_rgbt_select_val.yaml",
    "night78": CFG / "anti_uav_rgbt_fixed_night78.yaml",
    "night160": CFG / "anti_uav_rgbt_fixed_night160.yaml",
}
MODELS = {
    "earlyfusion": FORK / "ultralytics/cfg/models/11-RGBT/yolo11-RGBT-earlyfusion.yaml",
    "mid-to-late-fusion": FORK / "ultralytics/cfg/models/11-RGBT/yolo11-RGBT-mid-to-late-fusion.yaml",
}
SEEDS = [42, 43, 44, 45]
GRID_IMGSZ = [640, 704]
GRID_CONF = [0.05, 0.08, 0.10]
GRID_IOU = [0.40]
LOG = OUT / "n1_progress.log"
SUMMARY = OUT / "n1_summary.json"


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def preflight() -> None:
    missing = [str(p) for p in [DATA_TRAIN, *EVAL_YAML.values(), *MODELS.values()] if not p.exists()]
    if missing:
        raise SystemExit(f"preflight: missing files {missing}")
    for split, y in EVAL_YAML.items():
        text = y.read_text(encoding="utf-8")
        root = next(l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("path:"))
        val = next(l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("val:"))
        target = Path(val) if Path(val).is_absolute() else Path(root) / val
        if target.suffix == ".txt":
            first = target.read_text(encoding="utf-8").splitlines()[0].strip()
            if not Path(first).exists():
                raise SystemExit(f"preflight: {split} list entry does not resolve: {first}")
        elif not target.exists():
            raise SystemExit(f"preflight: {split} val path does not resolve: {target}")
    # the fork must be the clean upstream (reliability defaults to 1 for human labels either way)
    import subprocess
    st = subprocess.run(["git", "-C", str(FORK), "status", "--short"], capture_output=True, text=True).stdout.strip()
    log(f"preflight ok; fork status: {'clean' if not st else st[:200]}")


def prepare_env() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "ultralytics_config"))
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    import numpy as np
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # numpy 2.x shim used by the fork's metrics code
    sys.path.insert(0, str(FORK))


def train(tag: str, model_yaml: Path, seed: int):
    from ultralytics import YOLO
    model = YOLO(str(model_yaml))
    t0 = time.time()
    metrics = model.train(
        data=str(DATA_TRAIN), imgsz=640, epochs=100, batch=8, workers=4, device="0",
        project=str(OUT), name=tag, exist_ok=True, seed=seed, deterministic=True, pretrained=True,
        optimizer="AdamW", use_simotm="RGBT", channels=4, close_mosaic=10, amp=True,
        lr0=0.001, warmup_epochs=3.0, pairs_rgb_ir=["images", "images_ir"], verbose=False, plots=False,
    )
    return metrics, (time.time() - t0) / 3600.0


def evaluate(ckpt: Path, split: str, imgsz: int, conf: float, iou: float) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(ckpt)).val(
        data=str(EVAL_YAML[split]), imgsz=imgsz, batch=1, conf=conf, iou=iou, device="0", verbose=False,
        plots=False, workers=0, project=str(OUT / "eval_runs"),
        name=f"{ckpt.parent.parent.name}_{split}_img{imgsz}_conf{conf:g}_iou{iou:g}", exist_ok=True,
        use_simotm="RGBT", channels=4, pairs_rgb_ir=["images", "images_ir"],
    )
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}


def val_fixed(tag: str, ckpt: Path) -> dict:
    rows = []
    for imgsz in GRID_IMGSZ:
        for conf in GRID_CONF:
            for iou in GRID_IOU:
                spec = {"ckpt": str(ckpt), "run_id": tag, "split": "val", "imgsz": imgsz, "conf": conf, "iou": iou}
                try:
                    metrics = evaluate(ckpt, "val", imgsz, conf, iou)
                    rows.append({"ok": True, "spec": spec, "metrics": metrics})
                    log(f"  [{tag}] val img={imgsz} conf={conf:g} iou={iou:g} mAP50-95={metrics['mAP50_95']:.4f}")
                except Exception as exc:  # noqa: BLE001
                    rows.append({"ok": False, "spec": spec, "error": repr(exc)})
                    log(f"  [{tag}] val img={imgsz} conf={conf:g} FAILED {exc!r}")
    good = [r for r in rows if r.get("ok")]
    best = max(good, key=lambda r: r["metrics"]["mAP50_95"])
    bs = best["spec"]
    fixed = []
    for split in ["val", "night78", "night160"]:
        metrics = evaluate(ckpt, split, bs["imgsz"], bs["conf"], bs["iou"])
        fixed.append({"split": split, "spec": {**bs, "split": split}, "metrics": metrics})
        log(f"  [{tag}] fixed {split} img={bs['imgsz']} conf={bs['conf']:g} mAP50-95={metrics['mAP50_95']:.4f}")
    return {
        "selection_policy": "select (imgsz, conf, iou) on val by mAP50_95 over the reduced 6-setting grid; freeze for night78/night160",
        "best_val": best, "fixed_evals": fixed, "selection_grid": rows,
        "recipe": {"optimizer": "AdamW", "lr0": 0.001, "warmup_epochs": 3.0, "epochs": 100, "batch": 8, "imgsz": 640,
                   "close_mosaic": 10, "amp": True, "deterministic": True, "data": str(DATA_TRAIN)},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight()
    prepare_env()
    import torch
    log(f"env: python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda} gpu {torch.cuda.get_device_name(0)}")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    for model_name, model_yaml in MODELS.items():
        for seed in SEEDS:
            tag = f"{model_name}_s{seed}"
            ckpt = OUT / tag / "weights" / "best.pt"
            out_json = OUT / f"val_fixed_{tag}.json"
            if out_json.exists():
                log(f"[skip] {tag}: val_fixed JSON exists")
                continue
            if not ckpt.exists():
                log(f"[train] {tag} start")
                try:
                    metrics, wall_h = train(tag, model_yaml, seed)
                except Exception as exc:  # noqa: BLE001
                    log(f"[train] {tag} FAILED: {exc!r}")
                    summary[tag] = {"error": repr(exc)}
                    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    continue
                summary[tag] = {"seed": seed, "wall_h": round(wall_h, 3),
                                "train_mAP50_95": float(metrics.box.map) if hasattr(metrics, "box") else None,
                                "ckpt": str(ckpt)}
                SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                log(f"[train] {tag} done in {wall_h:.2f} h")
            log(f"[select] {tag} start")
            result = val_fixed(tag, ckpt)
            out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            fx = {e["split"]: round(e["metrics"]["mAP50_95"], 4) for e in result["fixed_evals"]}
            summary.setdefault(tag, {})["val_fixed"] = fx
            SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log(f"[select] {tag} done {fx}")
    log("N1 complete")


if __name__ == "__main__":
    os.chdir(r"<YOLO11_ROOT>")
    main()
