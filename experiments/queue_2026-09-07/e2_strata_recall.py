"""E2: student recall and mAP by target-size stratum on the validation split.

Each checkpoint is evaluated at its own frozen validation-selected tuple (from val_fixed_*.json) on the
three size strata built by build_size_strata.py (max box side < 28 / 28-39 / >= 39 px; 157/254/168 images).
Recall is Ultralytics' mean recall at the frozen confidence. Output: e2_strata_recall.tsv + per-arm means.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import statistics as st
import sys
from pathlib import Path

FORK = Path(r"<FORK_ROOT>")
RUNS = Path(r"<YOLO11_ROOT>/outputs/sota_runs/cmcd_student")
DATA = Path(r"<YOLO11_ROOT>/data/keyframes_split")
HERE = Path(__file__).resolve().parent
OUT = HERE / "e2_strata"
STRATA = ["small", "medium", "large"]
ARMS = [
    ("late_cmcd_pseudo_relw_all", range(4), "RA-CMCD all-consensus"),
    ("late_cmcd_pseudo_reloff", range(42, 46), "Plain consensus"),
    ("late_cmcd_pseudo_singlebest", range(42, 46), "Single-best teacher"),
    ("late_cmcd_pseudo_soft_teacher", range(8), "Soft-Teacher-style"),
]


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with (HERE / "queue_progress.log").open("a", encoding="utf-8") as f:
        f.write("E2 " + line + "\n")


def yaml_for(stratum: str) -> Path:
    p = OUT / f"anti_uav_val_{stratum}.yaml"
    p.write_text(f"path: {DATA.as_posix()}\ntrain: images/val_{stratum}\nval: images/val_{stratum}\nnc: 1\nnames: [drone]\n", encoding="utf-8")
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "ultralytics_config")); os.environ.setdefault("WANDB_DISABLED", "true")
    import numpy as np
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    sys.path.insert(0, str(FORK))
    from ultralytics import YOLO
    yamls = {s: yaml_for(s) for s in STRATA}
    rows = []
    for arm, seeds, label in ARMS:
        for s in seeds:
            tag = f"{arm}_s{s}"
            vf = RUNS / f"val_fixed_{tag}.json"; ckpt = RUNS / "runs" / tag / "weights" / "best.pt"
            if not vf.exists() or not ckpt.exists():
                log(f"[skip] {tag}"); continue
            spec = json.loads(vf.read_text(encoding="utf-8"))["best_val"]["spec"]
            cache = OUT / f"{tag}.json"
            if cache.exists():
                res = json.loads(cache.read_text(encoding="utf-8"))
            else:
                res = {}
                for stratum in STRATA:
                    m = YOLO(str(ckpt)).val(data=str(yamls[stratum]), imgsz=spec["imgsz"], batch=1, conf=spec["conf"], iou=spec["iou"], device="0",
                                            verbose=False, plots=False, workers=0, project=str(OUT / "eval_runs"), name=f"{tag}_{stratum}", exist_ok=True,
                                            use_simotm="RGBT", channels=4, pairs_rgb_ir=["images", "images_ir"])
                    res[stratum] = {"recall": float(m.box.mr), "precision": float(m.box.mp), "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}
                    log(f"[eval] {tag} {stratum} recall={res[stratum]['recall']:.4f} mAP50-95={res[stratum]['mAP50_95']:.4f}")
                cache.write_text(json.dumps(res, indent=2), encoding="utf-8")
            for stratum, r in res.items():
                rows.append([label, tag, stratum, r["recall"], r["precision"], r["mAP50"], r["mAP50_95"]])
    with (HERE / "e2_strata_recall.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["arm", "tag", "stratum", "recall", "precision", "mAP50", "mAP50_95"]); w.writerows(rows)
    lines = ["# E2 student recall by stratum (validation split, frozen tuples)", "", "| arm | stratum | n seeds | recall | mAP@0.5:0.95 |", "|---|---|---|---|---|"]
    for arm, seeds, label in ARMS:
        for stratum in STRATA:
            rr = [r for r in rows if r[0] == label and r[2] == stratum]
            if rr:
                lines.append(f"| {label} | {stratum} | {len(rr)} | {st.mean(r[3] for r in rr):.4f} ± {st.stdev([r[3] for r in rr]) if len(rr) > 1 else 0:.4f} | {st.mean(r[6] for r in rr):.4f} |")
    (HERE / "e2_strata_recall.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("E2 complete")


if __name__ == "__main__":
    os.chdir(r"<YOLO11_ROOT>")
    main()
