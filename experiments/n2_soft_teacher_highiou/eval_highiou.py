"""N2 (TODO-N2-AP): recompute AP@0.5 / 0.75 / 0.85 / 0.90 and mAP@0.5:0.95 at each seed's frozen
validation-selected tuple for the Soft-Teacher-style arm (all eight seeds) and, as a protocol
consistency check, for the three recipe arms of Table II. No retraining; same evaluation path
as the validation-fixed protocol (RGBT, 4-channel, paired RGB/IR, batch 1).

Outputs: <YOLO11_ROOT>/outputs/sota_runs/cmcd_student/highiou_ap_v2/<tag>_highiou.json and
<this folder>/highiou_summary.tsv. Resumable per tag.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path

FORK = Path(r"<FORK_ROOT>")
RUNS = Path(r"<YOLO11_ROOT>/outputs/sota_runs/cmcd_student")
CFG = Path(r"<YOLO11_ROOT>/outputs/sota_runs/yolo11_rgbt_repro/cfg")
EVAL_YAML = {"val": CFG / "anti_uav_rgbt_select_val.yaml", "night78": CFG / "anti_uav_rgbt_fixed_night78.yaml", "night160": CFG / "anti_uav_rgbt_fixed_night160.yaml"}
OUT = RUNS / "highiou_ap_v2"
HERE = Path(__file__).resolve().parent
LOG = HERE / "highiou_progress.log"
ARMS = [
    ("late_cmcd_pseudo_soft_v2", range(4)),  # corrected Soft-Teacher-style arm (stage S1); the junction-built soft_teacher seeds are quarantined
    ("late_cmcd_pseudo_soft_teacher", range(8)),
    ("late_cmcd_pseudo_relw_all", range(4)),
    ("late_cmcd_pseudo_reloff", range(42, 46)),
    ("late_cmcd_pseudo_singlebest", range(42, 46)),
]


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def prepare_env() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "ultralytics_config"))
    os.environ.setdefault("WANDB_DISABLED", "true")
    import numpy as np
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    sys.path.insert(0, str(FORK))


def evaluate(ckpt: Path, split: str, imgsz: int, conf: float, iou: float) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(ckpt)).val(
        data=str(EVAL_YAML[split]), imgsz=imgsz, batch=1, conf=conf, iou=iou, device="0", verbose=False,
        plots=False, workers=0, project=str(OUT / "eval_runs"),
        name=f"{ckpt.parent.parent.name}_{split}_img{imgsz}_conf{conf:g}_iou{iou:g}", exist_ok=True,
        use_simotm="RGBT", channels=4, pairs_rgb_ir=["images", "images_ir"],
    )
    box = m.box
    all_ap = box.all_ap  # (nc, 10): IoU 0.50, 0.55, ..., 0.95
    per = {f"AP{50 + 5 * i}": float(all_ap[:, i].mean()) for i in range(all_ap.shape[1])} if len(all_ap) else {}
    return {"mAP50": float(box.map50), "mAP50_95": float(box.map), "mAP75": float(box.map75), **per}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_env()
    rows = []
    for arm, seeds in ARMS:
        for s in seeds:
            tag = f"{arm}_s{s}"
            vf = RUNS / f"val_fixed_{tag}.json"
            ckpt = RUNS / "runs" / tag / "weights" / "best.pt"
            out = OUT / f"{tag}_highiou.json"
            if not vf.exists() or not ckpt.exists():
                log(f"[skip] {tag}: missing val_fixed JSON or checkpoint"); continue
            if out.exists():
                res = json.loads(out.read_text(encoding="utf-8")); log(f"[cached] {tag}")
            else:
                d = json.loads(vf.read_text(encoding="utf-8")); spec = d["best_val"]["spec"]
                recorded = {e["split"]: e["metrics"] for e in d["fixed_evals"]}
                res = {"tag": tag, "frozen": {k: spec[k] for k in ("imgsz", "conf", "iou")}, "evals": {}}
                for split in ["val", "night78", "night160"]:
                    try:
                        met = evaluate(ckpt, split, spec["imgsz"], spec["conf"], spec["iou"])
                    except Exception as exc:  # noqa: BLE001
                        log(f"[eval] {tag} {split} FAILED {exc!r}"); met = {"error": repr(exc)}
                    met["recorded_mAP50_95"] = recorded.get(split, {}).get("mAP50_95")
                    res["evals"][split] = met
                    log(f"[eval] {tag} {split} mAP50-95={met.get('mAP50_95', float('nan')):.4f} (recorded {met['recorded_mAP50_95']}) AP75={met.get('AP75', float('nan')):.4f} AP90={met.get('AP90', float('nan')):.4f}")
                out.write_text(json.dumps(res, indent=2), encoding="utf-8")
            for split, met in res["evals"].items():
                rows.append([tag, split, met.get("mAP50"), met.get("mAP50_95"), met.get("recorded_mAP50_95"), met.get("AP75"), met.get("AP85"), met.get("AP90")])
    with (HERE / "highiou_summary.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["tag", "split", "mAP50", "mAP50_95", "recorded_mAP50_95", "AP75", "AP85", "AP90"]); w.writerows(rows)
    log("highiou complete")


if __name__ == "__main__":
    os.chdir(r"<YOLO11_ROOT>")
    main()
