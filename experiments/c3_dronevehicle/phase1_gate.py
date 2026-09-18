"""C3 phase 1, steps D and E: teacher predictions on U, class-wise WBF, and the label-level gate.

Protocol mirrors the Anti-UAV study:
  - the WBF tuple (cluster IoU, score mode, minimum models, post-fusion confidence) is selected on the official
    validation split by mAP@0.5:0.95 against human labels, then applied to U unchanged;
  - reliability is Eq. 3 with M = 3 and the three teacher priors;
  - the gate compares three scorings of the identical fused boxes on U against U's held-out human labels:
    plain WBF score, confidence-only, and Eq. 3.

Gate: Eq. 3 must beat plain WBF by at least 0.3 pp mAP@0.5:0.95 on U before any student is trained.

Usage: python phase1_gate.py [--limit N]   (--limit subsets U for a smoke run)
Outputs gate_result.json / gate_result.md and cached predictions under the run directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(r"<YOLO11_ROOT>/outputs/sota_runs/dronevehicle_c3")
FORK = Path(r"<FORK_ROOT>")
LOG = HERE / "phase1_progress.log"
sys.path.insert(0, str(HERE))
import rescore_eq3 as R  # evaluator, IoU helpers, class names

PRIORS = {"late": 1.00, "midlate": 0.80, "early": 0.65}  # damp by fusion depth, fixed before any evaluation
GRID = {"iou": [0.30, 0.40, 0.50, 0.55], "mode": ["avg", "weighted", "max"], "min_models": [2, 3], "post_conf": [0.03, 0.05, 0.07]}
GATE_PP = 0.30


def log(msg: str) -> None:
    line = "[%s] GATE %s" % (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def predict(name: str, ckpt: Path, split: str, conf=0.001, iou=0.6, imgsz=640):
    """Cache per-image detections for one teacher on one split."""
    dst = OUT / "preds" / split / name
    if (dst / "done.json").exists():
        log("%s/%s predictions cached" % (split, name))
        return dst
    sys.path.insert(0, str(FORK))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "ultralytics_config"))
    os.environ.setdefault("WANDB_DISABLED", "true")
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    from ultralytics import YOLO
    log("%s/%s predicting" % (split, name))
    m = YOLO(str(ckpt))
    # stream=True is required: without it every Results object for 14k images is held in RAM and the process dies
    n = 0
    for _ in m.predict(source=str(OUT / "data" / "images" / split), imgsz=imgsz, conf=conf, iou=iou, device="0", stream=True,
                       save=False, save_txt=True, save_conf=True, verbose=False, project=str(OUT / "preds" / split), name=name,
                       exist_ok=True, use_simotm="RGBT", channels=4, pairs_rgb_ir=["images", "images_ir"]):
        n += 1
        if n % 2000 == 0:
            log("  %s/%s %d images" % (split, name, n))
    log("  %s/%s %d images done" % (split, name, n))
    (dst / "done.json").write_text(json.dumps({"conf": conf, "iou": iou, "imgsz": imgsz}), encoding="utf-8")
    return dst


def load_dets(pred_dir: Path, teacher: str, stems):
    out = {}
    lab = pred_dir / "labels"
    for s in stems:
        f = lab / ("%s.txt" % s)
        dets = []
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) >= 6:
                    dets.append(R.Det(s, int(float(p[0])), tuple(float(v) for v in p[1:5]), float(p[5]), teacher))
        out[s] = dets
    return out


def fuse(by_teacher_per_image, stems, iou_thr, mode, min_models, post_conf):
    """Class-wise WBF returning fused boxes with every scoring variant."""
    total_prior = sum(PRIORS.values())
    fused = []
    for s in stems:
        grouped = {}
        for t, per in by_teacher_per_image.items():
            for d in per.get(s, []):
                grouped.setdefault(d.cls, []).append(d)
        for cls, dets in grouped.items():
            dets = sorted(dets, key=lambda d: -d.score)
            clusters = []
            for d in dets:
                hit = None
                for c in clusters:
                    if R.iou(d.box, c[0].box) >= iou_thr:
                        hit = c
                        break
                if hit is None:
                    hit = []
                    clusters.append(hit)
                hit.append(d)
            for c in clusters:
                srcs = {d.source for d in c}
                if len(srcs) < min_models:
                    continue
                w = np.array([max(d.score, 1e-6) * PRIORS[d.source] for d in c])
                xy = np.array([R.to_xyxy(d.box) for d in c])
                box = R.to_xywh(tuple((xy * w[:, None]).sum(0) / w.sum()))
                conf_mean = float(np.mean([d.score for d in c]))
                if mode == "avg":
                    fused_score = conf_mean
                elif mode == "max":
                    fused_score = float(max(d.score for d in c))
                else:
                    fused_score = float(w.sum() / total_prior)
                if fused_score < post_conf:
                    continue
                sup = len(srcs) / len(PRIORS)
                pri = sum(PRIORS[x] for x in srcs) / total_prior
                loc = mean([R.iou(d.box, box) for d in c]) if len(c) > 1 else 1.0
                fused.append({"image": s, "cls": cls, "box": box, "plain": fused_score,
                              "conf_only": conf_mean, "eq3": conf_mean * (0.45 * sup + 0.35 * pri + 0.20 * loc),
                              "n_sources": len(srcs), "loc": loc, "struct": 0.45 * sup + 0.35 * pri + 0.20 * loc})
    return fused


def evaluate(fused, gt, scoring):
    preds = [R.Det(f["image"], f["cls"], f["box"], f[scoring]) for f in fused]
    return R.evaluate(preds, gt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="subset U for a smoke run")
    args = ap.parse_args()

    ckpts = {k: Path(v) for k, v in json.loads((OUT / "teacher_ckpts.json").read_text(encoding="utf-8")).items()}
    assert set(ckpts) == set(PRIORS), "teacher set mismatch: %s" % sorted(ckpts)

    # --- D: predictions -------------------------------------------------------------------------------
    dirs = {sp: {n: predict(n, c, sp) for n, c in ckpts.items()} for sp in ("val", "U")}

    def gt_for(split):
        d = OUT / "data" / "labels" / split
        return {f.stem: [R.Det(f.stem, c, b, 1.0) for c, b, _ in R.read(f, False)] for f in sorted(d.glob("*.txt"))}

    gt_val = gt_for("val")
    val_stems = sorted(gt_val)
    val_dets = {n: load_dets(dirs["val"][n], n, val_stems) for n in ckpts}

    # --- select the WBF tuple on val -----------------------------------------------------------------
    sel_path = OUT / "wbf_tuple.json"
    if sel_path.exists():
        best = json.loads(sel_path.read_text(encoding="utf-8"))
        log("WBF tuple loaded: %s" % best["tuple"])
    else:
        log("selecting the WBF tuple on val over %d configurations"
            % (len(GRID["iou"]) * len(GRID["mode"]) * len(GRID["min_models"]) * len(GRID["post_conf"])))
        best = None
        for i in GRID["iou"]:
            for md in GRID["mode"]:
                for mm in GRID["min_models"]:
                    for pc in GRID["post_conf"]:
                        f = fuse(val_dets, val_stems, i, md, mm, pc)
                        if not f:
                            continue
                        m = evaluate(f, gt_val, "plain")
                        if best is None or m["map50_95"] > best["map50_95"]:
                            best = {"tuple": {"iou": i, "mode": md, "min_models": mm, "post_conf": pc}, **m, "n_boxes": len(f)}
                            log("  new best %s -> mAP50-95 %.4f (%d boxes)" % (best["tuple"], m["map50_95"], len(f)))
        sel_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    t = best["tuple"]

    # --- E: apply unchanged to U and run the gate ----------------------------------------------------
    gt_u = gt_for("U")
    u_stems = sorted(gt_u)
    if args.limit:
        u_stems = u_stems[:args.limit]
        gt_u = {k: gt_u[k] for k in u_stems}
    u_dets = {n: load_dets(dirs["U"][n], n, u_stems) for n in ckpts}
    fused = fuse(u_dets, u_stems, t["iou"], t["mode"], t["min_models"], t["post_conf"])
    log("U: %d images, %d fused boxes at the frozen tuple" % (len(u_stems), len(fused)))

    res = {"tuple": t, "n_images_U": len(u_stems), "n_fused": len(fused), "priors": PRIORS, "scorings": {}}
    for scoring in ("plain", "conf_only", "eq3"):
        m = evaluate(fused, gt_u, scoring)
        res["scorings"][scoring] = m
        log("U %-10s mAP50 %.4f  mAP50-95 %.4f" % (scoring, m["map50"], m["map50_95"]))
    d_plain = (res["scorings"]["eq3"]["map50_95"] - res["scorings"]["plain"]["map50_95"]) * 100
    d_conf = (res["scorings"]["eq3"]["map50_95"] - res["scorings"]["conf_only"]["map50_95"]) * 100
    res["delta_eq3_minus_plain_pp"] = d_plain
    res["delta_eq3_minus_conf_only_pp"] = d_conf
    res["gate_threshold_pp"] = GATE_PP
    res["gate_passed"] = bool(d_plain >= GATE_PP)
    (HERE / "gate_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    lines = ["# C3 label-level gate on DroneVehicle U (%s)" % dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "",
             "Teachers trained on L only; WBF tuple selected on the official validation split and applied to U unchanged.",
             "Tuple: %s. U: %d images, %d fused boxes." % (t, len(u_stems), len(fused)), "",
             "| scoring | mAP@0.5 | mAP@0.5:0.95 |", "|---|---:|---:|"]
    for k, v in res["scorings"].items():
        lines.append("| %s | %.4f | %.4f |" % (k, v["map50"], v["map50_95"]))
    lines += ["", "- Eq. 3 minus plain WBF: %+.2f pp mAP@0.5:0.95 (gate needs >= %.2f pp)" % (d_plain, GATE_PP),
              "- Eq. 3 minus confidence-only: %+.2f pp mAP@0.5:0.95 (this is the structural-terms contribution)" % d_conf,
              "", "**Gate %s**" % ("PASSED" if res["gate_passed"] else "FAILED")]
    (HERE / "gate_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("gate %s: eq3-plain %+.2f pp, eq3-conf_only %+.2f pp" % ("PASSED" if res["gate_passed"] else "FAILED", d_plain, d_conf))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
