"""Recompute the DroneVehicle decision-level check with the paper's Eq. 3, on the cached June predictions.

Why: the June pilot (DRONEVEHICLE_SECOND_DATASET_SUPPLEMENT/scripts/evaluate_d3_wbf_reliability.py) scored fused
boxes with an equal-weight additive variant

    r = 0.25 * (conf + support/M + mean_prior_of_supporting + loc_consistency)

while Eq. 3 of the paper is multiplicative with fixed weights and a sum-normalized prior fraction

    r = mean_conf * (0.45 * support/M + 0.35 * sum_prior_supporting/sum_prior_all + 0.20 * loc_consistency)

The supplement says the check "follows the same structural terms as Eq. (3)", which is true of the terms but not of
the functional form. This script rescores the identical fused boxes under both, so the claim can be stated exactly.

No training and no GPU: it reuses the cached per-teacher predictions on the 1,469-image validation split.
Outputs rescore_eq3.json and rescore_eq3.md in this folder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np

SUPP = Path(r"<PAPER_ROOT>/DRONEVEHICLE_SECOND_DATASET_SUPPLEMENT")
PRED_ROOT = SUPP / "results" / "d3_teacher_predictions"
GT_DIR = Path(r"<DRONEVEHICLE_ROOT>/yolo_dataset/labels/val")
HERE = Path(__file__).resolve().parent

# priors exactly as the June pilot used them (accuracy-derived, near uniform)
PILOT_PRIORS = {"mcwf_s42": 1.000, "gated_s42": 0.985, "add_s42": 0.979, "max_s42": 0.978, "attention_s42": 0.959}
# priors set by the paper's own principle: damp the denser detectors (prediction counts on val)
DENSITY = {"mcwf_s42": 45936, "gated_s42": 43602, "add_s42": 48849, "max_s42": 50780, "attention_s42": 50821}
DAMPED_PRIORS = {k: round(min(DENSITY.values()) / v, 3) for k, v in DENSITY.items()}
CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}
IOUS = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
WBF_IOU = 0.55


@dataclass
class Det:
    image: str
    cls: int
    box: tuple
    score: float
    source: str = ""
    rel: float = 1.0


def to_xyxy(b):
    x, y, w, h = b
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def to_xywh(b):
    a, c, d, e = b
    return ((a + d) / 2, (c + e) / 2, d - a, e - c)


def iou(a, b):
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def read(path, with_conf):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        box = tuple(float(v) for v in p[1:5])
        conf = float(p[5]) if with_conf and len(p) > 5 else 1.0
        out.append((cls, box, conf))
    return out


def load_gt():
    gt = {}
    for f in sorted(GT_DIR.glob("*.txt")):
        gt[f.stem] = [Det(f.stem, c, b, 1.0) for c, b, _ in read(f, False)]
    return gt


def load_preds():
    by_teacher = {}
    for t in PILOT_PRIORS:
        d = PRED_ROOT / t / "labels"
        dets = []
        for f in sorted(d.glob("*.txt")):
            dets += [Det(f.stem, c, b, s, t) for c, b, s in read(f, True)]
        by_teacher[t] = dets
    return by_teacher


def evaluate(preds, gt):
    """Class-wise AP at each IoU threshold, matching the pilot's evaluator."""
    aps = {thr: {} for thr in IOUS}
    for thr in IOUS:
        for cls in CLASS_NAMES:
            p = sorted([d for d in preds if d.cls == cls], key=lambda d: -d.score)
            n_gt = sum(1 for v in gt.values() for d in v if d.cls == cls)
            if n_gt == 0:
                aps[thr][cls] = 0.0
                continue
            used = {}
            tp, fp = np.zeros(len(p)), np.zeros(len(p))
            for i, d in enumerate(p):
                cand = [g for g in gt.get(d.image, []) if g.cls == cls]
                best, bi = 0.0, -1
                for j, g in enumerate(cand):
                    v = iou(d.box, g.box)
                    if v > best:
                        best, bi = v, j
                key = (d.image, cls, bi)
                if best >= thr and bi >= 0 and key not in used:
                    used[key] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            ctp, cfp = np.cumsum(tp), np.cumsum(fp)
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            for k in range(len(mpre) - 2, -1, -1):
                mpre[k] = max(mpre[k], mpre[k + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            aps[thr][cls] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"map50": mean(aps[0.50].values()), "map50_95": mean([mean(v.values()) for v in aps.values()]),
            "per_class_ap50": {CLASS_NAMES[c]: round(aps[0.50][c], 4) for c in CLASS_NAMES}}


def cluster(by_teacher, priors):
    """Class-wise WBF; returns one fused box list with every scoring variant attached."""
    grouped = {}
    for dets in by_teacher.values():
        for d in dets:
            grouped.setdefault((d.image, d.cls), []).append(d)
    fused = []
    total_prior = sum(priors.values())
    for (image, cls), dets in grouped.items():
        dets = sorted(dets, key=lambda d: -d.score)
        clusters = []
        for d in dets:
            hit = None
            for c in clusters:
                if iou(d.box, c[0].box) >= WBF_IOU:
                    hit = c
                    break
            if hit is None:
                hit = []
                clusters.append(hit)
            hit.append(d)
        for c in clusters:
            w = np.array([max(d.score, 1e-6) * priors[d.source] for d in c])
            xy = np.array([to_xyxy(d.box) for d in c])
            box = to_xywh(tuple((xy * w[:, None]).sum(0) / w.sum()))
            srcs = {d.source for d in c}
            sup = len(srcs) / len(priors)
            loc = mean([iou(d.box, box) for d in c]) if len(c) > 1 else 1.0
            conf = float(np.mean([d.score for d in c]))
            pri_sum = sum(priors[s] for s in srcs) / total_prior
            pri_mean = mean([priors[s] for s in srcs])
            fused.append({
                "image": image, "cls": cls, "box": box,
                "plain": conf,
                "pilot_additive": 0.25 * (min(1.0, conf) + sup + pri_mean + loc),
                "eq3": conf * (0.45 * sup + 0.35 * pri_sum + 0.20 * loc),
                "conf_only": conf,
            })
    return fused


def main():
    gt = load_gt()
    by_teacher = load_preds()
    print("cached predictions:", {k: len(v) for k, v in by_teacher.items()})
    print("ground-truth images:", len(gt), "boxes:", sum(len(v) for v in gt.values()))
    print("damped priors (min density / own density):", DAMPED_PRIORS)

    res = {"wbf_iou": WBF_IOU, "n_images": len(gt), "priors_pilot": PILOT_PRIORS, "priors_damped": DAMPED_PRIORS,
           "scorings": {}, "single_teachers": {}}
    print("\nsingle teachers on the surviving official labels:")
    for t, dets in by_teacher.items():
        m = evaluate(dets, gt)
        res["single_teachers"][t] = {"n": len(dets), **m}
        print("  %-14s n=%6d  mAP50 %.4f  mAP50-95 %.4f" % (t, len(dets), m["map50"], m["map50_95"]))

    for prior_name, priors in (("pilot", PILOT_PRIORS), ("damped", DAMPED_PRIORS)):
        fused = cluster(by_teacher, priors)
        print("\npriors=%s: %d fused boxes" % (prior_name, len(fused)))
        for scoring in ("plain", "pilot_additive", "eq3"):
            preds = [Det(f["image"], f["cls"], f["box"], f[scoring]) for f in fused]
            m = evaluate(preds, gt)
            res["scorings"]["%s/%s" % (prior_name, scoring)] = {"n": len(preds), **m}
            print("  %-16s mAP50 %.4f  mAP50-95 %.4f" % (scoring, m["map50"], m["map50_95"]))
    (HERE / "rescore_eq3.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    lines = ["# DroneVehicle decision-level rescoring: Eq. 3 versus the June equal-weight variant", "",
             "Identical fused boxes (class-wise WBF, IoU %.2f) on the 1,469-image validation split; only the ranking score changes." % WBF_IOU,
             "No training, no GPU: the cached June teacher predictions are reused.", "",
             "| priors | scoring | boxes | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---:|---:|---:|"]
    for k, v in res["scorings"].items():
        p, s = k.split("/")
        lines.append("| %s | %s | %d | %.4f | %.4f |" % (p, s, v["n"], v["map50"], v["map50_95"]))
    base = res["scorings"]["pilot/plain"]
    lines += ["", "Deltas against plain WBF at the pilot priors:", ""]
    for k, v in res["scorings"].items():
        if k.endswith("plain"):
            continue
        lines.append("- %s: %+.2f pp mAP@0.5, %+.2f pp mAP@0.5:0.95" % (k, (v["map50"] - base["map50"]) * 100, (v["map50_95"] - base["map50_95"]) * 100))
    (HERE / "rescore_eq3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ready-to-input LaTeX replacement for the supplement appendix table
    NAMES = {"mcwf_s42": "MCWF teacher", "gated_s42": "Gated teacher", "add_s42": "Add teacher",
             "max_s42": "Max teacher", "attention_s42": "Attention teacher"}
    pl, e3 = res["scorings"]["pilot/plain"], res["scorings"]["pilot/eq3"]
    tex = ["% generated by experiments/c3_dronevehicle/rescore_eq3.py (audit item N7)",
           r"\begin{table}[!t]", r"\centering",
           r"\caption{DroneVehicle reduced teacher-pool reliability validation. Five existing YOLO11-S fusion teachers "
           r"from the DroneVehicle project are merged by class-wise WBF on the official $1{,}469$-image validation split; "
           r"plain and reliability-rescored WBF use identical boxes and only the ranking score changes. The reliability "
           r"score is Eq.~(3) of the main paper, evaluated against the official DroneVehicle YOLO annotations.}",
           r"\label{tab:dronevehicle-reliability}", r"\qhnetcompact", r"\resizebox{\columnwidth}{!}{%",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"Policy & Predictions & mAP@0.5 & mAP@0.5:0.95 \\", r"\midrule"]
    for t in ("mcwf_s42", "gated_s42", "add_s42", "max_s42", "attention_s42"):
        v = res["single_teachers"][t]
        tex.append("%s & %s & %.4f & %.4f \\\\" % (NAMES[t], format(v["n"], ","), v["map50"], v["map50_95"]))
    tex += [r"\midrule",
            "Plain WBF & %s & %.4f & %.4f \\\\" % (format(pl["n"], ","), pl["map50"], pl["map50_95"]),
            "Reliability-rescored WBF (Eq.~3) & %s & \\textbf{%.4f} & \\textbf{%.4f} \\\\" % (format(e3["n"], ","), e3["map50"], e3["map50_95"]),
            r"\midrule",
            "Delta & -- & $+%.4f$ & $+%.4f$ \\\\" % (e3["map50"] - pl["map50"], e3["map50_95"] - pl["map50_95"]),
            r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""]
    (HERE / "dronevehicle_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\nwrote rescore_eq3.json, rescore_eq3.md and dronevehicle_table.tex")
    print("main-text deltas to use: +%.2f pp mAP@0.5, +%.2f pp mAP@0.5:0.95"
          % ((e3["map50"] - pl["map50"]) * 100, (e3["map50_95"] - pl["map50_95"]) * 100))


if __name__ == "__main__":
    main()
