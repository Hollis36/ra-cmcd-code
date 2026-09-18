"""Paired image-level bootstrap on GLOBAL AP for the DroneVehicle decision-level rescoring (plain WBF vs Eq. 3).

Replaces bootstrap_eq3.py (retired 2026-09-17). That script resampled a per-image AP proxy, which only sees how boxes
are ranked inside each image. Rescoring mainly moves boxes across images in the global ranking, so the proxy measured
+0.14 pp where the table's global mAP@0.5:0.95 difference is +1.18 pp. The same proxy family was already retired for
the student contrasts (it flipped the sign of a contrast there).

Construction. rescore_eq3.evaluate() matches detections in descending score order, per image and class, against
the best-IoU ground-truth box, with no fallback when that box is taken. Whether a detection is a true positive at a
threshold therefore depends only on higher-ranked detections of the same image and class, so it is unchanged when
whole images are resampled. The flags are computed once per scoring with exactly that rule (same IoU function, same
tie order); each bootstrap draw only assigns multiplicities to images and integrates precision-recall over the
weighted, globally ranked detections, exactly as the evaluator does. With every multiplicity equal to one the
construction must reproduce rescore_eq3.json; the script asserts this to 1e-9 before resampling.

Paired: both scorings are evaluated on the same draw. Outputs bootstrap_rescore_global.json / .md / _draws.npz.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # stay off the cores of the running job
    os.environ.setdefault(_v, "1")

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import rescore_eq3 as R

HERE = Path(__file__).resolve().parent
B = 10000
SEED = 20260910
SCORINGS = ("plain", "eq3")
T = len(R.IOUS)
C = len(R.CLASS_NAMES)


def build(fused, gt, units):
    """Per-detection unit index, class, scores and true-positive flags per scoring (fused order)."""
    uidx = {u: i for i, u in enumerate(units)}
    groups = defaultdict(list)
    for i, f in enumerate(fused):
        groups[(f["image"], f["cls"])].append(i)
    gts = defaultdict(list)
    for img, dets in gt.items():
        for g in dets:
            gts[(img, g.cls)].append(g.box)
    n = len(fused)
    img_i = np.array([uidx[f["image"]] for f in fused], dtype=np.int64)
    cls = np.array([f["cls"] for f in fused], dtype=np.int64)
    score = {s: np.array([f[s] for f in fused], dtype=np.float64) for s in SCORINGS}
    tp = {s: np.zeros((n, T), dtype=np.float64) for s in SCORINGS}
    for key, idxs in groups.items():
        g = gts.get(key, [])
        if not g:
            continue
        # IoU with the evaluator's own function; first maximum, as its strict '>' scan picks
        M = np.array([[R.iou(fused[i]["box"], b) for b in g] for i in idxs])
        best_j = M.argmax(axis=1)
        best_v = M[np.arange(len(idxs)), best_j]
        for s in SCORINGS:
            order = sorted(range(len(idxs)), key=lambda k: -fused[idxs[k]][s])  # stable, like the evaluator
            for t, thr in enumerate(R.IOUS):
                used = set()
                for k in order:
                    if best_v[k] >= thr and best_v[k] > 0 and best_j[k] not in used:
                        used.add(int(best_j[k]))
                        tp[s][idxs[k], t] = 1.0
    gt_uc = np.zeros((len(units), C), dtype=np.float64)
    for img, dets in gt.items():
        for d in dets:
            gt_uc[uidx[img], d.cls] += 1
    return img_i, cls, score, tp, gt_uc


def make_ap(img_i, cls, score, tp, gt_uc):
    """Return ap(scoring, weights) -> (mAP@0.5, mAP@0.5:0.95) with the evaluator's integration."""
    orders = {s: [np.flatnonzero(cls == c)[np.argsort(-score[s][cls == c], kind="stable")] for c in range(C)] for s in SCORINGS}
    flags = {s: [tp[s][o] for o in orders[s]] for s in SCORINGS}

    def ap(s, w):
        per = np.zeros((T, C))
        for c in range(C):
            n_gt = float(gt_uc[:, c] @ w)
            if n_gt == 0:
                continue  # the evaluator scores a class without ground truth as AP 0
            wd = w[img_i[orders[s][c]]]
            call = np.cumsum(wd)
            for t in range(T):
                ctp = np.cumsum(wd * flags[s][c][:, t])
                mrec = np.concatenate(([0.0], ctp / n_gt, [1.0]))
                mpre = np.concatenate(([0.0], ctp / np.maximum(call, 1e-9), [0.0]))
                mpre = np.maximum.accumulate(mpre[::-1])[::-1]
                i = np.flatnonzero(mrec[1:] != mrec[:-1])
                per[t, c] = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
        return per[0].mean(), per.mean()
    return ap


def main() -> None:
    gt = R.load_gt()
    fused = R.cluster(R.load_preds(), R.PILOT_PRIORS)
    units = sorted(set(gt) | {f["image"] for f in fused})
    print("resampling units (images): %d (ground-truth files %d), fused boxes %d" % (len(units), len(gt), len(fused)), flush=True)
    img_i, cls, score, tp, gt_uc = build(fused, gt, units)
    ap = make_ap(img_i, cls, score, tp, gt_uc)

    # exact reproduction of the table before any resampling
    table = json.loads((HERE / "rescore_eq3.json").read_text(encoding="utf-8"))["scorings"]
    ones = np.ones(len(units))
    point = {}
    for s in SCORINGS:
        m50, m5095 = ap(s, ones)
        ref = table["pilot/%s" % s]
        print("  %-5s reproduced mAP50 %.6f (table %.6f)  mAP50-95 %.6f (table %.6f)" % (s, m50, ref["map50"], m5095, ref["map50_95"]), flush=True)
        assert abs(m50 - ref["map50"]) < 1e-9 and abs(m5095 - ref["map50_95"]) < 1e-9, "construction does not reproduce the table"
        point[s] = {"map50": m50, "map50_95": m5095}

    rng = np.random.default_rng(SEED)
    draws = np.zeros((B, 2, 2))  # [draw, scoring, metric]
    for b in range(B):
        w = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(np.float64)
        for k, s in enumerate(SCORINGS):
            draws[b, k] = ap(s, w)
        if (b + 1) % 1000 == 0:
            print("  %d/%d draws" % (b + 1, B), flush=True)
    np.savez_compressed(HERE / "bootstrap_rescore_global_draws.npz", draws=draws, scorings=np.array(SCORINGS), metrics=np.array(["map50", "map50_95"]))

    res = {"B": B, "seed": SEED, "n_images": len(units), "n_fused_boxes": len(fused), "priors": "pilot",
           "construction": "global AP with the evaluator of rescore_eq3.py; true-positive flags precomputed per scoring; "
                           "paired image-level resampling; reproduces rescore_eq3.json exactly at unit multiplicities",
           "point": point, "contrast_eq3_minus_plain": {}}
    lines = ["# DroneVehicle rescoring: paired image-level bootstrap on global AP (%s)" % dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "",
             "B = %d, seed %d, %d validation images resampled with replacement; plain and Eq. 3 scorings share every draw." % (B, SEED, len(units)),
             "Same evaluator as the table (rescore_eq3.py), so the point estimates below are the table's values exactly.", "",
             "| metric | plain | Eq. 3 | difference (pp) | 95% CI (pp) | one-sided p | positive draws |", "|---|---:|---:|---:|---|---:|---:|"]
    for m, name in ((0, "map50"), (1, "map50_95")):
        d = draws[:, 1, m] - draws[:, 0, m]
        diff = point["eq3"][name] - point["plain"][name]
        lo, hi = np.percentile(d, [2.5, 97.5])
        c = {"point_diff": diff, "diff_ci95": [float(lo), float(hi)], "p_one_sided_eq3_le_plain": float((d <= 0).mean()),
             "positive_rate": float((d > 0).mean()), "draw_mean": float(d.mean())}
        res["contrast_eq3_minus_plain"][name] = c
        lines.append("| %s | %.4f | %.4f | %+.2f | [%+.2f, %+.2f] | %s | %.1f%% |" % (
            "mAP@0.5" if m == 0 else "mAP@0.5:0.95", point["plain"][name], point["eq3"][name], 100 * diff, 100 * lo, 100 * hi,
            "< 1e-4" if c["p_one_sided_eq3_le_plain"] == 0 else "%.4f" % c["p_one_sided_eq3_le_plain"], 100 * c["positive_rate"]))
    (HERE / "bootstrap_rescore_global.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (HERE / "bootstrap_rescore_global.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
