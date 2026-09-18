"""Paired image-level bootstrap for the DroneVehicle Eq. 3 rescoring (audit item N7).

Same design as the June stability check: resample the 1,469 validation images with replacement, compare plain WBF
scoring against Eq. 3 scoring on identical fused boxes through a per-image AP proxy. Reported as an auxiliary
stability check, not as a replacement for the global mAP deltas in rescore_eq3.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np

import rescore_eq3 as R

HERE = Path(__file__).resolve().parent
N_BOOT = 1000
SEED = 20260910


def ap_proxy(preds, gts, thr):
    """Per-image, per-class AP proxy: precision-recall area over the image's own detections."""
    out = []
    for cls in R.CLASS_NAMES:
        g = [d for d in gts if d.cls == cls]
        p = sorted([d for d in preds if d["cls"] == cls], key=lambda d: -d["score"])
        if not g and not p:
            continue
        if not g:
            out.append(0.0)
            continue
        used = set()
        tp = np.zeros(len(p))
        for i, d in enumerate(p):
            best, bi = 0.0, -1
            for j, gg in enumerate(g):
                if j in used:
                    continue
                v = R.iou(d["box"], gg.box)
                if v > best:
                    best, bi = v, j
            if best >= thr and bi >= 0:
                used.add(bi)
                tp[i] = 1
        if len(p) == 0:
            out.append(0.0)
            continue
        ctp = np.cumsum(tp)
        rec = ctp / len(g)
        prec = ctp / np.arange(1, len(p) + 1)
        out.append(float(np.sum(np.diff(np.concatenate(([0.0], rec))) * prec)))
    return mean(out) if out else 0.0


def main() -> None:
    raise SystemExit("RETIRED 2026-09-17: a per-image AP proxy cannot see rescoring across images (it measured +0.14 pp "
                     "where the global mAP@0.5:0.95 difference is +1.18 pp). Use bootstrap_rescore_global.py.")


def _retired_main() -> None:
    gt = R.load_gt()
    by_teacher = R.load_preds()
    fused = R.cluster(by_teacher, R.PILOT_PRIORS)
    per_image = {}
    for f in fused:
        per_image.setdefault(f["image"], []).append(f)
    images = sorted(gt)
    print("images %d, fused boxes %d" % (len(images), len(fused)))

    stats = {}
    for scoring in ("plain", "eq3"):
        d50, d5095 = {}, {}
        for img in images:
            boxes = [{"cls": f["cls"], "box": f["box"], "score": f[scoring]} for f in per_image.get(img, [])]
            g = gt.get(img, [])
            d50[img] = ap_proxy(boxes, g, 0.50)
            d5095[img] = mean([ap_proxy(boxes, g, t) for t in R.IOUS])
        stats[scoring] = (d50, d5095)
        print("  %-6s per-image proxy mean: AP50 %.4f  AP50-95 %.4f" % (scoring, mean(d50.values()), mean(d5095.values())))

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(images))
    draws = {"map50": [], "map50_95": []}
    for _ in range(N_BOOT):
        s = rng.choice(idx, len(idx), replace=True)
        sample = [images[i] for i in s]
        for key, k in (("map50", 0), ("map50_95", 1)):
            a = mean(stats["eq3"][k][i] for i in sample)
            b = mean(stats["plain"][k][i] for i in sample)
            draws[key].append(a - b)
    out = {"seed": SEED, "n_boot": N_BOOT, "n_images": len(images),
           "unit": "image-level resampling with replacement over a per-image AP proxy; auxiliary stability check",
           "observed_per_image_proxy": {
               "delta_map50": mean(stats["eq3"][0][i] for i in images) - mean(stats["plain"][0][i] for i in images),
               "delta_map50_95": mean(stats["eq3"][1][i] for i in images) - mean(stats["plain"][1][i] for i in images)}}
    for key, v in draws.items():
        a = np.array(v)
        out[key] = {"mean": float(a.mean()), "ci95_low": float(np.percentile(a, 2.5)),
                    "ci95_high": float(np.percentile(a, 97.5)), "p_delta_gt_0": float((a > 0).mean())}
        print("  delta %-9s mean %+.5f  CI [%+.5f, %+.5f]  P(delta>0) %.3f"
              % (key, a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5), (a > 0).mean()))
    (HERE / "bootstrap_eq3.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote bootstrap_eq3.json")


if __name__ == "__main__":
    main()
