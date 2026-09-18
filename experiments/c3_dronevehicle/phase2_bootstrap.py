"""Paired image-level bootstrap for the DroneVehicle student contrast, mirroring the Anti-UAV protocol.

Same scheme as the main study: each of B draws resamples the same test-image indices for every arm, each arm is
summarized by its four-seed mean AP on the drawn images, and the reported interval is the percentile interval of
the paired difference under the one-sided null that weighted <= plain.

Per-image AP is computed once per (arm, seed) from cached detections at that student's frozen setting, so the
bootstrap itself is CPU-only and fast. Detections are exported once per student and cached.

Outputs phase2_bootstrap.json / .md.
"""
from __future__ import annotations

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
LOG = HERE / "phase2_progress.log"
CLASSES = ["car", "truck", "bus", "van", "freight_car"]
IOUS = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
ARMS = ["plain", "weighted", "gt_only"]
SEEDS = [0, 1, 2, 3]
B = 10000
SEED = 20260913
CONTRASTS = [("weighted", "plain"), ("weighted", "gt_only"), ("plain", "gt_only")]


def log(msg: str) -> None:
    line = "[%s] P2-BOOT %s" % (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1 = a[:, 0:1] - a[:, 2:3] / 2, a[:, 1:2] - a[:, 3:4] / 2
    ax2, ay2 = a[:, 0:1] + a[:, 2:3] / 2, a[:, 1:2] + a[:, 3:4] / 2
    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    ua = (a[:, 2:3] * a[:, 3:4]) + (b[:, 2] * b[:, 3]) - inter
    return inter / np.maximum(ua, 1e-12)


def export(arm: str, seed: int) -> Path:
    """Cache test-split detections for one student at its frozen setting."""
    key = "%s_s%d" % (arm, seed)
    dst = OUT / "test_preds" / key
    if (dst / "done.json").exists():
        return dst
    cfg = json.loads((HERE / "phase2_eval.json").read_text(encoding="utf-8"))[key]["setting"]
    sys.path.insert(0, str(FORK))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "ultralytics_config"))
    os.environ.setdefault("WANDB_DISABLED", "true")
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    from ultralytics import YOLO
    log("exporting test detections for %s at %s" % (key, cfg))
    m = YOLO(str(OUT / "students" / key / "weights" / "best.pt"))
    n = 0
    for _ in m.predict(source=str(OUT / "eval" / "images" / "test"), imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"],
                       device="0", stream=True, save=False, save_txt=True, save_conf=True, verbose=False,
                       project=str(OUT / "test_preds"), name=key, exist_ok=True,
                       use_simotm="RGBT", channels=4, pairs_rgb_ir=["images", "images_ir"]):
        n += 1
    (dst / "done.json").write_text(json.dumps({**cfg, "n": n}), encoding="utf-8")
    log("  %s: %d images" % (key, n))
    return dst


def read_dets(d: Path, stems):
    lab = d / "labels"
    out = {}
    for s in stems:
        f = lab / ("%s.txt" % s)
        rows = []
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) >= 6:
                    rows.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
        out[s] = rows
    return out


def per_image_ap(dets, gts):
    """Mean over classes present of the image's own AP, averaged over IoU thresholds."""
    vals = []
    for cls in range(len(CLASSES)):
        g = np.array([r[1:] for r in gts if r[0] == cls], dtype=float).reshape(-1, 4)
        d = sorted([r for r in dets if r[0] == cls], key=lambda r: -r[5])
        if len(g) == 0 and len(d) == 0:
            continue
        if len(g) == 0:
            vals.append(0.0)
            continue
        db = np.array([r[1:5] for r in d], dtype=float).reshape(-1, 4)
        M = iou_mat(db, g) if len(d) else np.zeros((0, len(g)))
        per_thr = []
        for thr in IOUS:
            used = set()
            tp = np.zeros(len(d))
            for i in range(len(d)):
                j = int(np.argmax(M[i])) if M.shape[1] else -1
                if j >= 0 and M[i, j] >= thr and j not in used:
                    used.add(j)
                    tp[i] = 1
            if len(d) == 0:
                per_thr.append(0.0)
                continue
            ctp = np.cumsum(tp)
            rec = ctp / len(g)
            prec = ctp / np.arange(1, len(d) + 1)
            per_thr.append(float(np.sum(np.diff(np.concatenate(([0.0], rec))) * prec)))
        vals.append(mean(per_thr))
    return mean(vals) if vals else 0.0


def main() -> None:
    raise SystemExit("RETIRED 2026-09-13: this per-image AP proxy penalises detection count and flipped the sign of a "
                     "contrast on DroneVehicle. Use phase2_bootstrap_global.py.")


def _retired_main() -> None:
    gt_dir = OUT / "eval" / "labels" / "test"
    stems = sorted(p.stem for p in gt_dir.glob("*.txt"))
    gts = {}
    for s in stems:
        rows = []
        for line in (gt_dir / ("%s.txt" % s)).read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5:
                rows.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        gts[s] = rows
    log("test split: %d images, %d boxes" % (len(stems), sum(len(v) for v in gts.values())))

    cache = HERE / "phase2_per_image_ap.json"
    table = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    for arm in ARMS:
        for seed in SEEDS:
            key = "%s_s%d" % (arm, seed)
            if key in table:
                continue
            if not (OUT / "students" / key / "weights" / "best.pt").exists():
                continue
            d = read_dets(export(arm, seed), stems)
            table[key] = [per_image_ap(d[s], gts[s]) for s in stems]
            cache.write_text(json.dumps(table), encoding="utf-8")
            log("  %s per-image AP mean %.4f" % (key, mean(table[key])))

    arms_present = {a: [table["%s_s%d" % (a, s)] for s in SEEDS if "%s_s%d" % (a, s) in table] for a in ARMS}
    arms_present = {a: v for a, v in arms_present.items() if v}
    means = {a: np.mean(np.array(v), axis=0) for a, v in arms_present.items()}
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(stems))
    res = {"B": B, "seed": SEED, "n_images": len(stems),
           "n_seeds": {a: len(v) for a, v in arms_present.items()},
           "point": {a: float(m.mean()) for a, m in means.items()}, "contrasts": {}}
    draws = {}
    for _ in range(B):
        s = rng.choice(idx, len(idx), replace=True)
        for a, m in means.items():
            draws.setdefault(a, []).append(float(m[s].mean()))
    for a, b in CONTRASTS:
        if a not in means or b not in means:
            continue
        d = np.array(draws[a]) - np.array(draws[b])
        res["contrasts"]["%s_vs_%s" % (a, b)] = {
            "point_diff": float(means[a].mean() - means[b].mean()),
            "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_one_sided_a_le_b": float((d <= 0).mean()), "positive_rate": float((d > 0).mean())}
    (HERE / "phase2_bootstrap.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    lines = ["# DroneVehicle student contrasts, paired image-level bootstrap (%s)" % dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "",
             "B = %d, seed %d, shared test-image indices, arms summarized by their %s-seed mean per-image AP." % (B, SEED, res["n_seeds"]),
             "Official test split, %d images. Per-image AP is an AP proxy computed image by image, so absolute values" % len(stems),
             "differ from the Ultralytics mAP in PHASE2_RESULT.md; only the paired differences are interpreted.", "",
             "| contrast | diff (pp) | 95% CI (pp) | p one-sided | positive draws |", "|---|---:|---|---:|---:|"]
    for k, v in res["contrasts"].items():
        lines.append("| %s | %+.2f | [%+.2f, %+.2f] | %.4f | %.1f%% |" % (
            k.replace("_vs_", " vs "), 100 * v["point_diff"], 100 * v["diff_ci95"][0], 100 * v["diff_ci95"][1],
            v["p_one_sided_a_le_b"], 100 * v["positive_rate"]))
    (HERE / "phase2_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    log("bootstrap complete")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
