"""Paired image-level bootstrap on GLOBAL AP for the DroneVehicle student contrasts.

Why this replaces phase2_bootstrap.py: that script used a per-image AP proxy, whose precision is computed within
each image, so it penalises detection count rather than detection quality. On DroneVehicle (about 15 objects per
image, five classes, confidence floor 0.001) it disagreed with the Ultralytics metric and even flipped the sign of
the plain-versus-GT-only contrast. Its output must not be used.

Correct construction: greedy matching is performed inside each image, so whether a detection is a true positive at
a given IoU threshold does not depend on which images are resampled. Only the pooled ranking and the ground-truth
count change. So the true-positive flags are precomputed once per student, and each bootstrap draw just gathers the
sampled images' detections, sorts by confidence, and integrates precision-recall exactly as global AP does.

Outputs phase2_bootstrap_global.json / .md.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(r"<YOLO11_ROOT>/outputs/sota_runs/dronevehicle_c3")
FORK = Path(r"<FORK_ROOT>")
LOG = HERE / "phase2_progress.log"
CLASSES = ["car", "truck", "bus", "van", "freight_car"]
IOUS = np.arange(0.50, 0.951, 0.05)
ARMS = ["plain", "weighted", "gt_only", "eq3"]
SEEDS = [0, 1, 2, 3]
B = 2000
SEED = 20260913
CONTRASTS = [("weighted", "plain"), ("weighted", "gt_only"), ("plain", "gt_only"), ("eq3", "plain"), ("eq3", "weighted")]


def log(msg: str) -> None:
    line = "[%s] P2-GBOOT %s" % (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def iou_mat(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)), dtype=np.float32)
    dx1, dy1 = d[:, 0:1] - d[:, 2:3] / 2, d[:, 1:2] - d[:, 3:4] / 2
    dx2, dy2 = d[:, 0:1] + d[:, 2:3] / 2, d[:, 1:2] + d[:, 3:4] / 2
    gx1, gy1 = g[:, 0] - g[:, 2] / 2, g[:, 1] - g[:, 3] / 2
    gx2, gy2 = g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2
    iw = np.clip(np.minimum(dx2, gx2) - np.maximum(dx1, gx1), 0, None)
    ih = np.clip(np.minimum(dy2, gy2) - np.maximum(dy1, gy1), 0, None)
    inter = iw * ih
    return (inter / np.maximum(d[:, 2:3] * d[:, 3:4] + g[:, 2] * g[:, 3] - inter, 1e-12)).astype(np.float32)


def read_labels(p: Path, with_conf: bool):
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            q = line.split()
            if len(q) >= 5:
                rows.append((int(float(q[0])), float(q[1]), float(q[2]), float(q[3]), float(q[4]),
                             float(q[5]) if with_conf and len(q) > 5 else 1.0))
    return rows


def export(arm: str, seed: int) -> Path:
    """Cache test-split detections for one student at its frozen setting (self-contained since 2026-09-17:
    this script must not depend on the retired per-image-proxy script for its inputs)."""
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
    log("  %s: %d test images exported" % (key, n))
    return dst


def build(arm: str, seed: int, stems, gts):
    """Precompute per-detection (image index, class, confidence, TP flag at each IoU) for one student."""
    key = "%s_s%d" % (arm, seed)
    cache = OUT / "gboot" / ("%s.npz" % key)
    if cache.exists():
        z = np.load(cache)
        return z["img"], z["cls"], z["conf"], z["tp"]
    lab = OUT / "test_preds" / key / "labels"
    img_i, cls_i, conf_i, tp_i = [], [], [], []
    for i, s in enumerate(stems):
        dets = read_labels(lab / ("%s.txt" % s), True)
        if not dets:
            continue
        gt = gts[s]
        for c in range(len(CLASSES)):
            dc = [d for d in dets if d[0] == c]
            if not dc:
                continue
            dc.sort(key=lambda r: -r[5])
            db = np.array([r[1:5] for r in dc], dtype=np.float32)
            gb = np.array([r[1:5] for r in gt if r[0] == c], dtype=np.float32).reshape(-1, 4)
            M = iou_mat(db, gb)
            flags = np.zeros((len(dc), len(IOUS)), dtype=np.int8)
            for t, thr in enumerate(IOUS):
                used = np.zeros(len(gb), dtype=bool)
                for k in range(len(dc)):
                    if not len(gb):
                        break
                    order = np.argsort(-M[k])
                    for j in order:
                        if M[k, j] < thr:
                            break
                        if not used[j]:
                            used[j] = True
                            flags[k, t] = 1
                            break
            img_i.append(np.full(len(dc), i, dtype=np.int32))
            cls_i.append(np.full(len(dc), c, dtype=np.int8))
            conf_i.append(np.array([r[5] for r in dc], dtype=np.float32))
            tp_i.append(flags)
    img = np.concatenate(img_i) if img_i else np.zeros(0, np.int32)
    cls = np.concatenate(cls_i) if cls_i else np.zeros(0, np.int8)
    conf = np.concatenate(conf_i) if conf_i else np.zeros(0, np.float32)
    tp = np.concatenate(tp_i) if tp_i else np.zeros((0, len(IOUS)), np.int8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, img=img, cls=cls, conf=conf, tp=tp)
    log("  %s: %d detections cached" % (key, len(conf)))
    return img, cls, conf, tp


def ap_for_sample(img, cls, conf, tp, counts, gt_per_img_cls):
    """Global AP (mean over classes and IoU thresholds) for one resample given per-image multiplicities."""
    mult = counts[img]
    keep = mult > 0
    if not keep.any():
        return 0.0
    i2, c2, f2, t2 = img[keep], cls[keep], conf[keep], tp[keep]
    m2 = mult[keep]
    aps = []
    for c in range(len(CLASSES)):
        sel = c2 == c
        n_gt = float((gt_per_img_cls[:, c] * counts).sum())
        if n_gt == 0:
            continue
        if not sel.any():
            aps.append(0.0)
            continue
        order = np.argsort(-f2[sel])
        w = m2[sel][order].astype(np.float32)
        flags = t2[sel][order]
        per_thr = []
        for t in range(len(IOUS)):
            ctp = np.cumsum(flags[:, t] * w)
            call = np.cumsum(w)
            rec = ctp / n_gt
            prec = ctp / np.maximum(call, 1e-12)
            mrec = np.concatenate(([0.0], rec, [rec[-1]]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            for k in range(len(mpre) - 2, -1, -1):
                mpre[k] = max(mpre[k], mpre[k + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            per_thr.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
        aps.append(float(np.mean(per_thr)))
    return float(np.mean(aps)) if aps else 0.0


def main() -> None:
    gt_dir = OUT / "eval" / "labels" / "test"
    stems = sorted(p.stem for p in gt_dir.glob("*.txt"))
    gts = {s: read_labels(gt_dir / ("%s.txt" % s), False) for s in stems}
    gt_per_img_cls = np.zeros((len(stems), len(CLASSES)), dtype=np.float32)
    for i, s in enumerate(stems):
        for r in gts[s]:
            gt_per_img_cls[i, r[0]] += 1
    log("test split: %d images, %d boxes" % (len(stems), int(gt_per_img_cls.sum())))

    data = {}
    for arm in ARMS:
        for seed in SEEDS:
            key = "%s_s%d" % (arm, seed)
            # every arm and seed must be present: a silently skipped student would bias its arm's mean
            assert (OUT / "students" / key / "weights" / "best.pt").exists(), "missing checkpoint for %s" % key
            export(arm, seed)
            data[(arm, seed)] = build(arm, seed, stems, gts)

    # point estimates, for a sanity check against the Ultralytics numbers
    ones = np.ones(len(stems), dtype=np.float32)
    point = {}
    for arm in ARMS:
        vals = [ap_for_sample(*data[(arm, s)], ones, gt_per_img_cls) for s in SEEDS if (arm, s) in data]
        point[arm] = float(np.mean(vals))
        log("  %-9s global AP (four-seed mean) %.4f" % (arm, point[arm]))

    rng = np.random.default_rng(SEED)
    draws = {a: [] for a in ARMS}
    for b in range(B):
        counts = np.bincount(rng.integers(0, len(stems), len(stems)), minlength=len(stems)).astype(np.float32)
        for arm in ARMS:
            vals = [ap_for_sample(*data[(arm, s)], counts, gt_per_img_cls) for s in SEEDS if (arm, s) in data]
            draws[arm].append(float(np.mean(vals)))
        if (b + 1) % 200 == 0:
            log("  %d/%d draws" % (b + 1, B))

    res = {"B": B, "seed": SEED, "n_images": len(stems), "point_global_ap": point, "contrasts": {}}
    for a, bb in CONTRASTS:
        d = np.array(draws[a]) - np.array(draws[bb])
        res["contrasts"]["%s_vs_%s" % (a, bb)] = {
            "point_diff": point[a] - point[bb],
            "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_one_sided_a_le_b": float((d <= 0).mean()), "positive_rate": float((d > 0).mean())}
    (HERE / "phase2_bootstrap_global.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    lines = ["# DroneVehicle student contrasts, paired image-level bootstrap on global AP (%s)" % dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "",
             "B = %d, seed %d, shared test-image indices, each arm summarized by its four-seed mean global AP@0.5:0.95." % (B, SEED),
             "Official test split, %d images. True-positive flags are precomputed per image (greedy matching is" % len(stems),
             "within-image, so resampling changes only the pooled ranking and the ground-truth count).", "",
             "Four-seed mean global AP@0.5:0.95: " + ", ".join("%s %.4f" % (a, point[a]) for a in ARMS), "",
             "| contrast | diff (pp) | 95% CI (pp) | p one-sided | positive draws |", "|---|---:|---|---:|---:|"]
    for k, v in res["contrasts"].items():
        lines.append("| %s | %+.2f | [%+.2f, %+.2f] | %.4f | %.1f%% |" % (
            k.replace("_vs_", " vs "), 100 * v["point_diff"], 100 * v["diff_ci95"][0], 100 * v["diff_ci95"][1],
            v["p_one_sided_a_le_b"], 100 * v["positive_rate"]))
    (HERE / "phase2_bootstrap_global.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    log("global bootstrap complete")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
