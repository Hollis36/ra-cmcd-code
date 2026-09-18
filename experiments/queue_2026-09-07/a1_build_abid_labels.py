"""A1: recompute the 1,506 training consensus clusters from the cached teacher predictions with the
training-label WBF tuple, attach the per-cluster maximum member confidence, and write Abid et al.
(BMVC 2025) formula-matched weights as labels_scored for two student datasets:
  abid_soft     : all boxes, weight = min(1, (s*)^2)
  abid_soft_t04 : boxes with s* < 0.4 removed, same weights
with s* = max_conf * exp(-5 * (1 - C_loc)) * (1 + 0.1 * (models - 2)).

The recomputed boxes must reproduce metadata.jsonl of train_train_teacher_global (asserted), so the
comparison stays on the identical clusters. Datasets mirror cmcd_dataset_reloff (junctions for images).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"<YOLO11_ROOT>/scripts/sota")
import cross_model_wbf as W  # noqa: E402

ROOT = Path(r"<YOLO11_ROOT>/outputs/sota_runs")
FUSED = ROOT / "cross_model_wbf" / "fused" / "train_train_teacher_global"
PREDS = ROOT / "cross_model_wbf" / "preds" / "train"
REF = ROOT / "cmcd_dataset_reloff"
SOURCES = [  # name, pred dir, format, prior (training-label configuration)
    ("yolo11_dirdata", "yolo11_dirdata_best_img704_conf0.001_iou0.6", "yolo", 1.0),
    ("c2former", "c2former_best_img768_conf0.001_iou0.4", "icafusion", 0.65),
    ("cft", "cft_best_img640_conf0.001_iou0.4", "yolo", 0.45),
]
IOU_THR, MIN_MODELS, POST_CONF = 0.30, 2, 0.07
ALPHA, BETA, P = 5.0, 0.1, 2.0


def recluster():
    stems = W.image_stems("train")
    sizes = W.image_sizes("train", stems)
    per_source = [W.parse_pred_dir(PREDS / d / "labels", stems, fmt, sizes) for _, d, fmt, _ in SOURCES]
    weights = [w for *_, w in SOURCES]
    total_weight = sum(weights)
    out = {}
    for idx, stem in enumerate(stems):
        rows = []
        for si, src in enumerate(per_source):
            if len(src[idx]):
                rows.append(np.concatenate([src[idx], np.full((len(src[idx]), 1), si, dtype=np.float32)], axis=1))
        if not rows:
            continue
        dets = np.concatenate(rows, axis=0)
        dets = dets[np.argsort(-dets[:, 4])]
        used = np.zeros(len(dets), dtype=bool)
        boxes = []
        for i in range(len(dets)):
            if used[i]:
                continue
            ious = W.box_iou(dets[i:i + 1, :4], dets[:, :4]).reshape(-1)
            mask = (ious >= IOU_THR) & (~used); mask[i] = True
            cluster = dets[mask]; used[mask] = True
            source_ids = sorted(set(cluster[:, 6].astype(int).tolist()))
            if len(source_ids) < MIN_MODELS:
                continue
            wts = cluster[:, 4] * np.asarray([weights[int(s)] for s in cluster[:, 6]], dtype=np.float32)
            box = (cluster[:, :4] * wts[:, None]).sum(axis=0) / max(float(wts.sum()), 1e-9)
            score = float(wts.sum() / total_weight)
            if score < POST_CONF:
                continue
            loc = float(W.box_iou(cluster[:, :4], box[None, :]).reshape(-1).mean())
            mean_conf = float(cluster[:, 4].mean()); max_conf = float(cluster[:, 4].max())
            models = len(source_ids)
            rel = mean_conf * (0.45 * models / 3 + 0.35 * sum(weights[s] for s in source_ids) / total_weight + 0.20 * loc)
            spread = 1.0 - loc
            cf = math.exp(-ALPHA * spread) * (1.0 + BETA * (models - 2))
            s_star = max_conf * cf
            boxes.append({"xyxy": box.tolist(), "score": score, "mean_conf": mean_conf, "max_conf": max_conf, "models": models,
                          "loc": loc, "reliability": rel, "s_star": s_star, "abid_weight": min(1.0, s_star ** P)})
        out[stem] = boxes
    return out


def verify(out):
    meta = [json.loads(l) for l in (FUSED / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = sum(len(v) for v in out.values())
    assert n == len(meta), f"box count {n} != {len(meta)}"
    by_img = {}
    for m in meta:
        by_img.setdefault(m["image"], []).append(m)
    mism = 0
    for stem, boxes in out.items():
        ref = by_img.get(stem, [])
        assert len(ref) == len(boxes), stem
        for b in boxes:
            best = min(ref, key=lambda m: sum(abs(a - c) for a, c in zip(m["xyxy"], b["xyxy"])))
            if max(abs(a - c) for a, c in zip(best["xyxy"], b["xyxy"])) > 1e-4 or abs(best["reliability"] - b["reliability"]) > 1e-4:
                mism += 1
    print(f"verify: {n} boxes; mismatches {mism}")
    assert mism == 0, "recomputed clusters do not reproduce the released label set"


def mirror(dst: Path, src: Path):
    """Hard-link (fallback: copy) every file of src into dst. Junctions must NOT be used: Ultralytics resolves
    them to the target folder and then reads the target's labels (found 2026-09-08)."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not (dst / f.name).exists():
            try:
                os.link(f, dst / f.name)
            except OSError:
                shutil.copy2(f, dst / f.name)


def build_dataset(name: str, out, threshold: float | None):
    d = ROOT / f"cmcd_dataset_{name}"
    if d.exists():
        shutil.rmtree(d)
    for sub in ("images", "images_ir"):
        for split in ("train", "val"):
            mirror(d / sub / split, REF / sub / split)
    for sub in ("labels", "labels_ir", "labels_scored"):
        (d / sub / "train").mkdir(parents=True, exist_ok=True)
        shutil.copytree(REF / sub / "val", d / sub / "val", dirs_exist_ok=True)
    kept = total = 0
    stems = [p.stem for p in (REF / "labels" / "train").glob("*.txt")]
    for stem in stems:
        boxes = out.get(stem, [])
        lines_lab, lines_sc = [], []
        for b in boxes:
            total += 1
            if threshold is not None and b["s_star"] < threshold:
                continue
            kept += 1
            x1, y1, x2, y2 = b["xyxy"]; cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            lines_lab.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            lines_sc.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {b['score']:.6f} {b['abid_weight']:.6f} {b['models']} {b['models']}")
        for sub, lines in (("labels", lines_lab), ("labels_ir", lines_lab), ("labels_scored", lines_sc)):
            (d / sub / "train" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (d / "anti_uav_cmcd_student.yaml").write_text(f"# generated by a1_build_abid_labels.py\npath: {d.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [drone]\n", encoding="utf-8")
    ws = [b["abid_weight"] for boxes in out.values() for b in boxes if threshold is None or b["s_star"] >= threshold]
    (d / "manifest.json").write_text(json.dumps({"dataset": name, "boxes_total": total, "boxes_kept": kept, "threshold_on_s_star": threshold,
                                                 "weight": "min(1,(s*)^2)", "alpha": ALPHA, "beta": BETA, "p": P,
                                                 "weight_stats": {"min": min(ws), "mean": sum(ws) / len(ws), "max": max(ws)}}, indent=2), encoding="utf-8")
    print(f"{name}: kept {kept}/{total} boxes; weight mean {sum(ws)/len(ws):.3f}; yaml {d / 'anti_uav_cmcd_student.yaml'}")


if __name__ == "__main__":
    os.chdir(r"<YOLO11_ROOT>")
    clusters = recluster()
    verify(clusters)
    build_dataset("abid_soft", clusters, None)
    build_dataset("abid_soft_t04", clusters, 0.4)
