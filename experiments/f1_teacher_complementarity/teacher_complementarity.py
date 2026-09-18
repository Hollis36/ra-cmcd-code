"""F1: teacher-error complementarity from cached teacher predictions (no GPU).

Replaces the CBEG/CMCI motivation of the submitted version with a direct measurement on the
teacher pool: for every human box on a split, the best IoU each teacher achieves, the oracle
best-teacher-per-box IoU, how often the best teacher changes, the correlation of per-box IoU
between teachers, the same for the WBF consensus box, and the picture by target size.

Inputs (same caches and parsers as cross_model_wbf.py):
  <YOLO11_ROOT>/outputs/sota_runs/cross_model_wbf/preds/<split>/<source>/labels/<stem>.txt
  <YOLO11_ROOT>/outputs/sota_runs/yolo11_rgbt_repro/dataset_dircopy/{labels,images}/<folder>
  <YOLO11_ROOT>/outputs/sota_runs/cross_model_wbf/fused/<split>_<tag>/metadata.jsonl (consensus boxes)
Outputs: complementarity_<split>.json and complementarity_report.md in this folder.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics as st
from pathlib import Path

from PIL import Image

ROOT = Path(r"<YOLO11_ROOT>/outputs/sota_runs")
PREDS = ROOT / "cross_model_wbf" / "preds"
DATA = ROOT / "yolo11_rgbt_repro" / "dataset_dircopy"
HERE = Path(__file__).resolve().parent
CONF_FLOOR = 0.10  # detections below this confidence are ignored when taking a teacher's best box

SPLITS = {
    "val": {"folder": "val", "fused": "val_val_teacher_global", "sources": {
        "YOLO11-RGBT": ("yolo11_dirdata_best_img704_conf0.001_iou0.6", "yolo"),
        "C2Former": ("c2former_best_img768_conf0.001_iou0.4", "icafusion"),
        "CFT": ("cft_best_img640_conf0.001_iou0.4", "yolo"),
        "ICAFusion": ("ica_best_img640_conf0.001_iou0.4", "icafusion"),
    }},
    "night160": {"folder": "night_160", "fused": None, "sources": {}},  # filled at runtime from the cache listing
}
SIZE_STRATA = {"small (<28 px)": (0, 28), "medium (28-39 px)": (28, 39), "large (>=39 px)": (39, 10 ** 9)}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def xywh2xyxy(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def parse_pred_file(path: Path, fmt: str, size):
    rows = []
    if not path.exists():
        return rows
    W, H = size
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [float(x) for x in re.split(r"[\s,]+", line.strip()) if x]
        if len(parts) < 6:
            continue
        if fmt == "icafusion":
            _, x, y, w, h, conf = parts[:6]; box = [x, y, x + w, y + h]
            if max(abs(v) for v in box) > 2.0:
                box = [box[0] / W, box[1] / H, box[2] / W, box[3] / H]
        else:
            _, cx, cy, w, h, conf = parts[:6]; box = xywh2xyxy(cx, cy, w, h)
        box = [min(max(v, 0.0), 1.0) for v in box]
        if box[2] > box[0] and box[3] > box[1]:
            rows.append((box, conf))
    return rows


def discover_sources(split: str):
    d = PREDS / split
    found = {}
    for p in sorted(d.iterdir()):
        n = p.name
        if n.startswith("racmcd") or "valfixed" in n:
            continue
        if n.startswith("yolo11"):
            found["YOLO11-RGBT"] = (n, "yolo")
        elif n.startswith("c2former"):
            found["C2Former"] = (n, "icafusion")
        elif n.startswith("cft"):
            found["CFT"] = (n, "yolo")
        elif n.startswith("ica"):
            found["ICAFusion"] = (n, "icafusion")
    return found


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b)); den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def analyse(split: str) -> dict:
    cfg = SPLITS[split]
    sources = cfg["sources"] or discover_sources(split)
    folder = cfg["folder"]
    label_dir = DATA / "labels" / folder; image_dir = DATA / "images" / folder
    stems = sorted(p.stem for p in label_dir.glob("*.txt"))
    sizes = {}
    for s in stems:
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            ip = image_dir / f"{s}{ext}"
            if ip.exists():
                with Image.open(ip) as im:
                    sizes[s] = im.size
                break
    gts = []  # (stem, box, side_px)
    for s in stems:
        W, H = sizes.get(s, (640, 512))
        for line in (label_dir / f"{s}.txt").read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            b = xywh2xyxy(*map(float, p[1:5])); gts.append((s, b, max((b[2] - b[0]) * W, (b[3] - b[1]) * H)))
    per_teacher = {}
    for name, (dirname, fmt) in sources.items():
        cache = {}
        vals = []
        for s, b, _ in gts:
            if s not in cache:
                cache[s] = [r for r in parse_pred_file(PREDS / split / dirname / "labels" / f"{s}.txt", fmt, sizes.get(s, (640, 512))) if r[1] >= CONF_FLOOR]
            vals.append(max((iou(b, r[0]) for r in cache[s]), default=0.0))
        per_teacher[name] = vals
    fused_vals = None
    if cfg["fused"]:
        meta = ROOT / "cross_model_wbf" / "fused" / cfg["fused"] / "metadata.jsonl"
        if meta.exists():
            by_img = {}
            for line in meta.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line); by_img.setdefault(r["image"], []).append(xywh2xyxy(*r["xywh"]))
            fused_vals = [max((iou(b, fb) for fb in by_img.get(s, [])), default=0.0) for s, b, _ in gts]
    names = list(per_teacher)
    n = len(gts)
    oracle = [max(per_teacher[t][i] for t in names) for i in range(n)]
    best_teacher = max(names, key=lambda t: st.mean(per_teacher[t]))
    argmax = [max(names, key=lambda t: per_teacher[t][i]) for i in range(n)]
    out = {
        "split": split, "n_gt": n, "conf_floor": CONF_FLOOR, "sources": {k: v[0] for k, v in sources.items()},
        "mean_best_iou": {t: st.mean(per_teacher[t]) for t in names},
        "median_best_iou": {t: st.median(per_teacher[t]) for t in names},
        "frac_iou_ge_0.75": {t: sum(v >= 0.75 for v in per_teacher[t]) / n for t in names},
        "frac_iou_ge_0.90": {t: sum(v >= 0.90 for v in per_teacher[t]) / n for t in names},
        "oracle_mean_best_iou": st.mean(oracle),
        "oracle_frac_iou_ge_0.75": sum(v >= 0.75 for v in oracle) / n,
        "oracle_frac_iou_ge_0.90": sum(v >= 0.90 for v in oracle) / n,
        "best_single_teacher": best_teacher,
        "frac_boxes_where_best_teacher_is_not_the_best_single": sum(a != best_teacher for a in argmax) / n,
        "argmax_share": {t: sum(a == t for a in argmax) / n for t in names},
        "pairwise_iou_correlation": {f"{a}|{b}": corr(per_teacher[a], per_teacher[b]) for i, a in enumerate(names) for b in names[i + 1:]},
        "fused_consensus_mean_best_iou": st.mean(fused_vals) if fused_vals else None,
        "fused_consensus_frac_iou_ge_0.75": (sum(v >= 0.75 for v in fused_vals) / n) if fused_vals else None,
        "fused_consensus_frac_iou_ge_0.90": (sum(v >= 0.90 for v in fused_vals) / n) if fused_vals else None,
        "by_size": {},
    }
    for label, (lo, hi) in SIZE_STRATA.items():
        idx = [i for i, (_, _, side) in enumerate(gts) if lo <= side < hi]
        if not idx:
            continue
        out["by_size"][label] = {
            "n": len(idx),
            "mean_best_iou": {t: st.mean(per_teacher[t][i] for i in idx) for t in names},
            "oracle_mean_best_iou": st.mean(oracle[i] for i in idx),
            "frac_best_teacher_changes": sum(argmax[i] != best_teacher for i in idx) / len(idx),
            "fused_mean_best_iou": (st.mean(fused_vals[i] for i in idx) if fused_vals else None),
        }
    return out


def fmt_report(results: list[dict]) -> str:
    lines = ["# Teacher-error complementarity (F1), computed " + __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"), "",
             f"Best IoU per human box per teacher (detections with confidence >= {CONF_FLOOR}); oracle = best teacher chosen per box; consensus = WBF training-config fused boxes where cached.", ""]
    for r in results:
        names = list(r["mean_best_iou"])
        lines += [f"## {r['split']} ({r['n_gt']} human boxes)", "", "| quantity | " + " | ".join(names) + " | oracle | WBF consensus |", "|---|" + "---|" * (len(names) + 2)]
        def row(label, key, okey, fkey, pct=False):
            f = (lambda v: f"{100*v:.1f}%" if pct else f"{v:.3f}")
            return f"| {label} | " + " | ".join(f(r[key][t]) for t in names) + f" | {f(r[okey])} | " + (f(r[fkey]) if r.get(fkey) is not None else "-") + " |"
        lines.append(row("mean best IoU", "mean_best_iou", "oracle_mean_best_iou", "fused_consensus_mean_best_iou"))
        lines.append(row("boxes with IoU >= 0.75", "frac_iou_ge_0.75", "oracle_frac_iou_ge_0.75", "fused_consensus_frac_iou_ge_0.75", pct=True))
        lines.append(row("boxes with IoU >= 0.90", "frac_iou_ge_0.90", "oracle_frac_iou_ge_0.90", "fused_consensus_frac_iou_ge_0.90", pct=True))
        lines += ["", f"Best single teacher: {r['best_single_teacher']}. Boxes where another teacher is tighter: {100*r['frac_boxes_where_best_teacher_is_not_the_best_single']:.1f}%. Share of boxes each teacher is tightest on: " + ", ".join(f"{t} {100*v:.1f}%" for t, v in r["argmax_share"].items()) + ".", "",
                  "Pairwise correlation of per-box best IoU: " + ", ".join(f"{k} {v:.2f}" for k, v in r["pairwise_iou_correlation"].items()) + ".", "",
                  "| size stratum | n | " + " | ".join(names) + " | oracle | best teacher changes | WBF consensus |", "|---|---|" + "---|" * (len(names) + 3)]
        for label, b in r["by_size"].items():
            lines.append(f"| {label} | {b['n']} | " + " | ".join(f"{b['mean_best_iou'][t]:.3f}" for t in names) + f" | {b['oracle_mean_best_iou']:.3f} | {100*b['frac_best_teacher_changes']:.1f}% | " + (f"{b['fused_mean_best_iou']:.3f}" if b.get("fused_mean_best_iou") is not None else "-") + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = []
    for split in ("val", "night160"):
        if not (PREDS / split).exists():
            print("no cache for", split); continue
        r = analyse(split); results.append(r)
        (HERE / f"complementarity_{split}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(split, "done; mean best IoU:", {k: round(v, 3) for k, v in r["mean_best_iou"].items()}, "oracle", round(r["oracle_mean_best_iou"], 3))
    (HERE / "complementarity_report.md").write_text(fmt_report(results), encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
