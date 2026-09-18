"""Where does the reliability score actually act? Stratify consensus pseudo-labels by teacher disagreement.

The paper's own correlation analysis says the structural terms "can matter only where teachers disagree; on this
benchmark they rarely do" (1,467 of 1,506 Anti-UAV boxes have full three-source support). This script tests that
directly on two label sets with very different disagreement rates:

  anti-uav   1,506 consensus boxes, 3 teachers, mostly full support
  dronevehicle-U  ~343k consensus boxes, 3 teachers trained on L only, far more disagreement

For each stratum it reports how well the score separates matched from unmatched pseudo-labels, for three scorings:
  conf_only   mean member confidence
  eq3         the paper's Eq. 3
  eq3_struct  the bracketed structural factor alone (no confidence), to isolate the agreement terms

and how much of the total training weight lands on unmatched boxes (the quantity the mechanism is supposed to cut).

CPU only. Outputs strata.json and strata.md in this folder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
C3 = Path(r"<PAPER_ROOT>/experiments/c3_dronevehicle")
sys.path.insert(0, str(C3))
import rescore_eq3 as R  # IoU helpers and label reader

AU_META = Path(r"<YOLO11_ROOT>/outputs/sota_runs/cross_model_wbf/fused/train_train_teacher_global/metadata.jsonl")
AU_GT = Path(r"<YOLO11_ROOT>/outputs/sota_runs/yolo11_rgbt_repro/dataset_dircopy/labels/train")
DV = Path(r"<YOLO11_ROOT>/outputs/sota_runs/dronevehicle_c3")
MATCH, UNMATCH = 0.5, 0.2


def iou_to_gt(box_xyxy, gt_xyxy):
    if len(gt_xyxy) == 0:
        return 0.0
    a = np.array(box_xyxy, dtype=float)
    b = np.array(gt_xyxy, dtype=float)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return float((inter / np.maximum(ua, 1e-12)).max())


def load_antiuav():
    rows = [json.loads(l) for l in AU_META.read_text(encoding="utf-8").splitlines() if l.strip()]
    gt = {}
    out = []
    for r in rows:
        if r["image"] not in gt:
            f = AU_GT / ("%s.txt" % r["image"])
            boxes = []
            if f.exists():
                for line in f.read_text(encoding="utf-8").splitlines():
                    p = line.split()
                    if len(p) >= 5:
                        cx, cy, w, h = map(float, p[1:5])
                        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            gt[r["image"]] = np.array(boxes, dtype=float).reshape(-1, 4)
        best = iou_to_gt(r["xyxy"], gt[r["image"]])
        sup, pri, loc, conf = r["model_fraction"], r["source_weight_fraction"], r["loc_consistency"], r["mean_conf"]
        struct = 0.45 * sup + 0.35 * pri + 0.20 * loc
        out.append({"support": r["model_count"], "loc": loc, "conf": conf,
                    "conf_only": conf, "eq3": conf * struct, "eq3_struct": struct, "best_iou": best})
    return out


def load_dronevehicle():
    import phase1_gate as G
    ck = json.loads((DV / "teacher_ckpts.json").read_text(encoding="utf-8"))
    t = json.loads((DV / "wbf_tuple.json").read_text(encoding="utf-8"))["tuple"]
    gt_dir = DV / "data" / "labels" / "U"
    stems = sorted(p.stem for p in gt_dir.glob("*.txt"))
    dets = {n: G.load_dets(DV / "preds" / "U" / n, n, stems) for n in ck}
    fused = G.fuse(dets, stems, t["iou"], t["mode"], t["min_models"], t["post_conf"])
    gt = {}
    out = []
    for f in fused:
        if f["image"] not in gt:
            lf = gt_dir / ("%s.txt" % f["image"])
            per_cls = {}
            for c, b, _ in R.read(lf, False):
                per_cls.setdefault(c, []).append(R.to_xyxy(b))
            gt[f["image"]] = {k: np.array(v, dtype=float).reshape(-1, 4) for k, v in per_cls.items()}
        best = iou_to_gt(R.to_xyxy(f["box"]), gt[f["image"]].get(f["cls"], np.zeros((0, 4))))
        out.append({"support": f["n_sources"], "loc": f["loc"], "conf": f["conf_only"],
                    "conf_only": f["conf_only"], "eq3": f["eq3"], "eq3_struct": f["struct"], "best_iou": best})
    return out, fused


def summarize(items, label):
    keep = [x for x in items if x["best_iou"] >= MATCH or x["best_iou"] < UNMATCH]
    y = np.array([1 if x["best_iou"] >= MATCH else 0 for x in keep])
    res = {"label": label, "n_total": len(items), "n_used": len(keep),
           "n_matched": int(y.sum()), "n_unmatched": int((1 - y).sum())}
    if 0 < y.sum() < len(y):
        for s in ("conf_only", "eq3", "eq3_struct"):
            v = np.array([x[s] for x in keep], dtype=float)
            res["auroc_" + s] = float(roc_auc_score(y, v))
            w = v / max(v.sum(), 1e-12)
            res["noise_weight_share_" + s] = float(w[y == 0].sum())
        res["noise_weight_share_uniform"] = float((1 - y).sum() / len(y))
        res["auroc_gain_struct_over_conf"] = res["auroc_eq3"] - res["auroc_conf_only"]
    return res


def main() -> None:
    report = {}

    au = load_antiuav()
    sup_counts = {}
    for x in au:
        sup_counts[x["support"]] = sup_counts.get(x["support"], 0) + 1
    print("anti-uav support distribution:", sup_counts)
    strata = [("all", au), ("full support (3 sources)", [x for x in au if x["support"] == 3]),
              ("partial support (2 sources)", [x for x in au if x["support"] == 2])]
    locs = sorted(x["loc"] for x in au)
    q1 = locs[len(locs) // 4]
    strata.append(("lowest-quartile localization agreement", [x for x in au if x["loc"] <= q1]))
    report["anti_uav"] = [summarize(v, k) for k, v in strata if v]

    dv, fused = load_dronevehicle()
    print("dronevehicle-U fused boxes:", len(dv))
    dv_sup = {}
    for x in dv:
        dv_sup[x["support"]] = dv_sup.get(x["support"], 0) + 1
    print("dronevehicle-U support distribution:", dv_sup)
    report["dronevehicle_U"] = [summarize(dv, "all"),
                                summarize([x for x in dv if x["support"] == 3], "full support (3 sources)"),
                                summarize([x for x in dv if x["support"] == 2], "partial support (2 sources)")]
    locs = sorted(x["loc"] for x in dv)
    q1 = locs[len(locs) // 4]
    report["dronevehicle_U"].append(summarize([x for x in dv if x["loc"] <= q1], "lowest-quartile localization agreement"))
    report["dronevehicle_U"].append(summarize([x for x in dv if x["loc"] > q1], "upper three quartiles"))

    (HERE / "strata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Where the reliability score acts: disagreement strata", "",
             "Matched = IoU >= %.1f with a human training box, unmatched = IoU < %.1f; ambiguous boxes excluded." % (MATCH, UNMATCH),
             "AUROC separates matched from unmatched. `eq3_struct` is the bracketed agreement factor with confidence removed,",
             "so it isolates what the structural terms contribute on their own.", ""]
    for ds, rows in report.items():
        lines += ["## %s" % ds, "",
                  "| stratum | boxes | matched | unmatched | AUROC conf-only | AUROC Eq.3 | AUROC structure alone | noise weight: uniform -> Eq.3 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            if "auroc_eq3" not in r:
                lines.append("| %s | %d | %d | %d | -- | -- | -- | -- |" % (r["label"], r["n_used"], r["n_matched"], r["n_unmatched"]))
                continue
            lines.append("| %s | %d | %d | %d | %.4f | %.4f | %.4f | %.3f -> %.3f |" % (
                r["label"], r["n_used"], r["n_matched"], r["n_unmatched"], r["auroc_conf_only"], r["auroc_eq3"],
                r["auroc_eq3_struct"], r["noise_weight_share_uniform"], r["noise_weight_share_eq3"]))
        lines.append("")
    (HERE / "strata.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
