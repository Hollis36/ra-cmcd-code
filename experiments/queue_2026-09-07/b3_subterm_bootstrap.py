"""B3: four-seed mean-over-seeds paired image-level bootstrap for the leave-one-out contrasts.

Regimes: full RA-CMCD (s0-3), confidence-only, drop-support, drop-prior, drop-localization (s0-3 each),
Soft-Teacher-style (s0-7). Contrasts: full minus each variant, on night-78 and night-160, same
resampling scheme as the paper (B = 10,000, seed 20260515, shared indices, one-sided H0: delta <= 0),
plus Holm-adjusted p-values across the five contrasts per split and metric.

Phases: --phase export (GPU: per-seed prediction export at the frozen tuple, cached),
        --phase bootstrap (CPU only), --phase all.
Reuses the paper's helpers from the execution shim.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"<YOLO11_ROOT>/scripts/sota")
import numpy as np  # noqa: E402

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

import multiseed_ablation_bootstrap as mab  # noqa: E402
from paired_bootstrap_racmcd import ModelEvalSpec, collect_predictions  # noqa: E402
from cross_model_wbf import DEFAULT_YOLO_ROOT, image_sizes, image_stems, load_ground_truth  # noqa: E402

RUNS = Path(r"<YOLO11_ROOT>/outputs/sota_runs/cmcd_student")
HERE = Path(__file__).resolve().parent
LOG = HERE / "queue_progress.log"
ARMS = {
    "full": ("late_cmcd_pseudo_relw_all", "val_fixed_late_cmcd_pseudo_relw_all_s{s}.json", range(4), "racmcd_relw_all_s{s}_ms"),
    "conf_only": ("late_cmcd_pseudo_abl_conf_only", "val_fixed_abl_conf_only_s{s}.json", range(4), "b3_conf_only_s{s}_ms"),
    "drop_support": ("late_cmcd_pseudo_abl_drop_support", "val_fixed_abl_drop_support_s{s}.json", range(4), "b3_drop_support_s{s}_ms"),
    "drop_prior": ("late_cmcd_pseudo_abl_drop_prior", "val_fixed_abl_drop_prior_s{s}.json", range(4), "b3_drop_prior_s{s}_ms"),
    "drop_iou": ("late_cmcd_pseudo_abl_drop_iou", "val_fixed_abl_drop_iou_s{s}.json", range(4), "b3_drop_iou_s{s}_ms"),
    "soft_v2": ("late_cmcd_pseudo_soft_v2", "val_fixed_late_cmcd_pseudo_soft_v2_s{s}.json", range(4), "b3_soft_v2_s{s}_ms"),
    "abid_soft": ("late_cmcd_pseudo_abid_soft", "val_fixed_late_cmcd_pseudo_abid_soft_s{s}.json", range(4), "b3_abid_soft_s{s}_ms"),
}
# Holm families follow the tables: the four leave-one-out contrasts (sub-term table) and the two Table III baselines.
FAMILIES = {
    "subterm": [("full", "conf_only"), ("full", "drop_support"), ("full", "drop_prior"), ("full", "drop_iou")],
    "baselines": [("full", "soft_v2"), ("full", "abid_soft")],
}
CONTRASTS = [c for pairs in FAMILIES.values() for c in pairs]
SPLITS = ["night78", "night160"]


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] B3 {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def specs() -> dict[str, dict[str, ModelEvalSpec]]:
    out = {}
    for regime, (run_prefix, vf_pat, seeds, name_pat) in ARMS.items():
        out[regime] = {}
        for s in seeds:
            vf = RUNS / vf_pat.format(s=s)
            d = json.loads(vf.read_text(encoding="utf-8"))["best_val"]["spec"]
            out[regime][f"s{s}"] = ModelEvalSpec(name=name_pat.format(s=s), ckpt=RUNS / "runs" / f"{run_prefix}_s{s}" / "weights" / "best.pt",
                                                 imgsz=int(d["imgsz"]), conf=float(d["conf"]), iou=float(d["iou"]))
    return out


def collect(split: str, sp, force: bool):
    stems = image_stems(split); sizes = image_sizes(split, stems)
    gt_boxes, gt_classes = load_ground_truth(split, stems)
    regime_dets = {}
    for regime, seeds in sp.items():
        regime_dets[regime] = {}
        for seed, spec in seeds.items():
            if not spec.ckpt.exists():
                raise FileNotFoundError(spec.ckpt)
            regime_dets[regime][seed] = collect_predictions(spec, split, stems, sizes, force_export=force, yolo_root=DEFAULT_YOLO_ROOT)
        log(f"{split}: {regime} predictions ready ({len(seeds)} seeds)")
    return regime_dets, gt_boxes, gt_classes


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj = {}; running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p)); adj[k] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["export", "bootstrap", "all"], default="all")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260515)
    ap.add_argument("--force-export", action="store_true")
    args = ap.parse_args()
    os.chdir(r"<YOLO11_ROOT>")
    mab.CONTRASTS = CONTRASTS
    sp = specs()
    results = {"n_bootstrap": args.n_bootstrap, "seed": args.seed, "contrasts_note": "one-sided H0: full <= variant; Holm within family per split and metric (subterm: four leave-one-out contrasts; baselines: soft_v2 and abid_soft)",
               "families": {k: [f"{a}_vs_{b}" for a, b in v] for k, v in FAMILIES.items()}, "splits": {}}
    for split in SPLITS:
        regime_dets, gt_boxes, gt_classes = collect(split, sp, args.force_export)
        if args.phase == "export":
            continue
        log(f"{split}: bootstrap start (B={args.n_bootstrap})")
        res = mab.bootstrap_ablation(regime_dets, gt_boxes, gt_classes, n_bootstrap=args.n_bootstrap, seed=args.seed)
        for m in mab.METRICS:
            for fam, pairs in FAMILIES.items():
                keys = [f"{a}_vs_{b}" for a, b in pairs]
                adj = holm({k: res["contrasts"][k][m]["p_one_sided_a_le_b"] for k in keys})
                for k in keys:
                    res["contrasts"][k][m]["p_holm"] = adj[k]
                    res["contrasts"][k][m]["holm_family"] = fam
        results["splits"][split] = res
        log(f"{split}: bootstrap done")
        (HERE / "b3_subterm_bootstrap.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    if args.phase == "export":
        log("export phase complete"); return
    lines = ["# B3 paired bootstrap, full RA-CMCD minus each variant (four-seed means)", "",
             f"B = {args.n_bootstrap}, seed {args.seed}, shared image indices, one-sided p for H0: full <= variant, Holm within family "
             "(subterm: conf_only, drop_support, drop_prior, drop_iou; baselines: soft_v2, abid_soft).", ""]
    for split, res in results["splits"].items():
        lines += [f"## {split} ({res['n_images']} images)", "", "| contrast | metric | point diff (pp) | 95% CI (pp) | p one-sided | p Holm |", "|---|---|---|---|---|---|"]
        for k, v in res["contrasts"].items():
            for m in mab.METRICS:
                c = v[m]
                lines.append(f"| {k} | {m} | {100*c['point_diff']:+.2f} | [{100*c['diff_ci95'][0]:+.2f}, {100*c['diff_ci95'][1]:+.2f}] | {c['p_one_sided_a_le_b']:.4f} | {c['p_holm']:.4f} |")
        lines.append("")
    (HERE / "b3_subterm_bootstrap.md").write_text("\n".join(lines), encoding="utf-8")
    log("B3 complete")


if __name__ == "__main__":
    main()
