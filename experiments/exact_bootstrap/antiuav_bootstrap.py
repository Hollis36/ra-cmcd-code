"""Anti-UAV-RGBT bootstraps rebuilt on the validator's own statistics, so every point estimate is the table's value.

Replaces, contrast for contrast, the exported-prediction bootstraps of the paper (scripts in <YOLO11_ROOT>/scripts/sota):
  multiseed_ablation_bootstrap.py  reliability vs plain, reliability vs single-best, plain vs single-best
  b3_subterm_bootstrap.py          full minus each leave-one-out variant (Holm over four) and minus the two baselines
                                   (Soft-Teacher-style, Abid et al.; Holm over two)
  multiseed_paired_bootstrap.py    transfer: four-seed student mean minus the human-label baseline (night splits)
  val_transfer_bootstrap.py        the same transfer contrast on the validation split
  multiseed_sequence_bootstrap.py  sequence-level cluster bootstrap of transfer and reliability vs plain
Conventions kept from those scripts: B = 10,000; image-level seed 20260515 + split index (night-78 0, night-160 1),
validation 20260515; sequence-level the same seeds over video-sequence ids; every regime is the mean over its seeds;
one-sided p = (#{diff <= 0} + 1) / (B + 1); 95% percentile intervals. All regimes of a split share the same draws.

Needs stats/antiuav/*.npz from capture_stats.py. Outputs antiuav_exact_bootstrap.json / .md.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from exact_engine import ImageStats, units_of

HERE = Path(__file__).resolve().parent
STATS = HERE / "stats" / "antiuav"
B = 10000
SEED = 20260515
METRICS = ("mAP50", "mAP50_95")
ARMS = {
    "reliability": ["relw_all_s%d" % s for s in range(4)],
    "plain": ["reloff_s%d" % s for s in range(42, 46)],
    "single_best": ["singlebest_s%d" % s for s in range(42, 46)],
    "conf_only": ["conf_only_s%d" % s for s in range(4)],
    "drop_support": ["drop_support_s%d" % s for s in range(4)],
    "drop_prior": ["drop_prior_s%d" % s for s in range(4)],
    "drop_iou": ["drop_iou_s%d" % s for s in range(4)],
    "soft_v2": ["soft_v2_s%d" % s for s in range(4)],
    "abid_soft": ["abid_soft_s%d" % s for s in range(4)],
}
BASELINE = {"night78": "baseline_dirdata", "night160": "baseline_list", "val": "baseline_dirdata"}
FAMILIES = {  # (a, b) means a minus b; Holm is applied within subterm and baselines, as in the paper
    "ablation": [("reliability", "plain"), ("reliability", "single_best"), ("plain", "single_best")],
    "subterm": [("reliability", "conf_only"), ("reliability", "drop_support"), ("reliability", "drop_prior"), ("reliability", "drop_iou")],
    "baselines": [("reliability", "soft_v2"), ("reliability", "abid_soft")],
    "transfer": [("reliability", "baseline")],
}
HOLM = ("subterm", "baselines")


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def summarize(diff: np.ndarray, point: float, point_record: float) -> dict:
    return {"point_diff": point, "point_diff_from_records": point_record, "diff_mean": float(diff.mean()),
            "diff_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "p_one_sided": (float(np.count_nonzero(diff <= 0)) + 1.0) / (len(diff) + 1.0),
            "positive_rate": float(np.mean(diff > 0))}


def run_split(split: str, arms: dict, families: dict, weights_fn, seed: int, captures: dict) -> dict:
    units = units_of(STATS / ("relw_all_s0__%s.npz" % split))
    models = {m: ImageStats(STATS / ("%s__%s.npz" % (m, split)), units) for ms in arms.values() for m in ms}
    ones = np.ones(len(units))
    point_model = {m: dict(zip(METRICS, s.map(ones))) for m, s in models.items()}
    rec_model = {m: {"mAP50": captures["%s__%s" % (m, split)]["recorded_map50"], "mAP50_95": captures["%s__%s" % (m, split)]["recorded_map50_95"]}
                 for m in models}
    point = {a: {k: float(np.mean([point_model[m][k] for m in ms])) for k in METRICS} for a, ms in arms.items()}
    point_rec = {a: {k: float(np.mean([rec_model[m][k] for m in ms])) for k in METRICS} for a, ms in arms.items()}
    rng = np.random.default_rng(seed)
    names = list(arms)
    draws = np.zeros((B, len(names), 2))
    for b in range(B):
        w = weights_fn(rng)
        vals = {m: s.map(w) for m, s in models.items()}
        for i, a in enumerate(names):
            draws[b, i] = np.mean([vals[m] for m in arms[a]], axis=0)
        if (b + 1) % 2000 == 0:
            print("  %s: %d/%d draws" % (split, b + 1, B), flush=True)
    out = {"n_images": len(units), "seed": seed, "point_per_model": point_model, "recorded_per_model": rec_model,
           "point": point, "point_from_records": point_rec, "contrasts": {}}
    for fam, pairs in families.items():
        for a, b_ in pairs:
            key = "%s_vs_%s" % (a, b_)
            out["contrasts"][key] = {"family": fam}
            for k, name in enumerate(METRICS):
                d = draws[:, names.index(a), k] - draws[:, names.index(b_), k]
                out["contrasts"][key][name] = summarize(d, point[a][name] - point[b_][name], point_rec[a][name] - point_rec[b_][name])
        for name in METRICS:
            keys = ["%s_vs_%s" % pr for pr in pairs]
            adj = holm({k: out["contrasts"][k][name]["p_one_sided"] for k in keys})
            for k in keys:
                out["contrasts"][k][name]["p_holm_within_family"] = adj[k]
    return out


def main() -> None:
    captures = json.loads((STATS / "captures.json").read_text(encoding="utf-8"))
    sys.path.insert(0, r"<YOLO11_ROOT>/scripts/sota")
    from sequence_cluster_bootstrap import group_by_sequence  # the paper's own sequence grouping
    res = {"date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "B": B, "seed": SEED,
           "statistic": "Ultralytics val mAP recomputed from the validator's per-image statistics under image multiplicities",
           "image_level": {}, "sequence_level": {}}
    for si, split in enumerate(("night78", "night160")):
        arms = {**ARMS, "baseline": [BASELINE[split]]}
        n = len(units_of(STATS / ("relw_all_s0__%s.npz" % split)))
        res["image_level"][split] = run_split(
            split, arms, FAMILIES, lambda rng, n=n: np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64), SEED + si, captures)
        units = units_of(STATS / ("relw_all_s0__%s.npz" % split))
        seq_ids, seq_to_idx = group_by_sequence(units)
        members = [np.array(seq_to_idx[s]) for s in seq_ids]

        def seq_weights(rng, members=members, n=len(units)):
            w = np.zeros(n)
            for j in rng.integers(0, len(members), len(members)):
                w[members[int(j)]] += 1
            return w
        seq = run_split(split, {"reliability": ARMS["reliability"], "plain": ARMS["plain"], "baseline": [BASELINE[split]]},
                        {"sequence": [("reliability", "baseline"), ("reliability", "plain")]}, seq_weights, SEED + si, captures)
        seq["n_sequences"] = len(seq_ids)
        res["sequence_level"][split] = seq
        (HERE / "antiuav_exact_bootstrap.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    res["image_level"]["val"] = run_split(
        "val", {"reliability": ARMS["reliability"], "baseline": [BASELINE["val"]]}, {"transfer": FAMILIES["transfer"]},
        lambda rng: np.bincount(rng.integers(0, 586, 586), minlength=586).astype(np.float64), SEED, captures)
    assert res["image_level"]["val"]["n_images"] == 586
    (HERE / "antiuav_exact_bootstrap.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    lines = ["# Anti-UAV-RGBT bootstraps on the validator's own statistics (%s)" % res["date"], "",
             "B = %d; image-level seed 20260515 + split index (validation 20260515); sequence-level the same seeds over sequence ids." % B,
             "Point = today's re-validation (four-seed means); record = the stored val_fixed values behind the tables.", ""]
    for level in ("image_level", "sequence_level"):
        for split, r in res[level].items():
            lines += ["## %s, %s (%d images%s)" % (level.replace("_", "-"), split, r["n_images"],
                                                  ", %d sequences" % r["n_sequences"] if "n_sequences" in r else ""), "",
                      "| contrast | family | point (pp) | record (pp) | 95% CI (pp) | p one-sided | p Holm (family) | positive |",
                      "|---|---|---:|---:|---|---:|---:|---:|"]
            for k, v in r["contrasts"].items():
                c = v["mAP50_95"]
                lines.append("| %s | %s | %+.2f | %+.2f | [%+.2f, %+.2f] | %.4f | %s | %.3f |" % (
                    k, v["family"], 100 * c["point_diff"], 100 * c["point_diff_from_records"], 100 * c["diff_ci95"][0], 100 * c["diff_ci95"][1],
                    c["p_one_sided"], "%.4f" % c["p_holm_within_family"] if v["family"] in HOLM else "--", c["positive_rate"]))
            lines.append("")
    (HERE / "antiuav_exact_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
