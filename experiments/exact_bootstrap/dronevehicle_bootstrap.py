"""DroneVehicle student contrasts rebuilt on the validator's own statistics (replaces phase2_bootstrap_global.py).

The test-split table (phase2_eval.json) comes from Ultralytics `val`; captures made on 2026-09-17 reproduce every
student's recorded mAP exactly, so these point estimates are the table's values. Each student carries about a million
detections at confidence 0.001, so the students are evaluated in parallel processes over one precomputed matrix of
image multiplicities: every process reads the same draws, which keeps all contrasts paired.

B = 10,000, seed 20260913 (the DroneVehicle study's seed), 8,980 official test images; arms are four-seed means;
one-sided p = (#{diff <= 0} + 1) / (B + 1); 95% percentile intervals.
Outputs dronevehicle_exact_bootstrap.json / .md and draws/dronevehicle_<student>.npy.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from exact_engine import ImageStats, units_of

HERE = Path(__file__).resolve().parent
STATS = HERE / "stats" / "dronevehicle"
DRAWS = HERE / "draws"
B = 10000
SEED = 20260913
ARMS = {a: ["%s_s%d" % (a, s) for s in range(4)] for a in ("gt_only", "plain", "weighted", "eq3")}
CONTRASTS = [("weighted", "plain"), ("weighted", "gt_only"), ("plain", "gt_only"), ("eq3", "plain"), ("eq3", "weighted")]
METRICS = ("mAP50", "mAP50_95")


def units() -> list[str]:
    return units_of(STATS / "gt_only_s0__test.npz")


def worker(key: str) -> str:
    out = DRAWS / ("dronevehicle_%s.npy" % key)
    if out.exists():
        return key
    u = units()
    s = ImageStats(STATS / ("%s__test.npz" % key), u)
    W = np.load(DRAWS / "dronevehicle_multiplicities.npy", mmap_mode="r")
    res = np.zeros((B + 1, 2))
    res[0] = s.map(np.ones(len(u)))  # row 0: unit multiplicities
    for b in range(B):
        res[b + 1] = s.map(W[b].astype(np.float64))
    np.save(out, res)
    return key


def main() -> None:
    DRAWS.mkdir(exist_ok=True)
    u = units()
    wfile = DRAWS / "dronevehicle_multiplicities.npy"
    if not wfile.exists():
        rng = np.random.default_rng(SEED)
        n = len(u)
        W = np.zeros((B, n), dtype=np.uint8)
        for b in range(B):
            W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
        np.save(wfile, W)
    keys = [k for ms in ARMS.values() for k in ms]
    caps = json.loads((STATS / "captures.json").read_text(encoding="utf-8"))
    assert all(("%s__test" % k) in caps for k in keys), "capture every student first"
    t0 = dt.datetime.now()
    with Pool(processes=len(keys)) as pool:
        for k in pool.imap_unordered(worker, keys):
            print("[%s] %s done" % (dt.datetime.now().strftime("%H:%M:%S"), k), flush=True)
    vals = {k: np.load(DRAWS / ("dronevehicle_%s.npy" % k)) for k in keys}
    for k in keys:  # unit multiplicities must give the validator's value, which equals the table record
        c = caps["%s__test" % k]
        assert abs(vals[k][0, 1] - c["map50_95"]) < 1e-12 and abs(vals[k][0, 0] - c["map50"]) < 1e-12, k
    arm = {a: np.mean([vals[k] for k in ms], axis=0) for a, ms in ARMS.items()}  # [B+1, 2]
    rec = {a: {"mAP50": float(np.mean([caps["%s__test" % k]["recorded_map50"] for k in ms])),
               "mAP50_95": float(np.mean([caps["%s__test" % k]["recorded_map50_95"] for k in ms]))} for a, ms in ARMS.items()}
    res = {"date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "B": B, "seed": SEED, "n_images": len(u),
           "statistic": "Ultralytics val mAP recomputed from the validator's per-image statistics under image multiplicities",
           "point": {a: {"mAP50": float(v[0, 0]), "mAP50_95": float(v[0, 1])} for a, v in arm.items()}, "point_from_records": rec,
           "runtime_minutes": (dt.datetime.now() - t0).total_seconds() / 60, "contrasts": {}}
    lines = ["# DroneVehicle students, bootstrap on the validator's own statistics (%s)" % res["date"], "",
             "B = %d, seed %d, %d official test images; arms are four-seed means; point estimates are the table values." % (B, SEED, len(u)), "",
             "Four-seed means (mAP@0.5:0.95): " + ", ".join("%s %.4f" % (a, res["point"][a]["mAP50_95"]) for a in ARMS), "",
             "| contrast | metric | point (pp) | 95% CI (pp) | p one-sided | positive |", "|---|---|---:|---|---:|---:|"]
    for a, b in CONTRASTS:
        key = "%s_vs_%s" % (a, b)
        res["contrasts"][key] = {}
        for k, m in enumerate(METRICS):
            d = arm[a][1:, k] - arm[b][1:, k]
            c = {"point_diff": float(arm[a][0, k] - arm[b][0, k]), "diff_mean": float(d.mean()),
                 "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                 "p_one_sided": (float(np.count_nonzero(d <= 0)) + 1.0) / (B + 1.0), "positive_rate": float(np.mean(d > 0))}
            res["contrasts"][key][m] = c
            lines.append("| %s | %s | %+.2f | [%+.2f, %+.2f] | %.4f | %.3f |" % (key, m, 100 * c["point_diff"], 100 * c["diff_ci95"][0],
                                                                         100 * c["diff_ci95"][1], c["p_one_sided"], c["positive_rate"]))
    (HERE / "dronevehicle_exact_bootstrap.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (HERE / "dronevehicle_exact_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
