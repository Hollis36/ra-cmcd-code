"""Second computation path for bootstrap_rescore_global.py (read-only check).

The bootstrap evaluates each draw through image multiplicities and precomputed true-positive flags. Here the first
two draws are rebuilt from the same seed, the resampled validation set is materialised explicitly (every drawn image
copied as many times as it was drawn, with its ground truth), and the table's own evaluator rescore_eq3.evaluate()
is run on it. Both paths must agree, and the stored draws must equal the recomputed ones.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np

import bootstrap_rescore_global as G
import rescore_eq3 as R

HERE = Path(__file__).resolve().parent
N_CHECK = 2


def main() -> None:
    gt = R.load_gt()
    fused = R.cluster(R.load_preds(), R.PILOT_PRIORS)
    units = sorted(set(gt) | {f["image"] for f in fused})
    ap = G.make_ap(*G.build(fused, gt, units))
    stored = np.load(HERE / "bootstrap_rescore_global_draws.npz")["draws"]
    rng = np.random.default_rng(G.SEED)
    uidx = {u: i for i, u in enumerate(units)}
    worst = 0.0
    for b in range(N_CHECK):
        w = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        gt_x = {}
        for u in units:
            for k in range(w[uidx[u]]):
                gt_x["%s#%d" % (u, k)] = [R.Det("%s#%d" % (u, k), g.cls, g.box, 1.0) for g in gt.get(u, [])]
        for si, s in enumerate(G.SCORINGS):
            # copies of a detection stay adjacent in list order, so ties are broken as in the weighted construction
            preds = [R.Det("%s#%d" % (f["image"], k), f["cls"], f["box"], f[s]) for f in fused for k in range(w[uidx[f["image"]]])]
            m = R.evaluate(preds, gt_x)
            fast = ap(s, w.astype(np.float64))
            d = max(abs(m["map50"] - fast[0]), abs(m["map50_95"] - fast[1]),
                    abs(stored[b, si, 0] - fast[0]), abs(stored[b, si, 1] - fast[1]))
            worst = max(worst, d)
            print("draw %d %-5s explicit evaluator mAP50 %.8f mAP50-95 %.8f | weighted %.8f %.8f | stored %.8f %.8f | max diff %.2e"
                  % (b, s, m["map50"], m["map50_95"], fast[0], fast[1], stored[b, si, 0], stored[b, si, 1], d), flush=True)
    print("WORST DIFFERENCE %.2e -> %s" % (worst, "AGREE" if worst < 1e-9 else "DISAGREE"))


if __name__ == "__main__":
    main()
