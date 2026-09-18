"""Second computation path for the weighted engine (read-only check).

For a few models and draws, the drawn multiset of images is materialised by duplicating each image's captured
statistics, and the fork's own ap_per_class is run on it; the weighted engine must agree. Draws are image-level
(multiplicities from bincount) and, for Anti-UAV, sequence-level (whole sequences duplicated). Also checks the
engine at unit multiplicities against the mAP the validator reported at capture time.

Usage: python validate.py antiuav | dronevehicle
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from exact_engine import ImageStats, expanded_ultralytics_map, units_of

HERE = Path(__file__).resolve().parent


def main() -> None:
    ds = sys.argv[1]
    stats = HERE / "stats" / ds
    caps = json.loads((stats / "captures.json").read_text(encoding="utf-8"))
    if ds == "antiuav":
        picks = ["relw_all_s0__night78", "reloff_s42__night160", "singlebest_s43__night78", "baseline_list__night160", "relw_all_s1__val"]
    else:
        picks = ["weighted_s0__test", "plain_s1__test", "eq3_s2__test", "gt_only_s3__test"]
    rng = np.random.default_rng(7)
    worst_unit, worst_draw = 0.0, 0.0
    for key in picks:
        npz = stats / ("%s.npz" % key)
        units = units_of(npz)
        s = ImageStats(npz, units)
        m50, m = s.map(np.ones(len(units)))
        du = max(abs(m - caps[key]["map50_95"]), abs(m50 - caps[key]["map50"]))
        worst_unit = max(worst_unit, du)
        print("%-28s unit multiplicities: engine %.8f / %.8f  validator %.8f / %.8f  diff %.1e"
              % (key, m50, m, caps[key]["map50"], caps[key]["map50_95"], du), flush=True)
        n = len(units)
        draws = [np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64) for _ in range(2)]
        if ds == "antiuav" and "night" in key:
            sys.path.insert(0, r"<YOLO11_ROOT>/scripts/sota")
            from sequence_cluster_bootstrap import group_by_sequence
            ids, members = group_by_sequence(units)
            w = np.zeros(n)
            for j in rng.integers(0, len(ids), len(ids)):
                w[np.array(members[ids[int(j)]])] += 1
            draws.append(w)
        for i, w in enumerate(draws):
            fast = s.map(w)
            slow = expanded_ultralytics_map(npz, units, w)
            d = max(abs(fast[0] - slow[0]), abs(fast[1] - slow[1]))
            worst_draw = max(worst_draw, d)
            print("   draw %d (%s): engine %.8f / %.8f  expanded ap_per_class %.8f / %.8f  diff %.1e"
                  % (i, "sequence" if i == 2 else "image", fast[0], fast[1], slow[0], slow[1], d), flush=True)
    print("WORST unit-multiplicity diff %.1e, WORST draw diff %.1e -> %s"
          % (worst_unit, worst_draw, "AGREE" if max(worst_unit, worst_draw) < 1e-9 else "DISAGREE"))


if __name__ == "__main__":
    main()
