"""Weighted-image mAP with the exact arithmetic of Ultralytics ap_per_class / compute_ap (8.3.75 fork).

A bootstrap draw assigns each image a multiplicity. Ultralytics matches predictions to labels inside each image
(match_predictions uses only that image's IoU matrix), so a detection's true-positive row does not change when whole
images are resampled; only the pooled ranking and the label counts do. Evaluating a draw therefore needs no
re-matching: take the validator's captured true-positive matrix, keep the global confidence order Ultralytics uses
(np.argsort(-conf) on the concatenated statistics), weight every detection by its image's multiplicity, and repeat
ap_per_class's cumulative sums and 101-point interpolation. With all multiplicities equal to one this is
Ultralytics' own computation on the same arrays; validate.py checks drawn multiplicities against Ultralytics run on
explicitly duplicated statistics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

X101 = np.linspace(0, 1, 101)
EPS = 1e-16


class ImageStats:
    """One model on one split, re-indexed to a shared ordered list of image stems (the resampling units)."""

    def __init__(self, npz: Path, units: list[str]):
        z = np.load(npz)
        stems = [str(s) for s in z["stems"]]
        pos = {s: i for i, s in enumerate(units)}
        assert set(stems) == set(units), "%s: image set differs from the shared units" % npz.name
        remap = np.array([pos[s] for s in stems], dtype=np.int64)
        img = remap[z["img"]] if len(z["img"]) else z["img"].astype(np.int64)
        tgt_img = remap[z["target_img"]]
        conf, tp, pred_cls, target_cls = z["conf"], z["tp"], z["pred_cls"], z["target_cls"]
        order = np.argsort(-conf)  # exactly ap_per_class's sort, on the arrays in the validator's entry order
        self.tp = tp[order].astype(np.float64)
        self.img = img[order]
        cls_sorted = pred_cls[order]
        self.classes = np.unique(target_cls)
        self.idx = {int(c): np.flatnonzero(cls_sorted == c) for c in self.classes}
        n = len(units)
        self.nt = {int(c): np.bincount(tgt_img[target_cls == c], minlength=n).astype(np.float64) for c in self.classes}
        self.n_units = n

    def map(self, w: np.ndarray) -> tuple[float, float]:
        """(mAP@0.5, mAP@0.5:0.95) of the multiset of images given by multiplicities w (length n_units)."""
        aps = []
        for c in self.classes:
            c = int(c)
            n_l = float(self.nt[c] @ w)
            if n_l == 0:
                continue  # the class is absent from this multiset: not one of ap_per_class's unique classes
            ap = np.zeros(self.tp.shape[1])
            idx = self.idx[c]
            if len(idx):
                wd = w[self.img[idx]]
                keep = wd > 0
                if keep.any():
                    wd, t = wd[keep], self.tp[idx[keep]]
                    tpc = np.cumsum(wd[:, None] * t, axis=0)
                    fpc = np.cumsum(wd[:, None] * (1.0 - t), axis=0)
                    recall = tpc / (n_l + EPS)
                    precision = tpc / (tpc + fpc)
                    ones = np.ones((1, t.shape[1]))
                    mrec = np.concatenate((0 * ones, recall, ones), axis=0)
                    mpre = np.concatenate((ones, precision, 0 * ones), axis=0)
                    mpre = np.flip(np.maximum.accumulate(np.flip(mpre, 0), axis=0), 0)
                    for j in range(t.shape[1]):
                        ap[j] = np.trapezoid(np.interp(X101, mrec[:, j], mpre[:, j]), X101)
            aps.append(ap)
        if not aps:
            return 0.0, 0.0
        a = np.array(aps)
        return float(a[:, 0].mean()), float(a.mean())


def units_of(npz: Path) -> list[str]:
    return [str(s) for s in np.load(npz)["stems"]]


def expanded_ultralytics_map(npz: Path, units: list[str], w: np.ndarray) -> tuple[float, float]:
    """Second path: duplicate each image's captured statistics w times and run the fork's own ap_per_class."""
    import sys
    sys.path.insert(0, r"<FORK_ROOT>")
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    from ultralytics.utils.metrics import ap_per_class
    z = np.load(npz)
    stems = [str(s) for s in z["stems"]]
    pos = {s: i for i, s in enumerate(units)}
    remap = np.array([pos[s] for s in stems], dtype=np.int64)
    img = remap[z["img"]] if len(z["img"]) else z["img"].astype(np.int64)
    tgt_img = remap[z["target_img"]]
    # group detections and labels by image once (stable, so each image keeps the validator's within-image order)
    od, ot = np.argsort(img, kind="stable"), np.argsort(tgt_img, kind="stable")
    n = len(units)
    ds, de = np.searchsorted(img[od], np.arange(n)), np.searchsorted(img[od], np.arange(n), side="right")
    ts, te = np.searchsorted(tgt_img[ot], np.arange(n)), np.searchsorted(tgt_img[ot], np.arange(n), side="right")
    di, ti = [], []
    for u in range(n):  # units in order, each repeated as often as drawn, like a validator over the multiset
        k = int(w[u])
        if k:
            di.append(np.tile(od[ds[u]:de[u]], k))
            ti.append(np.tile(ot[ts[u]:te[u]], k))
    di, ti = np.concatenate(di), np.concatenate(ti)
    tp, conf, pc, tc = z["tp"][di], z["conf"][di], z["pred_cls"][di], z["target_cls"][ti]
    ap = ap_per_class(tp, conf, pc, tc)[5]
    return float(ap[:, 0].mean()), float(ap.mean())
