"""W1: data-driven check of the Eq. 3 weight allocation on the training split (no GPU, no validation data).

For each of the 1,506 released consensus boxes (metadata of train_train_teacher_global) the four score terms are
mean confidence s, support fraction S_sup, source-prior fraction S_pri, and localization consistency C_loc, and
r(w) = s * (w_sup * S_sup + w_pri * S_pri + w_loc * C_loc) with w on the simplex. Each box is labelled matched
(IoU >= 0.5 with a human training box), unmatched (IoU < 0.2) or ambiguous (excluded), exactly as in the paper.

Outputs (this folder): weight_fit.json, weight_fit.md, and figures/fig_weight_fit.{pdf,png} in ../../paper/figures.
  1. AUROC(matched vs unmatched) over the full simplex (step 0.01), separation, and Spearman rho against the fixed score.
  2. Flatness: fraction of the simplex within 0.001 / 0.005 AUROC of the maximum; position of the fixed allocation.
  3. Bootstrap (B = 1000 over boxes) of the arg-max allocation.
  4. Unconstrained logistic regression on [s, s*S_sup, s*S_pri, s*C_loc] and the confidence-only score as references.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(r"<YOLO11_ROOT>/outputs/sota_runs")
META = ROOT / "cross_model_wbf" / "fused" / "train_train_teacher_global" / "metadata.jsonl"
HUMAN = ROOT / "yolo11_rgbt_repro" / "dataset_dircopy" / "labels" / "train"
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent.parent / "paper" / "figures"
FIXED = np.array([0.45, 0.35, 0.20])
STEP = 0.01
SEED = 20260515


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box a (4,) against boxes b (n,4), normalized coordinates."""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1]); x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-12)


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    meta = [json.loads(l) for l in META.read_text(encoding="utf-8").splitlines() if l.strip()]
    human: dict[str, np.ndarray] = {}
    for m in meta:
        if m["image"] in human:
            continue
        f = HUMAN / f"{m['image']}.txt"
        boxes = []
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) >= 5:
                    cx, cy, w, h = map(float, p[1:5]); boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        human[m["image"]] = np.array(boxes, dtype=float).reshape(-1, 4)
    terms = np.array([[m["mean_conf"], m["model_fraction"], m["source_weight_fraction"], m["loc_consistency"]] for m in meta])
    best_iou = np.array([iou_xyxy(np.array(m["xyxy"], dtype=float), human[m["image"]]).max() if len(human[m["image"]]) else 0.0 for m in meta])
    label = np.where(best_iou >= 0.5, 1, np.where(best_iou < 0.2, 0, -1))
    counts = {"boxes": len(meta), "matched": int((label == 1).sum()), "unmatched": int((label == 0).sum()), "ambiguous": int((label == -1).sum())}
    fixed_r = np.array([m["reliability"] for m in meta])
    return terms, label, fixed_r, counts


def score(terms: np.ndarray, w: np.ndarray) -> np.ndarray:
    return terms[:, 0] * (terms[:, 1:] @ w)


def simplex(step: float) -> np.ndarray:
    n = int(round(1 / step)); pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append((i * step, j * step, 1 - i * step - j * step))
    return np.array(pts)


def main() -> None:
    terms, label, fixed_r, counts = load()
    keep = label >= 0
    T, y = terms[keep], label[keep]
    assert abs(score(terms, FIXED) - fixed_r).max() < 1e-9
    grid = simplex(STEP)
    auc = np.array([roc_auc_score(y, score(T, w)) for w in grid])
    sep = np.array([score(T, w)[y == 1].mean() - score(T, w)[y == 0].mean() for w in grid])
    rho = np.array([spearmanr(score(terms, w), fixed_r).correlation for w in grid])
    i_best = int(auc.argmax()); i_fixed = int(np.argmin(np.abs(grid - FIXED).sum(1)))
    conf_only = roc_auc_score(y, T[:, 0])
    # references: confidence only; unconstrained logistic regression (5-fold CV AUROC to avoid in-sample optimism)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X = np.column_stack([T[:, 0], T[:, 0] * T[:, 1], T[:, 0] * T[:, 2], T[:, 0] * T[:, 3]])
    lr = LogisticRegression(max_iter=5000, C=1.0)
    cv_prob = cross_val_predict(lr, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    lr.fit(X, y)
    coef = lr.coef_[0]
    struct = np.clip(coef[1:], 0, None); struct_alloc = (struct / struct.sum()).tolist() if struct.sum() > 0 else None
    # bootstrap of the arg-max allocation
    rng = np.random.default_rng(SEED); B = 1000; argmax = []
    idx1 = np.where(y == 1)[0]; idx0 = np.where(y == 0)[0]
    coarse = simplex(0.05)
    for _ in range(B):
        s = np.concatenate([rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))])
        a = [roc_auc_score(y[s], score(T[s], w)) for w in coarse]
        argmax.append(coarse[int(np.argmax(a))])
    argmax = np.array(argmax)
    res = {
        "counts": counts, "step": STEP, "n_grid": int(len(grid)),
        "fixed": {"w": FIXED.tolist(), "auroc": float(auc[i_fixed]), "separation": float(sep[i_fixed])},
        "best": {"w": grid[i_best].tolist(), "auroc": float(auc[i_best]), "separation": float(sep[i_best]), "rho_vs_fixed": float(rho[i_best])},
        "auroc_range": [float(auc.min()), float(auc.max())],
        "fraction_within_0.001": float((auc >= auc.max() - 0.001).mean()),
        "fraction_within_0.005": float((auc >= auc.max() - 0.005).mean()),
        "rho_min_over_simplex": float(rho.min()),
        "confidence_only_auroc": float(conf_only),
        "logistic": {"cv_auroc": float(roc_auc_score(y, cv_prob)), "coef_[s, s*sup, s*pri, s*loc]": coef.tolist(), "structural_allocation": struct_alloc},
        "bootstrap_argmax": {"B": B, "mean_w": argmax.mean(0).tolist(), "w_loc_ci95": np.percentile(argmax[:, 2], [2.5, 97.5]).tolist()},
        "worst_allocation": {"w": grid[int(auc.argmin())].tolist(), "auroc": float(auc.min())},
    }
    (HERE / "weight_fit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # figure: AUROC as a function of the localization weight (support/prior split is immaterial: both terms are 1 on 1,467 boxes)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": False, "figure.dpi": 300})
    wl = np.round(grid[:, 2], 2); xs = sorted(set(wl))
    env_hi = [auc[wl == x].max() for x in xs]; env_lo = [auc[wl == x].min() for x in xs]
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    ax.fill_between(xs, env_lo, env_hi, color="tab:blue", alpha=0.25, label="range over support/prior split")
    ax.plot(xs, env_hi, color="tab:blue", lw=1.2, label="best split at this $w_{\\mathrm{loc}}$")
    ax.axhline(conf_only, color="tab:gray", ls="--", lw=1, label=f"confidence only ({conf_only:.3f})")
    ax.plot([FIXED[2]], [auc[i_fixed]], marker="*", ms=9, color="tab:red", ls="none", label="fixed 0.45/0.35/0.20")
    ax.set_xlabel("localization weight $w_{\\mathrm{loc}}$ (support + prior $= 1 - w_{\\mathrm{loc}}$)")
    ax.set_ylabel("AUROC matched vs. unmatched")
    ax.set_ylim(min(env_lo) - 0.003, max(env_hi) + 0.003)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig_weight_fit.pdf"); fig.savefig(FIGDIR / "fig_weight_fit.png")

    md = [
        "# W1 weight fit on the training split", "",
        f"Boxes {counts['boxes']}: matched {counts['matched']}, unmatched {counts['unmatched']}, ambiguous {counts['ambiguous']} (excluded). Simplex step {STEP}, {len(grid)} allocations.", "",
        f"- fixed 0.45/0.35/0.20: AUROC {auc[i_fixed]:.4f}, separation {sep[i_fixed]:.3f}",
        f"- best allocation {np.round(grid[i_best], 2).tolist()}: AUROC {auc[i_best]:.4f}, separation {sep[i_best]:.3f}, Spearman vs fixed {rho[i_best]:.4f}",
        f"- AUROC range over the whole simplex [{auc.min():.4f}, {auc.max():.4f}]; {100 * res['fraction_within_0.001']:.1f}% of allocations within 0.001 of the maximum, {100 * res['fraction_within_0.005']:.1f}% within 0.005; worst {np.round(grid[int(auc.argmin())], 2).tolist()} at {auc.min():.4f}",
        f"- confidence only (no structural term): AUROC {conf_only:.4f}",
        f"- unconstrained logistic regression (5-fold CV): AUROC {res['logistic']['cv_auroc']:.4f}; structural allocation from its coefficients {None if struct_alloc is None else np.round(struct_alloc, 2).tolist()}",
        f"- bootstrap arg-max (B={B}, step 0.05): mean {np.round(argmax.mean(0), 2).tolist()}, 95% interval of w_loc [{res['bootstrap_argmax']['w_loc_ci95'][0]:.2f}, {res['bootstrap_argmax']['w_loc_ci95'][1]:.2f}]",
        f"- minimum Spearman rho against the fixed score over the simplex: {rho.min():.4f}",
    ]
    (HERE / "weight_fit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
