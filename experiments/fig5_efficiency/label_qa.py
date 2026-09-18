"""Label QA for the efficiency figure: every name label must sit inside its panel, clear of other markers, and
closer to its own marker than to any other (so a reader attaches it to the right point).

Usage: python label_qa.py            (reports the labels of gen_efficiency_tradeoff_v3 that fail)
The generator's own check (check_text_overlaps) only tests text against text.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from matplotlib.collections import PathCollection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def build_figure(offsets: dict | None = None):
    """Run the generator up to (not including) saving and return (module, figure)."""
    import gen_efficiency_tradeoff_v3 as G
    if offsets:
        G._OFFSETS.update(offsets)
    captured = {}

    def fake_save(fig, *a, **k):
        captured["fig"] = fig

    G.save_figure = fake_save
    G.main(["--outdir", str(HERE / "_qa_unused")])
    return G, captured["fig"]


def _rect_point_dist(r, x, y):
    dx = max(r.x0 - x, 0, x - r.x1)
    dy = max(r.y0 - y, 0, y - r.y1)
    return math.hypot(dx, dy)


def qa(fig, ratio: float = 0.8) -> list[str]:
    """Return a list of failures. A label fails if (1) its box leaves the axes, (2) it overlaps another point's
    marker, or (3) some other marker is not clearly farther from it than its own marker (d_own > ratio * d_other)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    fails = []
    for ax in fig.axes:
        pts = []  # (x_px, y_px, radius_px, label)
        for coll in ax.collections:
            offs = coll.get_offsets()
            if not isinstance(coll, PathCollection) or len(offs) != 1 or len(coll.get_sizes()) == 0:
                continue  # the Pareto shading is a PolyCollection
            x, y = ax.transData.transform(offs[0])
            s = coll.get_sizes()[0]
            pts.append((x, y, math.sqrt(s) / 2 * dpi / 72, coll))
        # name labels are the annotations placed in offset points from a data point (headers and callouts are not)
        labels = [t for t in ax.texts if t.get_text().strip() and getattr(t, "xycoords", None) == "data"
                  and getattr(t, "anncoords", None) == "offset points"]
        axbox = ax.get_window_extent(renderer)
        for t in labels:
            box = t.get_window_extent(renderer)
            name = t.get_text()
            own = ax.transData.transform(t.xy)
            if box.x0 < axbox.x0 - 0.5 or box.x1 > axbox.x1 + 0.5 or box.y0 < axbox.y0 - 0.5 or box.y1 > axbox.y1 + 0.5:
                fails.append("%s | %s: label crosses the panel border" % (ax.get_xlabel(), name))
            d_own = _rect_point_dist(box, *own)
            for x, y, r, coll in pts:
                if math.hypot(x - own[0], y - own[1]) < 1.0:
                    continue  # its own marker (or the star drawn on top of the base marker)
                d = _rect_point_dist(box, x, y)
                if d < r:
                    fails.append("%s | %s: label overlaps another marker at (%.0f, %.0f)" % (ax.get_xlabel(), name, x, y))
                elif d_own > ratio * d:
                    fails.append("%s | %s: own marker %.1f px away, another marker %.1f px away" % (ax.get_xlabel(), name, d_own, d))
    return fails


if __name__ == "__main__" and len(sys.argv) == 1:
    _, fig = build_figure()
    out = qa(fig)
    print("\n".join(out) if out else "label QA: all labels pass")


# ---------------------------------------------------------------------------------------------------------------
# Offset search. Labels are moved in place on the drawn figure (only text positions change, so no re-plotting);
# the result is printed as _OFFSETS entries and must be written into the generator and re-checked with qa().
# ---------------------------------------------------------------------------------------------------------------
DXS = (-10, -8, -6, -4, 0, 4, 6, 8, 10)
DYS = (-20, -18, -16, -14, -12, -10, -8, -4, -2, 0, 2, 4, 6, 8, 10, 12)


def _panel_objects(fig, ax, renderer):
    pts = []
    for coll in ax.collections:
        offs = coll.get_offsets()
        if isinstance(coll, PathCollection) and len(offs) == 1 and len(coll.get_sizes()):
            x, y = ax.transData.transform(offs[0])
            pts.append((x, y, math.sqrt(coll.get_sizes()[0]) / 2 * fig.dpi / 72))
    labels = [t for t in ax.texts if t.get_text().strip() and getattr(t, "xycoords", None) == "data"
              and getattr(t, "anncoords", None) == "offset points"]
    boxes = []  # fixed obstacles: callout box
    for t in ax.texts:
        if t.get_text().strip() and t not in labels and t.get_bbox_patch() is not None:
            boxes.append(t.get_bbox_patch().get_window_extent(renderer))
    arrows = []
    for t in ax.texts:
        ap = getattr(t, "arrow_patch", None)
        if ap is not None:
            try:
                paths, _ = ap._get_path_in_displaycoord()
                arrows.append((t, paths))
            except Exception:  # noqa: BLE001
                pass
    return pts, labels, boxes, arrows


def _evaluate(t, box, own, pts, other_boxes, arrows, axbox, ratio):
    """(hard_ok, association_ok, d_own) for one label box."""
    from matplotlib.transforms import Bbox
    pad = Bbox.from_extents(box.x0 - 1, box.y0 - 1, box.x1 + 1, box.y1 + 1)
    if box.x0 < axbox.x0 + 1 or box.x1 > axbox.x1 - 1 or box.y0 < axbox.y0 + 1 or box.y1 > axbox.y1 - 1:
        return False, False, 0
    d_own = _rect_point_dist(box, *own)
    d_other = []
    for x, y, r in pts:
        d = _rect_point_dist(box, x, y)
        if math.hypot(x - own[0], y - own[1]) < 1.0:
            if d < r + 0.5:
                return False, False, 0  # covers its own marker
            continue
        if d < r + 1.5:
            return False, False, 0
        d_other.append(d)
    for ob in other_boxes:
        if pad.overlaps(ob):
            return False, False, 0
    for owner, paths in arrows:
        if owner is t:
            continue
        if any(p.intersects_bbox(pad, filled=False) for p in paths):
            return False, False, 0
    assoc = all(d_own <= ratio * d for d in d_other)
    return True, assoc, d_own


def optimize(fig, ratio: float = 0.8, passes: int = 3, keep_weight: float = 0.35):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    chosen = {}
    for _ in range(passes):
        changed = False
        for ax in fig.axes:
            pts, labels, fixed_boxes, arrows = _panel_objects(fig, ax, renderer)
            if not labels:
                continue
            axbox = ax.get_window_extent(renderer)
            for t in labels:
                own = ax.transData.transform(t.xy)
                cur = (t.xyann[0], t.xyann[1], t.get_ha())
                others = [u.get_window_extent(renderer) for u in labels if u is not t]
                if t.get_text() == "base":  # the star sits on its marker; its leader line does the association
                    pts_eval = [p for p in pts if math.hypot(p[0] - own[0], p[1] - own[1]) >= 1.0 or p[2] < 5]
                else:
                    pts_eval = pts
                best = None
                orig = t._qa_orig if hasattr(t, "_qa_orig") else cur
                t._qa_orig = orig
                if t.get_text() == "base":
                    continue  # keeps its leader line and published offset
                ok, assoc, _ = _evaluate(t, t.get_window_extent(renderer), own, pts_eval, others + fixed_boxes, arrows, axbox, ratio)
                if ok and assoc:
                    continue  # a label that already passes is not moved
                for dx in DXS:
                    for dy in DYS:
                        ha = "right" if dx < 0 else ("left" if dx > 0 else "center")
                        if dx == 0 and abs(dy) < 8:
                            continue
                        t.xyann = (dx, dy); t.set_ha(ha)
                        box = t.get_window_extent(renderer)
                        ok, assoc, d_own = _evaluate(t, box, own, pts_eval, others + fixed_boxes, arrows, axbox, ratio)
                        if not ok:
                            continue
                        cost = d_own + keep_weight * math.hypot(dx - orig[0], dy - orig[1]) + (0 if assoc else 1000)
                        if ha != orig[2]:
                            cost += 1.0
                        if best is None or cost < best[0] - 1e-9:
                            best = (cost, dx, dy, ha, assoc)
                if best is None:
                    t.xyann = cur[:2]; t.set_ha(cur[2])
                    print("no admissible position for", ax.get_xlabel(), t.get_text())
                    continue
                t.xyann = (best[1], best[2]); t.set_ha(best[3])
                if (best[1], best[2], best[3]) != cur:
                    changed = True
                chosen[(ax.get_xlabel(), t.get_text())] = best
        fig.canvas.draw()
        if not changed:
            break
    return chosen


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "search":
    G, fig = build_figure()
    res = optimize(fig)
    names = {"early": "YOLO11 early", "mid-late": "YOLO11 mid-late", "base": "YOLO11-RGBT base"}
    keys = {"Parameters (M)": "params", "FLOPs (G)": "flops", "Latency (ms)": "lat"}
    for (panel, label), (cost, dx, dy, ha, assoc) in sorted(res.items()):
        k = (names.get(label, label), keys[panel])
        was = G._OFFSETS.get(k)
        print("%-30s %-5s -> (%d, %d, %r)  was %s  assoc=%s  cost=%.1f" % (k[0], k[1], dx, dy, ha, was, assoc, cost))
    print("remaining QA failures:", qa(fig))
