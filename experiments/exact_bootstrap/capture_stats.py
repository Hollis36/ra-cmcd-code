"""Capture the Ultralytics validator's own per-image matching statistics for every model a bootstrap compares.

Why: the tables report Ultralytics `val` mAP. The earlier Anti-UAV bootstrap recomputed global mAP from exported
prediction files, whose inference path differs slightly from `val`, so its point estimates missed the table by up to
0.55 pp per arm (audit H2); the DroneVehicle student bootstrap used an independent AP implementation (+1.81 against
+2.23 pp). Here every model is validated exactly as for its table entry (same checkpoint, data yaml, frozen
imgsz/conf/iou, batch size, workers), and a callback keeps the validator's per-image true-positive matrix, confidences
and classes. A bootstrap built on these statistics reproduces the table value of every model exactly.

For each capture the script asserts that Ultralytics' own ap_per_class on the captured arrays equals the mAP the
validator just reported, and records the difference from the value stored when the table was made.

Usage: python capture_stats.py antiuav | dronevehicle      (GPU; skips captures already on disk)
Output: stats/<dataset>/<model>__<split>.npz and stats/<dataset>/captures.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

HERE = Path(__file__).resolve().parent
FORK = Path(r"<FORK_ROOT>")
SOTA = Path(r"<YOLO11_ROOT>/scripts/sota")
CMCD = Path(r"<YOLO11_ROOT>/outputs/sota_runs/cmcd_student")
YREPRO = Path(r"<YOLO11_ROOT>/outputs/sota_runs/yolo11_rgbt_repro")
DV_OUT = Path(r"<YOLO11_ROOT>/outputs/sota_runs/dronevehicle_c3")
DV_HERE = Path(r"<PAPER_ROOT>/experiments/c3_dronevehicle")
LOG = HERE / "capture.log"


def log(msg: str) -> None:
    line = "[%s] %s" % (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def antiuav_models():
    """(key, checkpoint, val_fixed record, splits) for every Anti-UAV model any reported bootstrap uses."""
    arms = [("relw_all", "late_cmcd_pseudo_relw_all_s{s}", "val_fixed_late_cmcd_pseudo_relw_all_s{s}.json", range(4)),
            ("reloff", "late_cmcd_pseudo_reloff_s{s}", "val_fixed_late_cmcd_pseudo_reloff_s{s}.json", range(42, 46)),
            ("singlebest", "late_cmcd_pseudo_singlebest_s{s}", "val_fixed_late_cmcd_pseudo_singlebest_s{s}.json", range(42, 46)),
            ("conf_only", "late_cmcd_pseudo_abl_conf_only_s{s}", "val_fixed_abl_conf_only_s{s}.json", range(4)),
            ("drop_support", "late_cmcd_pseudo_abl_drop_support_s{s}", "val_fixed_abl_drop_support_s{s}.json", range(4)),
            ("drop_prior", "late_cmcd_pseudo_abl_drop_prior_s{s}", "val_fixed_abl_drop_prior_s{s}.json", range(4)),
            ("drop_iou", "late_cmcd_pseudo_abl_drop_iou_s{s}", "val_fixed_abl_drop_iou_s{s}.json", range(4)),
            ("soft_v2", "late_cmcd_pseudo_soft_v2_s{s}", "val_fixed_late_cmcd_pseudo_soft_v2_s{s}.json", range(4)),
            ("abid_soft", "late_cmcd_pseudo_abid_soft_s{s}", "val_fixed_late_cmcd_pseudo_abid_soft_s{s}.json", range(4))]
    out = []
    for arm, run, vf, seeds in arms:
        for s in seeds:
            splits = ["night78", "night160"] + (["val"] if arm == "relw_all" else [])
            out.append(("%s_s%d" % (arm, s), CMCD / "runs" / run.format(s=s) / "weights" / "best.pt", CMCD / vf.format(s=s), splits))
    # human-label baselines of the transfer contrasts (night-78 and validation: dir-data run; night-160: list-data run)
    out.append(("baseline_dirdata", YREPRO / "runs" / "late_fusion_s0_full_dirdata" / "weights" / "best.pt",
                YREPRO / "val_fixed_late_fusion_s0_full_dirdata.json", ["night78", "val"]))
    out.append(("baseline_list", YREPRO / "runs" / "late_fusion_s0_full" / "weights" / "best.pt",
                YREPRO / "val_fixed_late_fusion_s0_full.json", ["night160"]))
    return out


def run_antiuav() -> None:
    sys.path.insert(0, str(SOTA))
    import run_yolo11_rgbt_repro as RR  # noqa: E402  (same helpers as val_fixed_eval.py)
    from val_fixed_eval import data_yaml_for_split  # noqa: E402
    YOLO = RR.prepare_yolo(RR.DEFAULT_YOLO_ROOT)
    for key, ckpt, vf, splits in antiuav_models():
        assert ckpt.exists(), ckpt
        rec = json.loads(vf.read_text(encoding="utf-8"))
        spec = rec["best_val"]["spec"]
        recorded = {e["split"]: e["metrics"] for e in rec["fixed_evals"]}
        for split in splits:
            # kwargs copied from run_yolo11_rgbt_repro.eval_one, which produced every val_fixed record
            kwargs = dict(data=str(data_yaml_for_split(split)), imgsz=int(spec["imgsz"]), batch=1, conf=float(spec["conf"]),
                          iou=float(spec["iou"]), device="0", verbose=False, plots=False, workers=0,
                          project=str(HERE / "eval_runs"), name="%s_%s" % (key, split), exist_ok=True,
                          use_simotm="RGBT", channels=4, pairs_rgb_ir=["images", "images_ir"])
            capture(YOLO, "antiuav", key, ckpt, split, kwargs, recorded[split]["mAP50_95"], recorded[split]["mAP50"])


def run_dronevehicle() -> None:
    sys.path.insert(0, str(FORK))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(DV_OUT / "ultralytics_config"))
    os.environ.setdefault("WANDB_DISABLED", "true")
    from ultralytics import YOLO  # noqa: E402
    ev = json.loads((DV_HERE / "phase2_eval.json").read_text(encoding="utf-8"))
    for arm in ("gt_only", "plain", "weighted", "eq3"):
        for s in range(4):
            key = "%s_s%d" % (arm, s)
            st = ev[key]["setting"]
            ckpt = DV_OUT / "students" / key / "weights" / "best.pt"
            assert ckpt.exists(), ckpt
            # kwargs copied from phase2_eval.evaluate, which produced the test-split table entries
            kwargs = dict(data=str(DV_OUT / "eval_test.yaml"), imgsz=int(st["imgsz"]), batch=8, conf=float(st["conf"]),
                          iou=float(st["iou"]), device="0", verbose=False, plots=False, workers=0,
                          project=str(HERE / "eval_runs"), name="dv_%s_test" % key, exist_ok=True,
                          use_simotm="RGBT", channels=4, pairs_rgb_ir=["images", "images_ir"])
            capture(YOLO, "dronevehicle", key, ckpt, "test", kwargs, ev[key]["test"]["map50_95"], ev[key]["test"]["map50"])


def capture(YOLO, dataset: str, key: str, ckpt: Path, split: str, kwargs: dict, rec_map: float, rec_map50: float) -> None:
    out_dir = HERE / "stats" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / ("%s__%s.npz" % (key, split))
    index = out_dir / "captures.json"
    done = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    if npz.exists() and ("%s__%s" % (key, split)) in done:
        return
    from ultralytics.utils.metrics import ap_per_class  # noqa: E402
    box = {}

    def on_val_batch_start(validator):
        # The validator appends one stats entry per image, but skips images with neither labels nor predictions,
        # so entries cannot be matched to images by position. Record the image each entry belongs to:
        # _prepare_batch runs for every image just before that image's entry can be appended.
        if getattr(validator, "_entry_files", None) is not None:
            return
        validator._entry_files, validator._cur_im = [], None
        orig = validator._prepare_batch

        def prep(si, batch):
            validator._cur_im = batch["im_file"][si]
            return orig(si, batch)

        class RecList(list):
            def append(self, x):
                list.append(self, x)
                validator._entry_files.append(validator._cur_im)

        assert len(validator.stats["tp"]) == 0
        validator._prepare_batch = prep
        validator.stats["tp"] = RecList()

    def on_val_end(validator):
        st = validator.stats
        box["tp"] = [t.cpu().numpy().astype(bool) for t in st["tp"]]
        box["conf"] = [t.cpu().numpy().astype(np.float64) for t in st["conf"]]
        box["pred_cls"] = [t.cpu().numpy().astype(np.int64) for t in st["pred_cls"]]
        box["target_cls"] = [t.cpu().numpy().astype(np.int64) for t in st["target_cls"]]
        box["entry_files"] = list(validator._entry_files)
        box["im_files"] = list(validator.dataloader.dataset.im_files)

    model = YOLO(str(ckpt))
    model.add_callback("on_val_batch_start", on_val_batch_start)
    model.add_callback("on_val_end", on_val_end)
    m = model.val(**kwargs)
    n_img = len(box["im_files"])
    assert len(box["entry_files"]) == len(box["tp"]) and len(set(box["entry_files"])) == len(box["tp"]), "entry attribution failed"
    pos = {f: i for i, f in enumerate(box["im_files"])}
    eidx = [pos[f] for f in box["entry_files"]]  # image index of each stats entry
    img = np.concatenate([np.full(len(c), eidx[i], dtype=np.int32) for i, c in enumerate(box["conf"])])
    tgt_img = np.concatenate([np.full(len(c), eidx[i], dtype=np.int32) for i, c in enumerate(box["target_cls"])])
    tp = np.concatenate(box["tp"], 0) if len(img) else np.zeros((0, 10), bool)
    conf = np.concatenate(box["conf"]) if len(img) else np.zeros(0)
    pred_cls = np.concatenate(box["pred_cls"]) if len(img) else np.zeros(0, np.int64)
    target_cls = np.concatenate(box["target_cls"])
    # Ultralytics' own AP on the captured arrays must equal what the validator just reported
    ap = ap_per_class(tp, conf, pred_cls, target_cls)[5]
    map_now, map50_now = float(ap.mean()), float(ap[:, 0].mean())
    assert abs(map_now - float(m.box.map)) < 1e-12 and abs(map50_now - float(m.box.map50)) < 1e-12, "capture incomplete"
    stems = np.array([Path(f).stem for f in box["im_files"]])
    np.savez_compressed(npz, tp=tp, conf=conf, pred_cls=pred_cls, img=img, target_cls=target_cls, target_img=tgt_img, stems=stems)
    done["%s__%s" % (key, split)] = {"ckpt": str(ckpt), "split": split, "kwargs": {k: v for k, v in kwargs.items() if k not in ("project", "name")},
                                      "n_images": n_img, "n_dets": int(len(conf)), "map50_95": map_now, "map50": map50_now,
                                      "recorded_map50_95": rec_map, "recorded_map50": rec_map50,
                                      "diff_from_record": map_now - rec_map}
    index.write_text(json.dumps(done, indent=1), encoding="utf-8")
    log("%s %s/%s: %d images, %d dets, mAP50-95 %.6f (record %.6f, diff %+.2e)" % (dataset, key, split, n_img, len(conf), map_now, rec_map, map_now - rec_map))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    assert which in ("antiuav", "dronevehicle"), "usage: capture_stats.py antiuav|dronevehicle"
    os.chdir(r"<YOLO11_ROOT>")
    (run_antiuav if which == "antiuav" else run_dronevehicle)()
    log("%s capture complete" % which)
