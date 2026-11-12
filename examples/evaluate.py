"""Score a demo run against the dataset's own labels and ground-truth 3D points.

Two independent questions, both answered in the render pixel space:

1. **Held-out 2D accuracy.**  The pipeline triangulates the keypoints from the
   annotated frames only.  Projecting those 3D points into every *other* frame
   and comparing with that frame's label gives a 2D error that no annotated click
   has seen.  The same measurement is repeated with random view sets of the same
   budget, so the demo shows the selection rule, not just the estimator.

2. **3D shape.**  Where ground-truth 3D coordinates exist (ChArUco board
   coordinates in mm, YCBV `points_summary.json` in mm), the triangulated points
   are compared after a Sim(3) fit.  The SfM world is scale-free, so only the
   *shape* is checkable; leave-one-out fits make the residual honest (the point
   being scored is never part of the fit).

    python3 examples/evaluate.py --format charuco \
        --renders /data/scene/3dgs/renders --labels /data/scene/3dgs/renders/labels \
        --selection work/selection.json --points3d work/labels/points3d.csv \
        --trials 100 --out work/eval.json
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels_io  # noqa: E402
from renderlabel import geometry as G  # noqa: E402
from renderlabel import views as V  # noqa: E402


def _stats(errs):
    e = np.asarray(errs, float)
    if not len(e):
        return {"n_obs": 0}
    return {"n_obs": int(len(e)),
            "median": float(np.median(e)),
            "mean": float(np.mean(e)),
            "p90": float(np.percentile(e, 90)),
            "rms": float(np.sqrt(np.mean(e ** 2)))}


def _read_points3d(path, min_views=2):
    pts = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if int(r.get("n_views", min_views)) < min_views:
                continue
            pts[r["name"]] = np.array([float(r["X"]), float(r["Y"]), float(r["Z"])])
    return pts


def _stem(view):
    return os.path.splitext(view["image"])[0]


def _eval_projection(X, views, labels, exclude_stems):
    """Project X (name -> 3D) into every labelled, non-excluded frame."""
    per_point = defaultdict(list)
    for v in views:
        stem = _stem(v)
        if stem in exclude_stems or stem not in labels:
            continue
        lab = labels[stem]
        names = [n for n in X if n in lab]
        if not names:
            continue
        uv, ok = G.project_points(np.array([X[n] for n in names]), v)
        for i, n in enumerate(names):
            if not ok[i]:
                continue
            u, w = uv[i]
            if 0.0 <= u <= v["W"] - 1 and 0.0 <= w <= v["H"] - 1:
                per_point[n].append(math.hypot(u - lab[n][0], w - lab[n][1]))
    pooled = [e for es in per_point.values() for e in es]
    return pooled, per_point


def _umeyama_sim3(A, B):
    """(s, R, t) mapping A (N,3) onto B (N,3) least-squares."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n = len(A)
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    C = B0.T @ A0 / n
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_a = (A0 ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_a)
    t = mu_b - s * R @ mu_a
    return s, R, t


def _align_resid(A, B):
    s, R, t = _umeyama_sim3(A, B)
    return np.linalg.norm(s * (np.asarray(A, float) @ R.T) + t - np.asarray(B, float),
                          axis=1), s


def _gt3d_block(X, gt):
    names = [n for n in X if n in gt]
    gt_all = np.array([gt[n] for n in names], float)
    x_all = np.array([X[n] for n in names], float)
    if len(names) < 4:
        return {"n_points": len(names), "note": "too few points for a Sim(3) fit"}
    res, s = _align_resid(x_all, gt_all)
    loo = []
    for i in range(len(names)):
        keep = [j for j in range(len(names)) if j != i]
        s_i, R_i, t_i = _umeyama_sim3(x_all[keep], gt_all[keep])
        pred = s_i * R_i @ x_all[i] + t_i
        loo.append(float(np.linalg.norm(pred - gt_all[i])))
    return {"n_points": len(names), "names": names,
            "scale_mm_per_unit": s,
            "full_fit_px_mm": _stats(res),
            "loo_mm": _stats(loo)}


def _candidates(sel, views, points3d_seed):
    """The candidate frames used by the selection stage (pose + model only)."""
    with open(points3d_seed, encoding="utf-8") as fh:
        pts = json.load(fh).get("points", {})
    X = np.array([pts[n] for n in sorted(pts)])
    margin = float(sel.get("margin_px", 20.0))
    return [k for k, v in enumerate(views)
            if G.inside_image(G.project_points(X, v)[0], v, margin)]


def _sample_clicks(labels, views, sample_stems, X_names):
    """clicks[point] = [(view, (u,v))] for the sampled frames where the label exists."""
    clicks = {n: [] for n in X_names}
    for v in views:
        stem = _stem(v)
        if stem not in sample_stems or stem not in labels:
            continue
        lab = labels[stem]
        for n in X_names:
            if n in lab:
                clicks[n].append((v, lab[n]))
    return clicks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True, choices=("charuco", "ycbv"))
    ap.add_argument("--renders", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--points3d", required=True)
    ap.add_argument("--gt-points", default=None,
                    help="ycbv: points_summary.json; charuco reads the label descriptions")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-views", type=int, default=2,
                    help="minimum clicks for a point (applied to ours and to every trial)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    views = V.load_views(a.renders)
    labels = labels_io.load(a.format, a.labels)
    X = _read_points3d(a.points3d, a.min_views)
    with open(a.selection, encoding="utf-8") as fh:
        sel = json.load(fh)
    ann_files = list(sel.get("seed_files") or []) + list(sel.get("selected_files") or [])
    ann_stems = {os.path.splitext(f)[0] for f in ann_files}
    if not ann_stems:
        raise SystemExit("%s has no seed_files/selected_files" % a.selection)

    ours_pooled, ours_per_point = _eval_projection(X, views, labels, ann_stems)
    if not ours_pooled:
        raise SystemExit("no held-out observation: check --labels / --selection")
    out = {
        "dataset": a.format,
        "renders": os.path.abspath(a.renders),
        "selection": os.path.abspath(a.selection),
        "points3d": os.path.abspath(a.points3d),
        "n_annotated_frames": len(ann_stems),
        "n_points": len(X),
        "min_views": int(a.min_views),
        "ours": _stats(ours_pooled),
    }

    # ---------------------------------------------------------- random baseline
    pts3d_seed = sel.get("points3d")
    if pts3d_seed and os.path.isfile(pts3d_seed):
        cands = _candidates(sel, views, pts3d_seed)
    else:
        cands = [k for k, v in enumerate(views)
                 if _stem(v) in labels]
    n_views = len(ann_stems)
    rng = np.random.default_rng(a.seed)
    names = sorted(X)
    trial_med, trial_mean, trial_ok = [], [], []
    for _ in range(a.trials):
        pick = rng.choice(len(cands), size=min(n_views, len(cands)), replace=False)
        stems = {_stem(views[cands[i]]) for i in pick}
        clicks = _sample_clicks(labels, views, stems, names)
        Xr = {}
        for n in names:
            p, used, _res = G.triangulate_point(clicks[n])
            if p is not None and len(used) >= a.min_views:
                Xr[n] = p
        trial_ok.append(len(Xr))
        pooled, _ = _eval_projection(Xr, views, labels, stems)
        st = _stats(pooled)
        if st["n_obs"]:
            trial_med.append(st["median"])
            trial_mean.append(st["mean"])
    if trial_med:
        tm, tmean = np.array(trial_med), np.array(trial_mean)
        diff_med = float(out["ours"]["median"]) - tm
        diff_mean = float(out["ours"]["mean"]) - tmean
        out["random"] = {
            "trials": a.trials,
            "n_views": n_views,
            "n_candidates": len(cands),
            "seed": a.seed,
            "points_ok": {"median": float(np.median(trial_ok)),
                          "min": int(np.min(trial_ok)), "max": int(np.max(trial_ok))},
            "trial_median": {"median": float(np.median(tm)), "mean": float(np.mean(tm))},
            "trial_mean": {"median": float(np.median(tmean)),
                           "mean": float(np.mean(tmean))},
            "win_rate_median": float(np.mean(tm > out["ours"]["median"])),
            "win_rate_mean": float(np.mean(tmean > out["ours"]["mean"])),
            "diff_median": {"median": float(np.median(diff_med)),
                            "mean": float(np.mean(diff_med))},
            "diff_mean": {"median": float(np.median(diff_mean)),
                          "mean": float(np.mean(diff_mean))},
        }
    else:
        out["random"] = {"trials": a.trials, "note": "no trial produced an observation"}

    out["per_point"] = {n: _stats(v) for n, v in sorted(ours_per_point.items())}

    # --------------------------------------------------------------- 3D check
    if a.format == "charuco":
        gt = labels_io.charuco_board_points(a.labels)
    else:
        if not a.gt_points:
            raise SystemExit("ycbv needs --gt-points points_summary.json")
        gt = labels_io.ycbv_gt_points(a.gt_points)
    out["gt3d"] = _gt3d_block(X, gt)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    o, r = out["ours"], out.get("random", {})
    print("held-out labels: %d points, %d observations" % (len(X), o["n_obs"]))
    print("  ours    median %.3f px / mean %.3f / p90 %.3f"
          % (o["median"], o["mean"], o["p90"]))
    if "trial_median" in r:
        print("  random  %d trials with %d views: trial median %.3f px (mean %.3f), "
              "trial mean %.3f px (mean %.3f)"
              % (r["trials"], r["n_views"], r["trial_median"]["median"],
                 r["trial_median"]["mean"], r["trial_mean"]["median"],
                 r["trial_mean"]["mean"]))
        print("  win rate (median) %.0f%%   paired diff median %.3f px "
              "(median over trials) / %.3f px (mean over trials)"
              % (100 * r["win_rate_median"], r["diff_median"]["median"],
                 r["diff_median"]["mean"]))
    g = out["gt3d"]
    if "loo_mm" in g:
        print("3D vs GT (%d points, Sim(3) scale %.4f mm/unit): "
              "LOO median %.3f mm / mean %.3f mm"
              % (g["n_points"], g["scale_mm_per_unit"], g["loo_mm"]["median"],
                 g["loo_mm"]["mean"]))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
