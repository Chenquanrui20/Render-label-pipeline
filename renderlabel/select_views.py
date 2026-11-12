"""Predict the saturated view count and select the frames to annotate.

Two-step cold start:

    # 1) pose only -- pick the two seed frames (no observations needed)
    python -m renderlabel select <scene> --selection work/selection.json

    # 2) after annotating those two frames, triangulate a coarse X0, compute
    #    Theta_span, N_sat and greedily pick N frames (5 deg guard)
    python -m renderlabel select <scene> --annotations work/seed.json \
        --selection work/selection.json

The rule is the one validated in the paper:

    N_sat = ceil( Theta_span / (2 * theta_eff) )

where Theta_span is the largest pairwise angle between the ray directions from
the object to the *candidate* cameras (cameras that see every seed point inside
the image with a margin), and theta_eff is the per-view angular gain (11.5 deg is
the value fitted on the paper's seven sequences; 0000's own value is 5.9 deg).
"""
import json
import os

import numpy as np

from . import geometry as G
from . import views as V


def _load_ann(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return d.get("frames", {})


def _scene_of(renders):
    """Best guess of the scene dir for the printed hint: <scene>/3dgs/renders."""
    return os.path.dirname(os.path.dirname(os.path.abspath(renders)))


def _triangulate_annotations(ann, views, by_file, refine=False):
    """name -> (X, n_views, [residuals])."""
    names = sorted({n for fa in ann.values() for n in fa})
    out = {}
    for n in names:
        clicks = []
        for f, fa in ann.items():
            if n in fa and f in by_file:
                clicks.append((views[by_file[f]], fa[n]))
        X, used, res = G.triangulate_point(clicks, refine=refine)
        if X is not None:
            out[n] = (X, len(used), res)
    return out


def run(a):
    views = V.load_views(a.renders)
    by_file = V.index_by_file(views)
    theta_eff = float(a.theta_eff)
    guard = float(a.guard)
    margin = float(a.margin)
    os.makedirs(os.path.dirname(os.path.abspath(a.selection)), exist_ok=True)

    # ---------------------------------------------------------- seed stage
    if not a.annotations:
        if len(views) < 2:
            raise SystemExit("need at least 2 rendered frames for a seed pair, found %d"
                             % len(views))
        Q = G.pose_proxy_point(views)
        dirs = G.ray_dirs(views, Q)
        all_idx = list(range(len(views)))
        seed, ang = G.farthest_pair(dirs, all_idx)
        cap = float(getattr(a, "seed_max_angle", 160.0))
        if ang > cap:
            # A ~180 deg pair puts the object on the baseline: depth unobservable.
            seed, ang = G.pair_near_angle(dirs, all_idx, cap)
        payload = dict(stage="seeds", theta_eff_deg=theta_eff, guard_deg=guard,
                       proxy_point=Q.tolist(), seed_idx=seed,
                       seed_files=[views[k]["render_file"] for k in seed],
                       seed_ray_angle_deg=ang)
        with open(a.selection, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print("pose-only proxy X0 = [%.4f %.4f %.4f]" % tuple(Q))
        print("seed pair = %s, ray angle %.2f deg"
              % (payload["seed_files"], ang))
        print("wrote", a.selection)
        print("\nnext: annotate these two frames ->\n"
              "  python -m renderlabel annotate %s --renders %s --frames seed "
              "--selection %s --points-file points.txt --out %s"
              % (_scene_of(a.renders), a.renders, a.selection,
                 a.out_annot or "work/seed.json"))
        return 0

    # --------------------------------------------------------- solve stage
    ann = _load_ann(a.annotations)
    if not ann:
        raise SystemExit("%s has no annotated frames" % a.annotations)
    seed_idx = sorted({by_file[f] for f in ann if f in by_file})
    tri = _triangulate_annotations(ann, views, by_file, refine=a.refine)
    if len(tri) < 1:
        raise SystemExit("no point could be triangulated from the seed annotations")
    names = sorted(tri)
    X = np.array([tri[n][0] for n in names])
    X0 = X.mean(0)
    for n in names:
        print("  %-8s  views %2d  median residual %.3f px"
              % (n, tri[n][1], float(np.median(tri[n][2])) if len(tri[n][2]) else float("nan")))

    # candidate set: every seed point lands inside the image with a margin
    cands = []
    for k, v in enumerate(views):
        uv, ok = G.project_points(X, v)
        if ok.all() and G.inside_image(uv, v, margin):
            cands.append(k)
    if len(cands) < 2:
        raise SystemExit("fewer than 2 candidate frames see all points with a %.0f px margin" % margin)

    dirs = G.ray_dirs(views, X0)
    th = G.theta_span(dirs, cands)
    n_raw = th / (2.0 * theta_eff)
    n_sat = G.n_sat(th, theta_eff)
    start = [k for k in seed_idx if k in cands]
    sel, tmax = G.greedy_ang(dirs, max(n_sat, len(start)), cands, guard=guard, start=start)

    # persist the seed model for the annotation GUI guides
    p3d = a.points3d or os.path.join(os.path.dirname(os.path.abspath(a.selection)),
                                     "points3d_seed.json")
    with open(p3d, "w", encoding="utf-8") as fh:
        json.dump({"points": {n: [float(x) for x in tri[n][0]] for n in names},
                   "source": a.annotations,
                   "n_views": {n: tri[n][1] for n in names}}, fh, ensure_ascii=False, indent=2)

    payload = dict(stage="solved", theta_eff_deg=theta_eff, guard_deg=guard,
                   margin_px=margin, x0=X0.tolist(),
                   n_points=len(names), points=names,
                   n_candidates=len(cands), theta_span_deg=th,
                   n_raw=n_raw, n_sat=n_sat, n_selected=len(sel),
                   theta_max_deg=tmax, seed_idx=seed_idx,
                   seed_files=[views[k]["render_file"] for k in seed_idx],
                   selected_idx=sel,
                   selected_files=[views[k]["render_file"] for k in sel],
                   points3d=p3d)
    with open(a.selection, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("X0 (centroid of %d points) = [%.4f %.4f %.4f]" % (len(names), *X0))
    print("candidate frames = %d/%d (all points inside, margin %.0f px)"
          % (len(cands), len(views), margin))
    print("Theta_span = %.2f deg -> N_sat = ceil(%.2f / (2*%.2f)) = %d"
          % (th, th, theta_eff, n_sat))
    print("greedy ang selection (guard %.1f deg): %d frames, theta_max %.2f deg"
          % (guard, len(sel), tmax))
    print("selected:", payload["selected_files"])
    print("wrote", a.selection, "and", p3d)
    print("\nnext: annotate the selected frames ->\n"
          "  python -m renderlabel annotate %s --renders %s --frames selected "
          "--selection %s --points-file points.txt --init %s --out work/selected.json"
          % (_scene_of(a.renders), a.renders, a.selection, p3d))
    return 0
