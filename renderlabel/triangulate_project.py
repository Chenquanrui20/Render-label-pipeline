"""Triangulate the annotated points and project them into every rendered frame.

    python -m renderlabel triangulate <scene> --annotations work/seed.json work/selected.json

For each point the estimator is multi-view DLT (optionally LM-refined) over every
frame in which it was clicked, with gross mis-clicks rejected by a pair-hypothesis
inlier vote (>= 4 clicks).  The resulting 3D points are then projected into
all rendered frames, producing labelme-style JSON (one point shape per keypoint
that projects inside the frame), QC overlays, and a summary with per-point
residuals and frame coverage.
"""
import csv
import json
import os

import cv2
import numpy as np

from . import geometry as G
from . import views as V


def _merge_annotations(paths):
    """Merge annotation files; later files win on a frame/point collision."""
    names, frames = [], {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        for n in d.get("points", []):
            if n not in names:
                names.append(n)
        for f, fa in d.get("frames", {}).items():
            frames.setdefault(f, {}).update(fa)
    return names, frames


def _labelme(names, uv, view, image_rel, mask):
    return {
        "version": "5.0.0",
        "flags": {},
        "shapes": [{
            "label": n,
            "points": [[float(uv[k, 0]), float(uv[k, 1])]],
            "group_id": None,
            "shape_type": "point",
            "flags": {},
        } for k, n in enumerate(names) if mask[k]],
        "imagePath": image_rel,
        "imageData": None,
        "imageHeight": view["H"],
        "imageWidth": view["W"],
    }


def run(a):
    views = V.load_views(a.renders)
    by_file = V.index_by_file(views)
    names, frames = _merge_annotations(a.annotations)
    if not frames:
        raise SystemExit("no annotated frames in %s" % ", ".join(a.annotations))

    # ---- triangulate every annotated point
    X, used_views, resid, failed = {}, {}, {}, []
    for n in names:
        clicks = [(views[by_file[f]], fa[n]) for f, fa in frames.items()
                  if n in fa and f in by_file]
        p, used, res = G.triangulate_point(clicks, refine=a.refine)
        if p is None or len(used) < a.min_views:
            failed.append(n)
            continue
        X[n] = p
        used_views[n] = len(used)
        resid[n] = res
    if not X:
        raise SystemExit("no point has enough views (min_views=%d)" % a.min_views)
    keep = [n for n in names if n in X]
    Xa = np.array([X[n] for n in keep])

    # ---- project into every frame
    os.makedirs(a.out, exist_ok=True)
    if a.overlay:
        os.makedirs(a.overlay, exist_ok=True)
    img_dir = V.images_dir(a.renders)
    coverage = {n: 0 for n in keep}
    n_written = 0
    for k, v in enumerate(views):
        uv, ok = G.project_points(Xa, v)
        # a point is a label only when it actually falls inside this frame
        inside = (ok & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) &
                  (uv[:, 0] >= 0) & (uv[:, 0] <= v["W"] - 1) &
                  (uv[:, 1] >= 0) & (uv[:, 1] <= v["H"] - 1))
        rel = os.path.relpath(os.path.join(img_dir, v["render_file"]),
                              os.path.abspath(a.out))
        for i, n in enumerate(keep):
            if inside[i]:
                coverage[n] += 1
        lab = _labelme(keep, uv, v, rel, inside)
        stem = os.path.splitext(v["render_file"])[0]
        with open(os.path.join(a.out, stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump(lab, fh, ensure_ascii=False, indent=1)
        n_written += 1
        if a.overlay:
            img = cv2.imread(os.path.join(img_dir, v["render_file"]))
            if img is not None:
                for i, n in enumerate(keep):
                    if not inside[i]:
                        continue
                    x, y = int(round(uv[i, 0])), int(round(uv[i, 1]))
                    cv2.circle(img, (x, y), 4, (0, 220, 0), -1, cv2.LINE_AA)
                    cv2.putText(img, n, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 220, 0), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(a.overlay, stem + ".png"), img)

    # ---- reports
    with open(os.path.join(a.out, "points3d.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "X", "Y", "Z", "n_views",
                    "resid_median_px", "resid_mean_px", "resid_max_px"])
        for n in keep:
            r = resid[n]
            w.writerow([n] + ["%.6f" % x for x in X[n]] + [used_views[n],
                        "%.4f" % float(np.median(r)), "%.4f" % float(np.mean(r)),
                        "%.4f" % float(np.max(r))])
    ann_res = np.concatenate([resid[n] for n in keep if len(resid[n])])
    summary = dict(
        renders=a.renders, labels=a.out,
        n_frames=n_written, n_points=len(keep),
        points=keep, failed_points=failed,
        annotated_frames=len(frames),
        annotated_views_used={n: used_views[n] for n in keep},
        coverage_frames=coverage,
        coverage_frac={n: coverage[n] / float(n_written) for n in keep},
        resid_px=dict(median=float(np.median(ann_res)),
                      mean=float(np.mean(ann_res)),
                      max=float(np.max(ann_res))),
        refine=bool(a.refine), min_views=a.min_views)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("points: %d ok, %d failed%s" % (len(keep), len(failed),
          (" (" + ", ".join(failed) + ")") if failed else ""))
    print("annotated-view residuals: median %.3f px / mean %.3f / max %.3f"
          % (summary["resid_px"]["median"], summary["resid_px"]["mean"],
             summary["resid_px"]["max"]))
    print("frames labelled: %d -> %s" % (n_written, a.out))
    if a.overlay:
        print("overlays -> %s" % a.overlay)
    return 0
