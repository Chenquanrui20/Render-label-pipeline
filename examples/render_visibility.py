#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per (frame, point) visibility from an EWA z-test against the 3DGS Gaussians.

This is an optional helper for the demo video (`make_demo_video.py
--visibility`).  The pipeline itself does not need it: labels are projected
into every render regardless of occlusion.

Why not the renderer's depth output: the gaussian-splatting fork's
`expected_invdepth` is an alpha-weighted expected *inverse* depth accumulated
over the whole ray, not the first surface; on the demo scenes it reads ~0.5x
the true depth even at pixels whose visibility is proven by a click.

Why not a circle of radius 2*max_scale: elongated Gaussians then "cover"
pixels they never reach and 2-5% closer surface points get misread as
occluders.  Each Gaussian is projected with its real 2D covariance (EWA:
J W Sigma W^T J^T) and the pixel must lie inside 3 sigma.

A point is occluded when an opaque Gaussian (opacity > --min-opacity) covers
the point's pixel and is clearly in front:

    z_g < z_pt * (1 - tau)

Writes {frame_stem: {name: {"vis": bool, "z": .., "z_front": .., "ratio": ..}}}.

    render_visibility.py <scene> [--renders ...] [--points3d ...] \
        --out work/visibility.json [--tau 0.02] \
        [--annotations work/seed.json work/selected.json]

The optional --annotations prints a calibration line: clicks are ground truth
for visibility, so the fraction of clicked views the test calls visible is a
direct sanity check of the threshold.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


def read_ply(path):
    """positions, opacity logits, scales log, rotations (w,x,y,z) from a 3DGS PLY."""
    with open(path, "rb") as fh:
        head = b""
        while b"end_header" not in head:
            head += fh.read(1)
        head += fh.read(1)
        n, names = None, []
        for line in head.decode("ascii", "replace").splitlines():
            p = line.split()
            if len(p) == 3 and p[0] == "element" and p[1] == "vertex":
                n = int(p[2])
            if len(p) == 3 and p[0] == "property" and p[1] == "float":
                names.append(p[2])
        assert n, "no vertex element"
        data = np.frombuffer(fh.read(n * 4 * len(names)), dtype=np.float32)
        data = data.reshape(n, len(names))
    idx = {k: i for i, k in enumerate(names)}
    xyz = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float64)
    op = data[:, idx["opacity"]].astype(np.float64)
    sc = data[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]].astype(np.float64)
    rot = data[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]].astype(np.float64)
    return xyz, op, sc, rot


def quat_to_cov(rot, sc):
    """3x3 covariance per Gaussian, Sigma = R diag(s^2) R^T."""
    q = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    S2 = np.exp(sc * 2.0)
    RS = R * S2[:, None, :]                       # columns scaled
    return RS @ np.transpose(R, (0, 2, 1))


def read_points(path):
    names, X = [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            names.append(r["name"])
            X.append([float(r["X"]), float(r["Y"]), float(r["Z"])])
    return names, np.asarray(X, float)


def project_frame(gauss, msg):
    """Project all kept Gaussians into one frame with their 2D covariances."""
    fx, fy = float(msg["fx"]), float(msg["fy"])
    cx, cy = float(msg["cx"]), float(msg["cy"])
    R = np.asarray(msg["R_world2cam"], float)
    C = np.asarray(msg["camera_center_world"], float)
    Xc = (gauss["xyz"] - C) @ R.T
    z = Xc[:, 2]
    ok = z > 1e-6
    Xc, z = Xc[ok], z[ok]
    cov = np.einsum("ij,njk,lk->nil", R, gauss["cov"][ok], R)   # R Sigma R^T
    x, y = Xc[:, 0], Xc[:, 1]
    J = np.zeros((len(z), 2, 3))
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / (z * z)
    cov2 = np.einsum("nij,njk,nlk->nil", J, cov, J)
    cov2[:, 0, 0] += 0.3          # low-pass, as in the rasterizer
    cov2[:, 1, 1] += 0.3
    uu = fx * x / z + cx
    vv = fy * y / z + cy
    det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
    lam = 0.5 * (cov2[:, 0, 0] + cov2[:, 1, 1])
    rad = 3.0 * np.sqrt(lam + np.sqrt(np.maximum(0.0, lam ** 2 - det)))
    return {"uu": uu, "vv": vv, "z": z, "cov2": cov2,
            "rad": np.maximum(rad, 1.0)}


def front_at(proj, u, v, W, H):
    """Nearest opaque Gaussian whose 3-sigma ellipse covers (u,v)."""
    uu, vv = proj["uu"], proj["vv"]
    near = ((uu - u) ** 2 + (vv - v) ** 2) <= proj["rad"] ** 2
    if not near.any():
        return np.inf, 0
    dx, dy = uu[near] - u, vv[near] - v
    cov2 = proj["cov2"][near]
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    d2 = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / np.maximum(
        a * c - b * b, 1e-12)
    hit = (d2 <= 9.0) & (uu[near] > -1) & (uu[near] < W + 1) & \
          (vv[near] > -1) & (vv[near] < H + 1)
    if not hit.any():
        return np.inf, 0
    zz = proj["z"][near][hit]
    return float(zz.min()), int(hit.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--subdir", default="3dgs",
                    help="subdirectory holding output/ and renders/ (default 3dgs)")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--renders", default=None,
                    help="render directory (default <scene>/<subdir>/renders)")
    ap.add_argument("--points3d", default=None,
                    help="points3d.csv (default <scene>/<subdir>/labels/points3d.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.02,
                    help="occluded when z_front < z_pt*(1-tau)")
    ap.add_argument("--min-opacity", type=float, default=0.5)
    ap.add_argument("--annotations", nargs="*", default=[],
                    help="annotation JSONs: report click calibration")
    ap.add_argument("--frame-list", nargs="*", default=None,
                    help="restrict to these render files")
    a = ap.parse_args()

    scene = os.path.abspath(a.scene)
    sub = a.subdir
    renders = os.path.abspath(a.renders) if a.renders else os.path.join(scene, sub, "renders")
    points = (os.path.abspath(a.points3d) if a.points3d
              else os.path.join(scene, sub, "labels", "points3d.csv"))
    ply = os.path.join(scene, sub, "output", "point_cloud",
                       "iteration_%d" % a.iteration, "point_cloud.ply")
    xyz, op, sc, rot = read_ply(ply)
    keep = op > np.log(a.min_opacity / (1.0 - a.min_opacity))
    gauss = {"xyz": xyz[keep], "cov": quat_to_cov(rot[keep], sc[keep])}
    print("gaussians %d  kept (opacity>%.2f): %d"
          % (len(xyz), a.min_opacity, int(keep.sum())), flush=True)

    names, X = read_points(points)
    msgs = sorted(glob.glob(os.path.join(renders, "message", "*.json")))
    msgs = [m for m in msgs if not os.path.basename(m).startswith("_")]
    if a.frame_list:
        keepf = set(a.frame_list)
        msgs = [m for m in msgs
                if json.load(open(m, encoding="utf-8"))["render_file"] in keepf]

    clicks = {}
    for p in a.annotations:
        with open(p, encoding="utf-8") as fh:
            for img, pts in json.load(fh).get("frames", {}).items():
                clicks.setdefault(os.path.splitext(img)[0], {}).update(pts)

    vis, ratios_all, ratios_click = {}, [], []
    n_vis = n_tot = 0
    for fn in msgs:
        with open(fn, encoding="utf-8") as fh:
            m = json.load(fh)
        stem = os.path.splitext(m["render_file"])[0]
        fx, fy, cx, cy = m["fx"], m["fy"], m["cx"], m["cy"]
        W, H = int(m["width"]), int(m["height"])
        R = np.asarray(m["R_world2cam"], float)
        C = np.asarray(m["camera_center_world"], float)
        proj = project_frame(gauss, m)
        fv = {}
        for i, n in enumerate(names):
            Xc = R @ (X[i] - C)
            z = float(Xc[2])
            if z <= 1e-9:
                fv[n] = {"vis": False, "z": z, "z_front": None, "ratio": None,
                         "reason": "behind"}
                n_tot += 1
                continue
            u = fx * Xc[0] / z + cx
            v = fy * Xc[1] / z + cy
            if not (0 <= u < W and 0 <= v < H):
                fv[n] = {"vis": False, "z": z, "z_front": None, "ratio": None,
                         "reason": "outside"}
                n_tot += 1
                continue
            zf, nc = front_at(proj, u, v, W, H)
            ratio = (zf / z) if np.isfinite(zf) else None
            blocked = ratio is not None and ratio < 1.0 - a.tau
            fv[n] = {"vis": not blocked, "z": z,
                     "z_front": (None if not np.isfinite(zf) else zf),
                     "ratio": ratio, "reason": "", "n_front": nc}
            n_tot += 1
            n_vis += 0 if blocked else 1
            if ratio is not None:
                ratios_all.append(ratio)
                if stem in clicks and n in clicks[stem]:
                    ratios_click.append(ratio)
        vis[stem] = fv
        if len(vis) % 50 == 0:
            print("  %d/%d frames" % (len(vis), len(msgs)), flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(vis, fh, ensure_ascii=False, indent=1)
    r = np.asarray(ratios_all)
    print("frames %d  points %d  visible %.1f%%"
          % (len(vis), len(names), 100.0 * n_vis / max(n_tot, 1)))
    if r.size:
        print("z_front/z_pt (all): p1 %.3f  p5 %.3f  median %.3f"
              % (np.percentile(r, 1), np.percentile(r, 5), np.median(r)))
    rc = np.asarray(ratios_click)
    if rc.size:
        print("clicked views (n=%d): min %.4f  median %.4f  max %.4f  visible %d/%d"
              % (rc.size, rc.min(), np.median(rc), rc.max(),
                 int((rc >= 1.0 - a.tau).sum()), rc.size))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
