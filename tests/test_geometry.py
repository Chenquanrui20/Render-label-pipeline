"""Synthetic self-tests: triangulation is exact, the selection rules behave,
and the full triangulate->project stage writes correct labels.

    python -m unittest discover -s tests -v
"""
import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from renderlabel import geometry as G
from renderlabel import select_views, triangulate_project


def ring_views(n=8, radius=3.0, height=0.0, f=800.0, W=640, H=480):
    """n cameras on a circle around the origin, looking at the origin."""
    out = []
    up = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        a = 2 * np.pi * i / n
        C = np.array([radius * np.cos(a), radius * np.sin(a), height])
        z = -C / np.linalg.norm(C)                # camera looks along +z at the origin
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])                   # world -> camera
        t = -R @ C
        out.append(dict(image="%03d" % i, render_file="%03d.png" % i,
                        R=R, t=t, C=C, f=f, cx=W / 2.0, cy=H / 2.0, k1=0.0,
                        W=W, H=H))
    return out


def write_message(path, v):
    d = {"image_name": v["image"], "render_file": v["render_file"],
         "width": v["W"], "height": v["H"],
         "fx": v["f"], "fy": v["f"], "cx": v["cx"], "cy": v["cy"], "k1": 0.0,
         "R_cam2world": np.asarray(v["R"]).T.tolist(),
         "R_world2cam": np.asarray(v["R"]).tolist(),
         "T_world2cam": np.asarray(v["t"]).tolist(),
         "camera_center_world": np.asarray(v["C"]).tolist()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.views = ring_views()
        self.X = np.array([[0.10, -0.05, 0.02],
                           [-0.08, 0.12, -0.04],
                           [0.03, 0.02, 0.15]])

    def test_projection_roundtrip(self):
        v = self.views[0]
        uv, ok = G.project_points(self.X, v)
        self.assertTrue(ok.all())
        # back-project a ray and check it passes through X
        xn = G.undistort_norm(uv[0:1], v["f"], v["cx"], v["cy"], 0.0)[0]
        ray_cam = np.array([xn[0], xn[1], 1.0])
        ray_world = v["R"].T @ ray_cam
        ray_world /= np.linalg.norm(ray_world)
        origin = v["C"]
        d = np.cross(self.X[0] - origin, ray_world)
        self.assertLess(np.linalg.norm(d), 1e-9)

    def test_dlt_exact(self):
        X = np.array([0.1, -0.05, 0.02])
        idx = [0, 2, 4]
        clicks = []
        for k in idx:
            uv, _ = G.project_points(X.reshape(1, 3), self.views[k])
            clicks.append((self.views[k], uv[0]))
        est, used, res = G.triangulate_point(clicks)
        self.assertEqual(len(used), 3)
        self.assertLess(np.linalg.norm(est - X), 1e-9)
        self.assertLess(res.max(), 1e-7)

    def test_robust_rejection(self):
        # 4 clicks (3 good + 1 gross mis-click).  With only 3 clicks the problem
        # is ambiguous (any pair explains a point exactly), hence 4.
        X = np.array([0.1, -0.05, 0.02])
        idx = [0, 2, 4, 6]
        clicks = []
        for k in idx:
            uv, _ = G.project_points(X.reshape(1, 3), self.views[k])
            clicks.append((self.views[k], uv[0].copy()))
        clicks[1] = (clicks[1][0], clicks[1][1] + np.array([120.0, -90.0]))
        est, used, res = G.triangulate_point(clicks)
        self.assertEqual(len(used), 3)
        self.assertLess(np.linalg.norm(est - X), 1e-5)

    def test_farthest_pair_and_theta(self):
        X0 = np.zeros(3)
        dirs = G.ray_dirs(self.views, X0)
        pair, ang = G.farthest_pair(dirs, list(range(len(self.views))))
        # 8 cameras on a circle -> opposite pair, 180 deg
        self.assertAlmostEqual(ang, 180.0, places=6)
        th = G.theta_span(dirs, list(range(len(self.views))))
        self.assertAlmostEqual(th, 180.0, places=6)

    def test_n_sat(self):
        self.assertEqual(G.n_sat(180.0, 11.5), 8)     # 7.83 -> 8
        self.assertEqual(G.n_sat(74.94, 11.5), 4)     # 3.26 -> 4
        self.assertEqual(G.n_sat(0.0, 11.5), 1)

    def test_greedy_guard(self):
        dirs = G.ray_dirs(self.views, np.zeros(3))
        cands = list(range(len(self.views)))
        sel, tmax = G.greedy_ang(dirs, 6, cands, guard=30.0)
        self.assertEqual(len(sel), 6)
        a = G.angle_matrix(dirs, sel, sel)
        off = a[~np.eye(len(sel), dtype=bool)]
        self.assertGreaterEqual(off.min(), 30.0 - 1e-9)
        self.assertTrue(np.isfinite(tmax))

    def test_greedy_truncates_on_guard(self):
        # all views within 10 deg of each other -> the 40 deg guard stops it early
        dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (5, 1))
        dirs += np.linspace(0, 0.15, 5)[:, None] * np.array([[1.0, 0.0, 0.0]])
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        sel, _ = G.greedy_ang(dirs, 5, list(range(5)), guard=40.0)
        self.assertLess(len(sel), 5)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.renders = os.path.join(self.root, "renders")
        os.makedirs(os.path.join(self.renders, "images"))
        os.makedirs(os.path.join(self.renders, "message"))
        self.views = ring_views(n=6)
        import cv2
        for v in self.views:
            write_message(os.path.join(self.renders, "message", v["image"] + ".json"), v)
            img = np.zeros((v["H"], v["W"], 3), np.uint8)
            cv2.imwrite(os.path.join(self.renders, "images", v["render_file"]), img)
        self.X = np.array([[0.10, -0.05, 0.02],
                           [-0.08, 0.12, -0.04],
                           [0.03, 0.02, 0.15],
                           [0.0, 0.0, 0.0]])
        self.names = ["P1", "P2", "P3", "P4"]

    def tearDown(self):
        self.tmp.cleanup()

    def _annotate(self, view_idx, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        frames = {}
        for k in view_idx:
            v = self.views[k]
            uv, _ = G.project_points(self.X, v)
            uv = uv + rng.normal(0, noise, uv.shape)
            frames[v["render_file"]] = {n: uv[i].tolist()
                                        for i, n in enumerate(self.names)}
        return {"points": self.names, "frames": frames}

    def test_select_stage(self):
        selp = os.path.join(self.root, "selection.json")
        # stage 1: pose only -> the seed pair (the real cold start)
        a = SimpleNamespace(renders=self.renders, annotations=None, selection=selp,
                            theta_eff=11.5, guard=5.0, margin=1.0, refine=False,
                            seed_max_angle=160.0, points3d=None, out_annot=None)
        select_views.run(a)
        with open(selp) as fh:
            seed_sel = json.load(fh)
        self.assertEqual(seed_sel["stage"], "seeds")
        self.assertLessEqual(seed_sel["seed_ray_angle_deg"], 160.0 + 1e-6)
        f2i = {v["render_file"]: i for i, v in enumerate(self.views)}
        seed_idx = [f2i[f] for f in seed_sel["seed_files"]]
        self.assertEqual(len(seed_idx), 2)

        # stage 2: annotate exactly those two frames, then solve
        ann = self._annotate(seed_idx, noise=0.2)
        annp = os.path.join(self.root, "seed.json")
        with open(annp, "w") as fh:
            json.dump(ann, fh)
        a.annotations = annp
        select_views.run(a)
        with open(selp) as fh:
            sel = json.load(fh)
        self.assertEqual(sel["stage"], "solved")
        self.assertEqual(sel["n_points"], 4)
        self.assertEqual(sel["n_selected"], len(sel["selected_files"]))
        self.assertGreaterEqual(sel["n_selected"], 2)
        # 6 cameras around 360 deg -> Theta_span near 180 -> N_sat = 8, capped by
        # the 6 candidates; the selection must not exceed the candidate count
        self.assertLessEqual(sel["n_selected"], sel["n_candidates"])

    def test_triangulate_project(self):
        ann = self._annotate([0, 2, 4], noise=0.2)
        annp = os.path.join(self.root, "seed.json")
        with open(annp, "w") as fh:
            json.dump(ann, fh)
        out = os.path.join(self.root, "labels")
        over = os.path.join(self.root, "overlay")
        a = SimpleNamespace(renders=self.renders, annotations=[annp], out=out,
                            overlay=over, refine=False, min_views=2)
        triangulate_project.run(a)
        labs = [f for f in os.listdir(out) if f.endswith(".json") and f != "summary.json"]
        self.assertEqual(len(labs), 6)
        with open(os.path.join(out, "summary.json")) as fh:
            s = json.load(fh)
        self.assertEqual(s["n_points"], 4)
        self.assertEqual(s["failed_points"], [])
        # accuracy
        with open(os.path.join(out, "points3d.csv"), encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        est = np.array([[float(r["X"]), float(r["Y"]), float(r["Z"])] for r in rows])
        # rows are in `keep` order = names order P1..P4
        err = np.linalg.norm(est - self.X, axis=1)
        self.assertLess(err.max(), 0.01)
        # labelme structure
        with open(os.path.join(out, "000.json")) as fh:
            lab = json.load(fh)
        self.assertEqual(lab["imageWidth"], 640)
        self.assertEqual(len(lab["shapes"]), 4)
        self.assertEqual(lab["shapes"][0]["shape_type"], "point")


if __name__ == "__main__":
    unittest.main()
