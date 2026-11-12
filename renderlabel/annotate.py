"""Interactive point annotation on rendered frames (OpenCV window).

Usage (inside a desktop session):

    python -m renderlabel annotate <scene> --frames seed  --points-file points.txt \
        --selection work/selection.json --out work/seed.json
    python -m renderlabel annotate <scene> --frames selected --points-file points.txt \
        --selection work/selection.json --init work/points3d_seed.json \
        --out work/selected.json

Keys
    left click   place the current point at the cursor (auto-advances)
    1..9         jump to that point index
    n / p        next / previous frame (saves first)
    a            adopt the projected guide positions for all missing points
    u            undo the last click on this frame
    r            clear this frame
    s            save now
    q / Esc      save and quit

Guides: when --init is given (the seed-triangulated 3D points), their projection
is drawn as a small ring.  Clicking near a guide still records the real click;
`a` copies all guide positions at once, which is the fast path when the model is
already good.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

from . import geometry as G
from . import views as V

WIN = "renderlabel annotate"


def load_points_file(path):
    names = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    if not names:
        raise SystemExit("empty points file: %s" % path)
    return names


def load_annotations(path):
    if not path or not os.path.isfile(path):
        return {"points": [], "frames": {}}
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    d.setdefault("frames", {})
    return d


def save_annotations(path, names, frames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"points": list(names),
               "frames": {k: {n: [float(p[0]), float(p[1])] for n, p in v.items()}
                          for k, v in frames.items()},
               "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)


def _guides(init, names, view):
    """Projected 3D guide positions for this frame (dict name -> (u,v))."""
    if not init:
        return {}
    X, keep = [], []
    for n in names:
        p = init.get(n)
        if p is not None:
            X.append(p)
            keep.append(n)
    if not X:
        return {}
    uv, ok = G.project_points(np.asarray(X, float), view)
    return {n: tuple(uv[i]) for i, n in enumerate(keep) if ok[i]}


class App:
    def __init__(self, views, idx, names, images, out, frames_ann, init, max_width=1600):
        self.views = views
        self.order = idx
        self.names = names
        self.images = images
        self.out = out
        self.ann = frames_ann          # render_file -> {name: [u,v]}
        self.init = init               # name -> [X,Y,Z]
        self.i = 0                     # position in order
        self.pt = 0                    # current point index
        self.scale = 1.0
        self.base = None
        self.small = None
        self.mx = self.my = -1
        self.max_width = max_width

    # ------------------------------------------------------------- rendering
    def cur(self):
        return self.views[self.order[self.i]]

    def frame_ann(self):
        return self.ann.setdefault(self.cur()["render_file"], {})

    def load(self):
        v = self.cur()
        img = cv2.imread(self.images[self.i])
        if img is None:
            raise SystemExit("cannot read %s" % self.images[self.i])
        self.base = img
        h, w = img.shape[:2]
        self.scale = min(1.0, float(self.max_width) / w)
        self.small = cv2.resize(img, (int(w * self.scale), int(h * self.scale)),
                                interpolation=cv2.INTER_AREA) if self.scale < 1.0 else img

    def to_full(self, x, y):
        return x / self.scale, y / self.scale

    def draw(self):
        v = self.cur()
        canvas = self.small.copy()
        s = self.scale
        fa = self.frame_ann()
        guides = _guides(self.init, self.names, v)
        name = v["render_file"]
        for k, n in enumerate(self.names):
            g = guides.get(n)
            if g is not None:
                gx, gy = g[0] * s, g[1] * s
                cv2.circle(canvas, (int(gx), int(gy)), 4, (255, 200, 0), 1, cv2.LINE_AA)
            p = fa.get(n)
            if p is not None:
                x, y = int(p[0] * s), int(p[1] * s)
                col = (0, 220, 0) if k != self.pt else (0, 120, 255)
                cv2.circle(canvas, (x, y), 4, col, -1, cv2.LINE_AA)
                cv2.putText(canvas, n, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1, cv2.LINE_AA)
        n_done = sum(1 for n in self.names if n in fa)
        hud = ("%d/%d  %s   %d/%d points   cur=%s   [n/p] frame [1-9] point "
               "[a]dopt guides [u]ndo [r]eset [s]ave [q]uit"
               % (self.i + 1, len(self.order), name, n_done, len(self.names),
                  self.names[self.pt]))
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (30, 30, 30), -1)
        cv2.putText(canvas, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
        self._magnifier(canvas)
        cv2.imshow(WIN, canvas)

    def _magnifier(self, canvas):
        if self.mx < 0:
            return
        r, z = 30, 6
        x0, y0 = max(0, self.mx - r), max(0, self.my - r)
        x1, y1 = min(canvas.shape[1], self.mx + r), min(canvas.shape[0], self.my + r)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return
        patch = canvas[y0:y1, x0:x1]
        big = cv2.resize(patch, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        cx, cy = (self.mx - x0) * z, (self.my - y0) * z
        cv2.line(big, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
        cv2.line(big, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)
        H, W = canvas.shape[:2]
        bh, bw = min(big.shape[0], H), min(big.shape[1], W)
        if self.mx < W // 2:
            canvas[26:26 + bh, W - bw:W] = big[:bh, :bw]
        else:
            canvas[26:26 + bh, 0:bw] = big[:bh, :bw]

    # ---------------------------------------------------------------- events
    def on_mouse(self, event, x, y, flags, param):
        self.mx, self.my = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            u, v = self.to_full(x, y)
            self.frame_ann()[self.names[self.pt]] = [u, v]
            # advance to the next still-missing point
            missing = [k for k, n in enumerate(self.names) if n not in self.frame_ann()]
            self.pt = missing[0] if missing else min(self.pt + 1, len(self.names) - 1)
            save_annotations(self.out, self.names, self.ann)
        self.draw()

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.load()
        self.draw()
        while True:
            k = cv2.waitKey(30) & 0xFF
            if k == 255:
                continue
            if k in (ord("q"), 27):
                save_annotations(self.out, self.names, self.ann)
                break
            elif k == ord("n"):
                save_annotations(self.out, self.names, self.ann)
                self.i = min(self.i + 1, len(self.order) - 1)
                self.load(); self.draw()
            elif k == ord("p"):
                save_annotations(self.out, self.names, self.ann)
                self.i = max(self.i - 1, 0)
                self.load(); self.draw()
            elif k == ord("s"):
                save_annotations(self.out, self.names, self.ann)
                print("saved ->", self.out, flush=True)
            elif k == ord("u"):
                fa = self.frame_ann()
                done = [n for n in self.names if n in fa]
                if done:
                    fa.pop(done[-1])
                self.draw()
            elif k == ord("r"):
                self.ann[self.cur()["render_file"]] = {}
                self.pt = 0
                self.draw()
            elif k == ord("a"):
                g = _guides(self.init, self.names, self.cur())
                for n, p in g.items():
                    self.frame_ann().setdefault(n, [float(p[0]), float(p[1])])
                save_annotations(self.out, self.names, self.ann)
                self.draw()
            elif ord("1") <= k <= ord("9"):
                j = k - ord("1")
                if j < len(self.names):
                    self.pt = j
                    self.draw()
        cv2.destroyAllWindows()
        save_annotations(self.out, self.names, self.ann)
        print("wrote", self.out, flush=True)


def main(a):
    renders = a.renders
    views = V.load_views(renders)
    if a.points_file:
        names = load_points_file(a.points_file)
    elif a.points:
        names = [s.strip() for s in a.points.split(",") if s.strip()]
    else:
        raise SystemExit("give --points-file or --points P1,P2,...")
    idx = V.resolve_frames(a.frames, views, a.selection)
    if not idx:
        raise SystemExit("no frames selected")
    images = [V.frame_path(renders, views[k]) for k in idx]
    init = {}
    if a.init and os.path.isfile(a.init):
        with open(a.init, encoding="utf-8") as fh:
            init = json.load(fh).get("points", {})
    ann = load_annotations(a.out)
    app = App(views, idx, names, images, a.out, ann["frames"], init,
              max_width=a.max_width)
    app.run()
    return 0
