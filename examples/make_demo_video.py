"""Compose the demo video for one dataset.

Five stages, in the order the pipeline runs:

    01 reconstruction   COLMAP sparse points + camera trajectory (rotating)
    02 rendering        real frame vs 3DGS render of the same pose (wipe)
    03 view selection   candidate frames -> seed pair -> N_sat -> greedy
                        picks, animated
    04 triangulation    the clicked frames, then the 3D bundle (rays from
                        the selected cameras to the triangulated points)
    05 projection       every render with the projected labels

Only stage 05 uses the run's labels; the few numbers on screen come from the
run's JSON (selection.json) or from the COLMAP model, never typed by hand.

This is a presentation video: it does not show accuracy results (the numbers
live in eval.json / summary.md and in the visualize.py gallery).

    python3 examples/make_demo_video.py --format charuco \
        --scene /data/charuco/train/0000 \
        --renders /data/charuco/train/0000/3dgs/renders \
        --labels /data/charuco/train/0000/3dgs/renders/labels \
        --selection work/selection.json --points3d work/labels/points3d.csv \
        --annotations work/seed.json work/selected.json \
        --sparse /data/charuco/train/0000/sparse/0 \
        --images /data/charuco/train/0000/images \
        --out work/gallery/demo.mp4
"""
import argparse
import glob
import json
import os
import sys
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels_io  # noqa: E402
import visualize as viz  # noqa: E402
from renderlabel import geometry as G  # noqa: E402
from renderlabel import views as V  # noqa: E402

FONT_MED = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
ORANGE = (232, 113, 10)      # RGB
BLUE = (64, 140, 235)
RED = (201, 42, 42)
INK = (238, 240, 243)
MUTED = (150, 158, 168)
DARK = (12, 14, 18)
CARD_TOP = (22, 26, 32)
CARD_BOT = (10, 12, 15)


@lru_cache(maxsize=64)
def _font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_MED, size)


def _pil(bgr):
    return Image.fromarray(cv2.cvtColor(np.ascontiguousarray(bgr),
                                        cv2.COLOR_BGR2RGB))


def _bgr(im):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _ease(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def _clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def _dark_card(W, H):
    y = np.linspace(0, 1, H)[:, None]
    arr = (np.array(CARD_TOP) * (1 - y) + np.array(CARD_BOT) * y)
    arr = np.repeat(arr[:, None, :], W, axis=1).astype(np.uint8)
    im = Image.fromarray(arr)
    return im, ImageDraw.Draw(im)


def _caption(img, stage, title, sub=None):
    """Bottom caption bar: stage label, title, optional detail line."""
    h = 112 if sub else 86
    a = np.full((h, 1, 1), 0.80)
    a[:18, 0, 0] = np.linspace(0.0, 0.80, 18)
    reg = img[-h:].astype(np.float32)
    img[-h:] = (reg * (1 - a) + np.array([16, 18, 22], np.float32) * a
                ).astype(np.uint8)
    img[-h:-h + 2] = ORANGE[::-1]  # thin accent line (BGR)
    im = _pil(img)
    d = ImageDraw.Draw(im)
    d.text((42, H_ - h + 12), stage, font=_font(17, True), fill=ORANGE)
    d.text((42, H_ - h + 34), title, font=_font(25, True), fill=INK)
    if sub:
        d.text((42, H_ - h + 68), sub, font=_font(16), fill=MUTED)
    d.text((W_ - 42, H_ - h + 14), "render-label pipeline",
           font=_font(14), fill=(110, 118, 128), anchor="ra")
    return _bgr(im)


def _chip(img, xy, text, fg=(255, 255, 255), bg=(0, 0, 0), alpha=0.55,
          size=17):
    im = _pil(img).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = _font(size, True)
    x, y = xy
    bb = d.textbbox((0, 0), text, font=f)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.rounded_rectangle([x, y, x + w + 26, y + h + 16], radius=8,
                        fill=bg + (int(255 * alpha),))
    d.text((x + 13, y + 8 - bb[1]), text, font=f, fill=fg + (255,))
    im = Image.alpha_composite(im, layer).convert("RGB")
    return _bgr(im)


def _fit_bg(img, W, H):
    """Contain-fit on a blurred, darkened version of the same image."""
    h, w = img.shape[:2]
    s = min(W / float(w), H / float(h))
    nw, nh = int(round(w * s)), int(round(h * s))
    fg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (0, 0), 26)
    bg = (bg.astype(np.float32) * 0.30).astype(np.uint8)
    x, y = (W - nw) // 2, (H - nh) // 2
    bg[y:y + nh, x:x + nw] = fg
    return bg


# --------------------------------------------------------------------------
# Compositor: segments with a short fade through black between them
# --------------------------------------------------------------------------
class Compositor:
    def __init__(self, path, W, H, fps):
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps), (W, H))
        if not self.w.isOpened():
            raise SystemExit("cannot open %s" % path)
        self.W, self.H, self.fps = W, H, fps
        self.path = path
        self.last = None
        self.n = 0

    def _write(self, fr):
        self.w.write(fr)
        self.n += 1

    def segment(self, name, gen):
        first = next(gen)
        if first is None:
            return
        if self.last is not None:
            black = np.zeros_like(first)
            for k in range(6):
                u = _ease((k + 1) / 6.0)
                self._write((self.last * (1 - u)).astype(np.uint8))
            for k in range(6):
                u = _ease(k / 6.0)
                self._write((first * u).astype(np.uint8))
        self._write(first)
        self.last = first
        for fr in gen:
            self._write(fr)
            self.last = fr
        print("  segment %-10s %5.1f s total %5.1f s"
              % (name, 0, self.n / self.fps))

    def close(self):
        if self.last is not None:
            for k in range(12):
                u = _ease((k + 1) / 12.0)
                self._write((self.last * (1 - u)).astype(np.uint8))
        self.w.release()
        print("wrote %s  (%.1f s)" % (self.path, self.n / self.fps))


# --------------------------------------------------------------------------
# 01 reconstruction
# --------------------------------------------------------------------------
def _ortho(P, mu, axes, theta, W, H, top_clear=36, bot_clear=130, pad=0.10,
           scale=None, robust=False):
    """Orthographic projection about the third principal axis.

    Pass `scale` when several point sets must share one framing (cameras and
    control points in the same shot).  `robust` frames on the 0.3-99.7
    percentile extent so a handful of stray SfM points cannot shrink the cloud.
    """
    Q = np.asarray(P, float) @ axes.T
    c, s = np.cos(theta), np.sin(theta)
    X = Q[:, 0] * c - Q[:, 1] * s
    Y = Q[:, 0] * s + Q[:, 1] * c
    Z = Q[:, 2]
    hh = H - top_clear - bot_clear
    if scale is None:
        if robust and len(X) > 20:
            xlo, xhi = np.percentile(X, [0.3, 99.7])
            ylo, yhi = np.percentile(Y, [0.3, 99.7])
            ex = max(abs(xlo), abs(xhi)) + 1e-9
            ey = max(abs(ylo), abs(yhi)) + 1e-9
        else:
            ex = np.abs(X).max() + 1e-9
            ey = np.abs(Y).max() + 1e-9
        scale = min((W * (1 - 2 * pad)) / (2 * ex),
                    (hh * (1 - 2 * pad)) / (2 * ey))
    px = (X * scale + W / 2).astype(np.int32)
    py = (-Y * scale + top_clear + hh / 2).astype(np.int32)
    return px, py, Z, scale


def _scatter(canvas, xs, ys, cols, r=1):
    H, W = canvas.shape[:2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            yy, xx = ys + dy, xs + dx
            m = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            canvas[yy[m], xx[m]] = cols[m]


def _cloud_frame(px, py, Z, cols, mu_axes=None, cams=None, cam_sel=None,
                 highlight=None, W=1280, H=720, cam_path=True):
    y = np.linspace(0, 1, H)[:, None, None]
    bg = (np.array([16, 19, 24], np.float32) * (1 - y) +
          np.array([7, 8, 10], np.float32) * y)
    canvas = np.repeat(bg, W, axis=1).astype(np.uint8)
    z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    fac = (0.40 + 0.60 * z)[:, None]
    order = np.argsort(Z)
    cols = np.clip(np.asarray(cols, np.float32) * fac, 0, 255).astype(np.uint8)
    # a cheap glow: same points, 35% colour, one pixel wider
    glow = np.clip(np.asarray(cols, np.float32) * 0.35, 0, 255).astype(np.uint8)
    _scatter(canvas, px[order], py[order], glow[order], r=2)
    _scatter(canvas, px[order], py[order], cols[order], r=1)
    if cams is not None:
        cpx, cpy, cz = cams
        if cam_path and len(cpx) > 1:
            pts = np.stack([cpx, cpy], 1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, (78, 72, 60), 1, cv2.LINE_AA)
        for x, yy in zip(cpx, cpy):
            cv2.circle(canvas, (int(x), int(yy)), 2, (158, 128, 84), 1,
                       cv2.LINE_AA)
        if cam_sel is not None:
            for k, (x, yy) in enumerate(zip(cpx, cpy)):
                if k in cam_sel:
                    cv2.circle(canvas, (int(x), int(yy)), 4,
                               (235, 140, 64), -1, cv2.LINE_AA)
    if highlight is not None:
        hx, hy, hz = highlight
        for x, yy in zip(hx, hy):
            cv2.drawMarker(canvas, (int(x), int(yy)), (60, 60, 220),
                           cv2.MARKER_DIAMOND, 9, 1, cv2.LINE_AA)
    return canvas


def recon_frames(sparse_dir, views, n_frames, W, H):
    import pycolmap
    rec = pycolmap.Reconstruction(sparse_dir)
    P = np.array([np.asarray(pt.xyz) for pt in rec.points3D.values()], float)
    C = np.array([np.asarray(pt.color) for pt in rec.points3D.values()],
                 np.uint8)
    npts = len(P)
    cams = np.array([v["C"] for v in views], float)
    # drop the long tail of stray SfM points before framing
    d = np.linalg.norm(P - cams.mean(0), axis=1)
    keep = d <= np.percentile(d, 99.0) * 1.5 + 1e-9
    P, C = P[keep], C[keep]
    if len(P) > 45000:
        idx = np.linspace(0, len(P) - 1, 45000).astype(int)
        P, C = P[idx], C[idx]
    C = cv2.cvtColor(C.reshape(1, -1, 3), cv2.COLOR_RGB2BGR).reshape(-1, 3)
    allp = np.vstack([P, cams])
    mu = allp.mean(0)
    _, _, Vt = np.linalg.svd(allp - mu, full_matrices=False)
    for f in range(n_frames):
        u = f / max(1, n_frames - 1)
        # a slow, partial turn (~60 deg): a full turntable reads as a spin
        theta = np.deg2rad(-110 + 60 * _ease(u))
        _, _, _, sc_p = _ortho(P, mu, Vt, theta, W, H, robust=True)
        _, _, _, sc_c = _ortho(cams, mu, Vt, theta, W, H)
        sc = min(sc_p, sc_c)
        px, py, Z, _ = _ortho(P, mu, Vt, theta, W, H, scale=sc)
        cpx, cpy, cz, _ = _ortho(cams, mu, Vt, theta, W, H, scale=sc)
        frame = _cloud_frame(px, py, Z, C, cams=(cpx, cpy, cz), W=W, H=H)
        frame = _caption(frame, "01 · Sparse reconstruction",
                         "COLMAP SfM — sparse points and camera trajectory",
                         "Points coloured by depth; the view rotates")
        yield frame


# --------------------------------------------------------------------------
# 02 render: real vs 3DGS wipe
# --------------------------------------------------------------------------
def render_frames(render_dir, image_dir, views, n_pairs, W, H, sec_each):
    n = len(views)
    idx = [int(round((k + 0.5) * n / n_pairs)) % n for k in range(n_pairs)]
    fps = 30
    for j, k in enumerate(idx):
        v = views[k]
        real = cv2.imread(os.path.join(image_dir, v["image"]))
        ren = cv2.imread(os.path.join(render_dir, "images", v["render_file"]))
        if real is None or ren is None:
            continue
        real = _fit_bg(real, W, H)
        ren = _fit_bg(ren, W, H)
        nf = int(sec_each * fps)
        for f in range(nf):
            u = f / max(1, nf - 1)
            # a long, eased wipe: hold the capture, slide to the render,
            # hold the render
            if u < 0.12:
                frame = real.copy()
                x = 0
            elif u > 0.88:
                frame = ren.copy()
                x = W
            else:
                x = int(W * _ease((u - 0.12) / 0.76))
                frame = real.copy()
                frame[:, :x] = ren[:, :x]
                cv2.line(frame, (x, 0), (x, H), (245, 245, 245), 2)
            la = _clamp01(1.0 - (x / W - 0.35) / 0.30)
            ra = _clamp01((x / W - 0.55) / 0.30)
            if la > 0.02:
                frame = _chip(frame, (36, 30), "Real capture", alpha=0.55 * la)
            if ra > 0.02:
                frame = _chip(frame, (W - 190, 30), "3DGS render", alpha=0.55 * ra)
            frame = _caption(frame, "02 · Pose-reused rendering",
                             "3DGS trained once; every camera pose re-rendered",
                             "The same pose: capture on the left, render on the right")
            yield frame


# --------------------------------------------------------------------------
# 03 selection animation (matplotlib)
# --------------------------------------------------------------------------
def selection_frames(sel, views, W, H, fps, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9, "axes.titlesize": 10, "axes.edgecolor": "#888888",
        "axes.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
        "figure.facecolor": "#fbfaf7", "axes.facecolor": "#fbfaf7",
    })
    C = np.array([v["C"] for v in views], float)
    cands = viz._candidates(sel, views)
    Cs = C[cands]
    mu = Cs.mean(0)
    _, sv, Vt = np.linalg.svd(Cs - mu, full_matrices=False)
    P = (Cs - mu) @ Vt.T
    x0 = np.asarray(sel["x0"], float)
    p0 = (x0 - mu) @ Vt.T
    pos = {k: j for j, k in enumerate(cands)}
    sel_idx = list(sel["selected_idx"])
    seed_idx = [k for k in sel["seed_idx"] if k in pos]
    dirs = G.ray_dirs(views, x0)[cands]
    m = dirs.mean(0)
    m /= max(np.linalg.norm(m), 1e-12)
    a0 = np.array([1.0, 0.0, 0.0])
    if abs(m @ a0) > 0.9:
        a0 = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(m, a0)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(m, e1)
    alpha = np.degrees(np.arccos(np.clip(dirs @ m, -1, 1)))
    az = np.arctan2(dirs @ e2, dirs @ e1)

    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0.055, 0.22, 0.47, 0.56])
    axp = fig.add_axes([0.60, 0.235, 0.28, 0.525], projection="polar")
    lim = float(np.abs(P[:, :2]).max()) * 1.15 + 1e-9
    fig.text(0.055, 0.945, "View selection", fontsize=16, color="#1a1a1a")
    fig.text(0.055, 0.905, "Pose and model only: seed pair, saturation count, greedy coverage",
             fontsize=9.5, color="#666666")
    t_all = 8.6
    txt_msg = fig.text(0.055, 0.845, "", fontsize=11, color="#333333")
    txt_sub = fig.text(0.055, 0.805, "", fontsize=10, color="#555555")
    nf = int(seconds * fps)
    for f in range(nf):
        t = f / fps / seconds * t_all
        ax.clear()
        axp.clear()
        ax.scatter(P[:, 0], P[:, 1], s=5, c="#9aa0a6",
                   alpha=0.30 + 0.35 * _ease(t / 0.9), linewidths=0)
        axp.scatter(az, alpha, s=5, c="#9aa0a6",
                    alpha=0.30 + 0.35 * _ease(t / 0.9), linewidths=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Camera centres", fontsize=10)
        axp.set_rlim(0, max(alpha.max() * 1.15, 1.0))
        axp.set_title("Viewing rays from X0", fontsize=10)
        axp.grid(color="#e6e6e3", linewidth=0.6)

        # seed pair
        a_seed = _ease((t - 0.9) / 0.5)
        if a_seed > 0:
            for k in seed_idx:
                q = P[pos[k]]
                ax.scatter([q[0]], [q[1]], s=70, marker="D", facecolors="none",
                           edgecolors="#e8710a", linewidths=1.5,
                           alpha=a_seed, zorder=6)
                axp.scatter([az[pos[k]]], [alpha[pos[k]]], s=70, marker="D",
                            facecolors="none", edgecolors="#e8710a",
                            linewidths=1.5, alpha=a_seed, zorder=6)
        a_x0 = _ease((t - 1.5) / 0.5)
        if a_x0 > 0:
            ax.scatter([p0[0]], [p0[1]], s=150, marker="*", c="#c92a2a",
                       alpha=a_x0, zorder=7)
        # selected, one by one
        k_show = 0
        for j in range(len(sel_idx)):
            if t >= 3.0 + 0.55 * j:
                k_show = j + 1
        for j, k in enumerate(sel_idx[:k_show]):
            q = P[pos[k]]
            ax.plot([p0[0], q[0]], [p0[1], q[1]], color="#8fb8e8", lw=0.8,
                    alpha=0.9)
            ax.scatter([q[0]], [q[1]], s=46, c="#1f6fd0", edgecolors="white",
                       linewidths=0.8, zorder=5)
            ax.annotate(str(j + 1), (q[0], q[1]), xytext=(4, 4),
                        textcoords="offset points", fontsize=8,
                        color="#114a8b")
            axp.scatter([az[pos[k]]], [alpha[pos[k]]], s=46, c="#1f6fd0",
                        edgecolors="white", linewidths=0.8, zorder=5)
            axp.annotate(str(j + 1), (az[pos[k]], alpha[pos[k]]),
                         xytext=(3, 2), textcoords="offset points",
                         fontsize=8, color="#114a8b")
        # stage text (persistent artists, updated in place)
        if t < 0.9:
            txt_msg.set_text("① %d candidate frames — pose and model only"
                             % len(cands))
        elif t < 3.0:
            txt_msg.set_text("② Seed pair: the widest pair under the "
                             "pose-only proxy (orange diamonds)")
        else:
            txt_msg.set_text("③ N_sat = ceil(Theta_span / (2 · theta_eff)) = %d"
                             % sel["n_sat"])
        if t > 3.0:
            txt_sub.set_text("Selected %d / %d views"
                             % (k_show, sel["n_selected"]))
        else:
            txt_sub.set_text("")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        frame = _caption(
            frame, "03 · View selection",
            "N_sat = ceil(Theta_span / (2 · theta_eff)) = %d" % sel["n_sat"],
            "Greedy angular coverage")
        yield frame
    plt.close(fig)


# --------------------------------------------------------------------------
# 04 few-view triangulation
# --------------------------------------------------------------------------
def _demo_points(path, max_resid=5.0):
    """Points the demo draws: drop those whose own triangulation residual is
    too large (a 2-view point can be several pixels off)."""
    names, X, nv, res = viz._read_points3d(path)
    keep = [i for i, r in enumerate(res) if r <= max_resid]
    return [names[i] for i in keep], X[keep]


def _crop_tile(img, view, X, names, clicks, tw, th, vis=None):
    uv, ok = G.project_points(X, view)
    pts = [uv[i] for i in range(len(names)) if ok[i]]
    if clicks:
        pts += [np.array([u, w]) for u, w in clicks.values()]
    if len(pts) < 2:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    pts = np.asarray(pts, float)
    pad = 24
    x0 = max(0, int(pts[:, 0].min()) - pad)
    y0 = max(0, int(pts[:, 1].min()) - pad)
    x1 = min(view["W"], int(pts[:, 0].max()) + pad)
    y1 = min(view["H"], int(pts[:, 1].max()) + pad)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    s = min(tw / crop.shape[1], th / crop.shape[0])
    out = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                     interpolation=cv2.INTER_CUBIC)
    canvas = np.full((th, tw, 3), 16, np.uint8)
    oy, ox = (th - out.shape[0]) // 2, (tw - out.shape[1]) // 2
    canvas[oy:oy + out.shape[0], ox:ox + out.shape[1]] = out
    for i, n in enumerate(names):
        if not ok[i]:
            continue
        x = int((uv[i, 0] - x0) * s) + ox
        y = int((uv[i, 1] - y0) * s) + oy
        if 0 <= x < tw and 0 <= y < th:
            if vis is not None and n not in vis:
                cv2.circle(canvas, (x, y), 3, (170, 170, 170), 1, cv2.LINE_AA)
                continue
            cv2.circle(canvas, (x, y), 4, (60, 220, 80), 1, cv2.LINE_AA)
    if clicks:
        for u, w in clicks.values():
            x = int((u - x0) * s) + ox
            y = int((w - y0) * s) + oy
            if 0 <= x < tw and 0 <= y < th:
                cv2.circle(canvas, (x, y), 2, (60, 60, 255), -1, cv2.LINE_AA)
    return canvas


def triangulate_frames(views, sel, points3d, labels, clicks_by_file, W, H,
                       fps, max_resid=5.0):
    by_file = V.index_by_file(views)
    names, X = _demo_points(points3d, max_resid)
    keep = set(names)
    files = list(sel["selected_files"])
    img_dir = os.path.join(render_dir_global, "images")
    cols = 4 if len(files) > 3 else len(files)
    rows = int(np.ceil(len(files) / cols))
    tw = int((W - 80 - 20 * (cols - 1)) / cols)
    avail_h = H - 130 - 160
    th = int((avail_h - 34 * (rows - 1)) / rows)
    th = min(th, int(tw * 1.05))
    tiles = []
    for f in files:
        v = views[by_file[f]]
        img = cv2.imread(os.path.join(img_dir, v["render_file"]))
        click = {n: uv for n, uv in (clicks_by_file.get(f) or {}).items()
                 if n in keep}
        lab = labels.get(viz._stem(v))
        vis = set(lab) if lab else None
        tiles.append(_crop_tile(img, v, X, names, click, tw, th, vis=vis))

    t_tiles = 2.0
    n_tile = int(t_tiles * fps)
    for k in range(n_tile):
        u = k / max(1, n_tile - 1)
        im = _pil(np.full((H, W, 3), 16, np.uint8))
        d = ImageDraw.Draw(im)
        d.text((40, 46), "04 · Few-view triangulation", font=_font(18, True),
               fill=ORANGE)
        d.text((40, 74), "Control points clicked on the selected frames",
               font=_font(26, True), fill=INK)
        show = max(1, int(np.ceil(_ease(u) * len(tiles))))
        grid_h = rows * th + (rows - 1) * 34
        y0 = 130 + max(0, (avail_h - grid_h) // 2)
        for j, tile in enumerate(tiles[:show]):
            r, c = divmod(j, cols)
            x = 40 + c * (tw + 20)
            y = y0 + r * (th + 34)
            im.paste(_pil(tile), (x, y))
            d.text((x, y + th + 6), "View %d" % (j + 1), font=_font(15),
                   fill=(200, 205, 212))
        frame = _bgr(im)
        frame = _caption(frame, "04 · Few-view triangulation",
                         "Selected views × control points → multi-view DLT",
                         "Red dots: clicks · green: visible label · grey: occluded")
        yield frame

    # 3D bundle
    cams = np.array([v["C"] for v in views], float)
    sel_c = np.array([views[by_file[f]]["C"] for f in files], float)
    allp = np.vstack([cams, X])
    mu = allp.mean(0)
    _, _, Vt = np.linalg.svd(allp - mu, full_matrices=False)
    n_b = int(3.0 * fps)
    for k in range(n_b):
        u = k / max(1, n_b - 1)
        theta = np.deg2rad(-30 + 60 * _ease(u))
        _, _, _, sc = _ortho(allp, mu, Vt, theta, W, H)
        cpx, cpy, cz, _ = _ortho(cams, mu, Vt, theta, W, H, scale=sc)
        sx, sy, sz, _ = _ortho(sel_c, mu, Vt, theta, W, H, scale=sc)
        px, py, Z, _ = _ortho(X, mu, Vt, theta, W, H, scale=sc)
        frame = _cloud_frame(px, py, Z, np.full((len(X), 3), (60, 60, 230),
                                              np.uint8), W=W, H=H)
        # selected cameras + rays
        for j in range(len(files)):
            cv2.circle(frame, (int(sx[j]), int(sy[j])), 5, (10, 113, 232),
                       -1, cv2.LINE_AA)
            for q in range(len(X)):
                cv2.line(frame, (int(sx[j]), int(sy[j])),
                         (int(px[q]), int(py[q])), (90, 70, 40), 1,
                         cv2.LINE_AA)
        for q in range(len(X)):
            cv2.drawMarker(frame, (int(px[q]), int(py[q])), (70, 70, 235),
                           cv2.MARKER_DIAMOND, 10, 1, cv2.LINE_AA)
        frame = _caption(frame, "04 · Few-view triangulation",
                         "Selected views × control points → multi-view DLT",
                         "Thin lines: rays from the selected cameras · diamonds: triangulated 3D points")
        yield frame


# --------------------------------------------------------------------------
# 05 projection fly-through
# --------------------------------------------------------------------------
def _frame_quality(view, X, names, lab):
    """Median disagreement with the frame's reference label, and n."""
    if not lab:
        return float("inf"), 0
    uv, ok = G.project_points(X, view)
    errs = []
    for i, n in enumerate(names):
        if not ok[i] or n not in lab:
            continue
        u, w = uv[i]
        if 0 <= u < view["W"] and 0 <= w < view["H"]:
            errs.append(float(np.hypot(u - lab[n][0], w - lab[n][1])))
    return (float(np.median(errs)) if errs else float("inf")), len(errs)


def _demo_frame_indices(views, labels, X, names, max_frames, min_vis=4):
    """Frames for the fly-through: the longest contiguous run whose label
    agreement is close to the sequence median.  A drifted reference stretch
    would make good labels look wrong, so it is left out of the demo."""
    q = [_frame_quality(v, X, names, labels.get(viz._stem(v))) for v in views]
    valid = [(i, m) for i, (m, n) in enumerate(q)
             if n >= min_vis and np.isfinite(m)]
    if not valid:
        step = max(1, int(np.ceil(len(views) / float(max_frames))))
        return list(range(0, len(views), step))[:max_frames]
    med = float(np.median([m for _i, m in valid]))
    thr = max(1.2, 1.35 * med)
    good = [i for i, m in valid if m <= thr]
    runs, cur = [], [good[0]]
    for i in good[1:]:
        if i - cur[-1] <= 3:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    run = max(runs, key=len)
    if len(run) <= max_frames:
        return run
    step = len(run) / float(max_frames)
    return [run[min(len(run) - 1, int(round(k * step)))]
            for k in range(max_frames)]


def projection_frames(views, labels, points3d, max_frames, W, H, fps,
                      max_resid=5.0, point_names=True):
    names, X = _demo_points(points3d, max_resid)
    todo = _demo_frame_indices(views, labels, X, names, max_frames)
    for i in todo:
        v = views[i]
        full = cv2.imread(os.path.join(render_dir_global, "images",
                                       v["render_file"]))
        if full is None:
            continue
        img, s = viz._resize(full, W)
        ref = labels.get(viz._stem(v))
        vis = set(ref) if ref else None
        viz._draw_view(img, v, X, names, s, ref=None, vis=vis,
                       draw_names=False, legend=False)
        viz._paste_zoom(img, full, v, X, names, None, None, frac=0.34,
                        max_inset=420, vis=vis, draw_names=point_names)
        frame = _fit_bg(img, W, H)
        frame = _caption(
            frame, "05 · Full projection",
            "Every triangulated point projected into every rendered frame",
            "Green: visible label · grey: occluded / not observed")
        yield frame


# --------------------------------------------------------------------------
# intro card
# --------------------------------------------------------------------------
def intro_frames(title, subtitle, stage_labels, seconds, W, H, fps):
    im, d = _dark_card(W, H)
    y = 128
    d.text((84, y), "RENDER-LABEL PIPELINE  ·  DEMO", font=_font(16, True),
           fill=ORANGE)
    d.text((84, y + 30), title, font=_font(46, True), fill=INK)
    d.text((84, y + 96), subtitle, font=_font(21), fill=(190, 196, 204))
    d.rectangle([84, y + 146, 250, y + 150], fill=ORANGE)
    yy = y + 190
    for i, s in enumerate(stage_labels):
        d.text((84 + i * 224, yy), "0%d" % (i + 1), font=_font(15, True),
               fill=BLUE)
        d.text((84 + i * 224, yy + 22), s, font=_font(17), fill=(210, 215, 222))
    d.text((84, H - 138), "Reconstruction → Rendering → View selection → "
           "Triangulation → Projection",
           font=_font(17), fill=(160, 168, 178))
    d.text((84, H - 96), "Only the clicks are substituted by existing labels; "
           "every other step is the pipeline itself.",
           font=_font(14), fill=(110, 118, 128))
    nf = int(seconds * fps)
    for _ in range(nf):
        yield _bgr(im)


# ---------------------------------------------------------------- plumbing
render_dir_global = None
W_ = H_ = 0


def main(argv=None):
    global render_dir_global, W_, H_
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True,
                    choices=("charuco", "ycbv", "labelme"))
    ap.add_argument("--renders", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", required=True, help="the real captured frames")
    ap.add_argument("--sparse", required=True, help="COLMAP model dir")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--points3d", required=True)
    ap.add_argument("--annotations", nargs="*", default=[])
    ap.add_argument("--visibility", default=None,
                    help="visibility JSON {frame: {name: {vis: bool}}}; points "
                         "marked false are drawn grey instead of green")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(argv)

    render_dir_global = a.renders
    W_, H_ = a.width, a.height
    views = V.load_views(a.renders)
    labels = labels_io.load(a.format, a.labels)
    if a.visibility:
        with open(a.visibility, encoding="utf-8") as fh:
            visj = json.load(fh)
        labels = {stem: {n: uv for n, uv in pts.items()
                         if visj.get(stem, {}).get(n, {}).get("vis", True)}
                  for stem, pts in labels.items()}
    with open(a.selection, encoding="utf-8") as fh:
        sel = json.load(fh)
    names, X, _nv, _res = viz._read_points3d(a.points3d)
    clicks_by_file = {}
    for p in a.annotations:
        with open(p, encoding="utf-8") as fh:
            clicks_by_file.update(json.load(fh).get("frames", {}))

    title = a.title or "Render-Label Pipeline"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    comp = Compositor(a.out, W_, H_, a.fps)

    comp.segment("intro", intro_frames(
        title, "From an orbit capture to labels on every rendered frame",
        ["Reconstruction", "Rendering", "View selection", "Triangulation",
         "Projection"],
        2.2, W_, H_, a.fps))
    comp.segment("recon", recon_frames(a.sparse, views,
                                       int(5.0 * a.fps), W_, H_))
    comp.segment("render", render_frames(a.renders, a.images, views, 1,
                                         W_, H_, 6.5))
    comp.segment("select", selection_frames(sel, views, W_, H_, a.fps, 8.6))
    comp.segment("triangulate", triangulate_frames(
        views, sel, a.points3d, labels, clicks_by_file, W_, H_, a.fps))
    comp.segment("project", projection_frames(
        views, labels, a.points3d, int(12.5 * a.fps), W_, H_, a.fps,
        point_names=(a.format != "charuco")))
    comp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
