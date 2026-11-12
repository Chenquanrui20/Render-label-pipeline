"""Compose the demo video for one dataset.

Five stages, in the order the pipeline runs:

    01 重建    COLMAP sparse points + camera trajectory (rotating)
    02 渲染     real frame vs 3DGS render of the same pose (wipe)
    03 选视图   candidate frames -> seed pair -> Theta_span -> N_sat ->
                greedy picks, animated
    04 交会    the clicked frames, then the 3D bundle (rays from the
                selected cameras to the triangulated control points)
    05 投影    every render with the projected labels

Only stage 05 uses the run's labels; every number on screen is read from the
run's JSON (selection.json / eval.json / labels/summary.json) or from the
COLMAP model, never typed by hand.

    python3 examples/make_demo_video.py --format charuco \
        --scene /data/charuco/train/0000 \
        --renders /data/charuco/train/0000/3dgs/renders \
        --labels /data/charuco/train/0000/3dgs/renders/labels \
        --selection work/selection.json --points3d work/labels/points3d.csv \
        --annotations work/seed.json work/selected.json \
        --eval work/eval.json --sparse /data/charuco/train/0000/sparse/0 \
        --images /data/charuco/train/0000/images \
        --out work/gallery/demo_full.mp4
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
        theta = np.deg2rad(-155 + 300 * _ease(u))
        _, _, _, sc_p = _ortho(P, mu, Vt, theta, W, H, robust=True)
        _, _, _, sc_c = _ortho(cams, mu, Vt, theta, W, H)
        sc = min(sc_p, sc_c)
        px, py, Z, _ = _ortho(P, mu, Vt, theta, W, H, scale=sc)
        cpx, cpy, cz, _ = _ortho(cams, mu, Vt, theta, W, H, scale=sc)
        frame = _cloud_frame(px, py, Z, C, cams=(cpx, cpy, cz), W=W, H=H)
        frame = _caption(frame, "01 · 稀疏重建",
                         "COLMAP SfM：%d 个位姿，%s 个三维点"
                         % (len(views), format(npts, ",")),
                         "点云按深度着色，旋转视角")
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
            if u < 0.25:
                frame = real.copy()
                x = 0
            elif u > 0.75:
                frame = ren.copy()
                x = W
            else:
                x = int(W * _ease((u - 0.25) / 0.5))
                frame = real.copy()
                frame[:, :x] = ren[:, :x]
                cv2.line(frame, (x, 0), (x, H), (245, 245, 245), 2)
            la = _clamp01(1.0 - (x / W - 0.35) / 0.30)
            ra = _clamp01((x / W - 0.55) / 0.30)
            if la > 0.02:
                tmp = _chip(frame, (36, 30), "真实拍摄", alpha=0.55 * la)
                frame = tmp
            if ra > 0.02:
                tmp = _chip(frame, (W - 190, 30), "3DGS 渲染", alpha=0.55 * ra)
                frame = tmp
            frame = _caption(frame, "02 · 复用位姿渲染",
                             "3DGS 训练 + 复用 COLMAP 位姿：%d 帧全部渲染" % len(views),
                             "视角 %d / %d" % (k + 1, len(views)))
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
    fig.text(0.055, 0.945, "冷启动选帧", fontsize=16, color="#1a1a1a")
    fig.text(0.055, 0.905, "只用位姿与模型：先种子对，再预测饱和数量，再贪心覆盖",
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
        ax.set_title("camera centres", fontsize=10)
        axp.set_rlim(0, max(alpha.max() * 1.15, 1.0))
        axp.set_title("viewing rays from X0", fontsize=10)
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
            txt_msg.set_text("① 候选帧 %d 个：只看位姿与模型" % len(cands))
        elif t < 3.0:
            txt_msg.set_text("② 种子对：位姿代理下张得最开的一对（橙色菱形）")
        else:
            txt_msg.set_text("③ N_sat = ⌈%.1f° ÷ (2 × %.1f°)⌉ = %d"
                             % (sel["theta_span_deg"], sel["theta_eff_deg"],
                                sel["n_sat"]))
        if t > 3.0:
            txt_sub.set_text("已选 %d / %d     theta_max = %.2f°"
                             % (k_show, sel["n_selected"],
                                sel["theta_max_deg"]))
        else:
            txt_sub.set_text("")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        frame = _caption(
            frame, "03 · 选视图",
            "Theta_span %.1f° → N_sat = %d（theta_eff %.1f°）"
            % (sel["theta_span_deg"], sel["n_sat"], sel["theta_eff_deg"]),
            "贪心覆盖，5° 守卫；theta_max = %.2f°"
            % sel["theta_max_deg"])
        yield frame
    plt.close(fig)


# --------------------------------------------------------------------------
# 04 few-view triangulation
# --------------------------------------------------------------------------
def _crop_tile(img, view, X, names, clicks, tw, th):
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
            cv2.circle(canvas, (x, y), 4, (60, 220, 80), 1, cv2.LINE_AA)
    if clicks:
        for u, w in clicks.values():
            x = int((u - x0) * s) + ox
            y = int((w - y0) * s) + oy
            if 0 <= x < tw and 0 <= y < th:
                cv2.circle(canvas, (x, y), 2, (60, 60, 255), -1, cv2.LINE_AA)
    return canvas


def triangulate_frames(views, sel, points3d, labels, clicks_by_file, W, H,
                       fps):
    by_file = V.index_by_file(views)
    names, X, _nv, _res = viz._read_points3d(points3d)
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
        click = (clicks_by_file.get(f) or {})
        tiles.append(_crop_tile(img, v, X, names, click, tw, th))

    t_tiles = 2.6
    n_tile = int(t_tiles * fps)
    for k in range(n_tile):
        u = k / max(1, n_tile - 1)
        im = _pil(np.full((H, W, 3), 16, np.uint8))
        d = ImageDraw.Draw(im)
        d.text((40, 46), "04 · 少量交会", font=_font(18, True), fill=ORANGE)
        d.text((40, 74), "只在选中的 %d 帧上点控制点" % len(files),
               font=_font(26, True), fill=INK)
        show = max(1, int(np.ceil(_ease(u) * len(tiles))))
        grid_h = rows * th + (rows - 1) * 34
        y0 = 130 + max(0, (avail_h - grid_h) // 2)
        for j, tile in enumerate(tiles[:show]):
            r, c = divmod(j, cols)
            x = 40 + c * (tw + 20)
            y = y0 + r * (th + 34)
            im.paste(_pil(tile), (x, y))
            d.text((x, y + th + 6), "视角 %d" % (j + 1), font=_font(15),
                   fill=(200, 205, 212))
        frame = _bgr(im)
        frame = _caption(frame, "04 · 少量交会",
                         "%d 个视角 × %d 个点 → 多视图 DLT" % (len(files), len(X)),
                         "红点是点击，绿圈是交会后重投影回去的位置")
        yield frame

    # 3D bundle
    cams = np.array([v["C"] for v in views], float)
    sel_c = np.array([views[by_file[f]]["C"] for f in files], float)
    allp = np.vstack([cams, X])
    mu = allp.mean(0)
    _, _, Vt = np.linalg.svd(allp - mu, full_matrices=False)
    n_b = int(4.4 * fps)
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
        frame = _caption(frame, "04 · 少量交会",
                         "%d 个视角 × %d 个点 → 多视图 DLT" % (len(files), len(X)),
                         "细线：选中相机的视线；菱形：交会出的三维控制点")
        yield frame


# --------------------------------------------------------------------------
# 05 projection fly-through
# --------------------------------------------------------------------------
def projection_frames(views, labels, points3d, max_frames, W, H, fps):
    names, X, _nv, _res = viz._read_points3d(points3d)
    n = len(views)
    step = max(1, int(np.ceil(n / float(max_frames))))
    todo = list(range(0, n, step))
    for k, i in enumerate(todo):
        v = views[i]
        full = cv2.imread(os.path.join(render_dir_global, "images",
                                       v["render_file"]))
        if full is None:
            continue
        img, s = viz._resize(full, W)
        ref = labels.get(viz._stem(v))
        viz._draw_view(img, v, X, names, s, ref=ref, draw_names=False,
                       legend=False)
        viz._paste_zoom(img, full, v, X, names, ref, None, frac=0.34,
                        max_inset=420)
        frame = _fit_bg(img, W, H)
        frame = _caption(
            frame, "05 · 全量投影",
            "把 %d 个三维点投影到全部 %d 帧" % (len(X), n),
            "帧 %d / %d" % (i + 1, n))
        yield frame


# --------------------------------------------------------------------------
# intro / result cards
# --------------------------------------------------------------------------
def intro_frames(title, subtitle, n_frames, n_points, focal, stage_labels,
                 seconds, W, H, fps):
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
    stats = "%d 渲染帧   ·   %d 控制点   ·   共享内参 f = %.1f px" \
            % (n_frames, n_points, focal)
    d.text((84, H - 138), stats, font=_font(17), fill=(160, 168, 178))
    d.text((84, H - 96), "仅点击一步由数据集已有标签代替；其余为流水线本体",
           font=_font(14), fill=(110, 118, 128))
    nf = int(seconds * fps)
    for _ in range(nf):
        yield _bgr(im)


def result_frames(ev, sel, png, seconds, W, H, fps):
    im, d = _dark_card(W, H)
    d.text((84, 78), "结果", font=_font(38, True), fill=INK)
    d.text((84, 128), "留出集：所有未参与点击的帧", font=_font(16), fill=MUTED)
    o = ev["ours"]
    r = ev.get("random", {})
    g = ev.get("gt3d", {})
    rows = [
        ("留出集 2D 误差（中位 / 均值）",
         "%.2f / %.2f px" % (o["median"], o["mean"]),
         "n = %s 个观测" % format(o["n_obs"], ",")),
    ]
    if "trial_median" in r:
        rows.append(("随机选帧，同样预算（中位 / 均值）",
                     "%.2f / %.2f px" % (r["trial_median"]["median"],
                                         r["trial_mean"]["median"]),
                     "中位判据胜率 %.0f%%，均值判据 %.0f%%"
                     % (100 * r["win_rate_median"], 100 * r["win_rate_mean"])))
    if "loo_mm" in g:
        rows.append(("三维形状，Sim(3) 留一（中位 / 均值）",
                     "%.2f / %.2f mm" % (g["loo_mm"]["median"],
                                         g["loo_mm"]["mean"]),
                     "%d 个控制点" % g["n_points"]))
    yy = 190
    for label, big, sub in rows:
        d.text((84, yy), label, font=_font(16), fill=MUTED)
        d.text((84, yy + 24), big, font=_font(34, True), fill=INK)
        d.text((84, yy + 70), sub, font=_font(14), fill=(140, 148, 158))
        yy += 130
    if png and os.path.isfile(png):
        img = cv2.imread(png)
        if img is not None:
            tw = int(W * 0.46)
            s = tw / img.shape[1]
            th = int(img.shape[0] * s)
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            x = W - tw - 54
            y = max(150, (H - th) // 2 - 30)
            im.paste(_pil(img), (x, y))
    d.text((84, H - 72), "像素误差只在同一渲染域内可比；中位与均值同时报告。",
           font=_font(14), fill=(120, 128, 138))
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
    ap.add_argument("--format", required=True, choices=("charuco", "ycbv"))
    ap.add_argument("--renders", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", required=True, help="the real captured frames")
    ap.add_argument("--sparse", required=True, help="COLMAP model dir")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--points3d", required=True)
    ap.add_argument("--annotations", nargs="*", default=[])
    ap.add_argument("--eval", default=None)
    ap.add_argument("--errors-png", default=None,
                    help="errors.png from visualize.py (result card inset)")
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
    with open(a.selection, encoding="utf-8") as fh:
        sel = json.load(fh)
    names, X, _nv, _res = viz._read_points3d(a.points3d)
    clicks_by_file = {}
    for p in a.annotations:
        with open(p, encoding="utf-8") as fh:
            clicks_by_file.update(json.load(fh).get("frames", {}))

    msg = next(iter(views))
    title = a.title or ("ArUco · ChArUco 序列" if a.format == "charuco"
                        else "YCBV · seq00_room")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    comp = Compositor(a.out, W_, H_, a.fps)

    comp.segment("intro", intro_frames(
        title, "从一次环绕拍摄，到全部渲染帧的标签",
        len(views), len(names), msg["f"],
        ["重建", "渲染", "选视图", "少量交会", "全量投影"],
        4.0, W_, H_, a.fps))
    comp.segment("recon", recon_frames(a.sparse, views,
                                       int(8.0 * a.fps), W_, H_))
    comp.segment("render", render_frames(a.renders, a.images, views, 4,
                                         W_, H_, 2.0))
    comp.segment("select", selection_frames(sel, views, W_, H_, a.fps, 8.6))
    comp.segment("triangulate", triangulate_frames(
        views, sel, a.points3d, labels, clicks_by_file, W_, H_, a.fps))
    comp.segment("project", projection_frames(
        views, labels, a.points3d, int(12.5 * a.fps), W_, H_, a.fps))
    if a.eval and os.path.isfile(a.eval):
        with open(a.eval, encoding="utf-8") as fh:
            ev = json.load(fh)
        comp.segment("result", result_frames(ev, sel, a.errors_png, 5.5,
                                             W_, H_, a.fps))
    comp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
