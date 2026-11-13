"""Demo visualizations: what the pipeline actually did, on the images.

Produces, under `--out` (a gallery directory):

    overlays/       the seed frames, the selected frames, and an evenly spaced
                    sample of held-out frames whose label agreement is at or
                    below the sequence median (where the reference label is
                    reliable).  Green = the projected label written by stage
                    07, grey = occluded / not labelled, yellow cross = the
                    dataset's own label on that render, red dot = a click.
    selection.png   camera centres (PGA projection), selected frames numbered;
                    the angular span, N_sat and theta_max in the caption.
    errors.png      random-baseline trial distributions vs ours (median/mean).
    per_point.png   held-out error per control point (median bar, mean dot).
    demo.mp4        every render with the projected labels (only with --video).
    report.html     one page that shows all of the above.

This script only reads the outputs of stages 04-07 plus the dataset labels; it
does not change any number.  The dataset labels are **render-domain** references
(ChArUco detection on renders / YCBV optical-flow labels), so the pixel errors
shown here are only comparable inside this render set.

    python3 examples/visualize.py --format charuco \
        --renders /data/scene/3dgs/renders \
        --labels /data/scene/3dgs/renders/labels \
        --selection work/selection.json --points3d work/labels/points3d.csv \
        --eval work/eval.json --annotations work/seed.json work/selected.json \
        --out work/gallery --video
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels_io  # noqa: E402
from renderlabel import geometry as G  # noqa: E402
from renderlabel import views as V  # noqa: E402

GREEN = (80, 220, 60)
YELLOW = (0, 230, 255)
RED = (60, 60, 255)
GRAY = (200, 200, 200)


def _stem(v):
    return os.path.splitext(v["image"])[0]


def _read_points3d(path):
    names, X, nviews, resid = [], [], [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            names.append(r["name"])
            X.append([float(r["X"]), float(r["Y"]), float(r["Z"])])
            nviews.append(int(r["n_views"]))
            resid.append(float(r["resid_median_px"]))
    return names, np.asarray(X, float), nviews, resid


def _text(img, s, org, scale, color, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def _place_labels(img, items, fs, color):
    """Greedy collision-avoiding label placement with a dark outline."""
    boxes = []
    H, W = img.shape[:2]
    for x, y, s in items:
        (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        offsets = [(9, -8), (9, 14), (-w - 9, -8), (-w - 9, 14),
                   (9, -20), (-w - 9, -20), (16, 26), (-w - 16, 26),
                   (9, 34), (-w - 9, 34), (9, -32), (-w - 9, -32)]
        placed = None
        for dx, dy in offsets:
            bx0, by0 = x + dx, y + dy - h
            bx1, by1 = bx0 + w, y + dy + 2
            if bx0 < 2 or by0 < 2 or bx1 > W - 2 or by1 > H - 2:
                continue
            if any(not (bx1 < b[0] or bx0 > b[2] or by1 < b[1] or by0 > b[3])
                   for b in boxes):
                continue
            placed = (bx0, by0, bx1, by1, x + dx, y + dy)
            break
        if placed is None:
            placed = (x + 9, y - 8 - h, x + 9 + w, y - 6, x + 9, y - 8)
        boxes.append(placed[:4])
        _text(img, s, (placed[4], placed[5]), fs, color, 1)


def _draw_view(img, view, X, names, scale, origin=(0, 0), ref=None, clicks=None,
               header="", sub="", font_scale=None, legend=True, draw_names=True,
               vis=None):
    """Draw projected labels (green), occluded points (grey), reference labels
    (yellow crosses) and clicks (red).

    ``vis`` is the set of points visible in this frame; points outside it are
    drawn as hollow grey circles instead of green crosses.  When ``vis`` is
    None and ``ref`` is given, the reference label doubles as the visible set.

    Image coordinates are ``(uv - origin) * scale``: origin lets the same
    routine draw a crop (the zoom inset) without re-projecting.
    """
    ox, oy = origin
    if vis is None and ref:
        vis = set(ref.keys())
    uv, ok = G.project_points(X, view)
    inside = (ok & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) &
              (uv[:, 0] >= 0) & (uv[:, 0] <= view["W"] - 1) &
              (uv[:, 1] >= 0) & (uv[:, 1] <= view["H"] - 1))
    uv = (uv - np.array([ox, oy], float)) * scale
    fs = font_scale or max(0.42, 0.5 * scale)

    if ref:
        arm = max(6, int(round(8 * max(0.8, scale))))
        for n, (u, w) in ref.items():
            x = int(round((u - ox) * scale))
            y = int(round((w - oy) * scale))
            cv2.line(img, (x - arm, y), (x + arm, y), YELLOW, 1, cv2.LINE_AA)
            cv2.line(img, (x, y - arm), (x, y + arm), YELLOW, 1, cv2.LINE_AA)
    if clicks:
        for n, (u, w) in clicks.items():
            cv2.circle(img, (int(round((u - ox) * scale)),
                             int(round((w - oy) * scale))),
                       2, RED, -1, cv2.LINE_AA)
    items = []
    for i, n in enumerate(names):
        if not inside[i]:
            continue
        x, y = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if vis is not None and n not in vis:
            cv2.circle(img, (x, y), 4, GRAY, 1, cv2.LINE_AA)
            continue
        cv2.circle(img, (x, y), 5, GREEN, 2, cv2.LINE_AA)
        cv2.line(img, (x - 3, y), (x + 3, y), GREEN, 1, cv2.LINE_AA)
        cv2.line(img, (x, y - 3), (x, y + 3), GREEN, 1, cv2.LINE_AA)
        items.append((x, y, n))
    if draw_names:
        _place_labels(img, items, fs, GREEN)

    if header:
        _text(img, header, (10, 24), 0.7, (255, 255, 255), 2)
    if sub:
        _text(img, sub, (10, 48), 0.52, (220, 220, 220), 1)
    if legend:
        _text(img, "green o visible label   grey o occluded / not labelled"
              "   yellow + dataset label   red . click",
              (10, img.shape[0] - 12), 0.46, GRAY, 1)


def _paste_zoom(canvas, full, view, X, names, ref, clicks, frac=0.46,
                max_inset=620, vis=None, draw_names=True):
    """Crop around the drawn points, enlarge, and paste it top-right."""
    uv, ok = G.project_points(X, view)
    pts = [uv[i] for i in range(len(names)) if ok[i] and np.isfinite(uv[i]).all()]
    pts += [np.array([u, w]) for u, w in (ref or {}).values()]
    if len(pts) < 2:
        return
    pts = np.asarray(pts, float)
    pad = 26
    x0 = max(0, int(np.floor(pts[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - pad)
    x1 = min(view["W"], int(np.ceil(pts[:, 0].max())) + pad)
    y1 = min(view["H"], int(np.ceil(pts[:, 1].max())) + pad)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return
    crop = full[y0:y1, x0:x1]
    tw = min(max_inset, max(160, int(frac * canvas.shape[1])))
    s = tw / float(crop.shape[1])
    th = int(round(crop.shape[0] * s))
    if th > 0.62 * canvas.shape[0]:
        s *= 0.62 * canvas.shape[0] / th
        tw, th = int(round(crop.shape[1] * s)), int(round(crop.shape[0] * s))
    zoom = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_CUBIC)
    _draw_view(zoom, view, X, names, s, origin=(x0, y0), ref=ref,
               clicks=clicks, font_scale=0.45, legend=False, vis=vis,
               draw_names=draw_names)
    cv2.rectangle(zoom, (0, 0), (tw - 1, th - 1), (255, 255, 255), 1)
    _text(zoom, "zoom x%.1f" % s, (6, th - 8), 0.42, GRAY, 1)
    px = canvas.shape[1] - tw - 12
    py = 12
    canvas[py:py + th, px:px + tw] = zoom


def _resize(img, max_side):
    h, w = img.shape[:2]
    s = min(1.0, float(max_side) / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return img, s


def _frame_errors(view, X, names, ref):
    """(median, mean, n) of projected-vs-reference on one frame (NaN if none)."""
    if not ref:
        return float("nan"), float("nan"), 0
    uv, ok = G.project_points(X, view)
    errs = []
    for i, n in enumerate(names):
        if n in ref and ok[i] and np.isfinite(uv[i]).all():
            u, w = uv[i]
            if 0 <= u <= view["W"] - 1 and 0 <= w <= view["H"] - 1:
                errs.append(float(np.hypot(u - ref[n][0], w - ref[n][1])))
    if not errs:
        return float("nan"), float("nan"), 0
    return float(np.median(errs)), float(np.mean(errs)), len(errs)


# --------------------------------------------------------------- still frames
def write_overlays(views, labels, clicks_by_file, sel, X, names, out_dir,
                   max_side):
    os.makedirs(out_dir, exist_ok=True)
    seed = list(sel.get("seed_files") or [])
    chosen = list(sel.get("selected_files") or [])
    ann = set(seed) | set(chosen)

    # held-out frames: evenly spaced, drawn from the better half by label
    # agreement -- this is a gallery, so a drifted reference frame would only
    # show the reference's error, not the pipeline's output.
    held = [v for v in views if v["render_file"] not in ann]
    scored = []
    for v in held:
        m, _a, n = _frame_errors(v, X, names, labels.get(_stem(v)))
        if n > 0 and np.isfinite(m):
            scored.append((m, v))
    sample = []
    if scored:
        cut = float(np.median([m for m, _v in scored]))
        good = [v for m, v in scored if m <= cut]
        k = min(8, len(good))
        step = max(1, len(good) // k)
        sample = good[::step][:k]

    plan = ([(f, "seed") for f in seed] +
            [(f, "selected") for f in chosen] +
            [(v["render_file"], "heldout") for v in sample])
    by_file = V.index_by_file(views)
    wrote = []
    seen = set()
    for f, role in plan:
        if f in seen:
            continue
        seen.add(f)
        v = views[by_file[f]]
        full = cv2.imread(os.path.join(_img_dir, v["render_file"]))
        if full is None:
            continue
        img, s = _resize(full, max_side)
        stem = _stem(v)
        ref = labels.get(stem)
        clicks = clicks_by_file.get(f)
        m, a, n = _frame_errors(v, X, names, ref)
        if role == "heldout":
            sub = "held-out frame  n=%d  median %.2f px / mean %.2f px" % (n, m, a)
        else:
            sub = "%s frame (used for triangulation)" % role
        hdr = "%s  [%s]" % (v["render_file"], role)
        _draw_view(img, v, X, names, s, ref=ref, clicks=clicks,
                   header=hdr, sub=sub, draw_names=False)
        _paste_zoom(img, full, v, X, names, ref, clicks,
                    vis=set(ref) if ref else None)
        name = "%s_%s.png" % (stem, role)
        cv2.imwrite(os.path.join(out_dir, name), img)
        wrote.append(dict(file=name, stem=stem, role=f, role_label=role,
                          median=m if np.isfinite(m) else None,
                          mean=a if np.isfinite(a) else None, n=n,
                          render=v["render_file"]))
    return wrote


# ------------------------------------------------------------------- video
def write_video(views, labels, X, names, out_path, max_side, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    first = cv2.imread(os.path.join(_img_dir, views[0]["render_file"]))
    h, w = first.shape[:2]
    s = min(1.0, float(max_side) / max(h, w))
    vw, vh = int(round(w * s)), int(round(h * s))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (vw, vh))
    if not writer.isOpened():
        raise SystemExit("cannot open video writer for %s" % out_path)
    for k, v in enumerate(views):
        full = cv2.imread(os.path.join(_img_dir, v["render_file"]))
        if full is None:
            continue
        img = cv2.resize(full, (vw, vh), interpolation=cv2.INTER_AREA) \
            if (full.shape[1], full.shape[0]) != (vw, vh) else full.copy()
        ref = labels.get(_stem(v))
        m, a, n = _frame_errors(v, X, names, ref)
        sub = ("frame %d/%d" % (k + 1, len(views)) +
               ("   held-out median %.2f px / mean %.2f px (n=%d)" % (m, a, n)
                if np.isfinite(m) else ""))
        _draw_view(img, v, X, names, s, ref=ref, header=v["render_file"],
                   sub=sub, draw_names=False)
        _paste_zoom(img, full, v, X, names, ref, None, frac=0.42,
                    max_inset=520)
        writer.write(img)
        if (k + 1) % 100 == 0:
            print("  video %d/%d" % (k + 1, len(views)))
    writer.release()
    print("  wrote", out_path)


# -------------------------------------------------------------- matplotlib
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e4e4e4", "grid.linewidth": 0.6,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })
    return plt


def _candidates(sel, views):
    with open(sel["points3d"], encoding="utf-8") as fh:
        pts = json.load(fh).get("points", {})
    X = np.array([pts[n] for n in sorted(pts)])
    margin = float(sel.get("margin_px", 20.0))
    return [k for k, v in enumerate(views)
            if G.inside_image(G.project_points(X, v)[0], v, margin)]


def plot_selection(sel, views, out_png):
    plt = _mpl()
    C = np.array([v["C"] for v in views], float)
    cands = _candidates(sel, views)
    Cs = C[cands]
    mu = Cs.mean(0)
    _, sv, Vt = np.linalg.svd(Cs - mu, full_matrices=False)
    P = (Cs - mu) @ Vt.T
    x0 = np.asarray(sel["x0"], float)
    p0 = (x0 - mu) @ Vt.T
    pos = {k: j for j, k in enumerate(cands)}
    var = sv ** 2 / max(1e-12, (sv ** 2).sum()) * 100

    sel_idx = list(sel["selected_idx"])
    seed_idx = set(sel["seed_idx"])

    fig = plt.figure(figsize=(9.2, 4.3))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(P[:, 0], P[:, 1], s=6, c="#9aa0a6", alpha=0.55,
               linewidths=0, label="candidate frame (%d)" % len(cands))
    for k in sel_idx:
        q = P[pos[k]]
        ax.plot([p0[0], q[0]], [p0[1], q[1]], color="#8fb8e8", lw=0.7)
    q = P[[pos[k] for k in sel_idx]]
    ax.scatter(q[:, 0], q[:, 1], s=42, c="#1f6fd0", edgecolors="white",
               linewidths=0.8, zorder=4, label="selected (%d)" % len(sel_idx))
    for r, k in enumerate(sel_idx):
        qq = P[pos[k]]
        ax.annotate(str(r + 1), (qq[0], qq[1]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color="#114a8b")
    seed_q = P[[pos[k] for k in seed_idx if k in pos]]
    ax.scatter(seed_q[:, 0], seed_q[:, 1], s=70, marker="D",
               facecolors="none", edgecolors="#e8710a", linewidths=1.4,
               zorder=5, label="seed pair")
    ax.scatter([p0[0]], [p0[1]], s=130, marker="*", c="#c92a2a", zorder=6,
               label="X0 (seed triangulation)")
    lim = float(np.abs(P[:, :2]).max()) * 1.12 + 1e-9
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("PC1 (%.0f%% var)" % var[0])
    ax.set_ylabel("PC2 (%.0f%% var)" % var[1])
    ax.set_title("camera centres, PC1-PC2")

    # right: the ray directions from X0 -- the only quantity the selector uses
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
    axp = fig.add_subplot(1, 2, 2, projection="polar")
    axp.scatter(az, alpha, s=6, c="#9aa0a6", alpha=0.55, linewidths=0)
    j = [pos[k] for k in sel_idx]
    axp.scatter(az[j], alpha[j], s=42, c="#1f6fd0", edgecolors="white",
                linewidths=0.8, zorder=4)
    for r, k in enumerate(sel_idx):
        axp.annotate(str(r + 1), (az[pos[k]], alpha[pos[k]]),
                     xytext=(4, 2), textcoords="offset points", fontsize=7,
                     color="#114a8b")
    axp.set_rlim(0, max(alpha.max() * 1.15, 1.0))
    axp.set_title("viewing rays from X0\n(r = angle from mean ray, deg)",
                  fontsize=9)
    axp.grid(color="#e4e4e4", linewidth=0.6)

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Camera geometry: Theta_span %.2f deg, N_sat = %d "
                 "(theta_eff %.1f deg), picked %d, theta_max %.2f deg"
                 % (sel["theta_span_deg"], sel["n_sat"], sel["theta_eff_deg"],
                    sel["n_selected"], sel["theta_max_deg"]), fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out_png)


def plot_errors(ev, out_png):
    plt = _mpl()
    r = ev.get("random", {})
    o = ev["ours"]
    tm = r.get("trial_median_values")
    tmean = r.get("trial_mean_values")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, vals, ours, lab in [
            (axes[0], tm, o["median"], "trial median"),
            (axes[1], tmean, o["mean"], "trial mean")]:
        if vals:
            ax.hist(vals, bins=18, color="#b7c6da", edgecolor="white",
                    linewidth=0.6, label="random view sets (%d trials)" % len(vals))
        ax.axvline(ours, color="#c92a2a", lw=1.8, label="ours (%.2f px)" % ours)
        ax.set_xlabel("held-out error over all points / frames (%s), px" % lab)
        ax.set_ylabel("trials")
        ax.legend(frameon=False, fontsize=8)
    wr_m = r.get("win_rate_median")
    wr_a = r.get("win_rate_mean")
    axes[0].set_title("trial medians  —  ours wins %.0f%%" % (100 * wr_m)
                      if wr_m is not None else "trial medians")
    axes[1].set_title("trial means  —  ours wins %.0f%%" % (100 * wr_a)
                      if wr_a is not None else "trial means")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out_png)


def plot_per_point(ev, out_png):
    plt = _mpl()
    pp = ev.get("per_point", {})
    items = sorted(pp.items(), key=lambda kv: kv[1].get("median", 0))
    names = [k for k, _ in items]
    med = [v.get("median", np.nan) for _, v in items]
    mean = [v.get("mean", np.nan) for _, v in items]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(names) + 1.2))
    ax.barh(y, med, height=0.62, color="#b7c6da", edgecolor="white",
            linewidth=0.6, label="median")
    ax.scatter(mean, y, s=16, c="#c92a2a", zorder=3, label="mean")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("held-out 2D error per point, px")
    ax.set_title("Per-point held-out error (median bar, mean dot)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out_png)


# -------------------------------------------------------------------- html
def write_html(out_dir, run_name, sel, ev, overlays, has_video):
    o = ev["ours"]
    r = ev.get("random", {})
    g = ev.get("gt3d", {})
    rows = [
        ("renders", "%d" % sel["n_candidates"]),
        ("Theta_span", "%.2f&deg;" % sel["theta_span_deg"]),
        ("N_sat = ceil(Theta/(2&theta;<sub>eff</sub>))",
         "%d (theta_eff %.1f&deg;)" % (sel["n_sat"], sel["theta_eff_deg"])),
        ("selected views", "%d (theta_max %.2f&deg;)" % (sel["n_selected"],
                                                         sel["theta_max_deg"])),
        ("held-out ours (median / mean)",
         "%.2f / %.2f px" % (o["median"], o["mean"])),
    ]
    if "trial_median" in r:
        rows += [
            ("random trial medians (median / mean over trials)",
             "%.2f / %.2f px" % (r["trial_median"]["median"],
                                 r["trial_median"]["mean"])),
            ("random trial means (median / mean over trials)",
             "%.2f / %.2f px" % (r["trial_mean"]["median"],
                                 r["trial_mean"]["mean"])),
            ("win rate (median / mean criterion)",
             "%.0f%% / %.0f%%" % (100 * r["win_rate_median"],
                                  100 * r["win_rate_mean"])),
        ]
    if "loo_mm" in g:
        rows.append(("3D vs GT, Sim(3) LOO (median / mean)",
                     "%.2f / %.2f mm" % (g["loo_mm"]["median"],
                                         g["loo_mm"]["mean"])))
    table = "\n".join("<tr><td>%s</td><td>%s</td></tr>" % (k, v) for k, v in rows)
    cards = []
    for ov in overlays:
        cap = "%s — %s" % (ov["render"], ov["role_label"])
        if ov["role_label"] == "heldout" and ov["n"]:
            cap += " — median %.2f px, mean %.2f px (n=%d)" % (
                ov["median"], ov["mean"], ov["n"])
        cards.append(
            '<figure><a href="overlays/%s"><img src="overlays/%s" '
            'loading="lazy"></a><figcaption>%s</figcaption></figure>'
            % (ov["file"], ov["file"], cap))
    video_html = ('<h2>Video</h2><video controls width="100%" '
                  'src="demo.mp4"></video>') if has_video else ""
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>demo — %(name)s</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
     max-width:1180px;margin:2rem auto;padding:0 1rem;color:#222;
     background:#fdfcfb;line-height:1.55}
h1{font-size:1.5rem;margin-bottom:.2rem} h2{margin-top:2rem;font-size:1.15rem}
table{border-collapse:collapse;margin:.8rem 0}
td{border-bottom:1px solid #e5e5e5;padding:.3rem .8rem .3rem 0;font-size:.9rem}
td:first-child{color:#666}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
      gap:14px}
figure{margin:0;background:#fff;border:1px solid #e8e8e8;border-radius:8px;
       overflow:hidden}
figure img{width:100%%;display:block}
figcaption{font-size:.78rem;color:#555;padding:.45rem .6rem}
.note{font-size:.85rem;color:#666;background:#f4f2ee;border-radius:8px;
      padding:.7rem .9rem}
code{font-size:.85em}
</style></head><body>
<h1>%(name)s demo — stages 04–07 on a 3DGS render set</h1>
<p class="note">The clicks (stages 04/06) are copied from the dataset's
existing <b>render-domain labels</b>; every other step is the pipeline itself.
Pixel errors are comparable only inside the same render domain. Green circles
are the labels written by stage 07 (visible in the reference visibility set);
grey circles are points that are occluded or not labelled in that frame;
yellow crosses are the dataset's reference labels; red dots are the clicks
used for triangulation. Labels are written by pure geometric projection --
occlusion masking is not part of the demo.</p>
<table>%(table)s</table>
<h2>View-selection geometry</h2>
<p><a href="selection.png"><img src="selection.png" style="width:100%%"></a></p>
<h2>Error distributions</h2>
<p><a href="errors.png"><img src="errors.png" style="width:100%%"></a></p>
<p><a href="per_point.png"><img src="per_point.png" style="width:100%%"></a></p>
%(video)s
<h2>Labels on rendered frames</h2>
<div class="grid">%(cards)s</div>
<p class="note">Full numbers: <a href="../summary.md">summary.md</a> ·
evaluation JSON: <a href="../eval.json">eval.json</a> ·
stage-07 output: <a href="../labels/">labels/</a></p>
</body></html>
""" % dict(name=run_name, table=table, cards="\n".join(cards),
           video=video_html)
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("  wrote", path)


# ---------------------------------------------------------------- plumbing
_img_dir = None


def main(argv=None):
    global _img_dir
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True,
                    choices=("charuco", "ycbv", "labelme"))
    ap.add_argument("--renders", required=True)
    ap.add_argument("--labels", required=True,
                    help="charuco/labelme: labelme dir; ycbv: points_states.csv")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--points3d", required=True)
    ap.add_argument("--eval", default=None, help="eval.json (for the figures)")
    ap.add_argument("--annotations", nargs="*", default=[],
                    help="seed.json / selected.json, to draw the clicks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-side", type=int, default=960)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--name", default=None, help="title of report.html")
    a = ap.parse_args(argv)

    views = V.load_views(a.renders)
    _img_dir = V.images_dir(a.renders)
    labels = labels_io.load(a.format, a.labels)
    names, X, nviews, resid = _read_points3d(a.points3d)
    with open(a.selection, encoding="utf-8") as fh:
        sel = json.load(fh)
    clicks_by_file = {}
    for p in a.annotations:
        with open(p, encoding="utf-8") as fh:
            clicks_by_file.update(json.load(fh).get("frames", {}))
    os.makedirs(a.out, exist_ok=True)

    print("[overlays]")
    overlays = write_overlays(views, labels, clicks_by_file, sel, X, names,
                              os.path.join(a.out, "overlays"), a.max_side)

    if sel.get("x0") is not None:
        plot_selection(sel, views, os.path.join(a.out, "selection.png"))
    if a.eval:
        with open(a.eval, encoding="utf-8") as fh:
            ev = json.load(fh)
        print("[figures]")
        plot_errors(ev, os.path.join(a.out, "errors.png"))
        plot_per_point(ev, os.path.join(a.out, "per_point.png"))
    else:
        ev = None

    has_video = bool(a.video) or os.path.isfile(
        os.path.join(a.out, "demo.mp4"))
    if a.video:
        print("[video]")
        write_video(views, labels, X, names, os.path.join(a.out, "demo.mp4"),
                    a.video_side, a.fps)

    if ev is not None:
        write_html(a.out, a.name or a.format, sel, ev, overlays, has_video)
    print("done ->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
