"""Geometry core: pinhole projection, DLT triangulation, LM refinement, and the
saturation/view-selection rules used by the pipeline.

All functions operate on plain numpy arrays and camera dicts with the keys

    R (3,3)  world -> camera rotation
    t (3,)   world -> camera translation
    C (3,)   camera centre in world coordinates
    f, cx, cy, k1   intrinsics (k1 = radial distortion; renders use k1 = 0)
    W, H     image size in pixels

The selection rules follow the paper implementation exactly:
  * ray direction  = unit vector from the object point X to the camera centre,
  * Theta_span     = largest pairwise angle between candidate ray directions,
  * N_sat          = ceil(Theta_span / (2 * theta_eff)),
  * greedy angular coverage starting from the farthest pair, with a hard
    minimum-angle guard between selected rays.
"""
import numpy as np

# ---------------------------------------------------------------- projection


def undistort_norm(uv, f, cx, cy, k1, iters=10):
    """Pixel -> normalised image coordinates (iterative inverse of the k1 model).

    The renders in this pipeline are undistorted (k1 = 0), but the function is
    kept for sources that still carry a radial term.
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    xd = (uv[:, 0] - cx) / f
    yd = (uv[:, 1] - cy) / f
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        x = xd / (1.0 + k1 * r2)
        y = yd / (1.0 + k1 * r2)
    return np.stack([x, y], 1)


def project_norm(xn, f, cx, cy, k1):
    """Normalised coordinates -> pixels (applies the same k1 model)."""
    xn = np.asarray(xn, np.float64).reshape(-1, 2)
    r2 = (xn ** 2).sum(1)
    xd = xn * (1.0 + k1 * r2)[:, None]
    return np.stack([f * xd[:, 0] + cx, f * xd[:, 1] + cy], 1)


def project_points(X, view):
    """World points (K,3) -> pixels (K,2) and a validity mask (K,).

    A point is invalid when it falls behind the camera (z <= 0).
    """
    X = np.asarray(X, np.float64).reshape(-1, 3)
    Xc = X @ np.asarray(view["R"], np.float64).T + np.asarray(view["t"], np.float64)
    z = Xc[:, 2]
    ok = z > 1e-9
    uv = np.full((len(X), 2), np.nan)
    if ok.any():
        uv[ok, 0] = view["f"] * Xc[ok, 0] / z[ok] + view["cx"]
        uv[ok, 1] = view["f"] * Xc[ok, 1] / z[ok] + view["cy"]
    return uv, ok


def inside_image(uv, view, margin=0.0):
    """True when every point is inside the image with `margin` px to the border."""
    if not np.isfinite(uv).all():
        return False
    return bool(uv[:, 0].min() >= margin and uv[:, 1].min() >= margin and
                uv[:, 0].max() <= view["W"] - 1 - margin and
                uv[:, 1].max() <= view["H"] - 1 - margin)


# ------------------------------------------------------------- triangulation


def triangulate_dlt(obs_n, views):
    """Multi-view DLT. obs_n: (V,2) normalised observations, views: list of camera dicts."""
    A = np.empty((2 * len(views), 4), np.float64)
    for i, v in enumerate(views):
        R = np.asarray(v["R"], np.float64)
        t = np.asarray(v["t"], np.float64).reshape(3)
        P1 = np.concatenate([R[0], t[0:1]])
        P2 = np.concatenate([R[1], t[1:2]])
        P3 = np.concatenate([R[2], t[2:3]])
        x, y = obs_n[i]
        A[2 * i] = x * P3 - P1
        A[2 * i + 1] = y * P3 - P2
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    if abs(h[3]) < 1e-12:
        return None
    return h[:3] / h[3]


def refine_lm(X0, obs_px, views):
    """Levenberg-Marquardt refinement of one 3D point against its pixel clicks."""
    from scipy.optimize import least_squares

    def resid(x):
        out = []
        for v, uv in zip(views, obs_px):
            Xc = np.asarray(v["R"]) @ x + np.asarray(v["t"]).reshape(3)
            if Xc[2] <= 1e-9:
                return np.full(2 * len(views), 1e6)
            xn = (Xc[:2] / Xc[2]).reshape(1, 2)
            out.append(project_norm(xn, v["f"], v["cx"], v["cy"], v.get("k1", 0.0))[0] - uv)
        return np.concatenate(out)

    r = least_squares(resid, X0, method="lm", xtol=1e-12, ftol=1e-12)
    return r.x


def _solve_subset(clicks, idx, refine):
    vs = [clicks[i][0] for i in idx]
    obs = np.array([clicks[i][1] for i in idx], float)
    obs_n = np.vstack([undistort_norm(o.reshape(1, 2), v["f"], v["cx"], v["cy"],
                                      v.get("k1", 0.0))[0] for o, v in zip(obs, vs)])
    X = triangulate_dlt(obs_n, vs)
    if X is None:
        return None
    if refine:
        X = refine_lm(X, obs, vs)
    return X


def triangulate_point(clicks, refine=False, reject_px=3.0):
    """Triangulate one point from `clicks = [(view, (u,v)), ...]`.

    Robustness: every view pair proposes a hypothesis; the hypothesis that keeps
    the most clicks within `reject_px` wins (ties broken by the lower median
    residual), and the point is finally re-solved on that inlier set.  With three
    views and one mis-click the problem is ambiguous by construction (any two
    views explain a point exactly), so rejection starts at four clicks; below
    that all clicks are used.

    Returns (X, used_clicks, residuals_px) or (None, [], []).
    """
    clicks = [(v, np.asarray(uv, float)) for v, uv in clicks
              if uv is not None and np.isfinite(np.asarray(uv, float)).all()]
    n = len(clicks)
    if n < 2:
        return None, [], []
    if n < 4:
        X = _solve_subset(clicks, list(range(n)), refine)
        if X is None:
            return None, [], []
        res = _residuals(X, clicks)
        return X, list(clicks), res

    best = None
    for i in range(n):
        for j in range(i + 1, n):
            X = _solve_subset(clicks, [i, j], refine=False)
            if X is None:
                continue
            res = _residuals(X, clicks)
            inl = res <= reject_px
            cnt = int(inl.sum())
            med = float(np.median(res[inl])) if cnt else float("inf")
            score = (cnt, -med)
            if best is None or score > best[0]:
                best = (score, inl)
    if best is None or best[0][0] < 2:
        return None, [], []
    used = [k for k in range(n) if best[1][k]]
    if len(used) < 3:                 # no majority to reject against
        used = list(range(n))
    X = _solve_subset(clicks, used, refine)
    if X is None:
        return None, [], []
    res = _residuals(X, [clicks[k] for k in used])
    return X, [clicks[k] for k in used], res


def _residuals(X, clicks):
    out = []
    for v, uv in clicks:
        uv_p, ok = project_points(np.asarray(X).reshape(1, 3), v)
        out.append(np.linalg.norm(uv_p[0] - uv) if ok[0] else np.inf)
    return np.array(out)


# ------------------------------------------------------- view selection rules


def ray_dirs(views, X):
    """Unit vectors from the object point X to every camera centre (the only
    directional quantity the selection stage is allowed to use)."""
    C = np.array([v["C"] for v in views], np.float64)
    d = C - np.asarray(X, np.float64).reshape(3)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def pose_proxy_point(views):
    """Least-squares optical-axis intersection - a pose-only stand-in for X0.

    Used before any annotation exists, solely to pick the two seed frames.
    """
    P = np.zeros((3, 3))
    b = np.zeros(3)
    for v in views:
        d = np.asarray(v["R"], np.float64).T @ np.array([0.0, 0.0, 1.0])
        M = np.eye(3) - np.outer(d, d)
        P += M
        b += M @ np.asarray(v["C"], np.float64)
    return np.linalg.solve(P, b)


def farthest_pair(dirs, cands):
    """The candidate pair with the largest angle between its ray directions."""
    cands = list(cands)
    D = dirs[cands]
    iu = np.triu_indices(len(cands), 1)
    a = np.degrees(np.arccos(np.clip(D @ D.T, -1, 1)))
    i = int(np.argmax(a[iu]))
    return [cands[iu[0][i]], cands[iu[1][i]]], float(a[iu[0][i], iu[1][i]])


def pair_near_angle(dirs, cands, target):
    """The candidate pair with the largest angle not exceeding `target`.

    Fallback for a full 360-degree capture, where the farthest pair is ~180
    degrees: two rays that are exactly opposite put the object on the baseline,
    so its depth is unobservable and triangulating the seed point is degenerate.
    Returns ([i, j], angle).
    """
    cands = list(cands)
    D = dirs[cands]
    iu = np.triu_indices(len(cands), 1)
    a = np.degrees(np.arccos(np.clip(D @ D.T, -1, 1)))[iu]
    ok = a <= target
    if not ok.any():
        return farthest_pair(dirs, cands)
    i = int(np.argmax(np.where(ok, a, -1.0)))
    return [cands[iu[0][i]], cands[iu[1][i]]], float(a[i])


def theta_max(dirs, cands, sel):
    """Coverage radius: the worst candidate's angle to its nearest selected view."""
    if not sel:
        return float("nan")
    a = np.degrees(np.arccos(np.clip(dirs[list(cands)] @ dirs[list(sel)].T, -1, 1)))
    return float(a.min(1).max())


def greedy_ang(dirs, k, cands, guard=5.0, start=None):
    """Greedy angular coverage.

    Starts from `start` (or the farthest pair), then repeatedly adds the frame
    whose nearest selected ray has the largest angle, subject to that angle being
    at least `guard` degrees.  If no candidate satisfies the guard the search
    stops (hard truncation, matching the deployment code) - it never relaxes the
    guard.  Returns (sorted selected indices, theta_max).
    """
    cands = list(cands)
    if start is None:
        start, _ = farthest_pair(dirs, cands)
    sel = [s for s in start if s in cands][:2]
    if len(sel) < 2:
        sel = [s for s in start if s in cands][:1] or [cands[0]]
    while len(sel) < k and len(sel) < len(cands):
        cm = np.degrees(np.arccos(np.clip(dirs[cands] @ dirs[sel].T, -1, 1))).min(1)
        pos = {s: j for j, s in enumerate(cands)}
        for s in sel:
            if s in pos:
                cm[pos[s]] = -1.0
        ok = cm >= guard
        if not ok.any():
            break
        sel.append(cands[int(np.argmax(np.where(ok, cm, -1.0)))])
    return sorted(sel), theta_max(dirs, cands, sel)


def theta_span(dirs, cands):
    """Largest pairwise angle between candidate ray directions (the cone to cover)."""
    cands = list(cands)
    D = dirs[cands]
    if len(D) < 2:
        return 0.0
    iu = np.triu_indices(len(D), 1)
    return float(np.degrees(np.arccos(np.clip((D @ D.T)[iu], -1, 1))).max())


def n_sat(theta_span_deg, theta_eff_deg):
    """Predicted saturated view count: ceil(Theta_span / (2 * theta_eff))."""
    if theta_eff_deg <= 0:
        raise ValueError("theta_eff must be > 0")
    return max(1, int(np.ceil(theta_span_deg / (2.0 * theta_eff_deg) - 1e-9)))


def angle_matrix(dirs, idx_a, idx_b):
    """Pairwise ray angles (degrees) between two index lists."""
    a = np.degrees(np.arccos(np.clip(dirs[list(idx_a)] @ dirs[list(idx_b)].T, -1, 1)))
    return a
