"""Camera/pose I/O.

Two on-disk formats are supported:

* **render message** (written by `renderlabel render`): one JSON per rendered
  frame, `<renders>/message/<name>.json`, with the pose in several equivalent
  representations so it can never be misread.  This is the canonical format for
  every later stage.
* **cameras.json** (GaussianObject style): a flat list with `position` (camera
  centre) and `rotation` (camera-to-world).  Written for external annotation
  tools; `load_cameras_json` accepts it too.
"""
import glob
import json
import os

import numpy as np


def _from_message(d):
    """One message dict -> view dict (with consistency checks).

    `T_world2cam` may be omitted when the camera centre is present: it is exactly
    ``-R @ C``, so a message that only stores the centre is still unambiguous.
    """
    R = np.asarray(d["R_world2cam"], np.float64)
    C = np.asarray(d["camera_center_world"], np.float64)
    if "R_cam2world" in d:
        err_R = float(np.abs(np.asarray(d["R_cam2world"]).T - R).max())
        if err_R > 1e-6:
            raise ValueError("%s: R_cam2world.T != R_world2cam (%.2e)"
                             % (d.get("image_name", "?"), err_R))
    if d.get("T_world2cam") is not None:
        t = np.asarray(d["T_world2cam"], np.float64).reshape(3)
        err_t = float(np.abs(-R @ C - t).max())
        if err_t > 1e-5:
            raise ValueError("%s: tvec != -R @ C (%.2e)" % (d.get("image_name", "?"), err_t))
    else:
        t = -R @ C
    return dict(image=d["image_name"],
                render_file=d.get("render_file", d["image_name"] + ".png"),
                R=R, t=t, C=C,
                f=float(d["fx"]), cx=float(d["cx"]), cy=float(d["cy"]),
                k1=float(d.get("k1", 0.0)),
                W=int(d["width"]), H=int(d["height"]))


def load_views(renders_dir):
    """Load every frame under <renders>/message/*.json, sorted by file name."""
    files = sorted(glob.glob(os.path.join(renders_dir, "message", "*.json")))
    files = [p for p in files if not os.path.basename(p).startswith("_")]
    if not files:
        raise SystemExit("no render messages under %s/message" % renders_dir)
    out = []
    for p in files:
        with open(p, encoding="utf-8") as fh:
            out.append(_from_message(json.load(fh)))
    return out


def load_cameras_json(path):
    """GaussianObject-style cameras.json -> view list.

    Expected keys per entry: img_name, width, height, position (C2W), rotation
    (C2W), and either fx/fy/cx/cy or f (+ image_size).  k1 defaults to 0.
    """
    with open(path, encoding="utf-8") as fh:
        entries = json.load(fh)
    out = []
    for e in entries:
        Rc2w = np.asarray(e["rotation"], np.float64)
        C = np.asarray(e["position"], np.float64)
        R = Rc2w.T
        t = -R @ C
        f = float(e.get("fx", e.get("f")))
        cx = float(e.get("cx", e["width"] / 2.0))
        cy = float(e.get("cy", e["height"] / 2.0))
        out.append(dict(image=os.path.splitext(e["img_name"])[0],
                        render_file=e["img_name"], R=R, t=t, C=C,
                        f=f, cx=cx, cy=cy, k1=float(e.get("k1", 0.0)),
                        W=int(e["width"]), H=int(e["height"])))
    return out


def save_cameras_json(views, path):
    """Write the GaussianObject-style list (camera-to-world) for external tools."""
    entries = [{
        "id": i,
        "img_name": v["render_file"],
        "width": v["W"], "height": v["H"],
        "position": [float(x) for x in v["C"]],
        "rotation": [[float(x) for x in r] for r in np.asarray(v["R"]).T],
        "fx": float(v["f"]), "fy": float(v["f"]),
        "cx": float(v["cx"]), "cy": float(v["cy"]),
        "k1": float(v.get("k1", 0.0)),
    } for i, v in enumerate(views)]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=1)


def index_by_file(views):
    return {v["render_file"]: k for k, v in enumerate(views)}


def resolve_frames(spec, views, selection=None):
    """Turn a --frames argument into a list of view indices.

    Accepted forms:
      * ``seed``     -> the two seed frames stored in selection.json
      * ``selected`` -> the N_sat frames stored in selection.json
      * ``all``      -> every view
      * comma-separated frame names or 0-based indices
      * a path to a text file with one frame name per line
    """
    by_file = index_by_file(views)
    if spec in ("seed", "selected", "all"):
        if spec == "all":
            return list(range(len(views)))
        if not selection:
            raise SystemExit("--frames %s needs --selection (run `select` first)" % spec)
        with open(selection, encoding="utf-8") as fh:
            sel = json.load(fh)
        key = "seed_files" if spec == "seed" else "selected_files"
        names = sel.get(key)
        if not names:
            raise SystemExit("%s has no %s" % (selection, key))
        return [by_file[n] for n in names]
    if os.path.isfile(spec):
        names = [l.strip() for l in open(spec, encoding="utf-8") if l.strip()]
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for n in names:
        if n in by_file:
            out.append(by_file[n])
        else:
            try:
                out.append(int(n))
            except ValueError:
                raise SystemExit("unknown frame %r (not a render_file name or index)" % n)
    return out


def images_dir(renders_dir):
    return os.path.join(renders_dir, "images")


def frame_path(renders_dir, v):
    return os.path.join(images_dir(renders_dir), v["render_file"])
