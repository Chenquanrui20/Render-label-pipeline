"""Demo helpers: read an existing 2D label set in the two dataset formats.

These are **not part of the seven pipeline stages**.  They exist so the demos can
replace the interactive GUI (step 04/06) with clicks copied from a label set that
already exists for every rendered frame, and so the produced labels can be scored
against that set.

Both loaders return ``{frame_stem: {point_name: (u, v)}}`` where ``frame_stem``
is the render image name without its extension (``0000_0001.jpg`` ->
``0000_0001``, ``000123.png`` -> ``000123``).

Formats
    charuco   labelme JSON, one file per render, produced by ArUco/ChArUco
              detection on the rendered images.  Only ``shape_type == "point"``
              shapes are kept (the ``board`` rectangle is not a control point).
    ycbv      ``labels_render/points_states.csv`` with one row per
              (frame, keypoint); ``state == "visible"`` rows carry the 2D label
              ``u_px, v_px`` on the rendered image.
"""
import csv
import json
import os


def _finite(u, v):
    return u == u and v == v and abs(u) < 1e12 and abs(v) < 1e12


def load_charuco(labels_dir):
    out = {}
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".json"):
            continue
        stem = os.path.splitext(fn)[0]
        with open(os.path.join(labels_dir, fn), encoding="utf-8") as fh:
            d = json.load(fh)
        pts = {}
        for s in d.get("shapes", []):
            if s.get("shape_type") != "point":
                continue
            u, v = s["points"][0][:2]
            if _finite(u, v):
                pts[s["label"]] = (float(u), float(v))
        if pts:
            out[stem] = pts
    return out


def load_ycbv(csv_path):
    out = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("state") != "visible":
                continue
            u, v = float(r["u_px"]), float(r["v_px"])
            if not _finite(u, v):
                continue
            out.setdefault(r["frame"], {})["kp" + r["kp"]] = (u, v)
    return out


def load(fmt, source):
    if fmt == "charuco":
        return load_charuco(source)
    if fmt == "labelme":          # generic labelme point dir, e.g. stage 07 output
        return load_charuco(source)
    if fmt == "ycbv":
        return load_ycbv(source)
    raise SystemExit("unknown label format %r" % fmt)


def charuco_board_points(labels_dir):
    """name -> [x, y, 0] mm, read from the shape descriptions."""
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(labels_dir, fn), encoding="utf-8") as fh:
            d = json.load(fh)
        out = {}
        for s in d.get("shapes", []):
            desc = (s.get("description") or "").strip()
            if not desc.startswith("board="):
                continue
            xyz = desc[len("board="):].split("mm")[0].strip("() ").split(",")
            if len(xyz) == 3:
                out[s["label"]] = [float(t) for t in xyz]
        if out:
            return out
    return {}


def ycbv_gt_points(points_summary):
    with open(points_summary, encoding="utf-8") as fh:
        d = json.load(fh)
    return {p["name"]: [float(x) for x in p["xyz_mm"]] for p in d["points"]}
