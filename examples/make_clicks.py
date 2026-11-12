"""Write GUI-annotation JSONs from an existing label set (demo shortcut).

The real pipeline clicks points by hand in `renderlabel annotate`.  A demo needs
to run unattended, so this script copies the 2D labels that the dataset already
provides for the chosen frames into the same JSON format the annotator writes:

    {"points": ["P1", ...],
     "frames": {"<render_file>": {"P1": [u, v], ...}, ...},
     "source": "...", "saved_at": "..."}

Only the frames named on the command line get clicks; the rest of the pipeline
is unchanged.  This is a demo convenience, **not** a replacement for annotation
on a new capture.

    python3 examples/make_clicks.py --format charuco \
        --renders /data/scene/3dgs/renders --labels /data/scene/3dgs/renders/labels \
        --frames-from work/selection.json --which seed --out work/seed.json
"""
import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels_io  # noqa: E402
from renderlabel import views as V  # noqa: E402


def _frames_from_selection(path, which):
    with open(path, encoding="utf-8") as fh:
        sel = json.load(fh)
    key = "seed_files" if which == "seed" else "selected_files"
    files = sel.get(key)
    if not files:
        raise SystemExit("%s has no %s -- run the matching select stage first"
                         % (path, key))
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True, choices=("charuco", "ycbv"))
    ap.add_argument("--renders", required=True,
                    help="render set directory (contains images/ and message/)")
    ap.add_argument("--labels", required=True,
                    help="charuco: labelme directory; ycbv: points_states.csv")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--frames", help="comma-separated render_file names")
    g.add_argument("--frames-from", help="selection.json written by `renderlabel select`")
    ap.add_argument("--which", choices=("seed", "selected"), default="seed",
                    help="which key of --frames-from to use (default seed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.frames_from:
        frames = _frames_from_selection(a.frames_from, a.which)
    else:
        frames = [s.strip() for s in a.frames.split(",") if s.strip()]

    views = V.load_views(a.renders)
    by_file = V.index_by_file(views)
    labels = labels_io.load(a.format, a.labels)
    missing = [f for f in frames if f not in by_file]
    if missing:
        raise SystemExit("frames not in the render set: %s" % missing[:5])

    order, frames_out = [], {}
    for f in frames:
        stem = os.path.splitext(views[by_file[f]]["image"])[0]
        pts = labels.get(stem)
        if not pts:
            print("  warning: no labels for %s (stem %s) -- frame skipped" % (f, stem))
            continue
        frames_out[f] = {n: [float(u), float(v)] for n, (u, v) in pts.items()}
        for n in pts:
            if n not in order:
                order.append(n)

    if not frames_out:
        raise SystemExit("none of the requested frames has a label")
    # ycbv keypoint names sort numerically; charuco keeps the label-file order
    if a.format == "ycbv":
        order.sort(key=lambda s: int(s[2:]) if s.startswith("kp") else 10 ** 9)

    payload = {"points": order, "frames": frames_out,
               "source": os.path.abspath(a.labels),
               "note": ("simulated clicks copied from an existing '%s' label set; "
                        "the interactive annotator was not used" % a.format),
               "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    counts = [len(v) for v in frames_out.values()]
    print("wrote %s: %d/%d frames, %d points (per-frame clicks %d..%d)"
          % (a.out, len(frames_out), len(frames), len(order), min(counts), max(counts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
