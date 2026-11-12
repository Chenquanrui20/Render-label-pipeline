"""Command line entry point: python -m renderlabel <stage> ...

Stages
    subset       pick N frames and build the undistorted 3DGS training scene
    render       render every COLMAP pose with the trained 3DGS model
    select       predict N_sat and choose which frames to annotate
    annotate     click the keypoints on rendered frames (OpenCV window)
    triangulate  triangulate the clicks and project labels into all renders
"""
import argparse
import os


def _default(scene, *parts):
    return os.path.join(scene, *parts)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="renderlabel")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("subset", help="build <scene>/3dgs for training")
    p.add_argument("scene")
    p.add_argument("-n", type=int, default=250, help="number of frames (default 250)")
    p.add_argument("--out", default=None)
    p.add_argument("--names-file", default=None,
                   help="explicit frame list, one name per line (overrides -n)")
    p.add_argument("--dry", action="store_true")

    p = sub.add_parser("render", help="render all COLMAP poses")
    p.add_argument("scene")
    p.add_argument("--gs-repo", default=os.environ.get("GS_REPO"),
                   help="path to the gaussian-splatting repository")
    p.add_argument("--subdir", default="3dgs")
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--camera", choices=("shared", "per-image"), default="shared")
    p.add_argument("--width", type=int, default=None,
                   help="default: the COLMAP image width")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fx", type=float, default=None,
                   help="default: median focal length of the COLMAP cameras")
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--limit", type=int, default=None, help="render only the first N")

    p = sub.add_parser("annotate", help="click points on rendered frames")
    p.add_argument("scene")
    p.add_argument("--renders", default=None)
    p.add_argument("--frames", default="seed",
                   help="seed | selected | all | name1,name2 | @file")
    p.add_argument("--selection", default=None)
    p.add_argument("--points-file", default=None)
    p.add_argument("--points", default=None, help="comma list, e.g. P1,P2,P3")
    p.add_argument("--init", default=None, help="points3d_seed.json for guide rings")
    p.add_argument("--out", default=None)
    p.add_argument("--max-width", type=int, default=1600)

    p = sub.add_parser("select", help="predict N_sat and choose frames")
    p.add_argument("scene")
    p.add_argument("--renders", default=None)
    p.add_argument("--annotations", default=None,
                   help="omit for the pose-only seed stage")
    p.add_argument("--selection", default=None)
    p.add_argument("--theta-eff", type=float, default=11.5,
                   help="per-view angular gain in degrees (default 11.5)")
    p.add_argument("--guard", type=float, default=5.0,
                   help="minimum ray angle between selected views (default 5)")
    p.add_argument("--seed-max-angle", type=float, default=160.0,
                   help="never pick a seed pair wider than this (default 160; a "
                        "pair at ~180 puts the object on the baseline and makes "
                        "the seed triangulation degenerate; 180 disables)")
    p.add_argument("--margin", type=float, default=20.0,
                   help="pixels a point must stay inside the image to count as visible")
    p.add_argument("--refine", action="store_true")
    p.add_argument("--points3d", default=None)
    p.add_argument("--out-annot", default=None, help="only used in the printed hint")

    p = sub.add_parser("triangulate", help="triangulate and project labels")
    p.add_argument("scene")
    p.add_argument("--renders", default=None)
    p.add_argument("--annotations", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--overlay", default=None,
                   help="overlay directory; pass '' to disable")
    p.add_argument("--refine", action="store_true", help="LM refinement per point")
    p.add_argument("--min-views", type=int, default=2)

    a = ap.parse_args(argv)

    if a.cmd == "subset":
        from . import subset3dgs
        return subset3dgs.run(a)

    if a.cmd == "render":
        from . import render_3dgs
        return render_3dgs.run(a)

    a.renders = a.renders or _default(a.scene, "3dgs", "renders")
    if a.cmd == "annotate":
        if a.out is None:
            a.out = _default(a.scene, "work", "seed.json")
        from . import annotate
        return annotate.main(a)
    if a.cmd == "select":
        if a.selection is None:
            a.selection = _default(a.scene, "work", "selection.json")
        from . import select_views
        return select_views.run(a)
    if a.cmd == "triangulate":
        if a.out is None:
            a.out = _default(a.scene, "3dgs", "labels")
        if a.overlay is None:
            a.overlay = _default(a.scene, "3dgs", "overlay")
        if a.overlay == "":
            a.overlay = None
        from . import triangulate_project
        return triangulate_project.run(a)
    return 2
