"""Audit the artefacts of stages 01-03 before running the annotation stages.

    python3 examples/check_stages.py --scene /data/seq \
        --sparse /data/seq/sparse/0 --subdir 3dgs --renders /data/seq/3dgs/renders

Prints, without writing anything:

    01  the COLMAP model: registered images, cameras, points, mean reprojection
        error, camera model (SIMPLE_RADIAL is expected *before* the subset step);
    02  the training scene: number of subset images and the trained point cloud
        (the subset model must be PINHOLE -- stock 3DGS rejects SIMPLE_RADIAL);
    03  the render set: frame count, the shared pinhole, and whether every
        message has its PNG.
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from renderlabel import views as V  # noqa: E402


def _model_stats(path):
    try:
        import pycolmap
    except ImportError:
        return None
    rec = pycolmap.Reconstruction(path)
    cams = list(rec.cameras.values())
    out = {"images": rec.num_images(), "cameras": rec.num_cameras(),
           "points3D": rec.num_points3D(),
           "models": sorted({c.model.name for c in cams})}
    try:
        out["mean_reproj_px"] = float(rec.compute_mean_reprojection_error())
    except Exception:
        pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sparse", default=None, help="default <scene>/sparse/0")
    ap.add_argument("--subdir", default="3dgs", help="training dir inside scene (default 3dgs)")
    ap.add_argument("--renders", default=None, help="default <scene>/<subdir>/renders")
    ap.add_argument("--iteration", type=int, default=30000)
    a = ap.parse_args(argv)

    scene = os.path.abspath(a.scene)
    sparse = a.sparse or os.path.join(scene, "sparse", "0")
    subdir = a.subdir
    renders = a.renders or os.path.join(scene, subdir, "renders")

    print("scene:", scene)
    print("\n[01] COLMAP model:", sparse)
    st = _model_stats(sparse)
    if st is None:
        print("     pycolmap not installed -- skipped")
    else:
        print("     %d images, %d cameras, %d points3D, model(s) %s"
              % (st["images"], st["cameras"], st["points3D"], ", ".join(st["models"])))
        if "mean_reproj_px" in st:
            print("     mean reprojection error %.3f px" % st["mean_reproj_px"])

    print("\n[02] 3DGS training scene:", os.path.join(scene, subdir))
    img_dir = os.path.join(scene, subdir, "images")
    if os.path.isdir(img_dir):
        print("     subset images: %d" % len(os.listdir(img_dir)))
    else:
        print("     missing", img_dir)
    sub_model = os.path.join(scene, subdir, "sparse", "0")
    st2 = _model_stats(sub_model) if os.path.isdir(sub_model) else None
    if st2 is not None:
        print("     subset model: %d images, model(s) %s (PINHOLE expected)"
              % (st2["images"], ", ".join(st2["models"])))
    ply = os.path.join(scene, subdir, "output", "point_cloud",
                       "iteration_%d" % a.iteration, "point_cloud.ply")
    if os.path.isfile(ply):
        print("     point cloud: %s (%.1f MB)"
              % (ply, os.path.getsize(ply) / 1e6))
    else:
        print("     point cloud MISSING:", ply)

    print("\n[03] render set:", renders)
    views = V.load_views(renders)
    v0 = views[0]
    print("     %d messages, shared camera fx=%.3f fy=%.3f c=(%.2f, %.2f) %dx%d"
          % (len(views), v0["f"], v0["f"], v0["cx"], v0["cy"], v0["W"], v0["H"]))
    missing = [v["render_file"] for v in views
               if not os.path.isfile(os.path.join(renders, "images", v["render_file"]))]
    print("     images present: %d/%d%s"
          % (len(views) - len(missing), len(views),
             ("; missing e.g. %s" % missing[:3]) if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
