"""Build <scene>/3dgs/ for 3DGS training: N equally spaced frames + an
undistorted PINHOLE COLMAP model.

Stock 3DGS only accepts PINHOLE / SIMPLE_PINHOLE, while a reconstruction made
with SIMPLE_RADIAL cameras (one per image, with k1) is rejected.  This stage

  1. picks N equally spaced registered images,
  2. copies the full model to COLMAP text and keeps only those image rows
     (verbatim, so ids and point2D ordering never shift),
  3. runs COLMAP's image undistorter on the sub-model,
  4. lays the result out as <scene>/3dgs/{images,sparse/0}.

The originals in <scene>/{images,sparse} are left untouched.  Requires pycolmap.

    python -m renderlabel subset <scene> -n 250
"""
import os
import shutil

import numpy as np


def _pycolmap():
    try:
        import pycolmap
    except ImportError:
        raise SystemExit("pycolmap is required for `subset`; install it with "
                         "`conda install -c conda-forge pycolmap`")
    return pycolmap


def select_names(pycolmap, rec, n):
    names = sorted(im.name for im in rec.images.values())
    if n >= len(names):
        return names
    idx = np.linspace(0, len(names) - 1, n).round().astype(int)
    assert len(set(idx.tolist())) == n, "equally-spaced picks collided"
    return [names[i] for i in idx]


def _write_subset_text(full_text_dir, sub_dir, keep_names):
    os.makedirs(sub_dir, exist_ok=True)

    keep_ids, used_cams, img_lines = set(), set(), []
    with open(os.path.join(full_text_dir, "images.txt")) as f:
        lines = iter(f)
        for line in lines:
            if line.startswith("#"):
                continue
            p2d_line = next(lines)
            p = line.split()
            image_id, name = int(p[0]), p[9]
            if name in keep_names:
                keep_ids.add(image_id)
                used_cams.add(int(p[8]))
                img_lines.append(line)
                img_lines.append(p2d_line)
    assert len(keep_ids) == len(keep_names), "%d != %d" % (len(keep_ids), len(keep_names))

    with open(os.path.join(sub_dir, "images.txt"), "w") as f:
        f.write("# Number of images: %d\n" % len(keep_ids))
        f.writelines(img_lines)

    n_cam = 0
    with open(os.path.join(full_text_dir, "cameras.txt")) as fi, \
            open(os.path.join(sub_dir, "cameras.txt"), "w") as fo:
        for line in fi:
            if line.startswith("#") or not line.strip():
                continue
            if int(line.split()[0]) in used_cams:
                fo.write(line)
                n_cam += 1
    assert n_cam == len(used_cams), "%d != %d" % (n_cam, len(used_cams))

    n_pt, n_dropped = 0, 0
    with open(os.path.join(full_text_dir, "points3D.txt")) as fi, \
            open(os.path.join(sub_dir, "points3D.txt"), "w") as fo:
        for line in fi:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            track = [(int(p[8 + 2 * k]), int(p[9 + 2 * k]))
                     for k in range((len(p) - 8) // 2)]
            kept = [t for t in track if t[0] in keep_ids]
            if not kept:
                n_dropped += 1
                continue
            fo.write(" ".join(p[:8] + [str(v) for (i, j) in kept for v in (i, j)]) + "\n")
            n_pt += 1
    print("  subset model: %d images, %d cameras, %d points3D (%d dropped)"
          % (len(keep_ids), n_cam, n_pt, n_dropped))
    return keep_ids


def _check_same_geometry(orig, sub, keep_names):
    by_name = {im.name: im for im in sub.images.values()}
    assert set(by_name) == set(keep_names)
    d_pose = d_p2d = 0.0
    for name in keep_names:
        a = next(im for im in orig.images.values() if im.name == name)
        b = by_name[name]
        d_pose = max(d_pose, np.abs(a.cam_from_world().matrix() -
                                    b.cam_from_world().matrix()).max())
        assert len(a.points2D) == len(b.points2D), "%s: point2D count changed" % name
        xy_a = np.array([q.xy for q in a.points2D])
        xy_b = np.array([q.xy for q in b.points2D])
        d_p2d = max(d_p2d, np.abs(xy_a - xy_b).max())
    print("  geometry check: max pose diff %.2e, max point2D diff %.2e px"
          % (d_pose, d_p2d))
    assert d_pose == 0.0 and d_p2d == 0.0, "text round-trip was not lossless"


def run(a):
    pycolmap = _pycolmap()
    scene_dir = os.path.abspath(a.scene)
    src_model = os.path.join(scene_dir, "sparse", "0")
    src_images = os.path.join(scene_dir, "images")
    dst = a.out or os.path.join(scene_dir, "3dgs")
    tmp = dst + ".tmp"

    rec = pycolmap.Reconstruction(src_model)
    print("[%s] source: %d images, %d cameras, %d points3D"
          % (os.path.basename(scene_dir), rec.num_images(), rec.num_cameras(),
             rec.num_points3D()))
    if a.names_file:
        keep = sorted(l.strip() for l in open(a.names_file, encoding="utf-8") if l.strip())
        missing = [k for k in keep if k not in {im.name for im in rec.images.values()}]
        assert not missing, "frames not in the COLMAP model: %s" % missing[:5]
    else:
        keep = select_names(pycolmap, rec, a.n)
    print("  selected %d frames: %s .. %s" % (len(keep), keep[0], keep[-1]))
    if a.dry:
        return 0

    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    full_text = os.path.join(tmp, "full_text")
    os.makedirs(full_text)
    rec.write_text(full_text)
    sub_text = os.path.join(tmp, "sub_text")
    _write_subset_text(full_text, sub_text, set(keep))
    shutil.rmtree(full_text)

    sub = pycolmap.Reconstruction(sub_text)
    _check_same_geometry(rec, sub, set(keep))

    out = os.path.join(tmp, "undist")
    pycolmap.undistort_images(output_path=out, input_path=sub_text,
                              image_path=src_images, image_names=[])
    urec = pycolmap.Reconstruction(os.path.join(out, "sparse"))
    assert urec.num_images() == len(keep), "%d != %d" % (urec.num_images(), len(keep))
    print("  undistorted: %d images, %d points, cameras %s"
          % (urec.num_images(), urec.num_points3D(),
             sorted({c.model.name for c in urec.cameras.values()})))

    for junk in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
        p = os.path.join(out, junk)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)

    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    os.rename(os.path.join(out, "images"), os.path.join(dst, "images"))
    os.makedirs(os.path.join(dst, "sparse"))
    os.rename(os.path.join(out, "sparse"), os.path.join(dst, "sparse", "0"))
    shutil.rmtree(tmp)
    print("  DONE ->", dst)
    return 0
