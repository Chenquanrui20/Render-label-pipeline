"""Render every COLMAP pose of a scene with a trained 3DGS model.

This reuses the reconstruction's poses (<scene>/sparse/0) and the 3DGS model in
<scene>/3dgs/output/point_cloud/iteration_N/point_cloud.ply, and writes

    <scene>/3dgs/renders/images/<image stem>.png
    <scene>/3dgs/renders/message/<image name>.json

The default camera is ONE shared pinhole for all views: fx = median focal of the
COLMAP cameras, principal point at the image centre, no distortion.  A single
shared camera is what makes the renders one consistent metric space - per-image
COLMAP intrinsics (SIMPLE_RADIAL, f drifting frame to frame) would put every view
in its own pixel space and break triangulation.

Runs inside the 3DGS environment (torch + the gaussian-splatting repo):

    GS_REPO=/path/to/gaussian-splatting python -m renderlabel render <scene>
    python -m renderlabel render <scene> --camera per-image   # only for PINHOLE sources
"""
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np
from PIL import Image

_MOD = {}


def _import_gs(gs_repo):
    if not gs_repo or not os.path.isdir(gs_repo):
        raise SystemExit("gaussian-splatting repo not found: %r (set GS_REPO or --gs-repo)"
                         % gs_repo)
    sys.path.insert(0, gs_repo)
    import torch
    import torchvision
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from utils.read_write_model import (read_cameras_binary, read_images_binary,
                                        read_points3D_binary)
    _MOD.update(torch=torch, torchvision=torchvision, render=render,
                Camera=Camera, GaussianModel=GaussianModel,
                read_cameras_binary=read_cameras_binary,
                read_images_binary=read_images_binary,
                read_points3D_binary=read_points3D_binary)


def _read_model(model_dir):
    return (_MOD["read_cameras_binary"](os.path.join(model_dir, "cameras.bin")),
            _MOD["read_images_binary"](os.path.join(model_dir, "images.bin")),
            _MOD["read_points3D_binary"](os.path.join(model_dir, "points3D.bin")))


class _Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def _model_kind(cam):
    return str(getattr(cam, "model", "")).split(".")[-1]


def _focal(cam):
    """fx (and fy) from a COLMAP camera record."""
    p = np.asarray(cam.params, float)
    kind = _model_kind(cam)
    if kind in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        return float(p[0]), float(p[0])
    if kind in ("PINHOLE", "RADIAL"):
        return float(p[0]), float(p[1])
    if kind in ("OPENCV_ROUND",):
        return float(p[0]), float(p[1])
    raise SystemExit("unsupported camera model %s" % kind)


def auto_intrinsics(cams):
    """Median focal and the (uniform) image size of the reconstruction."""
    if not cams:
        raise SystemExit("empty COLMAP model")
    sizes = {(int(c.width), int(c.height)) for c in cams.values()}
    if len(sizes) != 1:
        raise SystemExit("cameras have mixed sizes %s; pass --width/--height/--fx "
                         "explicitly" % sorted(sizes))
    W, H = sizes.pop()
    fx = float(np.median([_focal(c)[0] for c in cams.values()]))
    fy = float(np.median([_focal(c)[1] for c in cams.values()]))
    return W, H, fx, fy


def _make_camera(im, K, resolution, uid, image):
    torch, Camera = _MOD["torch"], _MOD["Camera"]
    W, H = resolution
    fx, fy, cx, cy = K
    R = np.transpose(np.asarray(im.qvec2rotmat(), float))
    T = np.asarray(im.tvec, float)
    FoVx = 2.0 * math.atan(W / (2.0 * fx))
    FoVy = 2.0 * math.atan(H / (2.0 * fy))
    cam = Camera(resolution=(W, H), colmap_id=im.camera_id, R=R, T=T,
                 FoVx=FoVx, FoVy=FoVy, depth_params=None, image=image,
                 invdepthmap=None, image_name=im.name, uid=uid, data_device="cuda")
    return cam, R, T


def _render_name(image_name):
    """Render file name: the COLMAP image stem plus .png (never name.png.png)."""
    return os.path.splitext(image_name)[0] + ".png"


def _render_parameters(cam, R, T, K, resolution, im, cam_info, iteration, bg):
    W, H = resolution
    fx, fy, cx, cy = K
    return OrderedDict([
        ("image_name", im.name), ("render_file", _render_name(im.name)),
        ("width", W), ("height", H),
        ("intrinsics_model", "PINHOLE"),
        ("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy), ("k1", 0.0),
        ("K", [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        ("distortion", None),
        ("qvec_wxyz", np.asarray(im.qvec).tolist()),
        ("tvec", np.asarray(im.tvec).tolist()),
        ("R_cam2world", R.tolist()),
        ("R_world2cam", np.transpose(R).tolist()),
        ("T_world2cam", T.tolist()),
        ("camera_center_world", cam.camera_center.detach().cpu().numpy().tolist()),
        ("colmap_camera_id", im.camera_id),
        ("source_camera_model", _model_kind(cam_info)),
        ("source_camera_params", np.asarray(cam_info.params).tolist()),
        ("source_camera_size", [int(cam_info.width), int(cam_info.height)]),
        ("source_image", im.name),
        ("gs_iteration", iteration), ("background_rgb", bg),
    ])


def run(a):
    torch, render, GaussianModel = (None, None, None)
    _import_gs(a.gs_repo)
    torch = _MOD["torch"]
    render = _MOD["render"]
    GaussianModel = _MOD["GaussianModel"]

    scene = os.path.abspath(a.scene)
    sub = a.subdir
    model_dir = os.path.join(scene, "sparse", "0")
    cams, imgs, pts = _read_model(model_dir)

    if a.fx is None or a.width is None or a.height is None:
        W0, H0, fx0, fy0 = auto_intrinsics(cams)
    W = a.width or W0
    H = a.height or H0
    fx = a.fx if a.fx is not None else fx0
    fy = a.fy if a.fy is not None else (fx if a.fx is not None else fy0)
    cx = a.cx if a.cx is not None else W / 2.0
    cy = a.cy if a.cy is not None else H / 2.0
    K = (fx, fy, cx, cy)
    background = [0.0, 0.0, 0.0]
    bg = torch.tensor(background, dtype=torch.float32, device="cuda")

    ply = os.path.join(scene, sub, "output", "point_cloud",
                       "iteration_%d" % a.iteration, "point_cloud.ply")
    if not os.path.isfile(ply):
        raise SystemExit("no trained model at %s" % ply)
    gaussians = GaussianModel(3)
    try:
        gaussians.load_ply(ply, use_train_test_exp=False)
    except TypeError:                      # older forks without exposure support
        gaussians.load_ply(ply)
    print("[%s] %d gaussians, %d poses" % (os.path.basename(scene),
                                           gaussians.get_xyz.shape[0], len(imgs)))
    print("  camera: shared %dx%d fx=%.3f fy=%.3f c=(%.2f, %.2f)"
          % (W, H, fx, fy, cx, cy))

    if a.camera == "per-image":
        bad = sorted({_model_kind(cams[i.camera_id]) for i in imgs.values()} - {"PINHOLE"})
        assert not bad, "--camera per-image needs PINHOLE sources, got %s" % bad

    out_root = os.path.join(scene, sub, "renders")
    img_dir = os.path.join(out_root, "images")
    msg_dir = os.path.join(out_root, "message")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msg_dir, exist_ok=True)
    dummy = Image.new("RGB", (1, 1), (0, 0, 0))

    items = sorted(imgs.values(), key=lambda i: i.name)
    if a.limit:
        items = items[:a.limit]
    for idx, im in enumerate(items):
        cam_info = cams[im.camera_id]
        if a.camera == "per-image":
            p = np.asarray(cam_info.params, float)
            K_use = (p[0], p[1], p[2], p[3])
            res_use = (int(cam_info.width), int(cam_info.height))
        else:
            K_use, res_use = K, (W, H)
        with torch.no_grad():
            cam, R, T = _make_camera(im, K_use, res_use, idx, dummy)
            try:
                out = render(cam, gaussians, _Pipe(), bg,
                             separate_sh=False, use_trained_exp=False)["render"]
            except TypeError:              # older forks without these kwargs
                out = render(cam, gaussians, _Pipe(), bg)["render"]
            rendering = torch.clamp(out, 0.0, 1.0)
            _MOD["torchvision"].utils.save_image(
                rendering, os.path.join(img_dir, _render_name(im.name)))
            params = _render_parameters(cam, R, T, K_use, res_use, im, cam_info,
                                        a.iteration, background)
            with open(os.path.join(msg_dir, im.name + ".json"), "w") as f:
                json.dump(params, f, indent=2)
            del cam, rendering
        if (idx + 1) % 100 == 0 or idx + 1 == len(items):
            print("  rendered %d/%d" % (idx + 1, len(items)), flush=True)

    summary = OrderedDict([
        ("scene", os.path.basename(scene)), ("scene_dir", scene),
        ("camera_mode", a.camera), ("n_rendered", len(items)),
        ("n_poses_in_sparse", len(imgs)),
        ("intrinsics", {"model": "PINHOLE", "fx": K[0], "fy": K[1],
                        "cx": K[2], "cy": K[3], "width": W, "height": H}),
        ("gs_model", ply), ("gs_iteration", a.iteration),
        ("background_rgb", background),
        ("images_dir", img_dir), ("message_dir", msg_dir),
    ])
    with open(os.path.join(msg_dir, "_intrinsics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  DONE: %d renders -> %s" % (len(items), out_root))
    return 0
