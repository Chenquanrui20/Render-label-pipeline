# 03 — Render every COLMAP pose

**Script:** `scripts/03_render.sh`
**Input:** `<SCENE>/sparse/0/`, `<SCENE>/3dgs/output/point_cloud/iteration_<ITER>/point_cloud.ply`
**Output:** `<SCENE>/3dgs/renders/images/<stem>.png`,
`<SCENE>/3dgs/renders/message/<image name>.json`

Every pose in the reconstruction gets one render. Because the poses come from
COLMAP, a render and the original real frame are *the same viewpoint* — the
renders form a new observation of the same capture, in a controlled pixel space.

## Command

```bash
SCENE=/data/box1 bash scripts/03_render.sh
```

This is:

```bash
GS_REPO=/path/to/gaussian-splatting \
python -m renderlabel render $SCENE --iteration 30000 --camera shared
```

It must run in the 3DGS environment (it imports torch and the
gaussian-splatting modules). The step script does this via `conda run`.

## One camera for all views

The default is a **single shared pinhole**:

```
fx = fy = median focal length of the COLMAP cameras
cx, cy = image centre
k1     = 0  (the undistorted images have no distortion)
```

This is the key to the whole method: all renders live in one metric pixel
space, so a 3D point triangulated from clicks in a few views reprojects exactly
into the others. Per-image COLMAP focal lengths would give every view its own
space and silently bias every label.

Override only with a reason:

```bash
python -m renderlabel render $SCENE --fx 1715 --cx 960 --cy 540
python -m renderlabel render $SCENE --camera per-image   # PINHOLE sources only
```

## The message file

One JSON per frame, next to the PNG. It carries the pose in three equivalent
forms plus the intrinsics, so no consumer can misread a convention:

```json
{
  "image_name": "frame_00042.png",
  "render_file": "frame_00042.png",
  "width": 1920, "height": 1080,
  "intrinsics_model": "PINHOLE",
  "fx": 1715.0, "fy": 1715.0, "cx": 960.0, "cy": 540.0, "k1": 0.0,
  "K": [[1715.0, 0.0, 960.0], [0.0, 1715.0, 540.0], [0.0, 0.0, 1.0]],
  "qvec_wxyz": [0.99, 0.01, -0.02, 0.03],
  "tvec": [-1.2, 0.3, 4.5],
  "R_cam2world": [[...], [...], [...]],
  "R_world2cam": [[...], [...], [...]],
  "T_world2cam": [-1.2, 0.3, 4.5],
  "camera_center_world": [1.1, -0.2, -4.4],
  "source_camera_model": "PINHOLE",
  "source_camera_params": [1715.0, 1715.0, 960.0, 540.0],
  "gs_iteration": 30000,
  "background_rgb": [0.0, 0.0, 0.0]
}
```

Conventions (OpenCV / COLMAP):

- `R_world2cam`, `T_world2cam` map a world point `X` to camera coordinates
  `Xc = R @ X + t`, camera looks along `+z`.
- `camera_center_world = -R.T @ t`.
- `R_cam2world = R_world2cam.T`.

`<renders>/message/_intrinsics.json` records the shared camera and the source
of the 3DGS model.

## Checks

```bash
ls $SCENE/3dgs/renders/message/*.json | wc -l   # == number of registered images
cat $SCENE/3dgs/renders/message/_intrinsics.json
```

Open a few PNGs. Typical 3DGS artefacts (floaters, blur at grazing angles) are
harmless for annotation; a completely wrong viewpoint or an empty image means
the wrong `.ply` or scene was used.

## Troubleshooting

| symptom | fix |
|---|---|
| `no trained model at ...` | check `ITER` matches the folder under `3dgs/output/point_cloud/`. |
| renders are black | background is black by design; check that the object is lit in the training images and that the 3DGS was trained on this scene. |
| renders do not line up with the real frames | the 3DGS was trained on a different model/folder. Step 02 must write `$SCENE/3dgs` from the same `sparse/0`. |
| `unsupported camera model` | the source has an exotic COLMAP model; rebuild the model as PINHOLE/SIMPLE_RADIAL in step 01. |
| `cameras have mixed sizes` | pass `--width/--height/--fx` explicitly. |

Next: [04 — two-view seed](04_seed_views.md)
