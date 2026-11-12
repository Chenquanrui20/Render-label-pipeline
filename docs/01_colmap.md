# 01 — COLMAP reconstruction

**Script:** `scripts/01_colmap.sh`
**Input:** `<SCENE>/images/`
**Output:** `<SCENE>/database.db`, `<SCENE>/sparse/0/{cameras,images,points3D}.bin`

The reconstruction supplies the camera poses that are later used to render the
3DGS and to place every label. Nothing downstream can be better than these
poses, so it is worth checking the model before continuing.

## Command

```bash
SCENE=/data/box1 bash scripts/01_colmap.sh
```

The script runs the three standard stages:

```bash
colmap feature_extractor \
    --database_path  $SCENE/database.db \
    --image_path     $SCENE/images \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1

colmap sequential_matcher --database_path $SCENE/database.db
colmap mapper --database_path $SCENE/database.db \
              --image_path $SCENE/images \
              --output_path $SCENE/sparse
```

`sequential_matcher` assumes the frames are in capture order (video frames).
For unordered photos use:

```bash
MATCHER=exhaustive SCENE=/data/box1 bash scripts/01_colmap.sh
```

## About `single_camera`

- `SINGLE_CAMERA=1` (default) fits **one** camera for all frames. This is right
  for a single video take and gives the most stable focal length.
- `SINGLE_CAMERA=0` refines one `SIMPLE_RADIAL` camera **per image**. Use it only
  if frames really come from different cameras / zoom settings, and be aware
  that per-frame focal lengths can drift and become degenerate.
- Either way, step 02 undistorts the model into a single PINHOLE camera per
  frame, and step 03 renders with one shared pinhole. The rest of the pipeline
  is identical.

## Checks

```bash
colmap model_analyzer --path $SCENE/sparse/0
```

Look at:

- **Registered images** — should be most of your frames. A low count means too
  little overlap (increase `FPS`), motion blur, or a textureless scene.
- **Mean reprojection error** — roughly below 1 px. Higher means the poses are
  not trustworthy; improve the capture rather than the settings.
- **Points3D** — tens of thousands for a normal object capture.

Two more checks:

```bash
ls $SCENE/sparse          # expect a directory "0"
ls $SCENE/sparse/0        # cameras.bin images.bin points3D.bin
```

## Troubleshooting

| symptom | fix |
|---|---|
| `sparse/0` missing | the mapper failed; check that consecutive frames overlap by ≥70%. For video use a denser `FPS` in step 00. |
| the model is in `sparse/1` | COLMAP wrote several disjoint models. Keep the largest and rename: `mv sparse/1 sparse/0` (after removing the old `0`). |
| few registered images | raise the extraction `FPS` so consecutive frames overlap more; try `MATCHER=exhaustive`; check for motion blur. |
| SIFT fails without a GPU | add `--SiftExtraction.use_gpu 0` and `--SiftMatching.use_gpu 0` to the two COLMAP calls. |
| very large image set | `exhaustive_matcher` is O(N²); use `sequential_matcher` or a vocab-tree matcher. |

Next: [02 — subset + 3DGS training](02_train_3dgs.md)
