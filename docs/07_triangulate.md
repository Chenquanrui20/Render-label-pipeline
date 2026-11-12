# 07 — Triangulate and project labels into every render

**Script:** `scripts/07_triangulate.sh`
**Input:** `<SCENE>/work/seed.json`, `<SCENE>/work/selected.json`
**Output:** `<SCENE>/3dgs/labels/`

```bash
SCENE=/data/box1 bash scripts/07_triangulate.sh
```

or directly:

```bash
python -m renderlabel triangulate $SCENE \
    --annotations $SCENE/work/seed.json $SCENE/work/selected.json \
    --out $SCENE/3dgs/labels --overlay $SCENE/3dgs/overlay
```

## Triangulation

For each point, all clicks across all annotation files are collected (later
files win on an identical frame/point pair, so `selected.json` overrides
`seed.json`). The point is solved by multi-view DLT in normalised coordinates.

**Outlier rejection.** With four or more clicks, every pair of views proposes a
hypothesis; the hypothesis that keeps the most clicks within 3 px wins (ties
broken by the lower median residual), and the point is re-solved on that inlier
set. With three clicks the problem is ambiguous — any two views explain a point
exactly — so no click is rejected below four clicks.

**Refinement.** `REFINE=1` / `--refine` adds a Levenberg–Marquardt refinement.
It changes the result by ~0.01 px on the paper's sequences and is off by default.

**Minimum views.** A point needs at least `--min-views` (default 2) clicks, or it
is listed in `failed_points`.

## Projection

The 3D points are projected into **every** rendered frame. A labelme file is
written per frame, containing one `point` shape per keypoint **that projects
inside the image**:

```json
{
  "version": "5.0.0",
  "flags": {},
  "shapes": [
    {"label": "P1", "points": [[912.4, 508.1]], "group_id": null,
     "shape_type": "point", "flags": {}}
  ],
  "imagePath": "../renders/images/frame_00013.png",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

Points that fall behind the camera or outside the frame are omitted from that
frame's file. `imagePath` is relative to the label directory and points at the
render PNG.

## Results

```
3dgs/labels/<frame>.json        one file per render
3dgs/labels/points3d.csv        name, X, Y, Z, n_views, residual median/mean/max
3dgs/labels/summary.json        coverage, residuals, failed points
3dgs/overlay/<frame>.png        labels drawn on the render (QC)
```

`summary.json`:

```json
{
  "n_frames": 946,
  "n_points": 8,
  "points": ["P1", "..."],
  "failed_points": [],
  "annotated_frames": 6,
  "annotated_views_used": {"P1": 6, "...": 5},
  "coverage_frames": {"P1": 812, "...": 640},
  "coverage_frac": {"P1": 0.858, "...": 0.676},
  "resid_px": {"median": 0.41, "mean": 0.44, "max": 1.2},
  "refine": false,
  "min_views": 2
}
```

## QC before you train on these labels

Open a few overlays. What to look for:

- **Residuals.** `resid_px` is the reprojection error of your *clicks* against
  the triangulated points, not the label error. Values around 0.3–1 px mean the
  clicks are consistent; several px mean mis-clicks or non-rigid content.
- **Coverage.** A point that exists in only 10% of the frames is either mostly
  occluded or its 3D position is wrong. Check an overlay at the edge of the
  coverage.
- **Edge frames.** 3DGS degrades at viewpoints far from the training subset.
  If labels look systematically off in a region, render with more training
  frames (step 02) rather than annotating more.

Remember that the labels live in the **render** pixel space: they are exact for
these renders, and their agreement with the real images is limited by the SfM
poses and the 3DGS geometry. If you need labels on the real images, project the
triangulated 3D points with the original per-image COLMAP cameras — the points
are in the COLMAP world frame, so this is a direct `project_points` call.

## Troubleshooting

| symptom | fix |
|---|---|
| `no point has enough views` | every point was clicked in fewer than `--min-views` frames; lower it or annotate more. |
| all residuals are huge | the annotation files were made against a different render set (frames were re-rendered with other intrinsics). Re-annotate or regenerate. |
| a point is in `failed_points` | it was clicked in fewer than two frames, or its clicks are inconsistent. |
| labels missing in many frames | the point projects outside them; check `coverage_frac` and the overlays. |

Back to the [README](../README.md) or the [FAQ](faq.md).
