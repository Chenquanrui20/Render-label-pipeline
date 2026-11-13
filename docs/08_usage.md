# 08 — Detailed usage guide

This is the long-form manual: everything from a raw capture to labelme labels,
including how to drive the pipeline from your own annotation tool and how to
render the demo video.  The per-stage reference documents (01–07) describe each
algorithm; this page is the operational walkthrough.

- [0. Ten-line version](#0-ten-line-version)
- [1. What you need](#1-what-you-need)
- [2. Scene layout](#2-scene-layout)
- [3. Install and verify](#3-install-and-verify)
- [4. Prepare the input](#4-prepare-the-input)
- [5. Stages 00–07](#5-stages-0007)
- [6. Annotation](#6-annotation)
- [7. Tuning the selection](#7-tuning-the-selection)
- [8. Reading the outputs](#8-reading-the-outputs)
- [9. Gallery and demo video](#9-gallery-and-demo-video)
- [10. Running on a pre-built 3DGS scene](#10-running-on-a-pre-built-3dgs-scene)
- [11. Troubleshooting](#11-troubleshooting)

---

## 0. Ten-line version

```bash
conda env create -f environment.yml && conda activate rlabel
SCENE=/data/box1 bash scripts/01_colmap.sh          # SfM
SCENE=/data/box1 bash scripts/02_train_3dgs.sh      # subset + 3DGS (gs env)
SCENE=/data/box1 bash scripts/03_render.sh          # one render per pose
SCENE=/data/box1 bash scripts/04_seed_views.sh      # GUI: 2 seed frames
SCENE=/data/box1 bash scripts/05_select_frames.sh   # N_sat + view selection
SCENE=/data/box1 bash scripts/06_annotate_selected.sh  # GUI: N selected frames
SCENE=/data/box1 bash scripts/07_triangulate.sh     # labels for every render
```

Steps 04 and 06 are the only interactive ones.

## 1. What you need

| side | requirement | notes |
|---|---|---|
| OS | Linux | the shell scripts assume bash |
| label side | Python ≥ 3.9, numpy, scipy, opencv (with GUI), pillow | `environment.yml` or `requirements.txt` |
| SfM | COLMAP CLI | `apt install colmap` or `conda install -c conda-forge colmap` |
| 3DGS side | NVIDIA GPU + CUDA, PyTorch, [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | separate env; see [02_train_3dgs.md](02_train_3dgs.md) |
| annotation | a desktop session (OpenCV window) | over SSH use `ssh -X` or VNC |

`pycolmap` is only needed by the subset stage (02); everything else runs
without it.

The inputs must show **one rigid object or board** from many angles, with
enough overlap for COLMAP (a 10–30 s orbit capture is typical).  The keypoints
are assumed to span a small volume compared with the camera distance, because
the cold start works from a single coarse 3D point.

## 2. Scene layout

One scene is one folder.  After the full pipeline:

```
<SCENE>/
  images/                       input frames (step 00 writes here)
  database.db                   COLMAP database          (step 01)
  sparse/0/                     COLMAP model             (step 01)
  3dgs/
    images/                     undistorted training subset (step 02)
    sparse/0/                   PINHOLE model of that subset (step 02)
    output/point_cloud/iteration_30000/point_cloud.ply   (3DGS training)
    renders/
      images/<stem>.png         one render per COLMAP pose (step 03)
      message/<stem>.json       pose + intrinsics of that render
    labels/                     labelme JSON + points3d.csv + summary.json (step 07)
    overlay/                    QC images of the projected labels (step 07)
  work/
    selection.json              seed pair, Theta_span, N_sat, selected frames
    points3d_seed.json          coarse 3D points from the seed clicks
    seed.json                   your clicks on the seed frames
    selected.json               your clicks on the selected frames
```

Everything the pipeline writes is under `3dgs/` and `work/`; the input images
and the COLMAP database are never modified.

## 3. Install and verify

### 3.1 Label environment (conda, recommended)

```bash
git clone <repo-url> render-label-pipeline
cd render-label-pipeline
conda env create -f environment.yml     # creates env "rlabel"
conda activate rlabel
pip install -e .                        # optional: the `renderlabel` command
```

### 3.2 Label environment (venv/pip alternative)

```bash
git clone <repo-url> render-label-pipeline
cd render-label-pipeline
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[subset]"              # add pycolmap for the subset stage
```

### 3.3 3DGS environment

Follow [02_train_3dgs.md](02_train_3dgs.md).  It installs `torch`, clones
`gaussian-splatting`, and builds its two CUDA submodules.  The scripts call
this env through `conda run -n gs`, so it never has to be active.

### 3.4 Verify

From the repository root, in the label env:

```bash
python -m unittest discover -s tests -v     # 12 tests, all pass, ~1 s
python -m renderlabel --help                # lists subset/render/select/annotate/triangulate
python -c "import cv2, numpy, scipy, PIL; print(cv2.__version__, numpy.__version__)"
```

A quick non-interactive end-to-end check (synthetic, no data, no GPU) is the
unit-test suite itself: `tests/test_geometry.py` runs the cold start and the
triangulation/projection path on generated cameras.

## 4. Prepare the input

### 4.1 From a video

```bash
mkdir -p /data/box1
VIDEO=/data/take1.mp4 FPS=2 SCENE=/data/box1 bash scripts/00_extract_frames.sh
```

`FPS=2` keeps two frames per second; 60–150 frames are plenty for one orbit.
The extraction refuses to write into a non-empty `images/`.

### 4.2 From an existing image folder

Put the images in `<SCENE>/images/` and go to step 01.  Names must sort in
capture order if you rely on the default subset selection (`frame_00001.png`,
…); otherwise pass an explicit list (`--names-file`).

## 5. Stages 00–07

### 00 — extract frames (optional)

`VIDEO=... FPS=2 SCENE=... bash scripts/00_extract_frames.sh` → `<SCENE>/images/`.
Skip if `images/` already exists.

### 01 — COLMAP SfM

```bash
SCENE=/data/box1 bash scripts/01_colmap.sh
# unordered photos instead of video frames:
MATCHER=exhaustive SCENE=/data/box1 bash scripts/01_colmap.sh
```

Runs `feature_extractor` (SIMPLE_RADIAL, single camera), `sequential_matcher`,
`mapper`; writes `database.db` and `sparse/0/`.  The database is reused if it
already exists.  Check: `colmap model_analyzer --path <SCENE>/sparse/0` reports
the number of registered images and a mean reprojection error (typically
0.3–1.5 px).

### 02 — subset + 3DGS training

```bash
SCENE=/data/box1 bash scripts/02_train_3dgs.sh
```

Equivalent to

```bash
python -m renderlabel subset <SCENE> -n 250
cd "$GS_REPO" && python train.py -s <SCENE>/3dgs \
    -m <SCENE>/3dgs/output --iterations 30000
```

The subset stage keeps 250 evenly spaced registered frames, writes a COLMAP
text model with `PINHOLE` cameras, and runs the image undistorter.  Training
takes ~30–60 min on a modern GPU.  Check that
`<SCENE>/3dgs/output/point_cloud/iteration_30000/point_cloud.ply` exists.

See [02_train_3dgs.md](02_train_3dgs.md) for CUDA/torch version matching and
memory notes.

### 03 — render every pose

```bash
SCENE=/data/box1 bash scripts/03_render.sh
```

Renders **every** pose of `sparse/0` with **one shared pinhole camera**
(`fx = fy = median focal length`, principal point at the image centre,
`k1 = 0`).  Output: `3dgs/renders/images/*.png` and
`3dgs/renders/message/*.json`.  Check: the number of message files equals the
number of registered images.

Why one camera: per-image COLMAP `SIMPLE_RADIAL` intrinsics give every view its
own pixel space; triangulation across views would then be inconsistent.  Use
`--camera per-image` only when the source model is already `PINHOLE` and you
know you need it.

### 04 — two-view cold start (GUI)

```bash
SCENE=/data/box1 bash scripts/04_seed_views.sh
```

First a pose-only pass picks the seed pair (the two cameras whose rays are
widest apart, capped at `SEED_MAX_ANGLE=160°`), writes
`work/selection.json` (stage `seeds`), and opens the annotator on exactly those
two frames.  Click every keypoint on both frames.

### 05 — predict N_sat and select frames

```bash
SCENE=/data/box1 bash scripts/05_select_frames.sh
```

Triangulates the seed clicks, estimates the object point `X0`, counts the
candidate frames (all seed points inside with `MARGIN` px), computes
`Theta_span` and `N_sat = ceil(Theta_span / (2·theta_eff))`, and greedily
selects views with a hard `GUARD` minimum angle.  Writes `work/selection.json`
(stage `solved`) and `work/points3d_seed.json` (used as guide rings in 06).
Read the console summary: number of candidates, `Theta_span`, `N_sat`, and how
many views were actually selected (`n_selected` can be smaller than `N_sat`
when the guard runs out of candidates).

### 06 — annotate the selected frames (GUI)

```bash
SCENE=/data/box1 bash scripts/06_annotate_selected.sh
```

Opens the annotator on the selected frames (the seed frames are preloaded, so
you only click the new ones).  The projection of the seed points is drawn as
guide rings; press `a` to adopt all guides where they are good and click only
where they are off.

### 07 — triangulate and project

```bash
SCENE=/data/box1 bash scripts/07_triangulate.sh
```

Multi-view DLT per point (robust rejection when ≥4 clicks), then projection
into **every** render.  Writes:

```
3dgs/labels/<frame>.json     labelme JSON, one point shape per visible keypoint
3dgs/labels/points3d.csv     name, X, Y, Z, n_views, residual median/mean/max
3dgs/labels/summary.json     coverage, residuals, failed points
3dgs/overlay/<frame>.png     QC overlay (set OVERLAY= to disable)
```

Inspect `summary.json` before training a detector: `resid_px` should be around
0.3–1 px on real captures; a point with a large residual means a bad click or a
frame/pose inconsistency, not a modelling problem.

`REFINE=1` adds an LM refinement (~0.01 px on the paper's sequences; off by
default) and `--min-views 3` drops points seen in fewer than three frames.

## 6. Annotation

### 6.1 Built-in GUI

`renderlabel annotate` opens an OpenCV window:

| key | action |
|---|---|
| left click | place the current point (auto-advances to the next) |
| `1`…`9` | jump to that point index |
| `n` / `p` | next / previous frame (saves first) |
| `a` | adopt the projected guide positions for every missing point |
| `u` | undo the last click on this frame |
| `r` | clear this frame |
| `s` | save now |
| `q` / Esc | save and quit |

The file is written after every action, so quitting never loses clicks; running
the same command again resumes from the existing file.  `--frames` accepts
`seed`, `selected`, `all`, a comma list (`000123.png,000456.png`), or
`@frames.txt`.

### 6.2 Using your own annotation tool (JSON contract)

Any tool can replace the GUI.  Two files are enough:

```json
{
  "points": ["P1", "P2", "P3"],
  "frames": {
    "000241.png": {"P1": [912.4, 508.1], "P2": [901.0, 533.2]},
    "000757.png": {"P1": [880.2, 500.7]}
  },
  "saved_at": "2026-11-13 12:00:00"
}
```

* **keys of `frames`** are the render image names as they appear in
  `3dgs/renders/message/<name>.json` (`image` / `render_file`), **with
  extension**;
* **coordinates** are pixels in the rendered image, `(u, v) ∈ [0, W) × [0, H)`,
  origin top-left;
* points absent from a frame are simply not observed there.

Give the seed file to `renderlabel select --annotations work/seed.json ...`,
then `renderlabel triangulate --annotations work/seed.json work/selected.json`.
Later files win on a duplicate (frame, point), so the selected round overrides
a seed click on the same frame.

To annotate from your own tool, give it the rendered images from
`3dgs/renders/images/` (not the original captures): the pixel space must be the
shared pinhole space, otherwise triangulation is biased.

## 7. Tuning the selection

All stage-05 knobs are in [scripts/config.sh](../scripts/config.sh) or can be
passed per run:

| variable | default | meaning |
|---|---|---|
| `THETA_EFF` | 11.5 | degrees of ray coverage one more view buys |
| `GUARD` | 5 | minimum pairwise ray angle between selected views |
| `MARGIN` | 20 | px a point must stay inside the image to count as visible |
| `SEED_MAX_ANGLE` | 160 | cap on the seed-pair angle (≈180° is degenerate) |

`theta_eff` was fitted on the paper's seven sequences.  For a different
capture rig (field of view, object size, noise) recalibrate it: annotate a
handful of extra views at several `N`, measure the held-out reprojection error,
and pick the `N` where it flattens; `theta_eff ≈ knee / 2` in degrees of
coverage.  The rule is a heuristic for *how many* views, not which ones — the
greedy cover chooses which.

The realised count can be below `N_sat`: with a hard guard the search stops
rather than relaxing it.  `selection.json` reports both `n_sat` and the number
of selected frames; if you need more views, lower `GUARD` (at the price of
narrow baselines) or add frames to the candidate pool.

## 8. Reading the outputs

* **`points3d.csv`** — the triangulated control points in the COLMAP world
  frame (arbitrary units).  `n_views` is how many clicks survived; `resid_*`
  are the reprojection residuals **on the annotated views** (they measure
  self-consistency, not accuracy).
* **`summary.json`** — `resid_px` over all annotated clicks, per-point
  `coverage_frames` (in how many renders the label exists), `failed_points`.
* **labelme JSONs** — one file per render, one `point` shape per keypoint that
  projects inside that frame; `imagePath` is relative to the label directory
  and points at the render.
* **The labels live in the render pixel space.**  Do not compare their pixel
  errors with errors measured on the original photographs; the intrinsics and
  the geometry source are different.  If you need labels on the real images,
  project the 3D points with `views`' own cameras — expect a larger error
  (SfM pose error + 3DGS geometry error) and no render-domain ground truth to
  check against.

## 9. Gallery and demo video

Two optional presentation scripts consume the same `work/` files.

### Gallery

```bash
RENDERS=<SCENE>/3dgs/renders
python3 examples/visualize.py --format labelme \
    --renders "$RENDERS" --labels <SCENE>/3dgs/labels \
    --selection <SCENE>/work/selection.json \
    --points3d <SCENE>/3dgs/labels/points3d.csv \
    --annotations <SCENE>/work/seed.json <SCENE>/work/selected.json \
    --out <SCENE>/work/gallery
```

Writes overlays, `selection.png`, `errors.png` (if `--eval` is given),
`per_point.png`, and `report.html`.

### Demo video

```bash
python3 examples/make_demo_video.py --format labelme \
    --renders "$RENDERS" --labels <SCENE>/3dgs/labels \
    --images <SCENE>/images --sparse <SCENE>/sparse/0 \
    --selection <SCENE>/work/selection.json \
    --points3d <SCENE>/3dgs/labels/points3d.csv \
    --annotations <SCENE>/work/seed.json <SCENE>/work/selected.json \
    --out <SCENE>/work/gallery/demo.mp4
```

1280×720 / 30 fps, five stages (reconstruction → render-vs-real wipe → view
selection → triangulation → projection).  All on-screen numbers are read from
the run's JSON; the video never reports accuracy.

**Visible vs occluded markers.**  Pass `--visibility visibility.json` to draw
occluded points as grey hollow circles instead of green crosses.  The file maps
render frame stems to points:

```json
{"000241": {"P1": {"vis": true}, "P5": {"vis": false}}}
```

Missing entries default to visible.  `examples/render_visibility.py` computes
such a file by projecting the 3DGS Gaussians into each render with their full
EWA covariance and z-testing the point against the nearest opaque surface:

```bash
python3 examples/render_visibility.py <SCENE> \
    --renders <SCENE>/3dgs/renders \
    --points3d <SCENE>/3dgs/labels/points3d.csv \
    --out <SCENE>/work/visibility.json
```

It reads `output/point_cloud/iteration_30000/point_cloud.ply` (override with
`--iteration`) and each render's `message/*.json` camera.  `--annotations
work/seed.json` prints a calibration line: the fraction of your clicked views
that the test calls visible (clicks are ground truth for visibility).

## 10. Running on a pre-built 3DGS scene

If the scene already has a reconstruction and a trained 3DGS (as in the
examples), skip 01–03 and start at 04.  Two details:

* The scripts assume `3dgs/` as the subdirectory.  To keep renders at the scene
  root instead, call the CLI directly:
  `python -m renderlabel render <SCENE> --subdir ""` (renders → `<SCENE>/renders`),
  and pass `--renders <SCENE>/renders` to `select`, `annotate`, `triangulate`.
* If the renders were produced with a different shared focal length, pass the
  same `--fx/--fy/--cx/--cy` when re-rendering; otherwise use the existing
  renders and messages as a consistent pair — do not mix.

The example scripts under [`examples/`](../examples/README.md) do exactly this
on two paper datasets with pre-existing 3DGS models.

## 11. Troubleshooting

| symptom | cause / fix |
|---|---|
| `SCENE is not set` | edit `scripts/config.sh` or prefix the command with `SCENE=/path` |
| `colmap: command not found` | install COLMAP (`apt install colmap`, or conda-forge) |
| COLMAP writes `sparse/1` | the mapper split the capture — use the largest model and rename it to `sparse/0` |
| 3DGS rejects `SIMPLE_RADIAL` | train with `-s <SCENE>/3dgs` (the undistorted subset), not `<SCENE>` |
| `pycolmap is required` | `pip install pycolmap` or `conda install -c conda-forge pycolmap`; only step 02 needs it |
| subset: `IndexError: Image with ID ... does not exist` | inconsistent model — a `points3D` track names a missing image (hand-edited or stitched `sparse/0`); pycolmap 4 refuses it. Rebuild the model; see [faq.md](faq.md) |
| annotate opens no window | no `DISPLAY`; use a desktop session or `ssh -X` |
| triangulate reports `failed_points` | fewer than `--min-views` clicks for that point, or clicks behind the camera |
| residuals much larger than ~1 px | a wrong click, or frames from a different pixel space (e.g. real images instead of renders) |
| a point has low coverage | it leaves the frame for part of the orbit; that is expected — labels exist only where it projects inside |

More per-stage notes: [01](01_colmap.md) · [02](02_train_3dgs.md) ·
[03](03_render.md) · [04](04_seed_views.md) · [05](05_select_frames.md) ·
[06](06_annotate_selected.md) · [07](07_triangulate.md) · [FAQ](faq.md)
