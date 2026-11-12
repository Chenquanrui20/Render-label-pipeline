# render-label-pipeline

**Saturation-aware view selection for render-based keypoint labels.**
Annotate a handful of rendered views, triangulate the keypoints, and project
exact labels into every frame of a COLMAP reconstruction — without annotating
thousands of images.

> **English abstract.** This repository packages a complete pipeline that turns a
> video or image set into dense keypoint labels: COLMAP structure-from-motion,
> 3D Gaussian Splatting on a fixed-size subset of the registered frames,
> re-rendering of *every* COLMAP pose with one shared pinhole camera, a two-view
> seed annotation, prediction of the saturated view count
> `N_sat = ceil(Theta_span / (2 * theta_eff))` plus greedy angular view
> selection, annotation of only those views, and multi-view DLT triangulation
> whose result is projected into all renders as labelme JSON.

## Why render the frames at all

- Every training frame exists as a render at **exactly the same pose**, in **one
  shared pixel space** (one focal length, one principal point). Multi-view
  triangulation is only well-posed when all views share a camera model; COLMAP's
  per-image `SIMPLE_RADIAL` intrinsics do not.
- Annotation effort scales with the number of *distinct viewpoints* needed, not
  with the number of frames. The selection rule predicts that number from pose
  geometry alone and picks well-spread views, so 4–8 annotated frames can label
  a whole capture.
- Labels are then produced by projection, so they are exact (up to triangulation
  error) and cost nothing per frame.

## Pipeline

```
 video / images
      |
      v
 [01] COLMAP SfM --------------------------> sparse/0   (poses + intrinsics)
      |
      v
 [02] pick N frames, undistort -------------> 3dgs/{images,sparse/0}
      |                                       |
      |                                       v
      |                                     train 3DGS (30k iters)
      v
 [03] render every COLMAP pose -------------> 3dgs/renders/{images,message}
      |
      +--> [04] 2-view cold start: pick the seed pair from poses alone,
      |         annotate those 2 frames (GUI)          -> work/seed.json
      |
      v
 [05] N_sat = ceil(Theta_span / (2*theta_eff)); greedy angular cover
      |                                             -> work/selection.json
      v
 [06] annotate the N_sat selected frames (GUI)       -> work/selected.json
      |
      v
 [07] multi-view DLT triangulation -> project into all renders
                                                     -> 3dgs/labels/*.json
```

Steps 04 and 06 are the only manual steps. Everything else runs headless.

## Requirements

| side | what | notes |
|---|---|---|
| label side | Python 3.9+, `numpy scipy opencv pillow pycolmap` | `environment.yml` / `requirements.txt` |
| 3DGS side | PyTorch + CUDA, [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) with submodules | separate conda env, see [docs/02_train_3dgs.md](docs/02_train_3dgs.md) |
| SfM | COLMAP (`colmap` CLI) | `apt install colmap` or `conda install -c conda-forge colmap` |

A CUDA GPU is required for 3DGS training and rendering. Annotation needs a
desktop session (OpenCV window; over SSH use `ssh -X` or a VNC session).

## Install

```bash
git clone <this-repo> render-label-pipeline
cd render-label-pipeline

# label side
conda env create -f environment.yml
conda activate rlabel

# optional: install as a package (otherwise run from the repo root)
pip install -e .
```

Then install the 3DGS environment exactly as in
[docs/02_train_3dgs.md](docs/02_train_3dgs.md). The scripts call it through
`conda run`, so both envs can coexist.

## Quick start

```bash
$EDITOR scripts/config.sh     # set SCENE, GS_REPO, POINTS, ...
```

A scene is one folder:

```
/data/box1/
  images/          # input frames (or run 00 to extract them from a video)
```

Run, in order:

```bash
bash scripts/01_colmap.sh            # SfM           -> sparse/0
bash scripts/02_train_3dgs.sh        # subset + train -> 3dgs/output/...
bash scripts/03_render.sh            # all poses     -> 3dgs/renders/
bash scripts/04_seed_views.sh        # GUI: annotate 2 frames
bash scripts/05_select_frames.sh     # N_sat + selection
bash scripts/06_annotate_selected.sh # GUI: annotate the selected frames
bash scripts/07_triangulate.sh       # labels for every render
```

`scripts/run_all.sh` runs the headless stages 01–03 and then prints the rest.
If your input is a video, run `VIDEO=/data/take1.mp4 bash scripts/00_extract_frames.sh`
first.

## What you get

```
<SCENE>/3dgs/labels/<frame>.json   labelme JSON, one "point" shape per keypoint
<SCENE>/3dgs/labels/points3d.csv   triangulated 3D points, residuals per point
<SCENE>/3dgs/labels/summary.json   coverage, residuals, failed points
<SCENE>/3dgs/overlay/<frame>.png   QC overlay of the projected labels
```

A label is written only for frames where the point actually projects inside the
image. `summary.json` reports, per point, how many annotated views were used and
in how many frames the label exists.

## The selection rule

Nothing in steps 04–05 looks at a 2D observation before the seed clicks:

1. **Seed pair (no annotations).** Fit the point that minimises the distance to
   all camera optical axes (a pose-only proxy for the object centre), then take
   the pair of frames whose rays are widest apart. The pair is capped at
   `SEED_MAX_ANGLE` (default 160°): two exactly opposite cameras put the object
   on the baseline, where depth is unobservable.
2. **Coarse X0.** Triangulate the seed clicks (multi-view DLT), take the
   centroid of the seed points.
3. **Candidates.** A frame is a candidate when every seed point projects inside
   the image with a `MARGIN` px border.
4. **Prediction.** `Theta_span` is the largest pairwise angle between the
   candidate rays *from X0*; `N_sat = ceil(Theta_span / (2*theta_eff))`.
   `theta_eff = 11.5°` is the value fitted on the paper's seven sequences;
   recalibrate it on your own data (see [docs/05_select_frames.md](docs/05_select_frames.md)).
5. **Selection.** Greedy angular cover: start from the seed pair, repeatedly add
   the candidate farthest from the selected set, subject to a hard `GUARD`
   minimum angle (5°). If no candidate satisfies the guard the search stops —
   it never relaxes the guard. The realised count can therefore be smaller than
   `N_sat`; `selection.json` reports both `n_sat` and `n_selected`.

## Configuration

All knobs live in [scripts/config.sh](scripts/config.sh) and can be overridden
per run (`SCENE=... MARGIN=40 bash scripts/05_select_frames.sh`):

| variable | default | meaning |
|---|---|---|
| `SCENE` | — | scene folder (required) |
| `N_FRAMES` | 250 | frames kept for 3DGS training |
| `ITER` | 30000 | 3DGS training iterations |
| `GS_REPO` | `$HOME/gaussian-splatting` | upstream 3DGS checkout |
| `POINTS` | `examples/points.txt` | keypoint names, one per line |
| `THETA_EFF` | 11.5 | degrees per view in the saturation rule |
| `GUARD` | 5 | minimum ray angle between selected views (deg) |
| `MARGIN` | 20 | px border a point must keep to count as visible |
| `SEED_MAX_ANGLE` | 160 | cap on the seed-pair ray angle (deg) |
| `LABEL_ENV` / `GS_ENV` | `rlabel` / `gs` | conda env names; set to `""` to use the current env |

## Documentation

| step | document |
|---|---|
| 00 frame extraction | [scripts/00_extract_frames.sh](scripts/00_extract_frames.sh) |
| 01 COLMAP | [docs/01_colmap.md](docs/01_colmap.md) |
| 02 subset + 3DGS training | [docs/02_train_3dgs.md](docs/02_train_3dgs.md) |
| 03 render all poses | [docs/03_render.md](docs/03_render.md) |
| 04 two-view seed | [docs/04_seed_views.md](docs/04_seed_views.md) |
| 05 saturation + selection | [docs/05_select_frames.md](docs/05_select_frames.md) |
| 06 annotate selected | [docs/06_annotate_selected.md](docs/06_annotate_selected.md) |
| 07 triangulate + project | [docs/07_triangulate.md](docs/07_triangulate.md) |
| troubleshooting | [docs/faq.md](docs/faq.md) |

## Tests

```bash
python -m unittest discover -s tests -v
```

Nine synthetic tests cover projection, exact DLT, outlier rejection, the
selection rules (farthest pair, Theta_span, N_sat, guard truncation), and two
end-to-end runs of the cold start and the label projection.

## Notes and limits

- **The labels live in the render pixel space** of the shared pinhole camera.
  That is the intended use: train/evaluate on renders of the same scene. Because
  the 3D points are recovered in the COLMAP world frame, you *can* also project
  them into the original real images with their own intrinsics — but then the
  accuracy is limited by the SfM poses and by the 3DGS geometry, and there is no
  render-domain ground truth to check against.
- The 3DGS must be trained on frames from the same reconstruction you render, or
  the renders will not line up with the poses. Step 02 enforces the undistorted
  PINHOLE subset for exactly this reason.
- Inputs should show one rigid object (or board) from many angles. The coarse
  X0 is a single point, so the rule assumes the keypoints span a small volume
  relative to the camera distance.
- This pipeline annotates; it does not train a detector. The labelme JSON is
  directly consumable by common keypoint-training toolchains.

## License

MIT — see [LICENSE](LICENSE).
