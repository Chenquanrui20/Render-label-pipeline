# FAQ and troubleshooting

## Environment

**The scripts call `conda run -n rlabel` / `-n gs`. I use venvs.**
Set `LABEL_ENV=""` and `GS_ENV=""` in `scripts/config.sh`; the scripts then use
whatever `python` is active. Activate the right env before each step.

**`conda run` prints nothing / buffers output.**
Keep `--no-capture-output` (already in `scripts/lib.sh`). If your conda is very
old, upgrade it or run the stage commands directly.

**`pycolmap` import fails.**
It is only needed for `subset` (step 02). Install it in the label env:
`conda install -c conda-forge pycolmap`. There is no pip wheel on all platforms.

**`could not load library libcudart.so`** when starting 3DGS training.
The torch wheel and the system CUDA runtime disagree. Install the torch wheel
matching your driver, or add the CUDA lib directory to `LD_LIBRARY_PATH`.

## Annotation

**No window opens over SSH.**
OpenCV needs an X display: `ssh -X user@host` (slow but works) or use a VNC /
remote-desktop session. `echo $DISPLAY` must be non-empty.

**The window is bigger than my screen.**
`--max-width 1200` scales the display; clicks are still stored in full-resolution
coordinates.

**I have more than 9 keypoints.**
The number keys only jump to the first nine; clicking cycles through all points
in `points.txt` order. Split very large point sets into several files and merge
the outputs, or just click in order.

**I mis-clicked and already saved.**
Re-run the same annotate command; it resumes. Click the point again to overwrite
it, or `u` to remove the last one on the current frame.

**The same physical point looks different across renders.**
3DGS view-dependent effects and floaters can shift an edge by a pixel or two.
Click the geometric feature (corner centre, dot centre), not the artefact.

## Reconstruction and training

**COLMAP registers only a fraction of my frames.**
More overlap: extract at a higher FPS; avoid fast motion / motion blur; add
texture. Try `MATCHER=exhaustive`. Check `colmap model_analyzer`.

**COLMAP wrote `sparse/1` instead of `sparse/0`.**
Several disjoint models. Keep the largest and rename it to `sparse/0`.

**3DGS says the camera model is unsupported.**
You trained on `<SCENE>` instead of `<SCENE>/3dgs`. The subset step exists to
convert `SIMPLE_RADIAL` cameras to undistorted `PINHOLE`.

**Training takes forever / runs out of memory.**
Lower `N_FRAMES` (e.g. 150) and/or resize the input images. Memory scales with
image resolution; 1080p with 250 frames needs a modern 24 GB card.

**Can I use my own 3DGS fork?**
Yes, if it keeps the official layout: `gaussian_renderer.render`,
`scene.cameras.Camera`, `scene.gaussian_model.GaussianModel` and the COLMAP
binary readers. The render call falls back when `separate_sh` /
`use_trained_exp` are not accepted, but a fork with a different `Camera`
constructor (e.g. the older `gaussian-splatting-pup` layout) is not supported.

## Selection

**`fewer than 2 candidate frames see all points with a N px margin`.**
Either the seed points span more of the image than `MARGIN` allows, or the seed
clicks are inconsistent. Lower `--margin` (e.g. 5) first; then re-check the seed
clicks. If the object is only partly visible in most frames, that is a capture
problem, not a parameter problem.

**`n_selected` is smaller than `n_sat`.**
The 5° guard stopped the greedy search. This is by design: never place two
views closer than the guard. Lower `GUARD` if your views really are that
redundant.

**`Theta_span` is ~180° and `N_sat` is large.**
That means the object is visible from all around, so 8 views at 11.5° is the
honest prediction. With a smaller annotation budget, raise `--theta-eff`
knowingly. You can also hand-edit `selected_files` in `selection.json` — step 06
follows it — but then the file no longer records what the rule predicted; change
`n_selected` to match so the record stays truthful.

## Labels

**Can I skip 3DGS and use real images?**
Yes, in principle: everything after step 03 only reads
`<renders>/message/*.json` + `<renders>/images/`. If you already have poses and
a **single shared** camera model for your real frames, writing message files in
the same schema lets you run steps 04–07 on them. The pipeline ships no converter
for that; the message schema is documented in [03_render.md](03_render.md).

**Are the labels valid on the real images?**
Not directly. They are exact in the render pixel space (one shared pinhole).
The triangulated 3D points are in the COLMAP world frame, so you can project
them into the real images with their own per-image intrinsics — accuracy is then
limited by the SfM poses and the 3DGS geometry. There is no render-domain ground
truth for that transfer.

**Why are some frames missing a point?**
The point projects outside the image (or behind the camera). Those shapes are
omitted on purpose; `summary.json` records the per-point coverage.

**Can I train a detector on these labels?**
Yes. The labelme JSON is directly consumable by common keypoint toolchains; the
`imagePath` points at the render PNG. Train and evaluate on the renders of the
same scene — that is the domain the labels are exact in.

## Disk and re-runs

**Which folders can I delete?**
`<SCENE>/3dgs/output` (retrain), `<SCENE>/3dgs/renders` (re-render),
`<SCENE>/3dgs/labels` + `overlay` (re-run step 07). Keep `work/` — it holds your
clicks.

**Can I re-run a later step after changing an earlier one?**
- Re-rendering invalidates annotations only if the intrinsics changed. Keep the
  shared camera fixed once you start annotating.
- Changing `N_FRAMES` requires retraining and re-rendering, but **not**
  re-annotating: the poses and shared camera are the same. `work/*.json` stays
  valid.
- Changing the COLMAP model changes the poses: everything after step 03 must be
  redone, including annotation.
