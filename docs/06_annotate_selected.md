# 06 — Annotate the selected frames

**Script:** `scripts/06_annotate_selected.sh`
**Input:** `<SCENE>/work/selection.json`, `<SCENE>/work/points3d_seed.json`
**Output:** `<SCENE>/work/selected.json`

```bash
SCENE=/data/box1 bash scripts/06_annotate_selected.sh
```

or directly:

```bash
python -m renderlabel annotate $SCENE \
    --frames selected --selection $SCENE/work/selection.json \
    --points-file points.txt \
    --init $SCENE/work/points3d_seed.json \
    --out $SCENE/work/selected.json
```

## Guides

The seed-triangulated 3D points are projected into each frame and drawn as small
blue rings. They are usually within a few pixels of the true position, so the
fast workflow is:

1. press **`a`** to adopt all guide positions in this frame,
2. fix only the points that are visibly off (click them again; the click
   overrides the adopted position),
3. `n` to the next frame, repeat.

If the guides are far off (bad seed clicks), do not use `a`; click normally.

## Keys

| key | action |
|---|---|
| left click | place the current point, auto-advance to the next missing one |
| `1`–`9` | jump to point *i* |
| `n` / `p` | next / previous frame (saves first) |
| `a` | adopt all guide positions for points that are still missing |
| `u` | undo the last placed point on this frame |
| `r` | clear this frame |
| `s` | save now |
| `q` / `Esc` | save and quit |

Every click auto-saves to `work/selected.json`, so a crash or a closed window
loses at most the current click. Re-running resumes exactly where you left off.

## Skipping points

You do **not** have to click a point that is occluded or out of frame in a
selected view. Leave it missing; step 07 triangulates each point from the frames
where it *was* clicked, and only writes labels for frames where it projects
inside the image. The HUD shows `k/m points` so you can see what is still
missing — press `q` when the remaining ones are genuinely invisible.

A point needs at least `--min-views` clicks overall (default 2) to appear at
all; the two seed frames plus the selected frames usually give 4–8.

## Output format

```json
{
  "points": ["P1", "P2", "..."],
  "frames": {
    "frame_00013.png": {"P1": [912.4, 508.1], "P2": [930.7, 515.3]},
    "frame_00088.png": {"P1": [1104.0, 621.9], "...": [0, 0]}
  },
  "saved_at": "2026-11-12 18:00:00"
}
```

All coordinates are in the **render image** pixel space (full resolution, not
the scaled display), origin top-left, `x` right, `y` down.

Next: [07 — triangulate and project](07_triangulate.md)
