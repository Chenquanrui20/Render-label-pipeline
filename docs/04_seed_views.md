# 04 — Two-view seed: pick two frames and annotate them

**Script:** `scripts/04_seed_views.sh`
**Input:** `<SCENE>/3dgs/renders/`
**Output:** `<SCENE>/work/selection.json` (stage `seeds`),
`<SCENE>/work/seed.json` (your clicks)

This is the cold start. No 3D point exists yet, so the two seed frames are picked
from **camera poses only**.

## Part 1 — pose-only seed selection (headless)

```bash
python -m renderlabel select $SCENE --selection $SCENE/work/selection.json
```

The script:

1. computes `Q`, the point that minimises the sum of squared distances to every
   camera's optical axis (a closed-form least-squares intersection),
2. takes the pair of cameras whose rays *from Q* are widest apart,
3. unless that angle exceeds `SEED_MAX_ANGLE` (default 160°).

Why the cap: two cameras exactly opposite (180°) put the object **on the
baseline**. The two rays to the object are then collinear and its depth is
unobservable — a two-view triangulation returns an arbitrary point along the
line. With a 160° cap the worst seed geometry still has a 20° triangulation
angle, which is well conditioned.

The result is written to `work/selection.json`:

```json
{
  "stage": "seeds",
  "theta_eff_deg": 11.5,
  "guard_deg": 5.0,
  "proxy_point": [0.12, -0.03, 0.05],
  "seed_idx": [12, 87],
  "seed_files": ["frame_00013.png", "frame_00088.png"],
  "seed_ray_angle_deg": 71.4
}
```

## Part 2 — annotate those two frames (GUI)

```bash
python -m renderlabel annotate $SCENE \
    --frames seed --selection $SCENE/work/selection.json \
    --points-file points.txt --out $SCENE/work/seed.json
```

The window shows the first seed frame. Click every keypoint once. The point
names come from `points.txt`; the current one is shown in the HUD.

| key | action |
|---|---|
| left click | place the current point at the cursor, auto-advance to the next missing one |
| `1`–`9` | jump to point *i* |
| `n` / `p` | next / previous frame (saves first) |
| `a` | adopt all projected guide positions (only useful once guides exist, i.e. step 06) |
| `u` | undo the last placed point on this frame |
| `r` | clear this frame |
| `s` | save now |
| `q` / `Esc` | save and quit |

The magnifier follows the cursor (6×, top-right corner). Use `n` to move to the
second seed frame and click the same physical points. Every click is
auto-saved, and re-running the command resumes from the existing file.

You do not have to click a point that is genuinely hidden in a seed frame — but
a point needs clicks in **both** seed frames to be triangulated, and only
triangulated points enter the candidate test in step 05. If more than a couple
of points are invisible in one of the two frames, pick a different seed pair by
hand:

```bash
python -m renderlabel annotate $SCENE --frames 13,88 --points-file points.txt \
    --out $SCENE/work/seed.json
```

`--frames` accepts a comma list of render file names (or plain indices into the
sorted frame list).

## Accuracy matters here

The seed clicks define `X0` and therefore the candidate set and `Theta_span`.
Click the same physical location (a corner, a marker centre, a distinctive
feature) as accurately as possible; sub-pixel clicks are the difference between
a label set with 0.5 px residuals and one with 5 px residuals. Zoom in with the
magnifier and take your time on these two frames — everything downstream is
projection.

## Checks

```bash
cat $SCENE/work/selection.json          # stage == "seeds", two seed_files
python -c "import json;d=json.load(open('$SCENE/work/seed.json'));print({k:len(v) for k,v in d['frames'].items()})"
```

Expect every point listed under both seed frames.

Next: [05 — predict N_sat and select frames](05_select_frames.md)
