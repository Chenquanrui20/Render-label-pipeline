# 05 — Predict the saturated view count and select frames

**Script:** `scripts/05_select_frames.sh`
**Input:** `<SCENE>/work/seed.json`
**Output:** `<SCENE>/work/selection.json` (stage `solved`),
`<SCENE>/work/points3d_seed.json`

```bash
SCENE=/data/box1 bash scripts/05_select_frames.sh
```

or directly:

```bash
python -m renderlabel select $SCENE --annotations $SCENE/work/seed.json \
    --selection $SCENE/work/selection.json \
    --theta-eff 11.5 --guard 5 --margin 20
```

## What happens

1. **Triangulate the seed clicks.** Each point clicked in both seed frames is
   solved by multi-view DLT (optionally LM-refined with `--refine`). The coarse
   point `X0` is the centroid of these points.
2. **Candidate frames.** A frame is a candidate if **every** triangulated seed
   point projects inside it with a `MARGIN` px border. This test uses poses and
   the model only — never a 2D observation from that frame.
3. **Angular span.** For each candidate, take the unit ray from `X0` to the
   camera centre. `Theta_span` is the largest angle between any two candidate
   rays — the cone the selected views have to cover.
4. **Prediction.**

   ```
   N_sat = ceil( Theta_span / (2 * theta_eff) )
   ```

   `theta_eff` is how many degrees of ray spread one view is worth. The default
   `11.5°` is the value fitted on the paper's seven sequences.
5. **Selection.** Greedy angular coverage: start from the seed pair, then
   repeatedly add the candidate whose nearest selected ray is farthest away,
   requiring at least `GUARD` (5°) between any two selected views. If no
   candidate satisfies the guard, **the search stops** — it never relaxes the
   guard. The realised count can therefore be smaller than `N_sat`; this happens
   when the candidate rays already lie closer together than the guard allows
   (for rays on one arc the ceiling is `floor(Theta_span / GUARD) + 1`, but for
   rays spread over a sphere there is no such closed form).

6. **Persist.** The coarse 3D points are written to
   `work/points3d_seed.json`; step 06 draws them as guide rings.

## Reading `selection.json`

```json
{
  "stage": "solved",
  "theta_eff_deg": 11.5,
  "guard_deg": 5.0,
  "margin_px": 20.0,
  "x0": [0.11, -0.02, 0.04],
  "n_points": 8,
  "points": ["P1", "..."],
  "n_candidates": 431,
  "theta_span_deg": 74.94,
  "n_raw": 3.26,
  "n_sat": 4,
  "n_selected": 4,
  "theta_max_deg": 27.1,
  "seed_files": ["frame_00013.png", "frame_00088.png"],
  "selected_files": ["frame_00013.png", "frame_00088.png", "...", "..."],
  "points3d": "/data/box1/work/points3d_seed.json"
}
```

- `n_sat` is the prediction; `n_selected` is what the guard actually allowed.
  When they differ, the guard is binding, not the saturation rule.
- `theta_max_deg` is the worst candidate's angle to its nearest selected view —
  the realised coverage radius. Smaller is better.

## Calibrating `theta_eff` on your own data

`11.5°` comes from one dataset family (7 sequences). It is a property of the
observation quality as much as of the geometry, so if you have the budget,
measure it once for your setup:

1. annotate a handful of extra frames (say 8–10 well-spread ones) with the same
   points;
2. for `N = 2..K`, take the first `N` frames of the greedy selection, triangulate
   from their clicks (`renderlabel.geometry.triangulate_point`), project the
   points into every **other** annotated frame (`project_points`) and record the
   median reprojection error;
3. find the `N` at which the error stops improving (the knee). Then
   `theta_eff ≈ Theta_span / (2 * N_knee)` for the `Theta_span` of that object.

A smaller `theta_eff` predicts more views (more annotation); a larger one fewer.
The rule is a budget predictor, not an accuracy guarantee: if the knee is flat,
prefer the smaller `N`.

## Tuning knobs

| flag | effect |
|---|---|
| `--theta-eff` | degrees per view; higher → fewer selected frames |
| `--guard` | minimum angle between views; higher → possibly fewer than `N_sat` |
| `--margin` | px border for candidacy; higher → fewer candidates (frames where the object is fully visible) |
| `--refine` | LM-refine the coarse points before computing `X0` (small effect) |

## Troubleshooting

| symptom | fix |
|---|---|
| `fewer than 2 candidate frames ... margin` | the seed points are spread too wide in the image, or the seed clicks are bad. Lower `--margin`, or re-click the seed frames. |
| `no point could be triangulated` | each point needs clicks in both seed frames; check `work/seed.json`. |
| `n_selected < n_sat` | the guard truncated the search. Expected for a small `Theta_span`; lower `GUARD` only if your views are genuinely redundant. |

Next: [06 — annotate the selected frames](06_annotate_selected.md)
