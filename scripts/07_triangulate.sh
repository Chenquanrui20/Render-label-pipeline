#!/usr/bin/env bash
# Step 07: triangulate the clicks and project labels into every rendered frame.
#
# Input : <SCENE>/work/{seed,selected}.json
# Output: <SCENE>/3dgs/labels/<frame>.json   labelme JSON, one point shape per keypoint
#         <SCENE>/3dgs/labels/points3d.csv   the 3D points + per-point residuals
#         <SCENE>/3dgs/labels/summary.json   coverage, residuals, failed points
#         <SCENE>/3dgs/overlay/<frame>.png   QC overlay (set OVERLAY= to disable)
#
#   SCENE=/data/box1 bash scripts/07_triangulate.sh
#
# Optional: REFINE=1 adds an LM refinement of each 3D point (changes the result
# by ~0.01 px on the paper's sequences; off by default).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_scene
ANN=()
[ -f "$SCENE/work/seed.json" ] && ANN+=("$SCENE/work/seed.json")
[ -f "$SCENE/work/selected.json" ] && ANN+=("$SCENE/work/selected.json")
[ ${#ANN[@]} -gt 0 ] || die "no annotations under $SCENE/work (run 04 and 06)"

OVERLAY="${OVERLAY-$SCENE/3dgs/overlay}"
args=(triangulate "$SCENE" --annotations "${ANN[@]}" --out "$SCENE/3dgs/labels")
if [ -n "$OVERLAY" ]; then
    args+=(--overlay "$OVERLAY")
else
    args+=(--overlay "")
fi
[ "${REFINE:-0}" = "1" ] && args+=(--refine)

banner "triangulate + project"
rl "${args[@]}"

echo
echo "labels -> $SCENE/3dgs/labels"
echo "check summary.json (residuals, coverage) and the overlays before training."
