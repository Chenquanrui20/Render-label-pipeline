#!/usr/bin/env bash
# YCBV demo: stages 04-07 on an existing 3DGS render set.
#
#   SCENE=/path/to/ycbv_little/data/train/seq00_room_20261012 \
#       bash examples/ycbv/run_demo.sh
#
# SCENE must contain  images/, colmap/sparse/0, points_summary.json,
# 3dgs_renders/{images,message} and labels_render/points_states.csv.
# The GUI clicks of steps 04/06 are simulated from the render labels
# (visible rows only); everything else is the real pipeline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python3}"
SCENE="${SCENE:?set SCENE to the YCBV sequence dir (contains 3dgs_renders/)}"
RENDERS="${RENDERS:-$SCENE/3dgs_renders}"
LABELS="${LABELS:-$SCENE/labels_render/points_states.csv}"
GT_POINTS="${GT_POINTS:-$SCENE/points_summary.json}"
WORK="${WORK:-$HERE/work}"
THETA_EFF="${THETA_EFF:-11.5}"
GUARD="${GUARD:-5}"
MARGIN="${MARGIN:-20}"
MIN_VIEWS="${MIN_VIEWS:-2}"

cd "$REPO"
mkdir -p "$WORK"

echo "=============================================================="
echo " ycbv demo: $SCENE"
echo " renders: $RENDERS"
echo " labels : $LABELS"
echo " work   : $WORK"
echo "=============================================================="

echo
echo "[stage 04] pose-only seed pair"
"$PY" -m renderlabel select "$SCENE" --renders "$RENDERS" \
    --selection "$WORK/selection.json" --points3d "$WORK/points3d_seed.json" \
    --theta-eff "$THETA_EFF" --guard "$GUARD" --margin "$MARGIN"

echo
echo "[stage 04] seed clicks (simulated from the render labels)"
"$PY" examples/make_clicks.py --format ycbv --renders "$RENDERS" \
    --labels "$LABELS" --frames-from "$WORK/selection.json" --which seed \
    --out "$WORK/seed.json"

echo
echo "[stage 05] Theta_span -> N_sat -> greedy selection"
"$PY" -m renderlabel select "$SCENE" --renders "$RENDERS" \
    --annotations "$WORK/seed.json" --selection "$WORK/selection.json" \
    --points3d "$WORK/points3d_seed.json" \
    --theta-eff "$THETA_EFF" --guard "$GUARD" --margin "$MARGIN"

echo
echo "[stage 06] clicks on the selected frames (simulated)"
"$PY" examples/make_clicks.py --format ycbv --renders "$RENDERS" \
    --labels "$LABELS" --frames-from "$WORK/selection.json" --which selected \
    --out "$WORK/selected.json"

echo
echo "[stage 07] triangulate + project into every render"
"$PY" -m renderlabel triangulate "$SCENE" --renders "$RENDERS" \
    --annotations "$WORK/seed.json" "$WORK/selected.json" \
    --out "$WORK/labels" --overlay "" --min-views "$MIN_VIEWS"

echo
echo "[eval] held-out 2D accuracy, random baseline, 3D vs GT points"
"$PY" examples/evaluate.py --format ycbv --renders "$RENDERS" \
    --labels "$LABELS" --selection "$WORK/selection.json" \
    --points3d "$WORK/labels/points3d.csv" --gt-points "$GT_POINTS" \
    --min-views "$MIN_VIEWS" --trials "${TRIALS:-100}" --out "$WORK/eval.json"

echo
"$PY" examples/summarize.py --run "ycbv:$WORK" --out "$WORK/summary.md"

echo
echo "done. labels: $WORK/labels ; summary: $WORK/summary.md"
