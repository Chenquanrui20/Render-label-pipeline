#!/usr/bin/env bash
# ArUco (ChArUco) demo: stages 04-07 on an existing 3DGS render set.
#
#   SCENE=/path/to/charuco/train/0000 bash examples/charuco/run_demo.sh
#
# SCENE must contain  sparse/0, images/, 3dgs/renders/{images,message,labels}.
# The GUI clicks of steps 04/06 are simulated from the render labels; see
# examples/make_clicks.py.  Everything else is the real pipeline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python3}"
SCENE="${SCENE:?set SCENE to the sequence dir (contains sparse/0, images/, 3dgs/renders)}"
RENDERS="${RENDERS:-$SCENE/3dgs/renders}"
LABELS="${LABELS:-$RENDERS/labels}"
WORK="${WORK:-$HERE/work}"
THETA_EFF="${THETA_EFF:-11.5}"
GUARD="${GUARD:-5}"
MARGIN="${MARGIN:-20}"
MIN_VIEWS="${MIN_VIEWS:-2}"

cd "$REPO"
mkdir -p "$WORK"

echo "=============================================================="
echo " charuco demo: $SCENE"
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
"$PY" examples/make_clicks.py --format charuco --renders "$RENDERS" \
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
"$PY" examples/make_clicks.py --format charuco --renders "$RENDERS" \
    --labels "$LABELS" --frames-from "$WORK/selection.json" --which selected \
    --out "$WORK/selected.json"

echo
echo "[stage 07] triangulate + project into every render"
"$PY" -m renderlabel triangulate "$SCENE" --renders "$RENDERS" \
    --annotations "$WORK/seed.json" "$WORK/selected.json" \
    --out "$WORK/labels" --overlay "" --min-views "$MIN_VIEWS"

echo
echo "[eval] held-out 2D accuracy, random baseline, 3D vs board"
"$PY" examples/evaluate.py --format charuco --renders "$RENDERS" \
    --labels "$LABELS" --selection "$WORK/selection.json" \
    --points3d "$WORK/labels/points3d.csv" --min-views "$MIN_VIEWS" \
    --trials "${TRIALS:-100}" --out "$WORK/eval.json"

echo
"$PY" examples/summarize.py --run "charuco:$WORK" --out "$WORK/summary.md"

echo
echo "done. labels: $WORK/labels ; summary: $WORK/summary.md"
