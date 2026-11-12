#!/usr/bin/env bash
# Step 04: two-view cold start.
#
# 1) pose only  -> pick the two seed frames whose rays are widest apart
#                  (no annotation needed; N_sat plays no role yet)
# 2) annotate   -> opens the GUI on exactly those two frames
#
#   SCENE=/data/box1 bash scripts/04_seed_views.sh
#
# Output: <SCENE>/work/selection.json   (stage "seeds")
#         <SCENE>/work/seed.json        (your clicks: frames -> {point: [u,v]})
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_scene
[ -d "$(renders_dir)" ] || die "no renders yet (run 03_render.sh)"
[ -f "$POINTS" ] || die "points file not found: $POINTS"
[ -n "${DISPLAY:-}" ] || die "annotate opens an OpenCV window; no DISPLAY (use ssh -X or a desktop session)"

banner "pose-only seed selection"
rl select "$SCENE" --selection "$(selection)" \
    --theta-eff "$THETA_EFF" --seed-max-angle "$SEED_MAX_ANGLE"

banner "annotate the two seed frames (GUI)"
echo "left click = place point | n/p = frame | u = undo | r = reset | a = adopt guides | q = save+quit"
rl annotate "$SCENE" \
    --frames seed --selection "$(selection)" \
    --points-file "$POINTS" \
    --out "$SCENE/work/seed.json"

echo
echo "next: bash scripts/05_select_frames.sh"
