#!/usr/bin/env bash
# Step 05: predict the saturated view count and select which frames to annotate.
#
#   N_sat = ceil( Theta_span / (2 * theta_eff) )
#
# Theta_span is the widest pairwise angle between the rays from the (coarse)
# object point to every candidate camera; a candidate is a frame that sees all
# seed points inside the image with a MARGIN px border.  The selection is a
# greedy angular cover started from the seed pair, with a hard GUARD minimum
# angle (the search stops rather than relaxing the guard).
#
# Input : <SCENE>/work/seed.json         (the two annotated seed frames)
# Output: <SCENE>/work/selection.json    (stage "solved")
#         <SCENE>/work/points3d_seed.json (coarse 3D points, used as GUI guides)
#
#   SCENE=/data/box1 bash scripts/05_select_frames.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_scene
SEED="$SCENE/work/seed.json"
[ -f "$SEED" ] || die "$SEED not found (run 04_seed_views.sh)"

banner "predict N_sat and select frames"
rl select "$SCENE" \
    --annotations "$SEED" \
    --selection "$(selection)" \
    --theta-eff "$THETA_EFF" --guard "$GUARD" --margin "$MARGIN" \
    --seed-max-angle "$SEED_MAX_ANGLE"

echo
echo "next: bash scripts/06_annotate_selected.sh"
