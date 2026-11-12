#!/usr/bin/env bash
# Step 02: pick N frames, build the undistorted scene, train 3DGS.
#
# Input : <SCENE>/images, <SCENE>/sparse/0
# Output: <SCENE>/3dgs/{images,sparse/0}         (undistorted PINHOLE model)
#         <SCENE>/3dgs/output/point_cloud/iteration_<ITER>/point_cloud.ply
#
#   SCENE=/data/box1 bash scripts/02_train_3dgs.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_scene
[ -d "$SCENE/sparse/0" ] || die "$SCENE/sparse/0 missing (run 01_colmap.sh)"
[ -d "$GS_REPO" ] || die "GS_REPO does not exist: $GS_REPO (see docs/02_train_3dgs.md)"

banner "build the 3DGS subset ($N_FRAMES frames, undistorted)"
rl subset "$SCENE" -n "$N_FRAMES"

banner "train 3DGS ($ITER iterations)"
mkdir -p "$SCENE/3dgs/output"
(
    cd "$GS_REPO"
    gs_python train.py \
        -s "$SCENE/3dgs" \
        -m "$SCENE/3dgs/output" \
        --iterations "$ITER"
)

PLY="$SCENE/3dgs/output/point_cloud/iteration_$ITER/point_cloud.ply"
[ -f "$PLY" ] || die "training finished but $PLY does not exist"
echo
echo "trained model -> $PLY"
echo "next: bash scripts/03_render.sh"
