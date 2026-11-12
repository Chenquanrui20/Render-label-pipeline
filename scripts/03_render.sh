#!/usr/bin/env bash
# Step 03: render every COLMAP pose with the trained 3DGS model.
#
# Input : <SCENE>/sparse/0 + <SCENE>/3dgs/output/.../point_cloud.ply
# Output: <SCENE>/3dgs/renders/images/<name>.png
#         <SCENE>/3dgs/renders/message/<name>.json   (pose + intrinsics per frame)
#
#   SCENE=/data/box1 bash scripts/03_render.sh
#
# One shared pinhole (median focal, centred principal point) is used for all
# views: per-image COLMAP intrinsics would give every view its own pixel space
# and break multi-view triangulation.  Use CAMERA=per-image only if the source
# model is already PINHOLE and you know why you need it.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_scene
[ -d "$SCENE/sparse/0" ] || die "$SCENE/sparse/0 missing (run 01_colmap.sh)"
[ -d "$GS_REPO" ] || die "GS_REPO does not exist: $GS_REPO (see docs/02_train_3dgs.md)"
export GS_REPO

CAMERA="${CAMERA:-shared}"
LIMIT="${LIMIT:-}"

banner "render all poses (camera=$CAMERA, iteration=$ITER)"
args=(render "$SCENE" --iteration "$ITER" --camera "$CAMERA")
[ -n "$LIMIT" ] && args+=(--limit "$LIMIT")
gs_rl "${args[@]}"

N=$(find "$SCENE/3dgs/renders/message" -maxdepth 1 -name '*.json' | wc -l)
echo
echo "renders -> $SCENE/3dgs/renders ($N messages)"
echo "next: bash scripts/04_seed_views.sh"
