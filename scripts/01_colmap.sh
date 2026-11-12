#!/usr/bin/env bash
# Step 01: structure-from-motion with COLMAP.
#
# Input : <SCENE>/images
# Output: <SCENE>/database.db, <SCENE>/sparse/0/{cameras,images,points3D}.bin
#
#   SCENE=/data/box1 bash scripts/01_colmap.sh
#
# Video frames are matched sequentially (neighbouring frames overlap); pass
# MATCHER=exhaustive for unordered photo sets.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_scene
command -v colmap >/dev/null 2>&1 || die "colmap not found (apt install colmap, or conda install -c conda-forge colmap)"
[ -d "$SCENE/images" ] || die "$SCENE/images does not exist (run 00_extract_frames.sh first)"

MATCHER="${MATCHER:-sequential}"
SINGLE_CAMERA="${SINGLE_CAMERA:-1}"   # 1 = one camera for all frames (recommended for video)

mkdir -p "$SCENE/sparse"
DB="$SCENE/database.db"

if [ ! -f "$DB" ]; then
    banner "feature extraction"
    colmap feature_extractor \
        --database_path "$DB" \
        --image_path "$SCENE/images" \
        --ImageReader.camera_model SIMPLE_RADIAL \
        --ImageReader.single_camera "$SINGLE_CAMERA"
else
    echo "database exists, skipping feature_extractor -> $DB"
fi

banner "matching ($MATCHER)"
case "$MATCHER" in
    sequential)
        colmap sequential_matcher --database_path "$DB" ;;
    exhaustive)
        colmap exhaustive_matcher --database_path "$DB" ;;
    *)
        die "MATCHER must be sequential or exhaustive, got $MATCHER" ;;
esac

banner "mapping"
colmap mapper \
    --database_path "$DB" \
    --image_path "$SCENE/images" \
    --output_path "$SCENE/sparse"

[ -d "$SCENE/sparse/0" ] || die "COLMAP produced no sparse/0. Check the mapper log:
  - too few frames / no overlap between consecutive frames
  - frames from several disconnected passes -> try MATCHER=exhaustive
  - if it wrote sparse/1 instead, keep the largest model and rename it to sparse/0"

banner "summary"
colmap model_analyzer --path "$SCENE/sparse/0" | head -20
echo
echo "sparse model -> $SCENE/sparse/0"
echo "next: bash scripts/02_train_3dgs.sh"
