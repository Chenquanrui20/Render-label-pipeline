#!/usr/bin/env bash
# Step 00 (optional): extract frames from a video so COLMAP has images.
# Skip this if <SCENE>/images already exists.
#
#   VIDEO=/data/take1.mp4 FPS=2 SCENE=/data/box1 bash scripts/00_extract_frames.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_scene
[ -n "${VIDEO:-}" ] || die "VIDEO is empty; set it in scripts/config.sh or export VIDEO=/path/to/video"
[ -f "$VIDEO" ] || die "video not found: $VIDEO"
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found (apt install ffmpeg)"

banner "extract frames at ${FPS} fps"
mkdir -p "$SCENE/images"
if [ -n "$(ls -A "$SCENE/images" 2>/dev/null)" ]; then
    die "$SCENE/images is not empty; refusing to mix two frame sets"
fi
ffmpeg -hide_banner -loglevel warning -i "$VIDEO" \
    -vf "fps=$FPS" "$SCENE/images/frame_%05d.png"
n=$(find "$SCENE/images" -maxdepth 1 -type f | wc -l)
echo "wrote $n frames -> $SCENE/images"
echo "next: bash scripts/01_colmap.sh"
