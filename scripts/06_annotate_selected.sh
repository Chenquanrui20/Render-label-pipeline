#!/usr/bin/env bash
# Step 06: annotate the selected frames.
#
# The seed points are projected into each frame and drawn as guide rings;
# `a` adopts every guide at once (fast path), or click each point yourself.
# The GUI auto-saves after every click and resumes from an existing file.
#
# Input : <SCENE>/work/selection.json (stage "solved"), points3d_seed.json
# Output: <SCENE>/work/selected.json
#
#   SCENE=/data/box1 bash scripts/06_annotate_selected.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_scene
[ -f "$(selection)" ] || die "no selection.json (run 05_select_frames.sh)"
[ -f "$SCENE/work/points3d_seed.json" ] || die "no points3d_seed.json (run 05_select_frames.sh)"
[ -n "${DISPLAY:-}" ] || die "annotate opens an OpenCV window; no DISPLAY (use ssh -X or a desktop session)"

banner "annotate N_sat selected frames (GUI)"
echo "left click = place point | n/p = frame | u = undo | r = reset | a = adopt guides | q = save+quit"
rl annotate "$SCENE" \
    --frames selected --selection "$(selection)" \
    --points-file "$POINTS" \
    --init "$SCENE/work/points3d_seed.json" \
    --out "$SCENE/work/selected.json"

echo
echo "next: bash scripts/07_triangulate.sh"
