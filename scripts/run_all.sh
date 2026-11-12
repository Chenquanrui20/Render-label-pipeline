#!/usr/bin/env bash
# Run every headless step in order: 01 -> 02 -> 03.
# Steps 04 and 06 open the annotation GUI and must be run by hand; 05 and 07
# are headless but depend on your clicks, so this script stops and prints them.
#
#   SCENE=/data/box1 bash scripts/run_all.sh
#
# SKIP_COLMAP=1 reuses an existing <SCENE>/sparse/0 (e.g. when you already ran
# step 01 yourself).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_scene

if [ "${SKIP_COLMAP:-0}" = "1" ]; then
    echo "SKIP_COLMAP=1 -> reusing $SCENE/sparse/0"
else
    bash "$HERE/01_colmap.sh"
fi
bash "$HERE/02_train_3dgs.sh"
bash "$HERE/03_render.sh"

cat <<EOF

== headless stages done ==

Now run, in this order:

  bash $HERE/04_seed_views.sh        # pick + annotate the two seed frames (GUI)
  bash $HERE/05_select_frames.sh     # predict N_sat and select frames
  bash $HERE/06_annotate_selected.sh # annotate the selected frames (GUI)
  bash $HERE/07_triangulate.sh       # triangulate + project labels
EOF
