# ---------------------------------------------------------------------------
# Edit this file once.  Every variable can also be overridden from the
# environment, e.g.
#
#     SCENE=/data/box1 N_FRAMES=200 bash scripts/02_train_3dgs.sh
# ---------------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Scene directory.  After step 01 it contains
#   images/  database.db  sparse/0/          (COLMAP output)
# and steps 02-04 add
#   3dgs/images 3dgs/sparse/0 3dgs/output 3dgs/renders
SCENE="${SCENE:-}"

# Optional: a video to extract frames from before COLMAP (step 00).
# Leave empty if you already have an images/ folder.
VIDEO="${VIDEO:-}"
FPS="${FPS:-2}"

# How many registered frames go into 3DGS training, and training iterations.
N_FRAMES="${N_FRAMES:-250}"
ITER="${ITER:-30000}"

# Upstream 3DGS repository: https://github.com/graphdeco-inria/gaussian-splatting
GS_REPO="${GS_REPO:-$HOME/gaussian-splatting}"

# Keypoint list, one name per line.
POINTS="${POINTS:-$ROOT/examples/points.txt}"

# Selection rule parameters.
THETA_EFF="${THETA_EFF:-11.5}"           # deg per view; recalibrate for your data
GUARD="${GUARD:-5}"                      # min ray angle between selected views
MARGIN="${MARGIN:-20}"                   # px a point must stay inside the frame
SEED_MAX_ANGLE="${SEED_MAX_ANGLE:-160}"  # never pick a seed pair wider than this

# Conda env names.  Set to "" to use the currently activated environment.
LABEL_ENV="${LABEL_ENV-rlabel}"
GS_ENV="${GS_ENV-gs}"
