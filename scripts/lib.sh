# Shared helpers; every step script starts with `source "$(dirname "$0")/lib.sh"`.
# shellcheck shell=bash

set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

die() { echo "error: $*" >&2; exit 1; }

# `python -m renderlabel` inside the label env (or the current env if LABEL_ENV="")
rl() {
    if [ -n "${LABEL_ENV:-}" ]; then
        conda run --no-capture-output -n "$LABEL_ENV" python -m renderlabel "$@"
    else
        python -m renderlabel "$@"
    fi
}

# python inside the 3DGS env (torch + the gaussian-splatting repo)
gs_python() {
    if [ -n "${GS_ENV:-}" ]; then
        conda run --no-capture-output -n "$GS_ENV" python "$@"
    else
        python "$@"
    fi
}

# `python -m renderlabel` inside the 3DGS env (rendering needs torch)
gs_rl() {
    if [ -n "${GS_ENV:-}" ]; then
        conda run --no-capture-output -n "$GS_ENV" python -m renderlabel "$@"
    else
        python -m renderlabel "$@"
    fi
}

require_scene() {
    [ -n "${SCENE:-}" ] || die "SCENE is not set. Edit scripts/config.sh, or run with SCENE=/path/to/scene"
    local abs
    abs="$(cd "$SCENE" 2>/dev/null && pwd)" || die "SCENE does not exist: $SCENE"
    SCENE="$abs"
    export SCENE
}

# everything the pipeline creates for a scene lives under these paths
work_dir()  { echo "$SCENE/work"; }
renders_dir() { echo "$SCENE/3dgs/renders"; }
selection() { echo "$SCENE/work/selection.json"; }

banner() {
    printf '\n== %s ==\n' "$*"
}
