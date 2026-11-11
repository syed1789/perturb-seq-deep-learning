#!/usr/bin/env bash
# Run the whole analysis in containers. Works with Docker (laptop, EC2) or
# Apptainer (HPC, where Docker is not permitted).
#
#   ./run_analysis.sh prefetch   # download input data (needs internet)
#   ./run_analysis.sh run        # run the analysis (default)
#   RUNTIME=apptainer ./run_analysis.sh run
#   DRY_RUN=1 ./run_analysis.sh  # print the commands without running them
set -euo pipefail
cd "$(dirname "$0")"
RUNTIME="${RUNTIME:-docker}"
DRY_RUN="${DRY_RUN:-0}"

run() { if [ "$DRY_RUN" = 1 ]; then echo "+ $*"; else "$@"; fi; }

# build_image <docker-tag> <build-context>
#   docker:    builds the image.
#   apptainer: expects ./<tag>.sif (build once: see HPC_AWS.md); builds it here
#              from the local Docker daemon if Docker happens to be available.
build_image() {
  local tag="$1" ctx="$2" sif="${1%%:*}.sif"
  case "$RUNTIME" in
    docker) run docker build -t "$tag" "$ctx" ;;
    apptainer)
      if [ ! -f "$sif" ]; then
        if command -v docker >/dev/null 2>&1; then
          run docker build -t "$tag" "$ctx"
          run apptainer build "$sif" "docker-daemon://$tag"
        else
          echo "Missing $sif. Build it on a machine with Docker (see HPC_AWS.md)." >&2
          [ "$DRY_RUN" = 1 ] || exit 1
        fi
      fi ;;
    *) echo "RUNTIME must be docker or apptainer" >&2; exit 1 ;;
  esac
}

# ctr <docker-ref-or-tag> <apptainer-source> <command...>
#   apptainer-source: a .sif path, or docker://<ref> for public registry images.
ctr() {
  local ref="$1" src="$2"; shift 2
  case "$RUNTIME" in
    docker) run docker run --rm -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e NUMBA_CACHE_DIR=/tmp \
              -v "$PWD":/work -w /work "$ref" "$@" ;;
    apptainer) run apptainer exec --bind "$PWD":/work --pwd /work "$src" "$@" ;;
  esac
}

step="${1:-run}"

IMG=perturb-seq-dl:latest
build_image "$IMG" .
case "$step" in
  prefetch) run bash scripts/download_data.sh ;;   # host curl; ~35 MB, MD5-verified
  run)
    [ -f data/adamson2016_10X001.h5ad ] || run bash scripts/download_data.sh
    ctr "$IMG" perturb-seq-dl.sif python scripts/preprocess.py
    ctr "$IMG" perturb-seq-dl.sif python scripts/train_classifier.py ;;
  *) echo "usage: $0 [prefetch|run]" >&2; exit 1 ;;
esac
