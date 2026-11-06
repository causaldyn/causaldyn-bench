#!/usr/bin/env bash
# One command reproduces the scorecard's committed cells (results/scorecard_reproduce.json): family
# 0's pilot worlds at rungs 0 to 2, regenerated from their seeds, scored through every arm the bench
# runs, and compared with the committed cells within the stated tolerance, digests by machine class
# (causaldyn_bench.scorecard.reproduce). In the image the Containerfile builds, the tools' arms run
# too; elsewhere the script names each one it skipped, and why.
#
#   scripts/reproduce.sh [--out DIR]
#
# PYTHON names the interpreter of an environment the bench is installed in; without it, `uv run`
# runs the environment uv.lock pins, as it stands. The exit status is 0 only where the cells
# reproduce.
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PJRT_NPROC=1
if [[ -n "${PYTHON:-}" ]]; then
  exec "$PYTHON" -m causaldyn_bench.scorecard.reproduce "$@"
fi
exec uv run --no-sync python -m causaldyn_bench.scorecard.reproduce "$@"
