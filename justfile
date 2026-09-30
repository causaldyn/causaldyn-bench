# Verification loop and paper-table generation for causaldyn-bench. `just check` runs exactly what
# ci.yml runs, in the same order, so a green check here is a green CI; `just paper-2` regenerates
# every table in paper P2 from one command, which is what makes those tables auditable rather than
# transcribed. `just types` checks the local `.venv`'s interpreter, with the `notebooks` group
# synced; ci.yml runs it on every supported one, because uv.lock resolves a newer numpy from 3.12.

default:
    @just --list

# The Python ladder, cheapest first. Stops at the first failure.
check: fmt lint types test

fmt:
    uv run ruff format --check .

lint:
    uv run ruff check .

types:
    uv run ty check

# addopts already carries -q; a second one suppresses the summary line entirely.
test:
    uv run pytest

fix:
    uv run ruff check --fix .
    uv run ruff format .

# ── Paper tables ──────────────────────────────────────────────────────────────

# Hours, not minutes -- 16 DML sweeps to G = 1280; under ~700 seeds the ladder does not resolve.
paper-1 seeds="960":
    uv run python -u -m causaldyn_bench.paper_one \
        --n-seeds {{seeds}} \
        --out results/paper1

# The same pipeline small enough to watch. Plumbing check only: 30 seeds resolve no ladder.
paper-1-smoke:
    uv run python -u -m causaldyn_bench.paper_one \
        --delta-windows 0.2 0.002 --g-windows 10 20 --g-points 4 --n-seeds 30 \
        --seeds 0 1 2 --n-samples 100 \
        --out results/paper1-smoke

# Most of a night -- 120 draws x 5 arms x 4 cluster counts x 3 topologies, then the q = 3
# quadrature to 531 441 points, where the nodes = 9 cells alone are hours. Run it detached.
paper-2 draws="120" boot="10000":
    uv run python -u -m causaldyn_bench.paper_two \
        --draws {{draws}} --boot {{boot}} --clusters 2 4 8 20 --nodes 4 5 6 7 8 9 \
        --out results/paper2

# The arm ordering again at 64-bit precision. Not a rounding check: the panel sampler's seed spends
# different bits at the wider dtype, so this is an independent panel stream. Only its g = 2 arm
# table is quoted, hence one cluster count and the cheapest quadrature grid.
paper-2-x64 draws="120" boot="10000":
    JAX_ENABLE_X64=1 uv run python -u -m causaldyn_bench.paper_two \
        --draws {{draws}} --boot {{boot}} --clusters 2 --nodes 4 \
        --out results/paper2-x64

# The same pipeline small enough to watch. Plumbing check only: 12 draws quote nothing. Two
# quadrature grids, so the refinement-residual column is exercised too.
paper-2-smoke:
    uv run python -u -m causaldyn_bench.paper_two \
        --draws 12 --boot 500 --clusters 2 --nodes 3 4 \
        --out results/paper2-smoke

# Minutes, not hours -- everything but Table 5 is closed form, and Table 5 is five seeds.
paper-3 seeds="11 12 13 14 15":
    uv run python -u -m causaldyn_bench.paper_three \
        --seeds {{seeds}} \
        --out results/paper3

# The same pipeline at one seed and a short ladder. Plumbing check only.
paper-3-smoke:
    uv run python -u -m causaldyn_bench.paper_three \
        --horizons 1000 10000 --mass-horizons 10000 100000 --seeds 11 \
        --out results/paper3-smoke

# P3.3 on the live emulator: hours on the service's single worker, resumable from its journal. Needs
# the BOPTEST stack up. SIGINT rather than TERM, so an interrupted episode still stops its test.
paper-3-live url="http://127.0.0.1:8000":
    JAX_ENABLE_X64=1 timeout -s INT 18000 uv run python -u -m causaldyn_bench.boptest_capped \
        --url {{url}} --out results/boptest_capped

# P3.3's post-hoc questions, NOT pre-registered: reads the scored run's results.json, then reruns
# fixed-estimate and frozen-learning probe episodes in all seven windows on the service's single
# worker. Needs the stack up.
paper-3-live-posthoc url="http://127.0.0.1:8000":
    JAX_ENABLE_X64=1 timeout -s INT 7200 uv run python -u \
        -m causaldyn_bench.boptest_capped_posthoc --url {{url}}

# L8.1, the heat pump through chc.prescribe: hours on the service's single worker, resumable from
# its journal. Needs the BOPTEST stack up. SIGINT rather than TERM, so an interrupted episode still
# stops its test.
boptest-prescribe-live url="http://127.0.0.1:8000":
    JAX_ENABLE_X64=1 timeout -s INT 14400 uv run python -u -m causaldyn_bench.boptest_prescribe \
        --url {{url}} --out results/boptest_prescribe

# D21, L8.1 rerun with the band, the weather and the schedule stated: hours. The design's six
# replicates are dealt to `workers` processes, one BOPTEST worker each (scale the service to match),
# each resumable from its own journal; the last step collects them. The library runs from an archive
# of its HEAD, so edits to the checkout cannot reach the run, and that commit is recorded.
boptest-rerun-live url="http://127.0.0.1:8000" workers="3":
    #!/usr/bin/env bash
    set -euo pipefail
    lib=../causal-hybrid-control
    commit=$(git -C "$lib" rev-parse --short=12 HEAD)
    snapshot=$(mktemp -d)
    trap 'rm -rf "$snapshot"' EXIT
    git -C "$lib" archive "$commit" src | tar -x -C "$snapshot"
    export PYTHONPATH="$snapshot/src" JAX_ENABLE_X64=1
    pids=()
    for w in $(seq 0 $(({{workers}} - 1))); do
        timeout -s INT 43200 uv run python -u -m causaldyn_bench.boptest_rerun --url {{url}} \
            --replicates $(seq "$w" {{workers}} 5) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    uv run python -u -m causaldyn_bench.boptest_rerun --url {{url}} --chc-commit "$commit"

# Hours, not minutes -- 8 horizons x (1 + 5 seeds x 3 widths x 2 optimisers) solves. Run it detached.
paper-4:
    uv run python -u -m causaldyn_bench.paper_four --out results/paper4

# Two horizons, one seed, one width. Plumbing check only: two points rank nothing.
paper-4-smoke:
    uv run python -u -m causaldyn_bench.paper_four \
        --horizons 0.30 0.76 --seeds 0 --widths 16 \
        --out results/paper4-smoke

# ── Tracks with their own report ──────────────────────────────────────────────

# Track L against DCBO -> results/track_l.{md,json}. Minutes: sixty prescribe fits and the scoring.
# At 64-bit, because the fit is load-bearing. The DCBO arms are read from results/track_l_dcbo.json,
# which scripts/dcbo_reference.py writes in an isolated environment (its docstring has the setup),
# so neither GPy nor numpy<2 ever enters this lockfile.
track-l:
    JAX_ENABLE_X64=1 uv run python -u -m causaldyn_bench.dcbo_track --out results

# Track O -> results/track_o.{md,json}. Minutes: 500 logs per design through each environment's own
# step, every evaluate_plan arm on each, and a 400 000-step online truth per plan. Needs the `gym`
# extra. At 64-bit, so the estimate and the truth score the same cost matrices.
track-o:
    JAX_ENABLE_X64=1 uv run python -u -m causaldyn_bench.ope_calibration --out results

# Track P -> results/track_p.{md,json}. Minutes: 300 runs per arm through each environment's own
# step, a null's up to 10 000 decisions. Needs the `gym` extra. At 64-bit, so the plans' gains are
# Track O's.
track-p:
    JAX_ENABLE_X64=1 uv run python -u -m causaldyn_bench.drift_calibration --out results

# Track Q -> results/track_q.{md,json}. 200 panels per world, three prescribe fits and an evaluation
# on each, dealt to `--workers` processes. At 64-bit, so the fitted channel is the one the truth is
# read against.
track-q:
    JAX_ENABLE_X64=1 uv run python -u -m causaldyn_bench.graph_errors --out results

# Track M v2, lift tests -> results/track_m2_lift.{md,json}. 500 histories of Heusch's world for each
# of six settings, one chc.lift fit on each, dealt to `--workers` processes. At 64-bit, so the fit
# reads the readouts at the precision the NumPy world draws them in.
track-m2-lift:
    JAX_ENABLE_X64=1 uv run python -u -m causaldyn_bench.lift_calibration --out results
