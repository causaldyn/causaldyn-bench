# Verification loop and paper-table generation for causaldyn-bench. `just check` runs exactly what
# ci.yml runs, in the same order, so a green check here is a green CI; `just paper-2` regenerates
# every table in paper P2 from one command, which is what makes those tables auditable rather than
# transcribed. There is deliberately no `ty` recipe: ci.yml does not type-check this repo (the
# notebooks and the BOPTEST client carry ten pre-existing diagnostics), and a recipe that fails on
# a clean tree teaches people to skip the ladder.

default:
    @just --list

# The Python ladder, cheapest first. Stops at the first failure.
check: fmt lint test

fmt:
    uv run ruff format --check .

lint:
    uv run ruff check .

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
