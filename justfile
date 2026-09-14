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

# Hours, not minutes -- 120 draws x 5 arms x 4 cluster counts x 3 topologies. Run it detached.
paper-2 draws="120" boot="10000":
    uv run python -u -m causaldyn_bench.paper_two \
        --draws {{draws}} --boot {{boot}} --clusters 2 4 8 20 --nodes 4 5 6 \
        --out results/paper2

# The same pipeline small enough to watch. Plumbing check only: 12 draws quote nothing.
paper-2-smoke:
    uv run python -u -m causaldyn_bench.paper_two \
        --draws 12 --boot 500 --clusters 2 --nodes 4 \
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
