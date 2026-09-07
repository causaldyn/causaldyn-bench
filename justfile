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
