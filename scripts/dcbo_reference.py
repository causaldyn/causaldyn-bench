"""Run the DCBO reference implementation on Track L's logs, outside the bench's environment.

The reference (github.com/neildhir/DCBO) needs ``GPy<1.14`` and therefore ``numpy<2``, and its
licence is ambiguous (MIT in ``LICENSE``, GPL-3.0-or-later in the README and ``setup.py``). So it is
never a dependency of the bench: this script runs in a throwaway environment, and only its output --
the interventions each method implemented -- is committed, as data that
:mod:`causaldyn_bench.dcbo_track` scores on the bench's own port of the SCMs.

The logs are drawn by that port (``src/causaldyn_bench/dcbo_scm.py``, loaded by path because it
imports nothing but NumPy) and written into the output, so the bench can check that the DCBO arm and
the CHC arm read the same data. The interventions DCBO explores are evaluated on the reference's own
SEM classes, which is what its API requires.

Seed ``s`` is log ``s`` and the reference's replicate ``s``: ``run_methods_replicates`` gives
replicate ``ex`` of a controlled experiment ``seed_anchor_points = ex + 1``, and this script calls
the per-method runner beneath it, ``run_all_opt_models``, with ``s + 1``. Through
``run_methods_replicates`` itself every single-replicate call would be replicate 0, and ABO and BO,
which never read the log, would return one run ten times.

    git clone https://github.com/neildhir/DCBO ~/.cache/chc-scratch/dcbo-src
    git -C ~/.cache/chc-scratch/dcbo-src checkout 85a9bdfa3552d2227fb03f481cb47c3b26b8bf33
    uv venv --python 3.10 ~/.cache/chc-scratch/dcbo/.venv
    uv pip install --python ~/.cache/chc-scratch/dcbo/.venv/bin/python "numpy<2" "scipy<=1.12" \\
        "GPy==1.13.2" emukit networkx pandas scikit-learn matplotlib seaborn tqdm paramz graphviz
    PYTHONPATH=~/.cache/chc-scratch/dcbo-src uv run --no-project \\
        --python ~/.cache/chc-scratch/dcbo/.venv/bin/python python scripts/dcbo_reference.py

``pygraphviz`` is not installed: it only builds the DAG in ``example_setups.py``, and building the
same edges into a ``networkx.MultiDiGraph`` directly needs no C extension.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from dcbo.experimental.experiments import (  # ty: ignore[unresolved-import]
    optimal_sequence_of_interventions,
    run_all_opt_models,
)
from dcbo.utils.dag_utils.graph_functions import (  # ty: ignore[unresolved-import]
    make_graphical_model,
)
from dcbo.utils.sem_utils.sem_estimate import build_sem_hat  # ty: ignore[unresolved-import]
from dcbo.utils.sem_utils.toy_sems import (  # ty: ignore[unresolved-import]
    NonStationaryDependentSEM,
    StationaryDependentSEM,
    StationaryIndependentSEM,
)
from dcbo.utils.sequential_intervention_functions import (  # ty: ignore[unresolved-import]
    get_interventional_grids,
)
from networkx import MultiDiGraph  # ty: ignore[unresolved-import]

DCBO_COMMIT = "85a9bdfa3552d2227fb03f481cb47c3b26b8bf33"
METHODS = ("DCBO", "CBO", "ABO", "BO")
TRIALS = 10  # number_of_trials: explorative interventions per step, as in the reference notebooks
ANCHORS = 100  # num_anchor_points, as in the reference notebooks
SEMS: dict[str, tuple[Any, str]] = {
    "stat": (StationaryDependentSEM, "dependent"),
    "ind": (StationaryIndependentSEM, "independent"),
    "nonstat": (NonStationaryDependentSEM, "dependent"),
}


def _port() -> Any:
    path = Path(__file__).resolve().parents[1] / "src" / "causaldyn_bench" / "dcbo_scm.py"
    spec = importlib.util.spec_from_file_location("dcbo_scm", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the SCM port from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _graph(name: str, horizon: int) -> Any:
    """``example_setups.setup_*_scm``'s DAG, as a MultiDiGraph with the DOT's node order."""
    topology = SEMS[name][1]
    extra = {"target_node": "Y"} if topology == "independent" else {}
    dot = make_graphical_model(0, horizon - 1, topology=topology, nodes=["X", "Z", "Y"], **extra)
    graph = MultiDiGraph()
    for parent, child in re.findall(r"(\w+_\d+) -> (\w+_\d+);", dot):
        graph.add_edge(parent, child)
    if name == "nonstat":  # nonstat_scm.ipynb, cell 5: the change point's extra edges
        graph.add_edge("X_0", "Z_1")
        graph.add_edge("Z_1", "Y_2")
    return graph


def _sem(name: str) -> Any:
    cls = SEMS[name][0]
    return cls(1) if name == "nonstat" else cls()


def _input_params(
    name: str,
    graph: Any,
    observations: dict[str, Any],
    domain: dict[str, list[float]],
    seed: int,
    change_points: list[bool] | None,
) -> dict[str, Any]:
    """``run_methods_replicates``' parameters at its defaults, but replicate ``seed``'s anchors."""
    return {
        "G": graph,
        "sem": SEMS[name][0],
        "base_target_variable": "Y",
        "observation_samples": observations,
        "intervention_domain": domain,
        "intervention_samples": None,
        "number_of_trials": TRIALS,
        "task": "min",
        "cost_type": 1,
        "n_restart": 1,
        "debug_mode": False,
        "optimal_assigned_blankets": None,
        "num_anchor_points": ANCHORS,
        "sample_anchor_points": True,
        "seed_anchor_points": seed + 1,
        "hp_i_prior": True,
        "args_sem": None,
        "manipulative_variables": None,
        "change_points": change_points,
    }


def _decisions(model: Any, horizon: int) -> list[dict[str, Any]]:
    """The decision per step, read the way ``Root._post_optimisation_assignments`` reads it."""
    out = []
    for t in range(horizon):
        outcomes = model.outcome_values[t]
        best = min(outcomes)
        variables = model.optimal_intervention_sets[t]
        levels = model.optimal_intervention_levels[t][variables][outcomes.index(best) - 1]
        out.append(
            {
                "variables": list(variables),
                "levels": [float(v) for v in np.asarray(levels).ravel()],
                "reported": float(best),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out", type=Path, default=Path("results/track_l_dcbo.json"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    port = _port()
    domain = {name: list(bounds) for name, bounds in port.DOMAIN.items()}
    sets = [tuple(s) for s in port.EXPLORATION_SETS]
    record: dict[str, Any] = {
        "dcbo_commit": DCBO_COMMIT,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "GPy": importlib.import_module("GPy").__version__,
            "emukit": importlib.import_module("emukit").__version__,
        },
        "settings": {
            "methods": list(METHODS),
            "number_of_trials": TRIALS,
            "num_anchor_points": ANCHORS,
            "n_observations": port.N_OBSERVATIONS,
            "horizon": port.HORIZON,
            "domain": domain,
            "exploration_sets": [list(s) for s in sets],
            "seeds": args.seeds,
            "seed_anchor_points": "seed + 1",
        },
        "reference_oracle": {},
        "runs": {},
    }
    for scm in port.SCMS:
        graph = _graph(scm.name, port.HORIZON)
        sem = _sem(scm.name)
        change_points = [t == 1 for t in range(port.HORIZON)] if scm.name == "nonstat" else None
        _, best_sets, best_values, _, _, _ = optimal_sequence_of_interventions(
            sets,
            get_interventional_grids(sets, domain, size_intervention_grid=port.GRID),
            sem.static(),
            sem.dynamic(),
            graph,
            port.HORIZON,
            model_variables=["X", "Z", "Y"],
            target_variable="Y",
        )
        record["reference_oracle"][scm.name] = {
            "sets": [list(s) for s in best_sets],
            "values": [float(v) for v in best_values],
        }
        record["runs"][scm.name] = {}
        for seed in args.seeds:
            log = port.sample_log(scm, seed)
            observations = {name: log[name] for name in ("X", "Z", "Y")}
            run: dict[str, Any] = {
                "log": {name: values.tolist() for name, values in observations.items()},
                "methods": {},
            }
            for method in METHODS:
                started = time.perf_counter()
                np.random.seed(seed)  # what run_methods_replicates does before its replicates
                models, _ = run_all_opt_models(
                    methods_list=[method],
                    input_params=_input_params(
                        scm.name, graph, observations, domain, seed, change_points
                    ),
                    exploration_sets=sets,
                    online=False,
                    use_di=False,
                    transfer_hp_o=False,
                    transfer_hp_i=False,
                    concat=False,
                    estimate_sem=True,
                    use_mc=False,
                    ground_truth=None,
                    n_obs_t=None,
                    make_sem_estimator=build_sem_hat,
                    number_of_trials_BO_ABO=TRIALS,
                )
                run["methods"][method] = {
                    "decisions": _decisions(models[0], port.HORIZON),
                    "seconds": time.perf_counter() - started,
                }
                print(scm.name, seed, method, run["methods"][method]["decisions"], flush=True)
            record["runs"][scm.name][str(seed)] = run
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
