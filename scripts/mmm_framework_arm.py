"""mmm-framework's own fit of its own data sets, outside the bench's environment.

mmm-framework 1.5.0 brings PyMC, NumPyro and JAX, so it is never a dependency of the bench: this
script runs in a throwaway environment (``scripts/mmm_framework_arm.txt``), reads one of the master
flat files the tool's wheel ships, which the bench holds unchanged in
``src/causaldyn_bench/scorecard/mmm_framework_worlds/``, and writes the tool's reading of each
channel as JSON, which the bench scores against the file's truth
(:mod:`causaldyn_bench.scorecard.mmm_framework`).

The tool is called as a black box, by the calls its README's Python example shows and no other:
``create_simple_mff_config`` with the sales as the KPI, the channels as media, the controls given
and the period dimension, the geography's beside it on a geo file; ``load_mff``;
``ModelConfigBuilder().bayesian_numpyro().with_chains(4).with_draws(2000).with_tune(1000)``, the
example's settings; ``BayesianMMM`` with ``TrendConfig(type=TrendType.LINEAR)``, the example's
trend; ``fit(random_seed=42)``, the example's seed; and ``MMMAnalyzer``'s
``compute_counterfactual_contributions`` and ``compute_channel_roi``. The example uses
``TrendConfig`` and ``TrendType`` without importing them; the script takes them from the package's
top level, or else from ``mmm_framework.config``, where the example's other configuration comes
from, and records which.

Each record carries the tool's version and those of what it runs on, the settings, each distinct
warning the fit raised with how often, the type of the counterfactual contributions' result, and
the ROI table as the analyser returned it, its index, columns and rows: each channel's spend,
contribution, ROI and the interval of its contribution, the reading the bench scores. Run offline,
the tool's caches in a scratch directory, one world a process::

    unshare -rn VENV/bin/python -I scripts/mmm_framework_arm.py MFF_CSV OUT_JSON KPI \\
        MEDIA CONTROLS GEO CHAINS DRAWS TUNE SEED

``MEDIA`` and ``CONTROLS`` comma-separated, ``GEO`` 1 for a geo file and 0 otherwise.
"""

import importlib.metadata
import json
import platform
import sys
import warnings
from collections import Counter
from importlib import import_module

import pandas as pd

VERSIONS = ("mmm-framework", "pymc", "pytensor", "numpyro", "jax", "numpy", "pandas")


def trend_classes():
    for where in ("mmm_framework", "mmm_framework.config"):
        module = import_module(where)
        if hasattr(module, "TrendConfig") and hasattr(module, "TrendType"):
            return where, module.TrendConfig, module.TrendType
    raise SystemExit("TrendConfig and TrendType are at neither place the example implies")


def jsonable(value):
    if isinstance(value, pd.DataFrame):
        return {
            "kind": "DataFrame",
            "index": [str(i) for i in value.index],
            "columns": [str(c) for c in value.columns],
            "values": [[jsonable(v) for v in row] for row in value.itertuples(index=False)],
        }
    if isinstance(value, pd.Series):
        return {
            "kind": "Series",
            "index": [str(i) for i in value.index],
            "values": [jsonable(v) for v in value.tolist()],
        }
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return {"kind": type(value).__name__, "repr": repr(value)[:2000]}


def main():
    path, out, kpi, media, controls, geo, chains, draws, tune, seed = sys.argv[1:11]
    from mmm_framework import BayesianMMM, ModelConfigBuilder  # ty: ignore[unresolved-import]
    from mmm_framework.analysis import MMMAnalyzer  # ty: ignore[unresolved-import]
    from mmm_framework.config import (  # ty: ignore[unresolved-import]
        DimensionType,
        create_simple_mff_config,
    )
    from mmm_framework.data_loader import load_mff  # ty: ignore[unresolved-import]

    where, TrendConfig, TrendType = trend_classes()
    dims = [DimensionType.PERIOD]
    if geo == "1":
        dims.append(DimensionType.GEOGRAPHY)
    config = create_simple_mff_config(
        kpi_name=kpi,
        media_names=media.split(","),
        control_names=[c for c in controls.split(",") if c],
        kpi_dimensions=dims,
    )
    panel = load_mff(pd.read_csv(path), config)
    model_config = (
        ModelConfigBuilder()
        .bayesian_numpyro()
        .with_chains(int(chains))
        .with_draws(int(draws))
        .with_tune(int(tune))
        .build()
    )
    model = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = model.fit(random_seed=int(seed))
    analyzer = MMMAnalyzer(model)
    contributions = analyzer.compute_counterfactual_contributions()
    roi = analyzer.compute_channel_roi()
    package = import_module("mmm_framework")
    record = {
        "tool": "mmm-framework",
        "version": getattr(package, "__version__", None),
        "versions": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in VERSIONS},
        },
        "file": path.rsplit("/", 1)[-1],
        "settings": {
            "kpi": kpi,
            "media": media.split(","),
            "controls": controls.split(","),
            "geo": geo == "1",
            "chains": int(chains),
            "draws": int(draws),
            "tune": int(tune),
            "seed": int(seed),
            "backend": "numpyro",
            "trend": "LINEAR",
            "trend_from": where,
        },
        "fit_result_type": type(results).__name__,
        "fit_warnings": dict(Counter(f"{w.category.__name__}: {w.message}" for w in caught)),
        "contributions_type": type(contributions).__name__,
        "roi": jsonable(roi),
    }
    with open(out, "w") as f:
        json.dump(record, f, indent=1)


if __name__ == "__main__":
    main()
