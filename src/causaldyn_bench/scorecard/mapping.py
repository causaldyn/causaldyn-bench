"""An arm's fitted response, mapped onto :mod:`chc.response`, and the check that the mapping is its
tool's own.

Where a tool's response for each channel is exactly a :class:`chc.response.Channel`, a carryover
kernel, a curve and a coefficient, the arm's record carries it under ``response``: up to 400 draws
of each channel's parameters in chc's terms, and the tool's own decomposition of the history to
check them by. :func:`channels` builds each draw's channels; :func:`residual` is the largest
relative gap between what they return on the history's spend and what the decomposition says they
return. A mapping is exact when the residual is within :data:`TOLERANCE`, far above the rounding of
two float64 implementations of one formula and far below any slip in the mapping, a kernel one week
too long or a scale off by a factor. Where the tool's response is not such a channel, the record
says why under ``response.unmapped`` and carries no draws.

The record's ``response``, beside ``unmapped``:

* ``kernel``, ``length`` and ``normalized``: the carryover kernel each channel's adstock goes
  through, ``GeometricAdstock``, and its fixed parameters;
* ``curve``: ``Tanh`` or ``Hill``, applied to the adstock;
* ``parameters``: for each channel, each of ``retention``, the curve's ``scale`` (and ``slope``)
  and ``coefficient``, a list over the draws;
* ``decomposition``: what the tool says each channel returned over the history's weeks in each draw,
  ``total``, a list over the draws for each channel; and in each week for the draws ``held``,
  ``weekly``, one list of weeks for each held draw.

The check reads the history's spend from the observation the record was fitted to, and runs at
float64, as every scored run does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from chc.response import Channel, GeometricAdstock, Hill, Saturation, Tanh

from causaldyn_bench.endogenous_mmm import Series
from causaldyn_bench.scorecard.observe import Observation

TOLERANCE = 1e-6
DRAWS = 400  # the most draws a record keeps
CURVES = {"Tanh": ("scale",), "Hill": ("scale", "slope")}  # each curve's parameters


def _float64() -> None:
    if jnp.zeros(()).dtype != jnp.float64:
        raise RuntimeError("a mapping is read at float64: set JAX_ENABLE_X64=1")


def _curve(name: str, values: Mapping[str, float]) -> Saturation:
    return Tanh(values["scale"]) if name == "Tanh" else Hill(values["scale"], values["slope"])


def channels(response: Mapping[str, Any], names: Sequence[str]) -> tuple[tuple[Channel, ...], ...]:
    """Each draw's channels, in the order of ``names``.

    Raises:
        ValueError: the response is unmapped, names a kernel or curve chc does not hold, or does not
            give every channel the same number of draws.
    """
    _float64()
    if "unmapped" in response:
        raise ValueError(f"the response is unmapped: {response['unmapped']}")
    if response["kernel"] != "GeometricAdstock":
        raise ValueError(f"no kernel {response['kernel']!r}: GeometricAdstock alone is mapped")
    if response["curve"] not in CURVES:
        raise ValueError(f"no curve {response['curve']!r}: one of {sorted(CURVES)}")
    curve = response["curve"]
    length, normalized = int(response["length"]), bool(response["normalized"])
    parameters = response["parameters"]
    if sorted(parameters) != sorted(names):
        raise ValueError(f"the response maps {sorted(parameters)}, not {sorted(names)}")
    takes = sorted({"retention", "coefficient", *CURVES[curve]})
    for name in names:
        if sorted(parameters[name]) != takes:
            raise ValueError(
                f"{name}'s parameters are {sorted(parameters[name])}; a {curve} "
                f"channel takes {takes}"
            )
    draws = {len(values) for name in names for values in parameters[name].values()}
    if len(draws) != 1 or not 1 <= next(iter(draws)) <= DRAWS:
        raise ValueError(f"every parameter takes one value a draw, 1 to {DRAWS}: {sorted(draws)}")
    out = []
    for i in range(next(iter(draws))):
        drawn = []
        for name in names:
            values = {key: float(column[i]) for key, column in parameters[name].items()}
            kernel = GeometricAdstock(values["retention"], length=length, normalized=normalized)
            drawn.append(Channel(kernel, _curve(curve, values), values["coefficient"]))
        out.append(tuple(drawn))
    return tuple(out)


def _gap(ours: Series, theirs: Series) -> float:
    """``max |ours - theirs|`` over the larger of their largest magnitudes, 0 where both are 0."""
    scale = max(float(np.max(np.abs(ours))), float(np.max(np.abs(theirs))))
    return 0.0 if scale == 0.0 else float(np.max(np.abs(ours - theirs))) / scale


def residual(response: Mapping[str, Any], observation: Observation) -> float:
    """The largest relative gap between the mapped channels' returns on the history's spend and the
    tool's decomposition: each draw's total over the history, and each held draw's every week.

    Raises:
        ValueError: the response cannot be built (:func:`channels`), or its decomposition does not
            cover every draw and every week of the history.
    """
    if observation.population is not None:
        raise ValueError("a geo observation's response is not mapped: national families alone")
    names = observation.channels
    drawn = channels(response, names)
    decomposition = response["decomposition"]
    held = [int(i) for i in decomposition["held"]]
    weeks = observation.sales.shape[0]
    worst = 0.0
    for c, name in enumerate(names):
        totals = np.asarray(decomposition["total"][name], dtype=float)
        weekly = np.asarray(decomposition["weekly"][name], dtype=float)
        if totals.shape != (len(drawn),) or weekly.shape != (len(held), weeks):
            raise ValueError(
                f"{name}'s decomposition is {totals.shape} totals and {weekly.shape} weeks, not "
                f"{(len(drawn),)} and {(len(held), weeks)}"
            )
        spend = observation.spend[:, c]
        returns = np.stack([np.asarray(draw[c](spend)) for draw in drawn])
        worst = max(
            worst,
            *(_gap(r.sum(keepdims=True), t[None]) for r, t in zip(returns, totals, strict=True)),
        )
        worst = max(worst, *(_gap(returns[i], w) for i, w in zip(held, weekly, strict=True)))
    return worst
