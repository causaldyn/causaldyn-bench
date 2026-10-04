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
* ``curve``: ``Tanh`` or ``Hill``, applied to the adstock: one name for every draw, or a list of
  one name a draw where a tool's draws mix curves;
* ``parameters``: for each channel, each of ``retention``, the curve's ``scale`` (and ``slope``)
  and ``coefficient``, a list over the draws. Where the draws mix curves, a draw whose curve takes
  no ``slope`` has null there. A draw's ``coefficient`` is a number, or a list over the history's
  weeks where the tool's coefficient moves over them;
* ``decomposition``: what the tool says each channel returned over the history's weeks in each draw,
  ``total``, a list over the draws for each channel; and in each week for the draws ``held``,
  ``weekly``, one list of weeks for each held draw.

A coefficient that moves is held past the history's last week at that week's (:class:`Walking`),
the one convention a return that counts all of its carryover needs (:mod:`.returns`).

The check reads the history's spend from the observation the record was fitted to, and runs at
float64, as every scored run does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import jax.numpy as jnp
import numpy as np
from chc.response import Channel, GeometricAdstock, Hill, Saturation, Tanh
from jax import Array
from jax.typing import ArrayLike

from causaldyn_bench.endogenous_mmm import Series
from causaldyn_bench.scorecard.observe import Observation

TOLERANCE = 1e-6
DRAWS = 400  # the most draws a record keeps
CURVES = {"Tanh": ("scale",), "Hill": ("scale", "slope")}  # each curve's parameters
KEYS = ("kernel", "length", "normalized", "curve", "parameters", "decomposition")


def _float64() -> None:
    if jnp.zeros(()).dtype != jnp.float64:
        raise RuntimeError("a mapping is read at float64: set JAX_ENABLE_X64=1")


@dataclass(frozen=True, eq=False)
class Drawn:
    """One channel's draws as a response gives them: each draw's curve, its parameters, and its
    coefficient, ``(draws,)``, or ``(draws, weeks)`` where it moves over the history's weeks."""

    curves: tuple[str, ...]
    retention: Series
    scale: Series
    slope: Series  # nan in a draw whose curve takes none
    coefficient: Series


@dataclass(frozen=True, eq=False)
class Walking:
    """A channel whose coefficient moves over the history's weeks: ``channel``'s return at a
    coefficient of 1, times each week's coefficient, held at the history's last week's past it."""

    channel: Channel
    path: Series  # (weeks,)

    def __call__(self, spend: ArrayLike) -> Array:
        returns = self.channel(spend)
        weeks = returns.shape[0]
        held = np.concatenate([self.path, np.full(max(weeks - self.path.size, 0), self.path[-1])])
        return jnp.asarray(held[:weeks]) * returns


def _coefficients(name: str, values: Sequence[Any]) -> Series:
    """A number a draw, or a path a draw, every path as long as the others."""
    if all(isinstance(value, Real) for value in values):
        return np.asarray(values, dtype=float)
    lengths = {len(value) if isinstance(value, Sequence) else -1 for value in values}
    if len(lengths) != 1 or min(lengths) < 1:
        raise ValueError(f"{name}'s coefficient is a number a draw or a path a draw, of one length")
    return np.asarray(values, dtype=float)


def drawn(response: Mapping[str, Any], names: Sequence[str]) -> dict[str, Drawn]:
    """Each channel's draws, under its name.

    Raises:
        ValueError: the response is unmapped or lacks a key, names a kernel or curve chc does not
            hold, does not give every channel and every parameter the same number of draws, gives a
            draw a slope its curve does not take, or mixes numbers and paths in a coefficient.
    """
    if "unmapped" in response:
        raise ValueError(f"the response is unmapped: {response['unmapped']}")
    missing = [key for key in KEYS if key not in response]
    if missing:
        raise ValueError(f"the response has no {missing}")
    if response["kernel"] != "GeometricAdstock":
        raise ValueError(f"no kernel {response['kernel']!r}: GeometricAdstock alone is mapped")
    named = response["curve"]
    listed = [named] if isinstance(named, str) else list(named)
    for curve in listed:
        if curve not in CURVES:
            raise ValueError(f"no curve {curve!r}: one of {sorted(CURVES)}")
    parameters = response["parameters"]
    if sorted(parameters) != sorted(names):
        raise ValueError(f"the response maps {sorted(parameters)}, not {sorted(names)}")
    kinds = sorted(set(listed))
    takes = sorted({"retention", "coefficient", *(key for kind in kinds for key in CURVES[kind])})
    for name in names:
        if sorted(parameters[name]) != takes:
            raise ValueError(
                f"{name}'s parameters are {sorted(parameters[name])}; a {' or '.join(kinds)} "
                f"channel takes {takes}"
            )
    draws = {len(values) for name in names for values in parameters[name].values()}
    if not isinstance(named, str):
        draws.add(len(listed))
    if len(draws) != 1 or not 1 <= next(iter(draws)) <= DRAWS:
        raise ValueError(
            f"every parameter, and a list of curves, takes one value a draw, 1 to {DRAWS}: "
            f"{sorted(draws)}"
        )
    count = next(iter(draws))
    curves = tuple(listed * count if isinstance(named, str) else listed)
    out = {}
    for name in names:
        columns = parameters[name]
        slopes = columns.get("slope", [None] * count)
        if any(
            (curve == "Hill") == (slope is None)
            for curve, slope in zip(curves, slopes, strict=True)
        ):
            raise ValueError(f"{name}'s slope is a number in a Hill draw and null in a Tanh one")
        out[name] = Drawn(
            curves,
            np.asarray(columns["retention"], dtype=float),
            np.asarray(columns["scale"], dtype=float),
            np.array([np.nan if slope is None else float(slope) for slope in slopes]),
            _coefficients(name, columns["coefficient"]),
        )
    return out


def _curve(name: str, scale: float, slope: float) -> Saturation:
    return Tanh(scale) if name == "Tanh" else Hill(scale, slope)


def _built(
    parsed: Mapping[str, Drawn], names: Sequence[str], length: int, normalized: bool
) -> tuple[tuple[Callable[[ArrayLike], Array], ...], ...]:
    out = []
    for i in range(len(parsed[names[0]].curves)):
        draw: list[Callable[[ArrayLike], Array]] = []
        for name in names:
            d = parsed[name]
            kernel = GeometricAdstock(float(d.retention[i]), length=length, normalized=normalized)
            curve = _curve(d.curves[i], float(d.scale[i]), float(d.slope[i]))
            if d.coefficient.ndim == 1:
                draw.append(Channel(kernel, curve, float(d.coefficient[i])))
            else:
                draw.append(Walking(Channel(kernel, curve, 1.0), d.coefficient[i]))
        out.append(tuple(draw))
    return tuple(out)


def channels(
    response: Mapping[str, Any], names: Sequence[str]
) -> tuple[tuple[Callable[[ArrayLike], Array], ...], ...]:
    """Each draw's channels, in the order of ``names``: a :class:`chc.response.Channel` where the
    coefficient is a number, a :class:`Walking` one where it is a path.

    Raises:
        ValueError: as :func:`drawn` does.
    """
    _float64()
    parsed = drawn(response, names)
    return _built(parsed, names, int(response["length"]), bool(response["normalized"]))


def _gap(ours: Series, theirs: Series) -> float:
    """``max |ours - theirs|`` over the larger of their largest magnitudes, 0 where both are 0."""
    scale = max(float(np.max(np.abs(ours))), float(np.max(np.abs(theirs))))
    return 0.0 if scale == 0.0 else float(np.max(np.abs(ours - theirs))) / scale


def residual(response: Mapping[str, Any], observation: Observation) -> float:
    """The largest relative gap between the mapped channels' returns on the history's spend and the
    tool's decomposition: each draw's total over the history, and each held draw's every week.

    Raises:
        ValueError: the response cannot be built (:func:`drawn`), a coefficient's path is not over
            the history's weeks, or its decomposition does not cover every channel, every draw and
            every week.
    """
    if observation.population is not None:
        raise ValueError("a geo observation's response is not mapped: national families alone")
    _float64()
    names = observation.channels
    parsed = drawn(response, names)
    weeks = observation.sales.shape[0]
    for name, d in parsed.items():
        if d.coefficient.ndim == 2 and d.coefficient.shape[1] != weeks:
            raise ValueError(
                f"{name}'s coefficient paths run {d.coefficient.shape[1]} weeks, not the "
                f"history's {weeks}"
            )
    decomposition = response["decomposition"]
    missing = [key for key in ("total", "held", "weekly") if key not in decomposition]
    if missing:
        raise ValueError(f"the decomposition has no {missing}")
    for key in ("total", "weekly"):
        if sorted(decomposition[key]) != sorted(names):
            raise ValueError(
                f"the decomposition's {key} maps {sorted(decomposition[key])}, not {sorted(names)}"
            )
    built = _built(parsed, names, int(response["length"]), bool(response["normalized"]))
    held = [int(i) for i in decomposition["held"]]
    worst = 0.0
    for c, name in enumerate(names):
        totals = np.asarray(decomposition["total"][name], dtype=float)
        weekly = np.asarray(decomposition["weekly"][name], dtype=float)
        if totals.shape != (len(built),) or weekly.shape != (len(held), weeks):
            raise ValueError(
                f"{name}'s decomposition is {totals.shape} totals and {weekly.shape} weeks, not "
                f"{(len(built),)} and {(len(held), weeks)}"
            )
        spend = observation.spend[:, c]
        returns = np.stack([np.asarray(draw[c](spend)) for draw in built])
        worst = max(
            worst,
            *(_gap(r.sum(keepdims=True), t[None]) for r, t in zip(returns, totals, strict=True)),
        )
        worst = max(worst, *(_gap(returns[i], w) for i, w in zip(held, weekly, strict=True)))
    return worst
