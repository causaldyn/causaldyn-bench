"""Family 13: PyMC-Marketing's home ground, worlds whose media are PyMC-Marketing's default class at
its default priors.

PyMC-Marketing 1.2.0's ``MMM`` (``pymc_marketing/mmm/mmm.py``) scales each channel's spend by its
largest week over the history and the target by its largest week (``DataDerivedScaling(method=
"max")``, lines 551-554, the reduction signed, ``scaling.py`` lines 138-140), puts the scaled spend
through its adstock and then its saturation (``adstock_first=True``, line 474), and multiplies the
contribution back by the target's scale. With its default components each channel's media in week
``t`` are

    m_{c,t} = beta_c T logistic(lam_c x_{c,t}),  logistic(z) = (1 - e^{-z}) / (1 + e^{-z}),
    x_{c,t} = sum_{l < L} alpha_c^l s_{c,t-l} / (S_c sum_{l < L} alpha_c^l),

``GeometricAdstock`` normalised (``components/adstock.py`` lines 215-216: ``normalize=True`` by
default, though ``transformers.geometric_adstock`` itself defaults to False, line 241), nothing
spent before the history, and ``LogisticSaturation`` (``components/saturation.py`` line 259), at
``S_c`` the channel's largest spend and ``T`` the target's scale. The media parameters are drawn
from the components' default priors:

* ``alpha_c ~ Beta(1, 3)`` (``components/adstock.py`` line 448);
* ``lam_c ~ Gamma(3, 1)``, shape 3 and rate 1 (``components/saturation.py`` line 262);
* ``beta_c ~ HalfNormal(2)`` (``components/saturation.py`` line 263).

Family 0, Track M v2, has this class too, at its own parameters, drawn on the range Heusch's three
channels span; here they are the tool's own priors. ``L`` is Track M v2's kernel length, 6, which
every arm is told.

* ``T`` (CHOICE) is the largest sales of the Track M v2 world beneath. The tool reads the largest
  sales of the history it is fitted to, which here include these media; a world at that scale
  solves ``T = max_t (b_t + T g_t)``, ``g_t`` the media per unit of ``T``, and exists only where
  ``g_t < 1`` in every week, in 3 of the first 30 pilot worlds. So a world's media are on a scale
  of its own, and in the arm's scaling its true ``beta_c`` is the drawn one times ``T`` over the
  world's own largest sales.

The world is then Track M v2's class at ``retention`` ``alpha_c``, ``saturation`` ``lam_c / S_c``
and ``effect`` ``beta_c T``: Track M v2's ``tanh(lam a / 2)`` is the logistic of ``lam a``, and the
adstock of the scaled spend is the adstock over ``S_c``. Every curve is concave, so the oracle is
exact. Beneath the media each world is Track M v2's drawn world for its seed: its base, its spend,
whose budget and bidding rules read the sales before media and so do not move, its measurement
layer and the quarter's processes.

On the first 30 pilot worlds media's share of the history's sales is 0.21 to 0.92, median 0.78,
above 0.6 in 24 of them, against 0.11 to 0.22, median 0.17, in the Track M v2 worlds beneath; each
channel's return per euro over the history, median and range, is pla 52 (1.5 to 104), meta 51 (2.7
to 290) and tv 27 (1.1 to 408), against Track M v2's 2.9 (1.7 to 4.7), 2.8 (1.7 to 4.8) and 3.0
(2.1 to 4.7). In the tool's model an intercept, the controls and the noise take the rest of a
target scaled to its largest week; on these worlds the priors put most of the sales in the media.
Family
14's alternative (:mod:`~causaldyn_bench.scorecard.robyn`) keeps the tool's form and gives each
channel the media the Track M v2 world beneath it returns over the history, so that only the
media's form is the tool's.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np

from causaldyn_bench.endogenous_mmm import MediaMixWorld, Series
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.continuation import shadow
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, still, truth

FAMILY = 13
# PyMC-Marketing 1.2.0's default priors: GeometricAdstock's alpha, LogisticSaturation's lam (shape
# and rate) and beta (the half-normal's sigma)
ALPHA = (1.0, 3.0)
LAM = (3.0, 1.0)
BETA = 2.0


def parameters(channels: tuple[str, ...], seed: int) -> tuple[Series, Series, Series]:
    """Each channel's ``alpha``, ``lam`` and ``beta`` for world ``seed``, from the default
    priors."""
    draw = rng(FAMILY, Role.PARAMETERS, seed)
    count = len(channels)
    alpha = draw.beta(*ALPHA, count)
    lam = draw.gamma(LAM[0], 1.0 / LAM[1], count)
    beta = np.abs(draw.normal(0.0, BETA, count))
    return alpha, lam, beta


def pymc_marketing(
    history: MediaMixWorld, alpha: Series, lam: Series, beta: Series
) -> MediaMixWorld:
    """``history`` with each channel's media PyMC-Marketing's at ``alpha``, ``lam`` and ``beta``,
    on the channel's largest spend and the history's largest sales, and the sales with them."""
    world = dataclasses.replace(
        history,
        retention=tuple(alpha.tolist()),
        saturation=tuple((lam / history.spend.max(axis=0)).tolist()),
        effect=tuple((beta * np.max(history.sales)).tolist()),
    )
    media = np.column_stack(
        [world._effect_of(c, history.spend[:, c]) for c in range(len(history.channels))]
    )
    return dataclasses.replace(world, media=media, sales=history.premedia + media.sum(axis=1))


class PyMCMarketing:
    """Family 13."""

    name = "PyMC-Marketing's home ground"
    stream = stream(FAMILY)
    labels = (
        "The world's media are PyMC-Marketing's default class, by the bench's generator: each "
        "channel's spend over its largest week through PyMC-Marketing 1.2.0's normalised geometric "
        "adstock and its logistic saturation, times the largest sales of the Track M v2 world "
        "beneath.",
        "Alpha, lam and beta are drawn from PyMC-Marketing's default priors, Beta(1, 3), Gamma(3, "
        "1) and HalfNormal(2), not from Track M v2's ranges.",
        "Every arm is told the kernel's length, Track M v2's.",
        "Beneath the media each world is a Track M v2 drawn world: its base, its spend, its "
        "measurement layer and its quarter.",
        TRACK_M2.labels[2],
    )
    pilots = MappingProxyType({"priors": pilot(FAMILY, 0)})
    scored = MappingProxyType({"priors": scored(FAMILY, 0)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in self.scored:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        base = generator.simulate(seed)
        history = pymc_marketing(base, *parameters(base.channels, seed))
        path = still(history)
        return World(
            environment,
            seed,
            generator,
            history,
            shadow(generator, history, seed, path),
            path,
            path,
        )

    def observe(self, world: World, k: int) -> Observation:
        return observation(FAMILY, world, k)

    def truth(self, world: World) -> Truth:
        return truth(world)


PYMC_MARKETING = PyMCMarketing()
