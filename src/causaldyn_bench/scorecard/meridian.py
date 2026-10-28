"""Family 12: Meridian's home ground, worlds whose media are in Meridian's own class, drawn from its
default priors.

Each channel's media in week ``t`` are Meridian 2.1.0's default media transform, a geometric
adstock and then a Hill:

    m_{c,t} = beta_c x_{c,t} / (x_{c,t} + K_c),
    x_{c,t} = sum_{i=0}^{8} w_{c,i} s_{c,t-i},  w_{c,i} = alpha_c^i / sum_{j=0}^{8} alpha_c^j,

with nothing spent before the history. In Meridian 2.1.0's source (``meridian/``):

* the adstock is ``model/adstock_hill.py``'s ``AdstockTransformer`` (``_adstock``, line 237): its
  window is ``max_lag + 1`` weeks (line 254), the weeks before the data padded with zeros (lines
  273-281), and its geometric weights ``alpha^l`` normalised to sum to 1 over the window (lines 195
  and 202-204), whatever the padding; ``max_lag`` is 8 by default (``model/spec.py`` line 556,
  ``constants.py`` line 236). The adstock comes before the Hill (``model/spec.py`` line 555,
  ``model/equations.py`` lines 118-125);
* the Hill is ``HillTransformer`` (``_hill``, lines 304-328), ``x^slope / (x^slope + ec^slope)``,
  read on each channel's media over its media scale (``model/transformers.py``'s
  ``MediaTransformer``, lines 66-88): the median of its positive weeks, over its population, which
  a national model's one geo cancels. As the Meridian arm reads it, each channel's media units are
  its spend, so the inflexion ``K_c = ec_c q_c``, ``q_c`` the median of the channel's positive
  weekly spend over the history: once drawn, it is the world's, read the same by a geo test's
  treated universe and the quarter;
* the default priors (``model/prior_distribution.py``): the decay ``alpha_m ~ Uniform(0, 1)``
  (line 355), ``ec_m ~ TruncatedNormal(0.8, 0.8)`` on ``[0.1, 10]`` (line 383), the slope fixed at
  1 (``Deterministic(1.0)``, line 419) and the ROI ``roi_m ~ LogNormal(0.2, 0.9)`` (line 448), the
  prior Meridian puts on a paid channel by default (``model/spec.py`` lines 989-1002);
* the ROI identity: under that prior the coefficient is set so the channel's incremental outcome
  is its ROI times its spend (``model/prior_sampler.py`` lines 182-184,
  ``model/equations.py``'s ``calculate_beta_x``, lines 392-476), the spend summed over every week
  of the data (``model/media.py`` line 148, ``data/input_data.py`` line 96), the outcome the
  transformed media summed over the model's weeks (``model/equations.py`` line 452), with no
  counterfactual but none (``model/media.py`` line 151) where no calibration period is set. So
  ``beta_c = roi_c sum_t s_{c,t} / sum_t x_{c,t} / (x_{c,t} + K_c)`` over the history's 156
  weeks: Meridian's ROI counts what the window's weeks return, the carryover of the history's
  last eight weeks past its end left out.

Every channel's ``alpha_c``, ``ec_c`` and ``roi_c`` are drawn from those priors, independently, from
the family's own stream. CHOICE: the drawn ROI sets the coefficient through that identity, so the
world is a draw from Meridian's own prior. The alternative is family 14's
(:mod:`~causaldyn_bench.scorecard.robyn`): each channel returning over the history what the Track M
v2 world beneath it returns, which family 14 takes since Robyn has no prior; here it would keep
Meridian's form and drop its ROI prior. On the first 30 pilot worlds media are a median 0.089 of
the history's sales, 0.038 at the tenth percentile and 0.170 at the ninetieth, 0.017 to 0.566,
where Track M v2's beneath them are a median 0.168; one world puts media above half its sales and
none above 0.6, nor on the first 100. Each channel's return per euro over the history
(``MediaMixWorld.roas``) is its drawn ROI, by the identity: a median 0.95 for pla, 1.02 for meta
and 1.09 for tv, from 0.10 to 38.1 over the 90 channels, the log-normal's tail. Its return on last
year's spend as the oracle reads it (:mod:`~causaldyn_bench.scorecard.returns`), which counts the
carryover past the history and leaves out last year's own carried in, is 0.96 to 1.05 times that:
a median 0.95, 1.01 and 1.12.

Beneath the media each world is Track M v2's drawn world for its seed: its base, its spend, whose
budget and bidding rules read the sales before media and so do not move, its measurement layer and
the quarter's processes. Every arm is told the kernel spans Meridian's window, ``L = max_lag + 1 =
9`` weeks, which is the world's whole carryover: the oracle, the quarter and the returns count all
of it. A slope-1 Hill of an adstock affine in the weekly spend, at a coefficient above 0, is
concave in it, so every cell is concave and the oracle is exact.

``scripts/meridian_form.py`` runs Meridian's own ``AdstockTransformer`` and ``HillTransformer`` in
the Meridian arm's pinned environment on worlds of this family and records what they return in
``results/meridian_form.json``; a test holds the world's transforms to that record within 1e-9.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from causaldyn_bench.endogenous_mmm import Curve, MediaMixWorld, Series
from causaldyn_bench.mmm_decision import drawn
from causaldyn_bench.scorecard.continuation import shadow
from causaldyn_bench.scorecard.family import Truth
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.seeds import Role, pilot, rng, scored, stream
from causaldyn_bench.scorecard.track_m2 import TRACK_M2, World, observation, still, truth

FAMILY = 12
MAX_LAG = 8  # Meridian's default max_lag: its adstock's window is MAX_LAG + 1 weeks
# Meridian 2.1.0's default priors on a paid media channel: the decay alpha_m's uniform bounds,
# ec_m's truncated normal (location, scale, lowest, highest), and roi_m's log-normal (mu, sigma)
DECAY = (0.0, 1.0)
EC = (0.8, 0.8, 0.1, 10.0)
ROI = (0.2, 0.9)


def adstocked(spend: ArrayLike, decay: float) -> Series:
    """Meridian's geometric adstock: each week's spend over the window of ``MAX_LAG + 1`` weeks,
    weighted ``decay^lag`` and normalised to sum to 1, nothing spent before the history."""
    spend = np.asarray(spend, dtype=float)
    weights = decay ** np.arange(MAX_LAG + 1, dtype=float)
    return np.convolve(spend, weights / weights.sum())[: spend.size]


def saturated(adstock: ArrayLike, inflexion: float) -> Series:
    """Meridian's Hill at its default slope of 1, at ``inflexion``: ``x / (x + inflexion)``."""
    a = np.asarray(adstock, dtype=float)
    return a / (a + inflexion)


def hill() -> Curve:
    """Meridian's Hill of slope 1 as the scorecard reads a curve, at its inflexion: concave from
    zero adstock."""
    return Curve(
        saturated,
        lambda adstock, inflexion: inflexion / (np.asarray(adstock, dtype=float) + inflexion) ** 2,
        concave=True,
    )


def media_scale(spend: ArrayLike) -> float:
    """Meridian's media scale of a channel on a national model: the median of its positive
    weeks."""
    spend = np.asarray(spend, dtype=float)
    return float(np.median(spend[spend > 0.0]))


@dataclass(frozen=True, kw_only=True)
class MeridianWorld(MediaMixWorld):
    """A history whose media are in Meridian's form: :attr:`retention` holds each channel's decay,
    :attr:`saturation` its inflexion, in thousands of euros of adstock, and :attr:`effect` its
    coefficient; the kernel is the normalised geometric one over Meridian's window, the
    :class:`~causaldyn_bench.endogenous_mmm.MediaMixWorld`'s at its length."""

    ec: tuple[float, ...]  # Meridian's ec_m: each inflexion over its channel's media scale
    scale: tuple[float, ...]  # each channel's media scale, the median of its positive weeks
    roi: tuple[float, ...]  # Meridian's roi_m, drawn: what the coefficient was set by

    def curves(self) -> tuple[Curve, ...]:
        return (hill(),) * len(self.channels)

    def _effect_of(self, column: int, spend: Series) -> Series:
        adstock = adstocked(spend, self.retention[column])
        return self.effect[column] * saturated(adstock, self.saturation[column])


def parameters(channels: tuple[str, ...], seed: int) -> tuple[Series, Series, Series]:
    """Each channel's decay, ec and ROI for world ``seed``, from Meridian's default priors: ec by
    the inverse of its truncated normal's distribution function, one uniform a channel."""
    draw = rng(FAMILY, Role.PARAMETERS, seed)
    n = len(channels)
    decay = draw.uniform(*DECAY, n)
    location, spread, lowest, highest = EC
    low, high = ndtr((lowest - location) / spread), ndtr((highest - location) / spread)
    ec = location + spread * ndtri(low + draw.uniform(0.0, 1.0, n) * (high - low))
    roi = draw.lognormal(*ROI, n)
    return decay, ec, roi


def meridian(history: MediaMixWorld, decay: Series, ec: Series, roi: Series) -> MeridianWorld:
    """``history`` with each channel's media in Meridian's form at ``decay`` and ``ec``, its
    coefficient set by Meridian's ROI identity at ``roi``, and the sales with them."""
    channels = range(len(decay))
    scale = np.array([media_scale(history.spend[:, c]) for c in channels])
    inflexion = ec * scale
    curve = np.column_stack(
        [
            saturated(adstocked(history.spend[:, c], float(decay[c])), float(inflexion[c]))
            for c in channels
        ]
    )
    effect = roi * history.spend.sum(axis=0) / curve.sum(axis=0)
    media = effect * curve
    kept = {f.name: getattr(history, f.name) for f in dataclasses.fields(MediaMixWorld)}
    kept |= {
        "media": media,
        "sales": history.premedia + media.sum(axis=1),
        "retention": tuple(decay.tolist()),
        "saturation": tuple(inflexion.tolist()),
        "effect": tuple(effect.tolist()),
        "kernel_length": MAX_LAG + 1,
        "curve": "hill",  # Meridian's of slope 1 (curves), no key of CURVES
    }
    return MeridianWorld(
        **kept, ec=tuple(ec.tolist()), scale=tuple(scale.tolist()), roi=tuple(roi.tolist())
    )


class Meridian:
    """Family 12."""

    name = "Meridian's home ground"
    stream = stream(FAMILY)
    labels = (
        "The world's media are Meridian's class, by the bench's generator: each channel's spend "
        "through Meridian 2.1.0's geometric adstock, normalised over its window of max_lag + 1 = 9 "
        "weeks, and its Hill of slope 1, at ec times the median of the channel's positive weeks; "
        "decay, ec and ROI drawn from Meridian's default priors, Uniform(0, 1), "
        "TruncatedNormal(0.8, 0.8) on [0.1, 10] and LogNormal(0.2, 0.9), and each channel's "
        "coefficient set by its ROI as Meridian's ROI prior sets it.",
        "Every arm is told the kernel spans Meridian's window, 9 weeks.",
        "Beneath the media each world is a Track M v2 drawn world.",
        TRACK_M2.labels[2],
    )
    pilots = MappingProxyType({"priors": pilot(FAMILY, 0)})
    scored = MappingProxyType({"priors": scored(FAMILY, 0)})

    def world(self, environment: str, seed: int) -> World:
        if environment not in self.scored:
            raise ValueError(f"{self.name} has no environment {environment!r}")
        generator = drawn(seed)
        base = generator.simulate(seed)
        history = meridian(base, *parameters(base.channels, seed))
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


MERIDIAN = Meridian()
