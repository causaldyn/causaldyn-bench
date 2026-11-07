"""The variance reductions against their closed forms on synthetic worlds of known covariance
(``scripts/control_variates.mac``), and the section each committed Track M v2 pilot report ends
with.

The synthetic worlds: four normal covariates of covariance ``S`` and mean ``MU``, two arms' regrets
``a = 1/5 + A' x + u`` and ``b = 1/10 + B' x + v``, the noise ``(u, v)`` apart from ``x`` of
covariance ``E``. Every Monte Carlo test draws from a fixed seed and reads its sample against a
band of four standard errors or more of the statistic it reads, so its own false-red rate is below
1e-4."""

import math

import numpy as np
import pytest
from scipy.stats import t as student

from causaldyn_bench.scorecard.scoring import COVARIATES, ArmScore, WorldRecord, score_world
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.variance import (
    HEADING,
    PILOTS,
    ROOT,
    Centre,
    Slope,
    _left_out,
    adjustment,
    controlled,
    fit,
    oracle_covariates,
    pairing,
    pilot_section,
    samples,
    written,
)

S = np.array(
    [[1.0, 0.5, 0.0, 0.0], [0.5, 2.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.25], [0.0, 0.0, 0.25, 1.0]]
)
MU = np.array([1.0, 2.0, 0.5, 1.0])
A = np.array([0.5, 0.0, 0.25, 0.0])
B = np.array([0.25, 0.2, 0.0, 0.0])
E = np.array([[0.1, 0.05], [0.05, 0.1]])
C = A - B
# the closed forms, exact in Maxima
PAIRING = 360 / 539  # 1 - var(d) / (var(a) + var(b))
R2 = 99 / 179  # of d on x
VZ = 1 / 10  # var(d - C' x)
SPREAD = 99 / 800  # C' S C


def _worlds(rng, n):
    """``n`` synthetic worlds: the two arms' regrets and the covariates."""
    x = MU + rng.standard_normal((n, 4)) @ np.linalg.cholesky(S).T
    noise = rng.standard_normal((n, 2)) @ np.linalg.cholesky(E).T
    return 0.2 + x @ A + noise[:, 0], 0.1 + x @ B + noise[:, 1], x


def _truth():
    """d's mean."""
    return 0.1 + MU @ C


def test_pairing_removes_the_share_of_an_unpaired_comparisons_variance_the_closed_form_gives():
    """50 000 worlds read the share to a standard error near 0.003; the band is 0.02."""
    a, b, _ = _worlds(np.random.default_rng(11), 50_000)
    paired = pairing(a, b, draws=20)
    assert paired.share == pytest.approx(PAIRING, abs=0.02)
    assert paired.worlds == 50_000
    against = pairing(a, a[::-1], draws=20)  # two arms on unrelated worlds: nothing to remove
    assert abs(against.share) < 0.03


def test_a_frozen_slope_leaves_the_mean_unbiased_and_its_variance_the_closed_form():
    """4 000 estimates on 40 worlds each: at the slope A - B their variance is 1/400, and at a
    slope frozen on a pilot of its own the general form ``var(d) - 2 beta' S C + beta' S beta``;
    the mean is d's within four standard errors, the variance within 10 % (4.5 of its standard
    errors) and the 95 % interval's coverage within 0.015 of 0.95 (4.4 of its)."""
    rng = np.random.default_rng(12)
    a, b, x = _worlds(rng, 40)
    pilot = fit(a - b, x)
    vd = A @ S @ A + B @ S @ B - 2 * A @ S @ B + VZ
    for slope, variance in (
        (Slope(COVARIATES, C, 0), VZ / 40),
        (pilot, (vd - 2 * pilot.beta @ S @ C + pilot.beta @ S @ pilot.beta) / 40),
    ):
        estimates, covered = [], 0
        for _ in range(4_000):
            a, b, x = _worlds(rng, 40)
            estimate = controlled(a - b, x, slope, Centre.known(MU))
            estimates.append(estimate.mean)
            covered += estimate.interval[0] <= _truth() <= estimate.interval[1]
            assert estimate.df == 39
        assert np.mean(estimates) == pytest.approx(_truth(), abs=4 * math.sqrt(variance / 4_000))
        assert np.var(estimates, ddof=1) == pytest.approx(variance, rel=0.1, abs=0.0)
        assert covered / 4_000 == pytest.approx(0.95, abs=0.015)
    assert math.isclose(VZ / 40, 1 / 400, rel_tol=1e-15)


def test_a_centre_read_on_other_worlds_adds_its_own_variance():
    """The covariates' mean read on 100 other worlds: the estimate's variance is
    1/400 + (A - B)' S (A - B) / 100 = 299/80000, half as much again as with the mean known, and
    the Welch-Satterthwaite interval holds d's mean 0.95 of the time (bands as above)."""
    rng = np.random.default_rng(13)
    slope = Slope(COVARIATES, C, 0)
    estimates, covered = [], 0
    for _ in range(4_000):
        a, b, x = _worlds(rng, 40)
        _, _, other = _worlds(rng, 100)
        estimate = controlled(a - b, x, slope, Centre.of(other))
        estimates.append(estimate.mean)
        covered += estimate.interval[0] <= _truth() <= estimate.interval[1]
        assert 39 < estimate.df < 139
    assert math.isclose(VZ / 40 + SPREAD / 100, 299 / 80_000, rel_tol=1e-15)
    assert np.var(estimates, ddof=1) == pytest.approx(299 / 80_000, rel=0.1, abs=0.0)
    assert np.mean(estimates) == pytest.approx(_truth(), abs=4 * math.sqrt(299 / 80_000 / 4_000))
    assert covered / 4_000 == pytest.approx(0.95, abs=0.015)


def test_a_slope_fitted_on_the_scored_worlds_biases_the_interval():
    """12 worlds, four covariates, 10 000 times: fitted on the scored worlds, the estimate's
    variance is Lavenberg and Welch's 5/3 times sigma^2 (1 - R^2) / n, 1/72; the textbook interval
    off its residuals holds d's mean about 0.89 of the time, and read as a frozen slope's is, 0.79;
    a slope frozen on a pilot of 40 other worlds holds it 0.95 of the time. Bands: the variance
    within 10 % (7 standard errors), each coverage within 0.02 (6 or more)."""
    rng = np.random.default_rng(14)
    a, b, x = _worlds(rng, 40)
    frozen = fit(a - b, x)
    n, p, draws = 12, 4, 10_000
    estimates, textbook, naive, held = [], 0, 0, 0
    truth = _truth()
    for _ in range(draws):
        a, b, x = _worlds(rng, n)
        d = a - b
        inside = fit(d, x)
        own = controlled(d, x, inside, Centre.known(MU))
        estimates.append(own.mean)
        naive += own.interval[0] <= truth <= own.interval[1]
        residual = d - d.mean() - (x - x.mean(axis=0)) @ inside.beta
        half = student.ppf(0.975, n - p - 1) * math.sqrt(residual @ residual / (n - p - 1) / n)
        textbook += abs(own.mean - truth) <= half
        other = controlled(d, x, frozen, Centre.known(MU))
        held += other.interval[0] <= truth <= other.interval[1]
    assert math.isclose((n - 2) / (n - p - 2) * VZ / n, 1 / 72, rel_tol=1e-15)
    assert np.var(estimates, ddof=1) == pytest.approx(1 / 72, rel=0.1, abs=0.0)
    assert textbook / draws == pytest.approx(0.89, abs=0.02)
    assert naive / draws == pytest.approx(0.79, abs=0.02)
    assert held / draws == pytest.approx(0.95, abs=0.02)


def test_the_left_out_share_is_breiman_and_freedmans_not_r_squared():
    """2 000 pilots of 40 worlds: the left-out errors' mean squares give the share
    1 - (1 - R^2) 37/33 = 2947/5907 = 0.499 (to 0.02, four standard errors), where R^2 is 0.553:
    the slope's own error is paid for. The adjusted value is d itself, so a frozen slope at A - B
    on a large sample removes R^2."""
    rng = np.random.default_rng(15)
    raw = adjusted = 0.0
    for _ in range(2_000):
        a, b, x = _worlds(rng, 40)
        r, e = _left_out(a - b, x)
        raw += float(r @ r)
        adjusted += float(e @ e)
    assert 1.0 - adjusted / raw == pytest.approx(2947 / 5907, abs=0.02)
    assert abs(1.0 - adjusted / raw - R2) > 0.03
    a, b, x = _worlds(rng, 50_000)
    frozen = adjustment(a - b, x, Slope(COVARIATES, C, 0), draws=20)
    assert frozen.share == pytest.approx(R2, abs=0.02)


def test_a_left_out_error_is_the_fit_to_the_other_worlds():
    rng = np.random.default_rng(16)
    a, b, x = _worlds(rng, 15)
    d = a - b
    raw, adjusted = _left_out(d, x)
    for i in range(d.size):
        keep = np.arange(d.size) != i
        slope = fit(d[keep], x[keep]).beta
        intercept = d[keep].mean() - x[keep].mean(axis=0) @ slope
        assert adjusted[i] == pytest.approx(d[i] - intercept - x[i] @ slope, rel=1e-9, abs=1e-12)
        assert raw[i] == pytest.approx(d[i] - d[keep].mean(), rel=1e-12, abs=1e-15)


def test_a_reduction_carries_its_bootstrap_interval_and_refuses_what_it_cannot_read():
    rng = np.random.default_rng(17)
    a, b, x = _worlds(rng, 40)
    for reduction in (pairing(a, b), adjustment(a - b, x), adjustment(a - b, x, fit(a - b, x))):
        low, high = reduction.interval
        assert low < reduction.share < high and reduction.worlds == 40
    assert pairing(a, b) == pairing(a, b)  # the resample is seeded
    with pytest.raises(ValueError, match="neither arm's regret moves"):
        pairing(np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="collinear"):
        adjustment(a - b, np.column_stack([x[:, :3], x[:, 0]]))
    with pytest.raises(ValueError, match="collinear"):
        fit(a - b, np.column_stack([x[:, :3], 2.0 * x[:, 1]]))
    with pytest.raises(ValueError, match="is a number"):
        adjustment(a - b, np.where(np.arange(40)[:, None] == 3, np.nan, x))
    with pytest.raises(ValueError, match="8 worlds or more"):
        adjustment((a - b)[:7], x[:7])
    with pytest.raises(ValueError, match="7 worlds or more"):
        fit((a - b)[:6], x[:6])


def _record(environment, i, regrets, covariates):
    arms = {
        arm: ArmScore(regret, None, None, 0, None, None, None, None, {})
        for arm, regret in regrets.items()
    }
    return WorldRecord(1, environment, 10 * i, i, True, 0, 1.0, 1.0, 3, 0, covariates, arms)


def test_samples_read_a_look_of_world_records_in_their_seeded_order():
    rng = np.random.default_rng(18)
    records = []
    for environment in ("walk", "trend"):
        a, b, x = _worlds(rng, 12)
        for i in reversed(range(12)):
            covariates = dict(zip(COVARIATES, x[i].tolist(), strict=True))
            records.append(_record(environment, i, {"first": a[i], "second": b[i]}, covariates))
    held = samples(records, 10)
    assert [s.environment for s in held] == ["walk", "trend"]
    for sample in held:
        ours = [r for r in records if r.environment == sample.environment]
        ours.sort(key=lambda r: r.index)
        assert np.array_equal(sample.regrets["first"], [r.arms["first"].regret for r in ours[:10]])
        assert np.array_equal(sample.covariates[:, 3], [r.covariates["price"] for r in ours[:10]])
    with pytest.raises(ValueError, match="misses"):
        samples(records, 13)


def test_the_oracle_covariates_are_the_world_records():
    seeds = TRACK_M2.pilots["drawn"][:3]
    held = oracle_covariates(TRACK_M2, "drawn", seeds)
    for row, seed in zip(held, seeds, strict=True):
        record = score_world(TRACK_M2, "drawn", seed, 0, {}, pilot=True)
        assert row.tolist() == [record.covariates[name] for name in COVARIATES]


def test_a_section_is_written_in_place_of_the_one_a_report_held():
    report = "# A pilot\n\n| a | b |\n"
    once = written(report, f"{HEADING}\n\nfirst\n")
    assert once == f"# A pilot\n\n| a | b |\n\n{HEADING}\n\nfirst\n"
    assert (
        written(once, f"{HEADING}\n\nsecond\n")
        == f"# A pilot\n\n| a | b |\n\n{HEADING}\n\nsecond\n"
    )


@pytest.mark.parametrize("name", PILOTS)
def test_each_committed_pilot_report_ends_with_its_variance_section(name):
    """The numbers in the report are the code's on the committed records:
    ``python -m causaldyn_bench.scorecard.variance`` rewrites them."""
    report = (ROOT / "results" / f"{name}.md").read_text()
    assert report.endswith("\n\n" + pilot_section(name))
