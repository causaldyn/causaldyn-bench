"""A family's report, read off synthetic records: every family's cells carry the same columns, each
reads the records' own numbers, the compute table reads only what the records hold, an arm's effect
shares are read against the spend shares, and a scored report prints the pre-registration it ran
under, whose hash a reader holds to the file."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from causaldyn_bench.scorecard.collinear import COLLINEAR
from causaldyn_bench.scorecard.confounded import CONFOUNDED
from causaldyn_bench.scorecard.drift import DRIFT
from causaldyn_bench.scorecard.family import number
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.meridian import MERIDIAN
from causaldyn_bench.scorecard.outside import OUTSIDE
from causaldyn_bench.scorecard.performance import PERFORMANCE
from causaldyn_bench.scorecard.preregistration import Stamp, verify
from causaldyn_bench.scorecard.pymc_marketing import PYMC_MARKETING
from causaldyn_bench.scorecard.report import markdown, report, shares
from causaldyn_bench.scorecard.robyn import ROBYN
from causaldyn_bench.scorecard.s_curves import S_CURVES
from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord
from causaldyn_bench.scorecard.short import SHORT
from causaldyn_bench.scorecard.simmmulator import Simmmulator
from causaldyn_bench.scorecard.spillover import SPILLOVER
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    0: TRACK_M2,
    1: DRIFT,
    2: CONFOUNDED,
    3: COLLINEAR,
    4: S_CURVES,
    5: PERFORMANCE,
    6: OUTSIDE,
    7: SHORT,
    8: GEO,
    9: SPILLOVER,
    10: FEW_GEOS,
    12: MERIDIAN,
    13: PYMC_MARKETING,
    14: ROBYN,
}
CHANNELS = ("pla", "meta", "tv")
SPEND = {"pla": 100.0, "meta": 300.0, "tv": 600.0}
ROI = {"pla": 4.0, "meta": 1.0, "tv": 0.5}  # effects 400, 300 and 300 against spends 1, 3 and 6
COLUMNS = {"regret", "failed", "moved", "crps", "mape", "coverage", "shares"}


def _family(f, tmp_path):
    return Simmmulator(tmp_path) if f == 15 else FAMILIES[f]


def _score(regret, *, roi=None, crps=None, mape=None, coverage=None, cost=None, failure=None):
    """An arm's score whose response, where ``roi`` gives each channel's draws' mean ROI, was read
    at ``coverage``, the share of the channels each central share held."""
    record = {"error": failure} | ({} if cost is None else {"cost": cost})
    read: dict[str, Any] = {}
    if roi is not None:
        held = coverage or {"80": 1.0, "90": 1.0, "95": 1.0}
        read = {
            "returns": {name: {"roi": [roi[name], 0.0, 0.0]} for name in CHANNELS},
            "roi_coverage": held,
            "mroi_coverage": held,
        }
    return ArmScore(regret, failure, None, 0, None, None, None, crps, record, mape=mape, **read)


def _record(family, environment, index, arms, *, pilot=True, k=0):
    return WorldRecord(
        family=family,
        environment=environment,
        seed=1000 + index,
        index=index,
        pilot=pilot,
        k=k,
        budget=1.0,
        best=1.0,
        channels=len(CHANNELS),
        oracle_bounds=0,
        covariates={},
        arms=arms,
        returns={name: {"roi": ROI[name], "mroi": ROI[name] / 2} for name in CHANNELS},
        spend=dict(SPEND),
    )


def _run(family, *, worlds=3, pilot=True):
    """Each of the family's first two environments' first ``worlds`` worlds at rungs 0 and 1."""
    records = []
    for environment in list(family.scored)[:2]:
        for i in range(worlds):
            arms = {
                "status quo": _score(0.2 + 0.01 * i),
                "rival": _score(
                    0.1 * i,
                    roi=ROI,
                    crps=0.05 * (i + 1),
                    mape=0.02 * (i + 1),
                    coverage={"80": 0.0, "90": 1 / 3, "95": 2 / 3},
                    cost={"cpu_seconds": 1800.0, "wall_seconds": 600.0, "peak_rss_bytes": 2**30},
                ),
            }
            for k in (0, 1):
                records.append(_record(number(family), environment, i, arms, pilot=pilot, k=k))
    return records


@pytest.mark.parametrize("f", [*FAMILIES, 15])
def test_every_familys_report_carries_every_column(f, tmp_path):
    family = _family(f, tmp_path)
    data = report(family, _run(family))
    environments = list(family.scored)[:2]
    assert [(c["k"], c["environment"]) for c in data["cells"]] == [
        (k, e) for k in (0, 1) for e in (*environments, "pooled")
    ]
    for cell in data["cells"]:
        pooled = cell["environment"] == "pooled"
        assert (cell["worlds"], cell["look"]) == (3, 3 * len(environments) if pooled else 3)
        assert set(cell["arms"]) == {"status quo", "rival"}
        for columns in cell["arms"].values():
            assert set(columns) == COLUMNS
    text = markdown(data)
    assert f"# Family {f}: {family.name}, pilot" in text
    for heading in ("CRPS", "MAPE, %", "coverage 80/90/95", "effect against spend share"):
        assert heading in text
    assert "## Compute, as recorded" in text
    assert json.loads(json.dumps(data, allow_nan=False)) == data


def test_a_cell_reads_the_records_own_numbers():
    rival = report(TRACK_M2, _run(TRACK_M2))["cells"][0]["arms"]["rival"]
    assert rival["regret"]["mean"] == pytest.approx(0.1, rel=1e-12, abs=0.0)
    assert rival["regret"]["low"] < 0.1 < rival["regret"]["high"]
    assert rival["crps"]["mean"] == pytest.approx(0.1, rel=1e-12, abs=0.0)
    assert rival["mape"]["mean"] == pytest.approx(0.04, rel=1e-12, abs=0.0)
    assert rival["coverage"] == {
        "roi_coverage": {"80": 0.0, "90": 1 / 3, "95": 2 / 3},
        "mroi_coverage": {"80": 0.0, "90": 1 / 3, "95": 2 / 3},
        "worlds": 3,
    }
    assert (rival["failed"], rival["moved"]) == (0, 0)
    quo = report(TRACK_M2, _run(TRACK_M2))["cells"][0]["arms"]["status quo"]
    assert (quo["crps"], quo["mape"], quo["coverage"], quo["shares"]) == (None,) * 4
    line = next(
        line
        for line in markdown(report(TRACK_M2, _run(TRACK_M2))).splitlines()
        if "| rival |" in line
    )
    assert "| +4.00 [" in line
    assert "| roi 0.000/0.333/0.667; mroi 0.000/0.333/0.667 | pull +0.000, miss 0.000 |" in line


def test_effect_shares_are_read_against_the_spend_shares():
    spend = np.array([SPEND[n] for n in CHANNELS])
    truth = np.array([ROI[n] for n in CHANNELS]) * spend
    # an arm whose effect shares are the spend shares: every channel the same ROI
    flat = dict.fromkeys(CHANNELS, 1.0)
    # one halfway: each channel's effect the mean of the truth's and the spend's shares, scaled
    halfway = {
        n: (truth[c] / truth.sum() + spend[c] / spend.sum()) / 2 / spend[c]
        for c, n in enumerate(CHANNELS)
    }
    records = [
        _record(
            3,
            "r-0.9",
            0,
            {
                "on spend": _score(0.0, roi=flat),
                "true": _score(0.0, roi=ROI),
                "halfway": _score(0.0, roi=halfway),
                "unread": _score(0.0),
            },
        )
    ]
    on_spend, half = shares(records, "on spend"), shares(records, "halfway")
    assert on_spend is not None and half is not None
    assert on_spend["pull"] == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert shares(records, "true") == {"pull": 0.0, "miss": 0.0, "channels": 3}
    assert half["pull"] == pytest.approx(0.5, rel=1e-12, abs=0.0)
    lean = spend / spend.sum() - truth / truth.sum()
    assert half["miss"] == pytest.approx(np.mean(np.abs(lean)) / 2, rel=1e-12, abs=0.0)
    assert shares(records, "unread") is None
    assert shares(records, "absent") is None


def test_the_compute_table_reads_only_what_the_records_hold():
    cost = {
        "cpu_model": "an x86-64 CPU",
        "threads": "1",
        "peak_rss_bytes": 3 * 2**30,
        "wall_seconds": 120.0,
        "cpu_seconds": 7200.0,
    }
    arms = [
        {"fitted": _score(0.0, cost=cost), "untimed": _score(0.0)},
        {
            "fitted": _score(0.0, cost=cost | {"peak_rss_bytes": 2**30, "wall_seconds": 60.0}),
            "untimed": _score(0.0),
        },
        # a rung whose fit is another rung's, as an arm that reads no test records it
        {"fitted": _score(0.0, cost={"fitted with k": 0}), "untimed": _score(0.0)},
    ]
    records = [_record(0, "drawn", i, a) for i, a in enumerate(arms)]
    table = report(TRACK_M2, records)["compute"]
    assert table["fitted"] == {
        "records": 3,
        "cpu_model": ["an x86-64 CPU"],
        "threads": ["1"],
        "peak_rss_gib": {"median": 2.0, "most": 3.0, "n": 2},
        "wall_seconds_a_fit": {"median": 90.0, "n": 2},
        "core_hours": {"total": 4.0, "n": 2},
    }
    assert table["untimed"] == {
        "records": 3,
        "cpu_model": None,
        "threads": None,
        "peak_rss_gib": None,
        "wall_seconds_a_fit": None,
        "core_hours": None,
    }
    text = markdown(report(TRACK_M2, records))
    assert "| fitted | 3 | an x86-64 CPU | 1 | 2.00 / 3.00 | 90.0 | 4.000 |" in text
    assert (
        "| untimed | 3 | not recorded | not recorded | not recorded | not recorded | not recorded |"
        in text
    )


def _preregistration(tmp_path):
    path = tmp_path / "design.md"
    path.write_text("# The design, fixed before the first scored world\n")
    return path


def test_a_scored_report_prints_its_preregistrations_hash_in_its_header_and_json(tmp_path):
    path = _preregistration(tmp_path)
    data = report(TRACK_M2, _run(TRACK_M2, pilot=False), preregistration=path)
    assert data["kind"] == "scored"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    printed = Stamp.from_line(markdown(data))
    assert printed == Stamp.from_json(data["preregistration"]) == Stamp(str(path), actual)
    assert verify(printed) == printed


def test_a_report_whose_preregistration_changed_or_is_gone_fails_the_check(tmp_path):
    path = _preregistration(tmp_path)
    printed = Stamp.from_line(
        markdown(report(TRACK_M2, _run(TRACK_M2, pilot=False), preregistration=path))
    )
    path.write_text(path.read_text() + "a line added after the run\n")
    with pytest.raises(ValueError, match="the file there is"):
        verify(printed)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        verify(printed)
    with pytest.raises(FileNotFoundError):
        report(TRACK_M2, _run(TRACK_M2, pilot=False), preregistration=path)


def test_a_scored_report_is_not_written_without_its_preregistration():
    with pytest.raises(ValueError, match="prints the pre-registration the run declares"):
        report(TRACK_M2, _run(TRACK_M2, pilot=False))
    pilot = report(TRACK_M2, _run(TRACK_M2))
    assert pilot["preregistration"] is None
    with pytest.raises(ValueError, match="one pre-registration line, not 0"):
        Stamp.from_line(markdown(pilot))
    with pytest.raises(ValueError, match="not both"):
        report(TRACK_M2, _run(TRACK_M2) + _run(TRACK_M2, pilot=False)[:1])


def _committed_reports():
    """Every committed report's JSON that names a kind, and its markdown beside it."""
    for path in sorted((ROOT / "results").rglob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("kind") in ("scored", "pilot") and "cells" in data:
            yield path, data


def test_every_committed_scored_report_holds_its_preregistrations_hash():
    """The run's report, once committed, is read here: a scored one prints its stamp in its JSON
    and its markdown, the two agree, and the file at the stamp's path hashes to it."""
    for path, data in _committed_reports():
        if data["kind"] != "scored":
            continue
        stamp = Stamp.from_json(data["preregistration"])
        assert Stamp.from_line(path.with_suffix(".md").read_text()) == stamp
        verify(stamp, ROOT)
