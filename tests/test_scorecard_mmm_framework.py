"""Family 21: mmm-framework's two data sets are read unchanged, by their digests, beside their
licence; the truth reader returns what each truth file states, and on the geo set the stated ROAS
is the stated contribution over the file's spend, as on the national set it is not; a swapped truth
file moves what a reading is scored against; and the bench's arm and the tool's record are read
into readings of the key's channels."""

import datetime
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from causaldyn_bench.observational_mmm import _adstock
from causaldyn_bench.scorecard.mmm_framework import (
    DIGESTS,
    ENVIRONMENTS,
    MMM_FRAMEWORK,
    NO_CELL,
    WHEEL_SHA256,
    AnswerKey,
    MffWorld,
    MmmFramework,
    Panel,
    Reading,
    Totalled,
    denominator,
    main,
    markdown,
    observational,
    read,
    read_mff,
    read_truth,
    score,
    tool_reading,
)

FAMILY = MmmFramework()
ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "mmm_framework_pilot"


@pytest.fixture(scope="module")
def worlds():
    return {e: FAMILY.world(e) for e in ENVIRONMENTS}


def _stated(environment):
    return json.loads((FAMILY.directory / f"{environment}_truth.json").read_text())


def test_the_files_are_the_wheels_and_carry_their_licence():
    shipped = sorted(p.name for p in FAMILY.directory.iterdir())
    assert shipped == sorted([*DIGESTS, "NOTICE"])
    for name, digest in DIGESTS.items():
        assert hashlib.sha256((FAMILY.directory / name).read_bytes()).hexdigest() == digest
    notice = (FAMILY.directory / "NOTICE").read_text()
    assert "A rival's worlds, labelled so" in notice
    assert f"mmm-framework {MMM_FRAMEWORK}" in notice and "Apache License, Version 2.0" in notice
    assert WHEEL_SHA256 in notice and all(f"{n}  " in notice for n in DIGESTS)
    assert all(digest in notice for digest in DIGESTS.values())
    assert "mmm_framework_worlds" in (ROOT / "NOTICE").read_text()


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_the_truth_reader_returns_what_the_file_states(worlds, environment):
    key, stated = FAMILY.truth(worlds[environment]), _stated(environment)
    assert key.channels == tuple(stated["channels"])
    assert key.contribution.tolist() == [stated["true_contribution"][c] for c in key.channels]
    assert key.roas.tolist() == [stated["true_roas"][c] for c in key.channels]
    assert (key.scenario, key.violates, key.representable) == (
        stated["scenario"],
        stated["violates"],
        stated["representable"],
    )
    if environment == "geo":
        assert key.geos == tuple(stated["geographies"])
        for name, table in (("contribution", key.contribution_by_geo), ("roas", key.roas_by_geo)):
            by_geo = stated[f"true_{name}_by_geo"]
            assert table is not None
            assert table.tolist() == [[by_geo[g][c] for c in key.channels] for g in key.geos]
    else:
        assert key.geos is key.contribution_by_geo is key.roas_by_geo is None
        assert dict(key.roles) == stated["notes"]["roles"]


def test_on_the_geo_set_the_stated_roas_is_the_contribution_over_the_files_spend(worlds):
    world = worlds["geo"]
    key = world.key
    assert key.contribution_by_geo is not None and key.roas_by_geo is not None
    spent = world.spend.sum(axis=0)  # (geos, channels)
    # every quotient below is within 4e-15 of the stated one: a few last places of a sum of 91
    np.testing.assert_allclose(key.contribution_by_geo / spent, key.roas_by_geo, rtol=4e-15, atol=0)
    np.testing.assert_allclose(denominator(world), 1.0, rtol=4e-15, atol=0)
    np.testing.assert_allclose(
        key.contribution_by_geo.sum(axis=0), key.contribution, rtol=2e-16, atol=0
    )


def test_on_the_national_set_it_is_not(worlds):
    shares = denominator(worlds["national"])
    assert shares.shape == (7,)
    assert np.all((shares > 0.80) & (shares < 0.91)), shares
    assert np.ptp(shares) > 0.09  # a share of its own a channel: no one scale of the spend


def test_the_panels_hold_every_cell_of_the_files(worlds):
    national, geo = worlds["national"].panel, worlds["geo"].panel
    assert (len(national.periods), national.geos, len(national.variables)) == (104, ("",), 21)
    assert (len(geo.periods), len(geo.geos), len(geo.variables)) == (91, 8, 6)
    assert national.periods[0].isoformat() == "2024-07-15"
    assert national["Sales"][0, 0] == 789.056468471508
    assert geo.geos == worlds["geo"].key.geos
    assert geo["Sales"][0, geo.geos.index("LA")] == 172.86213177603707
    assert worlds["national"].spend.shape == (104, 1, 7)
    assert worlds["geo"].spend.shape == (91, 8, 4)


def test_an_arm_reads_the_controls_the_roles_keep(worlds):
    roles = worlds["national"].key.roles
    assert worlds["national"].controls == tuple(
        name for name, role in roles.items() if role in ("confounder", "precision")
    )
    assert len(worlds["national"].controls) == 6
    assert worlds["geo"].controls == ("Price",)


def test_a_swapped_truth_moves_what_a_reading_is_scored_against(tmp_path, worlds):
    stated = _stated("national")
    stated["true_contribution"]["TV"] *= 2.0
    stated["true_roas"]["Search"] *= 0.5
    swapped = tmp_path / "national_truth.json"
    swapped.write_text(json.dumps(stated))
    key, moved = worlds["national"].key, read_truth(swapped)
    tv, search = key.channels.index("TV"), key.channels.index("Search")
    assert moved.contribution[tv] == 2.0 * key.contribution[tv]
    assert moved.roas[search] == 0.5 * key.roas[search]
    reading = Reading("fixed", key.contribution * 1.1, key.roas * 0.9)
    before, after = score(key, reading), score(moved, reading)
    np.testing.assert_allclose(before.contribution_error, 0.1, rtol=1e-14, atol=0)
    assert after.contribution_error[tv] == pytest.approx(1.1 / 2.0 - 1.0, rel=1e-14, abs=0.0)
    assert after.roas_error[search] == pytest.approx(0.9 / 0.5 - 1.0, rel=1e-14, abs=0.0)
    others = [c for c in range(len(key.channels)) if c not in (tv, search)]
    np.testing.assert_array_equal(after.roas_error[others], before.roas_error[others])
    world = MffWorld.of("national", worlds["national"].panel, moved)
    np.testing.assert_array_equal(FAMILY.truth(world).contribution, moved.contribution)


def test_a_file_not_the_recorded_one_is_not_read(tmp_path):
    for name in DIGESTS:
        shutil.copy(FAMILY.directory / name, tmp_path / name)
    assert read(tmp_path, "geo").key.scenario == "geo_heterogeneous"
    truth = tmp_path / "geo_truth.json"
    truth.write_bytes(truth.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=r"is not mmm-framework 1\.5\.0's"):
        read(tmp_path, "geo")
    with pytest.raises(ValueError, match="no environment 'markets'"):
        FAMILY.world("markets")


def _write(path, rows):
    header = "Period,Geography,Product,Campaign,Outlet,Creative,VariableName,VariableValue\n"
    path.write_text(header + "".join(",".join(row) + "\n" for row in rows))
    return path


def test_a_malformed_flat_file_is_refused(tmp_path):
    good = [
        ("2024-01-01", "", "", "", "", "", "Sales", "1"),
        ("2024-01-08", "", "", "", "", "", "Sales", "2"),
    ]
    assert read_mff(_write(tmp_path / "good.csv", good))["Sales"].tolist() == [[1.0], [2.0]]
    cases = {
        "twice": [*good, good[0]],
        "does not read": [*good, ("2024-01-08", "", "SKU", "", "", "", "TV", "1")],
        "misses a cell": [*good, ("2024-01-08", "", "", "", "", "", "TV", "1")],
        "a week apart": [good[0], ("2024-01-09", "", "", "", "", "", "Sales", "2")],
        "as nan": [good[0], ("2024-01-08", "", "", "", "", "", "Sales", "nan")],
    }
    for match, rows in cases.items():
        with pytest.raises(ValueError, match=match):
            read_mff(_write(tmp_path / "bad.csv", rows))
    (tmp_path / "header.csv").write_text("Period,Sales\n")
    with pytest.raises(ValueError, match="not a master flat file"):
        read_mff(tmp_path / "header.csv")


def test_a_key_states_each_channel_once():
    stated = _stated("geo")
    del stated["true_roas"]["TV"]
    with pytest.raises(ValueError, match="each channel's contribution and ROAS"):
        AnswerKey.parse(stated)
    stated = _stated("geo")
    del stated["true_roas_by_geo"]["LA"]["TV"]
    with pytest.raises(ValueError, match="every channel's roas in every geo"):
        AnswerKey.parse(stated)


def test_a_score_reads_each_channel_and_its_interval(worlds):
    key = worlds["geo"].key
    interval = np.column_stack([key.roas * 0.5, key.roas * 1.5])
    interval[0] = (key.roas[0] * 1.1, key.roas[0] * 1.2)  # misses the first channel
    scored = score(key, Reading("arm", key.contribution * 0.8, key.roas * 1.2, interval))
    np.testing.assert_allclose(scored.contribution_error, -0.2, rtol=1e-14, atol=0)
    assert scored.contribution == pytest.approx(0.2, rel=1e-14, abs=0.0)
    assert scored.roas == pytest.approx(0.2, rel=1e-14, abs=0.0)
    assert scored.covered == 0.75
    assert score(key, Reading("arm", key.contribution, key.roas)).covered is None
    with pytest.raises(ValueError, match="each of the 4 channels"):
        score(key, Reading("arm", key.contribution[:3], key.roas[:3]))


def test_the_arm_reads_the_geo_set_as_its_total(worlds):
    world = worlds["geo"]
    totalled = Totalled.of(world)
    np.testing.assert_array_equal(totalled.sales, world.sales.sum(axis=1))
    np.testing.assert_array_equal(totalled.spend, world.spend.sum(axis=1))
    np.testing.assert_array_equal(totalled.columns[:, 0], world.panel["Price"].mean(axis=1))
    design = totalled.controls()
    assert design.shape == (91, 1 + 1 + 6)
    np.testing.assert_array_equal(design[:, 0], 1.0)


def test_the_arm_recovers_channels_a_world_writes_exactly():
    """Two channels of independent spend and one control, the sales the arm's own model writes
    from them exactly: the fitted contributions and returns are the written ones."""
    rng = np.random.default_rng(21)
    weeks = 104
    periods = tuple(datetime.date(2024, 1, 1) + datetime.timedelta(weeks=w) for w in range(weeks))
    spend = rng.uniform(50.0, 150.0, size=(weeks, 2))
    control = rng.normal(size=weeks)
    returns = np.column_stack(
        [
            coefficient * np.tanh(_adstock(spend[:, c], retention, 6) / spend[:, c].mean())
            for c, (retention, coefficient) in enumerate(((0.4, 300.0), (0.7, 120.0)))
        ]
    )
    variables = {"A": spend[:, :1], "B": spend[:, 1:], "x": control[:, None]}
    panel = Panel(periods, ("",), {"Sales": np.zeros((weeks, 1)), **variables})
    contribution = returns.sum(axis=0)
    key = AnswerKey(
        scenario="written",
        description="",
        violates="",
        representable=True,
        channels=("A", "B"),
        contribution=contribution,
        roas=contribution / spend.sum(axis=0),
        geos=None,
        contribution_by_geo=None,
        roas_by_geo=None,
        roles={},
    )
    design = Totalled.of(MffWorld.of("national", panel, key)).controls()
    assert design.shape == (weeks, 1 + 1 + 6)
    sales = design @ np.linspace(1000.0, 10.0, design.shape[1]) + returns.sum(axis=1)
    world = MffWorld.of(
        "national", Panel(periods, ("",), {"Sales": sales[:, None], **variables}), key
    )
    scored = score(key, observational(world))
    # least squares on sales it writes exactly: the fit's own tolerance, not the data's
    np.testing.assert_allclose(scored.contribution_error, 0.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(scored.roas_error, 0.0, rtol=0, atol=1e-6)


def _record(world, scale=1.0):
    spent = world.spend.sum(axis=(0, 1))
    contribution = world.key.contribution * scale
    columns = ["Channel", "Total Spend", "Total Contribution", "ROI"]
    columns += ["Contribution HDI Low", "Contribution HDI High"]
    values = [
        [c, float(s), float(v), float(v / s), float(0.5 * v), float(2.0 * v)]
        for c, s, v in zip(world.key.channels, spent, contribution, strict=True)
    ]
    return {"roi": {"kind": "DataFrame", "columns": columns, "values": values[::-1]}}


def test_the_tools_record_is_read_in_the_keys_channels(worlds):
    world = worlds["geo"]
    reading = tool_reading(_record(world), world)
    assert reading.arm == "mmm-framework"
    np.testing.assert_array_equal(reading.contribution, world.key.contribution)
    assert reading.roas_interval is not None
    np.testing.assert_allclose(reading.roas_interval[:, 1], 2.0 * reading.roas, rtol=1e-15, atol=0)
    record = _record(world)
    record["roi"]["values"][0][1] *= 1.01
    with pytest.raises(ValueError, match="not the file's"):
        tool_reading(record, world)
    record = _record(world)
    record["roi"]["values"] = record["roi"]["values"][1:]
    with pytest.raises(ValueError, match="one row a channel"):
        tool_reading(record, world)


def test_the_committed_pilot_is_what_the_command_writes(tmp_path):
    """The bench's arm read again on both sets, and the tool's committed records read again, give
    the committed pilot: the arm's numbers within 1e-6 of them, as its channels are JAX's at
    either precision, the report's rounding and the tool's numbers exactly.

    The arm's own numbers hold within 1e-6 only on a machine that ends its fit where the pilot's
    machine did: its least-squares fit takes its Jacobian from differences of its residuals and
    stops on a relative change of 1e-8 in its cost, so another machine's last bits end it at
    another point of that valley. CI's runners read its returns up to 4.4e-4 apart, on numpy 2.4
    and 2.5 alike, so the arm is held within 5e-3 everywhere and the report is read where its
    numbers are the pilot's to 1e-6."""
    committed = json.loads(PILOT.with_suffix(".json").read_text())
    main(["--out", str(tmp_path / "pilot")])
    fresh = json.loads((tmp_path / "pilot.json").read_text())
    assert fresh["no_cell"] == committed["no_cell"] == dict(NO_CELL)
    assert fresh["tool"] == committed["tool"] and set(fresh["tool"]) == set(ENVIRONMENTS)
    same = True
    for new, old in zip(fresh["worlds"], committed["worlds"], strict=True):
        assert [a["arm"] for a in new["arms"]] == ["observational", "mmm-framework"]
        for a, b in zip(new["arms"], old["arms"], strict=True):
            rtol = 5e-3 if a["arm"] == "observational" else 1e-6
            for field in ("contribution", "roas"):
                np.testing.assert_allclose(a[field], b[field], rtol=rtol, atol=0)
                same &= bool(np.allclose(a[field], b[field], rtol=1e-6, atol=0))
    if fresh["x64"] and same:
        assert markdown(fresh) == PILOT.with_suffix(".md").read_text()
