"""Family 21's generated worlds: the files mmm-framework's generator wrote are read by the digests
of a manifest whose own digest the bench holds, beside their licence; the truth reader returns what
each key states, and the stated ROAS is the stated contribution over the file's spend or, where the
tool plants recording errors in that spend, over the true spend its key's notes state; a swapped
key moves what a reading is scored against; and the committed pilot has a cell of the bench's arm
and one of the tool's own fit on every world."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from causaldyn_bench.scorecard.mmm_framework import (
    MANIFEST,
    MANIFEST_SHA256,
    MARKETS,
    MMM_FRAMEWORK,
    NO_CELL,
    UNREAD_MARKETS,
    WHEEL_SHA256,
    MffWorld,
    MmmFramework,
    Reading,
    denominator,
    main,
    markets_markdown,
    read_market,
    read_truth,
    score,
)

FAMILY = MmmFramework()
ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "mmm_framework_markets"
MISRECORDED = ("spend_outliers", "mixed_data_errors")  # the tool's planted recording errors
PANELS = ("geo_clean", "geo_heterogeneous")


@pytest.fixture(scope="module")
def worlds():
    return {name: FAMILY.world(name) for name in MARKETS}


def _stated(name):
    return json.loads((FAMILY.markets / f"{name}_truth.json").read_text())


def test_the_files_are_the_generators_and_carry_their_licence():
    directory = FAMILY.markets
    manifest = json.loads((directory / MANIFEST).read_text())
    assert hashlib.sha256((directory / MANIFEST).read_bytes()).hexdigest() == MANIFEST_SHA256
    assert manifest["version"] == MMM_FRAMEWORK
    assert manifest["call"] == "mmm_framework.synth.generate_mff(name)"
    assert tuple(manifest["worlds"]) == MARKETS
    files = [f"{name}_{kind}" for name in MARKETS for kind in ("mff.csv", "truth.json")]
    assert sorted(p.name for p in directory.iterdir()) == sorted([*files, MANIFEST, "NOTICE"])
    for name in MARKETS:
        for kind in ("mff.csv", "truth.json"):
            data = (directory / f"{name}_{kind}").read_bytes()
            recorded = manifest["worlds"][name][kind]
            assert (len(data), hashlib.sha256(data).hexdigest()) == (
                recorded["bytes"],
                recorded["sha256"],
            )
    assert not set(MARKETS) & set(UNREAD_MARKETS)
    notice = (directory / "NOTICE").read_text()
    assert "A rival's worlds, labelled so" in notice
    assert f"mmm-framework {MMM_FRAMEWORK}" in notice and "Apache License, Version 2.0" in notice
    assert WHEEL_SHA256 in notice and f"MANIFEST.json  {MANIFEST_SHA256}" in notice
    assert "mmm_framework_markets" in (ROOT / "NOTICE").read_text()


@pytest.mark.parametrize("name", MARKETS)
def test_the_truth_reader_returns_what_each_key_states(worlds, name):
    key, stated = FAMILY.truth(worlds[name]), _stated(name)
    assert key.scenario == stated["scenario"] == name
    assert key.channels == tuple(stated["channels"])
    assert key.contribution.tolist() == [stated["true_contribution"][c] for c in key.channels]
    assert key.roas.tolist() == [stated["true_roas"][c] for c in key.channels]
    assert (key.violates, key.representable) == (stated["violates"], stated["representable"])
    if name in PANELS:
        assert key.geos == tuple(stated["geographies"]) and len(key.geos) == 4
        for kind, table in (("contribution", key.contribution_by_geo), ("roas", key.roas_by_geo)):
            by_geo = stated[f"true_{kind}_by_geo"]
            assert table is not None
            assert table.tolist() == [[by_geo[g][c] for c in key.channels] for g in key.geos]
    else:
        assert key.geos is None and worlds[name].panel.geos == ("",)
    weeks = len(worlds[name].panel.periods)
    assert weeks == (130 if name in PANELS else 156)


@pytest.mark.parametrize("name", MARKETS)
def test_the_stated_roas_is_the_contribution_over_the_spend_its_key_names(worlds, name):
    world = worlds[name]
    key = world.key
    if name in MISRECORDED:
        true = _stated(name)["notes"]["true_spend_sum"]
        spent = np.array([true[c] for c in key.channels])
        # within 4e-15 of it, as measured: exactly
        np.testing.assert_allclose(key.contribution / key.roas, spent, rtol=4e-15, atol=0)
        assert np.max(np.abs(denominator(world) - 1.0)) > 0.005  # the file's spend is not it
    else:
        # within 4e-15 of the file's spend: a few last places of a sum of 156 weeks (1.8e-15)
        np.testing.assert_allclose(denominator(world), 1.0, rtol=4e-15, atol=0)


@pytest.mark.parametrize("name", PANELS)
def test_on_the_panels_each_geography_is_stated_on_its_own_spend(worlds, name):
    world = worlds[name]
    key = world.key
    assert key.contribution_by_geo is not None and key.roas_by_geo is not None
    np.testing.assert_allclose(
        key.contribution_by_geo / world.spend.sum(axis=0), key.roas_by_geo, rtol=4e-15, atol=0
    )
    np.testing.assert_allclose(
        key.contribution_by_geo.sum(axis=0), key.contribution, rtol=2e-16, atol=0
    )


def test_an_arm_reads_the_controls_the_keys_roles_keep(worlds):
    roles = worlds["realistic"].key.roles
    assert worlds["realistic"].controls == tuple(
        name for name, role in roles.items() if role in ("confounder", "precision")
    )
    assert "brand_awareness" not in worlds["realistic"].controls
    # the key of dense_controls names no role in its notes: the arm is handed every candidate
    assert len(worlds["dense_controls"].controls) == 25
    assert worlds["clean"].controls == ("Price",)


def test_a_swapped_key_moves_what_a_reading_is_scored_against(tmp_path, worlds):
    stated = _stated("clean")
    stated["true_contribution"]["TV"] *= 2.0
    stated["true_roas"]["Search"] *= 0.5
    swapped = tmp_path / "clean_truth.json"
    swapped.write_text(json.dumps(stated))
    key, moved = worlds["clean"].key, read_truth(swapped)
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
    world = MffWorld.of("clean", worlds["clean"].panel, moved)
    np.testing.assert_array_equal(FAMILY.truth(world).contribution, moved.contribution)


def test_a_swapped_key_under_a_manifest_that_names_it_moves_the_familys_oracle(
    tmp_path, monkeypatch, worlds
):
    copy = tmp_path / "markets"
    shutil.copytree(FAMILY.markets, copy)
    stated = _stated("clean")
    stated["true_contribution"]["TV"] *= 2.0
    truth = copy / "clean_truth.json"
    truth.write_text(json.dumps(stated))
    manifest = json.loads((copy / MANIFEST).read_text())
    manifest["worlds"]["clean"]["truth.json"]["sha256"] = hashlib.sha256(
        truth.read_bytes()
    ).hexdigest()
    (copy / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="is not the recorded manifest"):
        MmmFramework(markets=copy).world("clean")
    monkeypatch.setattr(
        "causaldyn_bench.scorecard.mmm_framework.MANIFEST_SHA256",
        hashlib.sha256((copy / MANIFEST).read_bytes()).hexdigest(),
    )
    key, moved = worlds["clean"].key, MmmFramework(markets=copy).world("clean").key
    tv = key.channels.index("TV")
    assert moved.contribution[tv] == 2.0 * key.contribution[tv]
    others = [c for c in range(len(key.channels)) if c != tv]
    np.testing.assert_array_equal(moved.contribution[others], key.contribution[others])


def test_a_file_or_manifest_not_the_recorded_one_is_not_read(tmp_path):
    copy = tmp_path / "markets"
    shutil.copytree(FAMILY.markets, copy)
    assert MmmFramework(markets=copy).world("clean").key.scenario == "clean"
    truth = copy / "clean_truth.json"
    truth.write_bytes(truth.read_bytes().replace(b'"TV": ', b'"TV":  ', 1))
    with pytest.raises(ValueError, match="is not what mmm-framework's generator wrote"):
        read_market(copy, "clean")
    assert read_market(copy, "synergy").key.scenario == "synergy"
    manifest = copy / MANIFEST
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="is not the recorded manifest"):
        read_market(copy, "synergy")
    for name in UNREAD_MARKETS:
        with pytest.raises(ValueError, match=f"hold no '{name}'"):
            read_market(FAMILY.markets, name)


def test_the_committed_pilot_is_what_the_command_writes(tmp_path):
    """The bench's arm read again on every generated world, and the tool's committed records read
    again, give the committed pilot: the arm's numbers within 1e-6 of them, as its channels are
    JAX's at either precision, the report and the tool's numbers exactly; each world has a cell of
    both."""
    committed = json.loads(PILOT.with_suffix(".json").read_text())
    main(["--markets", "--out", str(tmp_path / "markets")])
    fresh = json.loads((tmp_path / "markets.json").read_text())
    assert fresh["no_cell"] == committed["no_cell"] == dict(NO_CELL)
    assert fresh["generator"] == committed["generator"]
    assert fresh["generator"]["manifest_sha256"] == MANIFEST_SHA256
    assert fresh["tool"] == committed["tool"] and tuple(fresh["tool"]) == MARKETS
    assert [w["environment"] for w in fresh["worlds"]] == list(MARKETS)
    for new, old in zip(fresh["worlds"], committed["worlds"], strict=True):
        assert [a["arm"] for a in new["arms"]] == ["observational", "mmm-framework"]
        for a, b in zip(new["arms"], old["arms"], strict=True):
            for field in ("contribution", "roas"):
                np.testing.assert_allclose(a[field], b[field], rtol=1e-6, atol=0)
    if fresh["x64"]:
        assert markets_markdown(fresh) == PILOT.with_suffix(".md").read_text()
