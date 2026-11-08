"""PyMC-Marketing's arm piloted at its pin: a record at the pin is scored and its lift rows counted,
and scoring refuses a record off the pin, a smoke run, another setting or a version-1 record; the
committed pilot names the pin file's versions, its arm refused no lift row, family 0's tests whose
sales did not fall are those the version-1 record counted as dropped, and its markdown is written
from its JSON."""

import json
from importlib.metadata import version

import jax
import numpy as np
import pytest

from causaldyn_bench.budget_regret import PILOTS
from causaldyn_bench.scorecard.observe import export
from causaldyn_bench.scorecard.pymc_marketing_pilot import (
    ARM,
    BENCH,
    CELLS,
    OUT,
    PACKAGES,
    RUNG,
    directory,
    markdown,
    pilot,
    pins,
)
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

SEED = 900  # family 0's first drawn pilot world


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Family 0's first drawn pilot world exported at the pilot's rung."""
    root = tmp_path_factory.mktemp("pilot")
    with jax.enable_x64(True):
        world = TRACK_M2.world("drawn", SEED)
        export(TRACK_M2.observe(world, RUNG), directory(root, TRACK_M2, "drawn"))
    return root


def _record(work, **changes):
    """A record of the exported world that plans the status quo, as the arm writes one."""
    path = directory(work, TRACK_M2, "drawn") / f"world_{SEED}.npz"
    with np.load(path) as saved:
        record = {
            "version": int(saved["version"]),
            "seed": SEED,
            "digest": str(saved["digest"]),
            "weekly": saved["status_quo"].tolist(),
            "optimiser_success": True,
            "flags": {"divergences": 0, "max_rhat": 1.002},
            "settings": {"setting": "static", "smoke": False},
            "lift_rows": int(saved["lift_x"].size),
            "lift_dropped": 0,
            "error": None,
            "versions": pins(),
        }
    record |= changes
    made = path.parent / ARM / path.with_suffix(".json").name
    made.parent.mkdir(exist_ok=True)
    made.write_text(json.dumps(record))
    return record


def test_the_pins_are_the_arm_files(tmp_path):
    pinned = pins()
    assert tuple(pinned) == PACKAGES and pinned["pymc-marketing"] == "1.3.0"
    short = tmp_path / "pins.txt"
    short.write_text("# a comment\npymc==6.3.2\n")
    with pytest.raises(ValueError, match="pins no version"):
        pins(short)


def test_family_0_is_the_budgets_runs_pilot():
    assert [e for f, e in CELLS if f is TRACK_M2] == [e.name for e in PILOTS]
    assert {e.name: list(e.seeds) for e in PILOTS} == {
        e: list(TRACK_M2.pilots[e]) for e in TRACK_M2.pilots
    }


def test_a_record_at_the_pin_is_scored_and_its_lift_counted(work):
    _record(work)
    with jax.enable_x64(True):
        data = pilot(work)
    (world,) = data["worlds"]
    assert (world["seed"], world["lift_dropped"], world["error"]) == (SEED, 0, None)
    assert world["tests"] == world["lift_rows"] == 4 * 3
    drawn = data["environments"][0]
    assert (drawn["environment"], drawn["fitted"], drawn["whole"]) == ("drawn", 1, 1)
    assert drawn["pilots"] == 40 and drawn["taken"] == 12 and drawn["dropped"] == 0
    assert all(e["fitted"] == 0 for e in data["environments"][1:])
    assert world["regret"][ARM] == world["regret"]["status quo"]
    assert "compute" not in data["reports"][0] and data["versions"] == pins()
    assert data["scored_under"]["causal-hybrid-control"] == version("causal-hybrid-control")
    assert "family 0's reference, 20 of its 20 pilot worlds" in markdown(data)


@pytest.mark.parametrize(
    ("changes", "refused"),
    [
        ({"versions": {**pins(), "pymc-marketing": "1.2.0"}}, "not the pin file's"),
        ({"smoke": "smoke, not a result"}, "smoke run"),
        ({"settings": {"setting": "static", "smoke": True}}, "smoke run"),
        ({"settings": {"setting": "shared", "smoke": False}}, "not 'static'"),
        ({"version": 1}, "version-1 record"),
    ],
)
def test_a_record_off_the_pin_or_a_smoke_run_is_refused(work, changes, refused):
    _record(work, **changes)
    with jax.enable_x64(True), pytest.raises(ValueError, match=refused):
        pilot(work)


@pytest.fixture(scope="module")
def committed():
    return json.loads(OUT.with_suffix(".json").read_text())


def test_the_committed_pilot_names_the_pin_and_refused_no_row(committed):
    assert committed["versions"] == pins() and committed["arm"] == ARM
    assert committed["rung"] == RUNG and committed["x64"]
    assert set(committed["scored_under"]) == set(BENCH)
    worlds = committed["worlds"]
    # a fit that raised holds no count; the geo arm's optimiser may raise after the fit
    assert worlds and all(w["lift_dropped"] == 0 for w in worlds if w["error"] is None)
    national = [w for w in worlds if w["family"] in (0, 13)]
    assert all(w["error"] is None and w["lift_rows"] == w["tests"] for w in national)
    # the rows 1.2.0 refused are among those taken
    assert sum(w["not_fallen"] for w in national) > 0
    for e in committed["environments"]:
        if e["family"] in (0, 13):
            assert e["fitted"] == e["whole"] == e["pilots"]
        assert e["dropped"] == 0 and e["tests_dropped"] in (None, 0)


def test_each_cell_is_the_mean_of_its_worlds(committed):
    for found in committed["reports"]:
        for cell in found["cells"]:
            seen = [
                w
                for w in committed["worlds"]
                if w["family"] == found["family"]
                and w["index"] < cell["worlds"]
                and cell["environment"] in ("pooled", w["environment"])
            ]
            assert len(seen) == cell["look"]
            for arm, columns in cell["arms"].items():
                mean = np.mean([w["regret"][arm] for w in seen])
                assert columns["regret"]["mean"] == pytest.approx(mean, rel=1e-12, abs=0.0)


def test_the_markdown_is_written_from_the_json(committed):
    assert markdown(committed) == OUT.with_suffix(".md").read_text()


def test_the_version_1_records_count_is_what_its_export_left_out(committed):
    """The budgets pilot at the same pin read version-1 exports of family 0's pilot worlds: each
    world's ``lift_dropped`` there is the tests of its version-2 export whose sales did not fall,
    and the arm took every one of them here."""
    first = json.loads((OUT.parent / "track_m2_budgets_pilot_pymc130.json").read_text())
    assert first["pymc_environment"]["pymc-marketing"] == committed["versions"]["pymc-marketing"]
    ours = {(w["environment"], w["seed"]): w for w in committed["worlds"] if w["family"] == 0}
    read = 0
    for environment in first["environments"]:
        for world in environment["worlds"]:
            mine, theirs = ours[(environment["environment"]["name"], world["seed"])], world["pymc"]
            assert theirs["lift_dropped"] == mine["not_fallen"]
            assert (
                theirs["lift_rows"] + theirs["lift_dropped"] == mine["tests"] == mine["lift_rows"]
            )
            assert mine["lift_dropped"] == 0
            read += 1
    assert read == len(ours) == 60
