# Report: the PyMC-Marketing arm moved to 1.3.0 (`cloud/pymc-arm-130`)

Status note, added when this file was written: the push problem described at the top of the report
below has since been fixed. `cloud/pymc-arm-130` is on GitHub at 0642c53, two commits on `main`
(b843176). The report follows verbatim; an appendix after it gives the exact `--smoke` records, the
commands that produced them, and the fitted worlds' records.

---

FINAL REPORT — pymc-arm-130 (causaldyn/causaldyn-bench)

**The push failed.** The git proxy refused it with a 403: "causaldyn/causaldyn-bench is not in this session's authorized repository set". The clone also had no remote configured, so I added `origin`. Both commits exist only locally on `cloud/pymc-arm-130`, and I saved them as:
- a bundle: `/tmp/claude-0/-home-user-repo/0784a782-0a3e-5ece-aa44-b0bfe74d581d/scratchpad/pymc-arm-130.bundle`
- patches: `/tmp/claude-0/-home-user-repo/0784a782-0a3e-5ece-aa44-b0bfe74d581d/scratchpad/patches/0001-feat-move-the-PyMC-Marketing-arm-to-1.3.0-s-signed-l.patch` and `/tmp/claude-0/-home-user-repo/0784a782-0a3e-5ece-aa44-b0bfe74d581d/scratchpad/patches/0002-docs-cite-PyMC-Marketing-1.3.0-s-source-for-family-1.patch`

To publish them, the repository has to be added to the session's sources.

**1. Verdict per item**
1. Re-pin: **done.** pymc-marketing goes 1.2.0 → 1.3.0, pytensor 3.3.4 → 3.3.6, pandas 3.0.6 → 3.1.0. h5netcdf, h5py and tzdata are added; pymc stays at 6.3.2. The header comment is kept, with the command and the date updated. No change to pyproject.toml or uv.lock.
2. Read 1.3.0's source and write it into the arm's header: **done.** The new paragraph cites file and line.
   - `mmm.py` 4127-4134: the method takes `likelihood=None`; `dist` survives as a deprecated keyword-only alias.
   - `lift_test.py` 180-207: `None` becomes `Prior("Normal")` (line 196); passing `dist` gives a DeprecationWarning (185-193).
   - `assert_monotonic` is no longer called (1.2.0 called it at line 399); it remains, deprecated, at 359-371.
   - Validation (line 569, lines 95-140): values must be finite and `sigma` positive (64-92). Gamma alone needs `delta_y > 0` and `delta_x > 0` (129-140).
   - Likelihood (611-619): `mu` is the signed model lift, `observed` the signed `delta_y`.
   - StudentT needs an `nu` from the caller (116-120).
   - National and geo use one class and one call: `exact_row_indices` (line 232) and `scale_lift_measurements` are unchanged from 1.2.0.
3. `pymc_marketing_lift.py`: **done.**
   - The rule is now: take a row if `x`, `delta_x`, `delta_y` and `sigma` are finite and `sigma > 0`.
   - Splitting a geo test across its geos is kept, because the geo path did not change.
   - The geo call no longer passes `dist=pmd.Normal`. That alias is deprecated, and the default is now that same Normal.
   - `lift_dropped` stays in the record.
4. Family 13's defaults: **done, nothing moved.**
   - GeometricAdstock, Beta(1, 3), `normalize=True`, LogisticSaturation, Gamma(3, 1), HalfNormal(2), max scaling and `adstock_first` are identical. The class bodies and transformer functions diff clean.
   - In the docstring I updated the version and the line numbers that shifted (mmm.py 581-584 and 504; adstock.py 326-327 and 559), plus the version line of `scripts/pymc_marketing_form.py`. `results/pymc_marketing_form.json` is untouched.
   - I re-ran the form script under 1.3.0 to a scratch path: 15 cases, defaults equal, transforms within 6.4e-15 relative (the test holds 1e-9).
5. Smoke and pilot: **partly.**
   - `--smoke` ran on two version-2 exports, with pymc_marketing's DeprecationWarnings turned into errors:
     - national world 900: error None, 12 lift rows, 0 dropped, optimiser success.
     - geo world 360003: error None, 60 rows, 12 tests taken, 0 dropped.
   - Full NUTS ran on pilot worlds 900-902 from the budgets `-pilot` export: all fitted, no error, 0 divergences.
   - **I wrote no results file.** The existing report path (`budget_regret score`) refuses to run without all 40 pilot worlds, and `external_arms` scores only Meridian and Robyn. Rather than edit the report code, I scored the three records in scratch with `budget_regret.score_world`. The digests matched and the oracle reproduces the committed pilot.
   - **`lift_dropped` is not 0 on these three worlds: it is 1, 2 and 0.** These are the rows the version-1 export itself left out for 1.2.0, and the budgets run's digests pin that export. The arm's own count is 0 on every world.

**2. Branch and commits** (`cloud/pymc-arm-130`, not pushed)
```
0642c53 docs: cite PyMC-Marketing 1.3.0's source for family 13's defaults
1af4c71 feat: move the PyMC-Marketing arm to 1.3.0's signed lift likelihood
```

**3. Tests**
- Fail-before (old tests against the new rule): 7 failed. These are the 5 in `test_pymc_marketing_lift.py` and the 2 in `test_pymc_marketing_likelihood.py`, with errors such as `KeyError: 'lift_x'` and "Mismatch at indices [5],[6],[7],[11]: True (ACTUAL), False (DESIRED)".
- Pass-after: both files, 8 passed. One test is new: a tested geo that spent nothing on the channel is refused.
- Scorecard tests (`tests/test_scorecard_*.py`): 533 passed, 1 skipped, 1 failed. The failure is `test_the_worlds_transforms_are_pymc_marketings_own`, where the world's spend differs in its last digit (`171.44408853726873 == 171.44408853726875`). It fails the same way on `main` and comes from the bench side, not PyMC.
- `ruff check`: all passed. `ruff format --check`: 198 files formatted. `ty check`: 2 diagnostics, both an unresolved matplotlib import in `notebooks/`, the same on `main`.
- Not run: the full suite, and family 8 or other families' arm fits beyond the smoke.

**4. Numbers old → new** (pilot, drawn worlds; regret per euro, PyMC-Marketing)

| World | 1.2.0 | 1.3.0 | Lift rows | Dropped (export / arm) |
|---|---|---|---|---|
| 900 | 0.1352 | 0.1147 | 11 | 1 / 0 |
| 901 | 0.0903 | 0.0930 | 10 | 2 / 0 |
| 902 | 0.0029 | 0.0092 | 12 | 0 / 0 |

In the 1.3.0 fits, the largest R-hat is at most 1.0031 and the smallest bulk ESS at least 680. On the version-1 export only the likelihood changed (Gamma → Normal), not the rows.

**5. Proposed CHANGELOG lines**
- Changed: the PyMC-Marketing arm runs PyMC-Marketing 1.3.0. Lift tests are read on the tool's default signed Normal, and tests whose sales rose are taken rather than dropped.
- Changed: geo exports' lift rows use the default likelihood instead of the deprecated `dist=pymc.dims.Normal`, which is the same Normal.
- Fixed: the arm's lift-row rule now matches the tool's own validation (finite values, positive error).

**6. Open questions**
1. Results file: should `budget_regret score` accept a subset of seeds (for example a `--seeds` option), or should all 40 drawn pilot worlds be fitted? Either makes `results/track_m2_external_pilot_pymc130.*` possible.
2. Version-1 records: `lift_dropped` includes the rows the pinned export left out. Keep that, or split it into export-dropped and arm-dropped?
3. Under 1.3.0, a Gamma likelihood refuses every test that cuts spend. So the pre-registered budgets run (1.2.0, Gamma) can only be reproduced from the old pin in git history. Should that be documented in `budget_regret.py`?
4. Splitting a geo test still counts a geo with zero spend in `m`, so the remaining rows hold (m−1)/m of the test's information. This behaviour predates the change, and a new test now records it. Should it be fixed?
5. Family 13's `labels` string and the `ALPHA`/`LAM` comment still say 1.2.0. I left them because the brief said "docstring only".
6. I did not read upstream issue #2958 itself; the header cites only the 1.3.0 source.
7. The bench environment needs a sibling `causal-hybrid-control` checkout. I used CHC `main` (1.0.0), while the lock records 0.23.0.

---

## Appendix: the exact runs behind the report

### Fail-before: the old tests against the new rule

```
FAILED tests/test_pymc_marketing_lift.py::test_the_arm_takes_a_row_whose_sales_moved_with_its_spend_and_counts_the_rest
FAILED tests/test_pymc_marketing_lift.py::test_on_a_geo_export_whose_tests_rose_the_arm_drops_them_itself
FAILED tests/test_pymc_marketing_lift.py::test_on_a_national_export_the_rows_are_the_ladders_and_the_arm_drops_the_one_that_rose
FAILED tests/test_pymc_marketing_lift.py::test_a_test_of_five_geos_splits_into_five_rows_that_sum_back_to_it
FAILED tests/test_pymc_marketing_lift.py::test_on_the_familys_geo_export_each_test_splits_into_its_five_geos
FAILED tests/test_pymc_marketing_likelihood.py::test_a_geo_exports_apportioned_rows_are_read_on_a_normal_likelihood
FAILED tests/test_pymc_marketing_likelihood.py::test_a_national_exports_rows_keep_the_tools_default_likelihood
```

### `--smoke`

Exports written by `causaldyn_bench.scorecard.observe.export`: `TRACK_M2.observe(TRACK_M2.world("drawn",
900), 4)` (national) and `GEO.observe(GEO.world("policy-0", GEO.pilots["policy-0"][3]), 4)` (geo,
world 360003). Each was fitted by

```
OMP_NUM_THREADS=1 uv run --no-project --python 3.12 --with-requirements scripts/pymc_marketing_arm.txt \
    python -u -W error::DeprecationWarning:pymc_marketing scripts/pymc_marketing_arm.py \
    --worlds DIR --out DIR/pymc --smoke
```

The script printed `world_900 None None` and `world_360003 None None` (no error). The records, 2
chains of 100 draws, PyMC-Marketing 1.3.0:

```
national world_900: {"error": null, "smoke": "smoke, not a result", "lift_rows": 12, "lift_dropped": 0, "optimiser_success": true, "flags": {"divergences": 0, "max_treedepth": 5, "max_rhat": 1.0367335348407793, "min_ess_bulk": 71.54555891638479}}
geo world_360003:   {"error": null, "smoke": "smoke, not a result", "lift_rows": 60, "lift_dropped": 0, "lift_tests": {"taken": 12, "dropped": 0}, "optimiser_success": true, "flags": {"divergences": 84, "max_treedepth": 116, "max_rhat": 3.070712283535505, "min_ess_bulk": 2.4378906324799363}}
```

The geo smoke's flags are those of a 100-draw run, which checks the plumbing and is not a result.

### The fitted worlds

Exported by `python -m causaldyn_bench.budget_regret export --work DIR --pilot --environments
drawn` (version 1, the budgets run's export), worlds 900, 901 and 902 fitted by
`scripts/pymc_marketing_arm.py` at the arm's full settings (4 chains, 1000 tuning and 1000 kept
draws). Each record, PyMC-Marketing 1.3.0:

```
world_900: {"error": null, "lift_rows": 11, "lift_dropped": 1, "optimiser_success": true, "divergences": 0, "max_rhat": 1.0030612614973602, "min_ess_bulk": 1041.2584989440306}
world_901: {"error": null, "lift_rows": 10, "lift_dropped": 2, "optimiser_success": true, "divergences": 0, "max_rhat": 1.00291097381794, "min_ess_bulk": 891.0627413210168}
world_902: {"error": null, "lift_rows": 12, "lift_dropped": 0, "optimiser_success": true, "divergences": 0, "max_rhat": 1.0028636824998638, "min_ess_bulk": 680.3938324650884}
```

In each, `lift_dropped` is the export's own count (the tests whose sales rose, left out for 1.2.0
and pinned by the budgets run's digests) plus the arm's, which is 0. Scored by
`causaldyn_bench.budget_regret.score_world`, each record's digest matched its world and the oracle
and the status quo's regret reproduced the committed pilot's.
