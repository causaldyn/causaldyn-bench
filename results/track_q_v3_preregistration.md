# Track Q, version 3 — pre-registration

Written on 2026-10-27 and committed with its harness, before any replicate of the scored run was
drawn. The code that will run is `causaldyn_bench.graph_errors_v3` at this document's commit,
unchanged from there to the run. The results will name this document's commit.

## Why a third version

Version 2 (`results/track_q_v2.md`) read the logger check's flag as a stop and found it no better
than a stop at the same rate that ignored the failures, in every world (least one-sided Fisher p
0.64). causal-hybrid-control now states why (its `validation/confounded_log_equivalence.mac`). A
confounder drawn afresh and Gaussian leaves the log of the states and the levers with the law of a
loop without it, so no test that reads only those columns has power above its size there.

A test that reads the logger's own draw can see it. The draw moves the lever apart from anything
the plant reads. So the channel it instruments, conditioned on the states alone, does not depend on
the graph, and its distance from the channel the plan was fitted on is a Durbin-Wu-Hausman contrast.
`chc.prescribe(..., dither=<column>)` reads that contrast in the plan's own regret metric through
`chc.misspecification.misspecification_cost` (`chc.decision.DitherCheck`, the library's ADR 0139).
Version 3 scores it as the stop.

## What does not change

All of these are version 2's:

- the nine worlds and their generator;
- the three arms (the true graph, the wrong one, the adjustment set chosen by fit);
- the panels (400 units x 12 periods, one to plan on and a later one to evaluate on);
- the decision and every `chc.prescribe` option;
- the tolerance, 0.8012;
- the margin, 0.0048, and the threshold, 0.0548;
- the three claims ("left" by the exceedance at a trusted step, "lost" on regret, "missed" by the
  evaluation interval);
- the silent failure, the stop's Fisher test and the gate.

Versions 1 and 2 still reproduce: every new argument of `causaldyn_bench.graph_errors` defaults to
their values, and the draws are the same with the draw logged or not
(`tests/test_graph_errors_v3.py`).

## What changes

### 1. The panels log the logger's draw

The generator already draws the incentive's fresh part, `fresh ~ N(0, 1.5^2)`. The incentive is

    sticky x last + sqrt(1 - sticky^2) x fresh + chase x demand + read x promotion.

Version 3 logs `fresh` as a column of both panels, named `fresh`. Logging it consumes no random
number, so every other column is bit-identical to version 2's at the same seed. The arm chosen by
fit does not see the column and reads no check: an analyst would not adjust for the logger's own
randomness.

### 2. The stop

The true and the wrong graph call `prescribe(..., dither="fresh")` on the panel they plan on.
**The stop refuses the plan and its evaluation when all three hold:**

1. the check is `tested`, not `void`;
2. the contrast's p-value is at most **0.05**;
3. the contrast's priced cost, `contrast.cost`, is above **0.1 x the gain the plan claims on its
   own model**.

The claimed gain is doing nothing's planning cost on the plan's fitted model, from the plan's
start, less the plan's task cost (`graph_errors.claimed_gain`). The factor 0.1 is version 1's
`MATERIAL`. A plan fails its regret claim when it loses more than 0.1 of the stakes on the true
plant. The stop asks the same of the contrast, in the plan's own units, with no knowledge of the
true plant. The cost is `(d' W d - tr(W S))/2`: what planning on the fitted channel would cost,
to second order, were the instrumented channel the plant's.

No flag is read on the later panel (`later_flag` is None). The logger check's flag is kept per
replicate as `logger_flag`, for a secondary reading.

### 3. The seed

**20261026.** Replicate `i` of world `w` draws its panel from `default_rng([20261026, w, i, 0])`
and its later panel from `[20261026, w, i, 1]`, `w` in version 1's world order.

## The run

* **200 replicates a world**, nine worlds, on the CPU, `JAX_ENABLE_X64=1`, four workers.
* causal-hybrid-control at the first commit whose `prescribe` takes `dither=` (`197c734` on its
  development branch, or the commit that carries it to main), imported from its source tree. The run records it with `--chc-commit`, and the results name it. The harness
  refuses to run on a library without `dither=`.
* Run once, by `just track-q-v3 --preregistration <this commit> --chc-commit <commit>`.

## Pass, fail and how they are read

1. **The gate (primary), as version 2's.** For each world, the stop's table is read over the true
   and the wrong graph's replicates, or over the wrong graph's alone where the two graphs are the
   same (none, sticky). It counts:
   - of the replicates whose claims failed, how many the stop cleared of every failure;
   - of the replicates whose claims held, how many the stop refused;
   - the one-sided Fisher exact p that the stop removes failures more often than it refuses sound
     claims, as a stop at the same rate that ignored the failures would.

   **The stop passes the gate if p <= 0.05 in at least one world.** Reported beside it, not a
   criterion: the worlds where p <= 0.05/9 = 0.0056.
2. **The size.** The contrast's rejection rate at 0.05 on the panels the true graph planned on,
   pooled over the eight worlds with an oracle, with its Clopper-Pearson interval and the number of
   void panels. **It passes if the rate is at most 0.07.** The bound is one-sided: the library's
   contrast is conservative on short panels (its ADR 0139), and a test that rejects less often than
   its level does not falsify the design.
3. **The verdict.** The stop is worth shipping as a stop if both 1 and 2 pass.

## Reported whatever the outcome

* The gate's table per world, with the verdict at 0.05 and the reading at 0.0056.
* The size, with the void count.
* Per world and arm that read the draw: panels tested and void, the contrast's rejection rate with
  its interval, the stop's rate, the mean priced cost and the mean margin it is held to.
* **Secondary readings, not criteria:**
  - the gate with the contrast alone as the stop, at any priced cost (the margin at zero);
  - the gate with the logger check's flag as the stop, version 2's, on these panels.
* Per world, every arm in both readings, in version 2's columns.
* The JSON, per replicate: version 2's columns, each arm's check (status, p, cost, its standard
  error, the noise term, the share, the relevance, the claimed gain) and the logger check's flag.
  Another margin can then be read without a rerun.
* Any departure from this document, as a dated amendment in the results. Wherever it can be
  computed, the result this document specifies is reported beside it.

## Expectations, stated before the run

These come from the diagnosis below and from version 2's result. They are not criteria.

* **Mediator, collider latent.** The wrong graph loses on regret on every replicate, as in
  version 2. Its fitted channel is the direct one, or a collider's. The instrumented channel is the
  total one, so the contrast rejects and its cost is a large share of the claimed gain: the stop
  fires. The true graph's contrast does not reject. **The gate passes in both worlds.**
* **Confounder.** The wrong graph leaves its tube on most replicates, as in version 2 (0.830).
  The contrast rejects there, but its priced cost is near 0.001 against a margin near 0.004, so
  the stop does not fire. The failures are the tube's, which reads the plan's states, not its
  regret. **The gate does not pass there.** In the secondary reading, with the contrast alone as
  the stop, it does.
* **Confounder AR.** Few failures at this tolerance, as in version 2. Whether the gate can pass
  there is open.
* **None, chase, sticky, collider observed, non-ancestor.** The wrong graph costs the fit little,
  the stop fires at most at the test's size, and the gate does not pass.
* **The size** is below 0.05: the contrast reads its variance high on panels of 11 transitions a
  unit.

## The diagnosis this design follows

The design was set after a prototype on the throwaway seed 1, replicates 0-1 of the none,
confounder and mediator worlds, on the library's development tree. None of those panels is in the
scored run. What it read:

* **none:** the contrast's share near 1, p 0.55-0.61.
* **confounder, wrong graph:** p of 3e-7 and 1e-10. The priced cost was 0.0012-0.0014 against a
  true regret of 0.00022-0.00026 and stakes near 0.04.
* **mediator, wrong graph:** p near 1.8e-11. The priced cost was 0.05-0.09 against a true regret
  of 0.077-0.116.
* **the true graphs:** p 0.80-0.93.

The margin's factor, 0.1, was fixed to version 1's `MATERIAL` before the prototype, and the
prototype did not move it. The expectation for the confounder world follows from it. The harness's
test runs 2 replicates of the mediator world on the throwaway seed 7 to check that the code runs end
to end. Its numbers are not used and were not read as a result.

## Limits

* The draw is the incentive's own fresh part, logged exactly. A logger that logs its draw with an
  error, or draws it from the state, is not scored.
* The contrast reads the plan's regret to second order at the plan. A wrong graph whose cost is
  large but far from the plan's actions is outside what it prices.
* "The true plant" is the bench's linear-Gaussian plant of each world, which the generator's panels
  are checked against. The worlds are CHC's by construction.
