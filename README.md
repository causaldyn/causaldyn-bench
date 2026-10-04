# causaldyn-bench

[![ci](https://github.com/causaldyn/causaldyn-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/causaldyn/causaldyn-bench/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![doi](https://zenodo.org/badge/DOI/10.5281/zenodo.22139814.svg)](https://doi.org/10.5281/zenodo.22139814)

A benchmark for **causal, constrained, dynamical decision-making** — it scores *predictions and
decisions*, not just one-step error. The point is to measure methods on the axis each deserves, so a
gradient-boosted tree can win one-step prediction while a causal, constrained controller wins the
closed-loop decision. Built on
[`causal-hybrid-control`](https://github.com/causaldyn/causal-hybrid-control).

## The tracks

| track | question | metric | expected winner |
|---|---|---|---|
| **A** one-step | `x_t, u_t → x_{t+1}` | RMSE | tree surrogate |
| **B** rollout | `x_t, u_{t:t+H} → x_{t+1:t+H}` | rollout RMSE | hybrid (physics anchors the horizon) |
| **C** counterfactual | `x_{t+1}(do(u))` under confounding | \|effect − truth\| | causal / Double ML |
| **D** control | `u_t = π(x_t)` under constraints | regret vs oracle | causal hybrid controller |
| **D-planner** planner vs model | the same objective planned by a gradient and by a sampler, on three models | regret on the true plant | *neither* — the model axis decides it |
| **E** systems | control-solve latency | ms (min of 7, after 2 warm-ups) | the known-only model — fewer terms to differentiate, same compiled solver |
| **F** structure | which lagged parents drive the target, under confounding + autocorrelation | F1 / control payoff | discovery-informed residual |
| **G** dynamic effect | the impulse response `∂x_{t+h}/∂u_t`, not just `h = 1` | IRF error / control payoff | structured (Levinson) IRF |
| **H** marketplace | offline incentive allocation when SUTVA fails through a shared equilibrium | regret vs equilibrium-aware oracle | de-confounded + equilibrium-aware |
| **I** sensitivity | control when **no adjustment set exists** — the assumed `Γ` is the only lever | worst-case closed-loop cost | a *calibrated* `Γ`, not the largest one |
| **D-causal** identification | the control channel of a *real* emulator, logged by a weather-compensated controller | \|8h step response − randomised reference\| | orthogonal (de-confounded) fit |
| **J** identification, non-building | the control channel of a *third-party* plant whose answer is known exactly (`Pendulum-v1`) | \|fitted gain − 3.0\| | orthogonal (de-confounded) fit |
| **K** delay identification | *when* the incentive acts, from a log whose confounder acts at a **different** lag | \|τ̂ − τ\| / closed-loop regret | adjusted local projection |
| **N** fold design | which cross-fitting **split** to use under network interference — the method is held fixed and only the folds vary | MSE vs a graph-blind unit split | *the design law, negatively* — it convicts the two splits a practitioner reaches for |
| **M** allocation over time | media budget across channels **and** weeks, from a log whose planner chased the season | lift over doing nothing, audited on the true plant | *the identification axis* — and the horizon axis only when the two channel orderings conflict |
| **L** sequential intervention | which variables to set, and to what, at each step of DCBO's three synthetic dynamic SCMs | regret vs the per-step oracle, each step against the arm's own history | *the reference methods, on their own ground* — the track says where CHC cannot follow, and where DCBO cannot |
| **O** off-policy evaluation | does `evaluate_plan`'s 95% interval cover a feedback plan's online cost, estimated from another policy's logs, on two Gymnasium plants CHC did not write | interval coverage over 500 replicates, against ±2 points of nominal | *the interval, not a method* — the track is the 0.8.0 evidence gate |
| **Q** graph errors | the graph handed to `prescribe` is wrong: does the logger check flag it before a plan or its evaluation fails silently | silent-failure rate, with the flag read as a stop, over 200 replicates of nine worlds | *the check, not a method* — it earns its stop on one error of eight |

**Track L** (`causaldyn_bench.dcbo_track`) tests the "no analogues" claim in both directions
against DCBO (Aglietti et al., NeurIPS 2021), the nearest academic statement of *which lever, how
much, when*. Its three synthetic SCMs are ported from the equations, not the code — the
repository's `LICENSE` says MIT while its README and `setup.py` say GPL-3.0-or-later — and the
reference itself runs only in an isolated environment (`scripts/dcbo_reference.py`, GPy 1.13 on
NumPy 1.26), entering the bench as data. The port replays the outcomes the reference recorded for
its own decisions, except in three CBO runs on `nonstat` whose history the reference re-drew with
noise. Every arm reads the same log per seed and is scored on the noise-free SCM, each step against
the best response to the arm's own history. Over ten seeds, mean total regret (`just track-l`): on
`stat` the CHC arm ties DCBO — `0.42` against `0.50`, paired difference `−0.08 [−0.30, 0.12]` —
from the log alone, without one experiment; on `ind` (`5.88` against `2.54`) and `nonstat` (`6.48`
against `1.39`) it loses to every reference method, because `prescribe` cannot state those
problems. It tracks a set point where DCBO minimises, sets every lever at every step, fits one
control-affine, time-invariant transition — `ind`'s optimum is an interior bump, `nonstat` changes
its equations mid-horizon — and runs no experiments. Nor are the graph-blind baselines a formality:
at ten explorative interventions a step, ABO beats DCBO on both stationary SCMs (`0.06` and
`1.17`). The other way round, DCBO cannot run on Track M's confounded media log at all: its `Root`
takes the true SEM as the oracle its experiments query, and a log has none to give.

**Track O** (`causaldyn_bench.ope_calibration`, needs the `gym` extra) asks whether
`chc.evaluate_plan`'s interval covers on plants the library did not write: `Pendulum-v1` hanging and
`MountainCarContinuous-v0` at its valley floor, both stepped through their own `step`, with a
Gaussian disturbance on every command. Each replicate logs 4000 transitions of a lightly damped
legacy operator, fits a linear plant to them by least squares, and evaluates two LQR plans; the
truth is each plan's online average cost over 400 000 steps. Over 500 replicates (`just track-o`),
with the fitted model and the smoothing correction trusted, every arm is within two points of
0.95: `"dr"` 0.944–0.962, `"fqe"` 0.944–0.954, and `"mis"` 0.930–0.964, at the edge on two of the
four plans. The default `model_error = 1` covers 1.000 everywhere, by design. Three things the
gate does not show. On the car's aggressive plan the certificate refused 107 of the 500 logs, so
that coverage is over the 393 it passed. FQE reads 3–5% high on the car (3.4–3.8 standard errors)
and still covers. And with the logger clipping 2% of steps (the stress row), FQE on the pendulum
covers 0.900, a bias of 0.018 its certificate cannot see, while `"mis"` and `"dr"` hold. See
`results/track_o.md`.

**Track P** (`causaldyn_bench.drift_calibration`, needs the `gym` extra) asks whether `chc.gate`'s
channel-drift monitor keeps its false-alarm bound on the same two environments. Track O's moderate
plan runs through each environment's own `step` with a logged Gaussian dither of a quarter of the
bound and process noise on the state, and the monitor reads the dither against the environment's
linearisation, whose drift is wrong away from the equilibrium. The plan's offset holds its mean
command at 0.1 of the bound, where 0.1% of the actions clip, or at 0.8, where 23–26% do on an
unmoved channel, and a clipped decision is read with its draw. Over 300 runs per arm at `A = 10³` (`just track-p`), on
both environments, both plans and both nulls — the channel as modelled, and grown by exactly the
radius — at most 0.013 of the runs alarmed within 100 decisions, where `α = H/A = 0.1` is allowed,
and every run alarmed within `10 A`, after 1.77–2.57 `A` on average. A channel moved to 1.25 times
the model's was caught on every run, after 96–101 decisions on average with the plan inside the
bound and 204–223 with it on the bound. Setting the clipped decisions' e-values to 0 instead, which
is also valid, caught it on 0.7% and 2.0% of the runs within 4000 decisions. See
`results/track_p.md`.

**Track Q** (`causaldyn_bench.graph_errors`) hands `prescribe` a wrong graph and asks whether the
logger check (ADR 0028, experimental) turns the error into a flag before a claim fails silently. On
the lifecycle market, 400 units by 12 periods and 200 replicates a world (`just track-q`), a plan
fails when its true mean path leaves the tolerance within the steps its certificate trusts, or when
its true regret passes the certificate's bound by a tenth of the stakes; an evaluation fails when
its interval misses. The check's size is 0.046 over 1600 true-graph panels, [0.036, 0.057]. It
earns its stop, removing failures more often than it refuses sound claims over the true and the
wrong graph (one-sided Fisher), on one error of eight: a persistent promotion the graph omits, which
the logger reads and which moves supply, shows in the lever's own past and is flagged on every
panel, and the stop removes the failures of 195 of 213 replicates while refusing 12 of 187 sound
ones (`p = 2·10⁻⁷⁵`). Three errors fail silently on every replicate with the flag at its size: the
same promotion drawn afresh each period (the check tests the columns the graph names, so an
unnamed column is a hidden one); orders, a mediator the graph turns into a parent of the incentive,
an orientation inside one Markov equivalence class, where the fitted channel reads −0.030 against
0.080 and the plan's regret is 2.4 times doing nothing's; and sessions made a parent of the
incentive while a latent that moves supply also drives them, where the channel reads 0.000 and the
plan buys nothing, as it predicts. Where the flag fires on every panel it can cost: a logger that
chases demand, one that keeps half its last incentive, and sessions whose other parent is adjusted
too all leave the claims sound but one, and the stop refuses them all. A lever declared a
non-ancestor is inert, since `prescribe` fits every lever it is given: that arm reads exactly as
the oracle. Picking the adjustment set by fit ties the oracle in seven worlds and loses exactly
where the wrong graph does. And the oracle fails too, on 9% of the persistent-promotion
replicates: an adjustment covariate stays out of the fitted drift, where `chc.dynamics_id`'s scope
note says a persistent driver belongs (R19: the worlds are CHC's). See `results/track_q.md`.

**Track N** (`causaldyn_bench.fold_design`) is the only track that varies nothing but the
cross-fitting split, and its result is an ordering whose useful half is negative. On `C_12` with two
clusters, over 120 draws: the Result 52 design **ties** the graph-blind split that keeps units
intact, while the two splits a practitioner reaches for first cost **+37%** MSE (contiguous graph
blocks) and **+66%** (Emmenegger-style neighbour exclusion). One number explains all three — the
fraction of edges left inside a fold: `0.50` designed, `0.46` random units, `0.83` contiguous.

The law also forecasts *where* it matters, and paired bootstrap intervals (`just paper-2`) say the
forecast holds: its mass ratio designed/contiguous is `0.720` on the cycle, `0.974` on a `3×4` torus
and `0.966` on a random cubic graph, and each of those lands **inside** the 95% interval of the
realised MSE ratio on both coefficients — six cells for six. And the whole effect is `O(1/g)` in the
number of independent clusters (`1.370, 1.139, 1.083, 1.047` across `g = 2, 4, 8, 20`), so fold
design is a **small-cluster-count** instrument — a handful of cities, not twenty replicas. The
intervals put a boundary on that: at 120 draws the contiguous cost separates from the baseline only
at `g = 2`, and the exclusion cost only up to `g = 8`, so past a handful of clusters a bad split is
not merely cheap but undetectable. Neighbour exclusion is worse than every alternative where it runs
and cannot run at all at `K = 2` on either denser graph: its hop-1 neighbourhood covers the training
fold. Buying validity by discarding data needs a split that is already graph-aware — the design it
was meant to replace.

**Track M** (`causaldyn_bench.allocation`) scores the MMM case study as a 2×2 in *identified?* ×
*forward-looking?*: the CHC schedule, the **same identified fit spent on this week alone**, an equal
split, and a whole-horizon plan fitted observationally — all at matched budget, all audited on the
true plant. The result is not the one the track was built to show. Over eight seeds, mean lift over
doing nothing: `CHC-adjusted 46.43`, `myopic-greedy 46.11`, `equal-split 44.09`, `naive-MMM 36.49`.
Adjusting for the season is worth **+9.9 and wins 8 of 8**; looking past this week is worth **+0.3
with its sign flipping 3/5**, inside `7%` of the mean lift at every seed but one (`+7.3` at seed 4).
The CHC schedule beats the equal split at 8 of 8, the myopic rule at 7 of 8.

And the design that flips it is one line, which is what makes the null a measurement rather than an
absence. A myopic rule loses when the carryover ordering **contradicts** the immediate one — not
merely because carryover exists. On the shipped plant `β_c/θ_c` ranks the channels `(1.29, 1.50,
1.60)` against `γ_c`'s `(0.50, 0.20, 0.35)`: the two disagree about the top channel but *agree about
which to drop*, and dropping it is most of the available gain. Re-parameterise so the best immediate
channel is the worst carryover channel — `β/θ` of `(0.10, 8.00, 1.60)` at unchanged `γ`, so the
myopic ordering is untouched by construction — and the CHC schedule wins **8 of 8** by `1.20…3.79`
(`+9.5%`). The equal split then catches the optimiser, because concentration has become the error.

**Track M v2** (`causaldyn_bench.endogenous_mmm`) is the media-mix world CHC did not write:
Heusch's (2026b) generator, written clean-room from his paper (CC-BY-4.0; his code carries no
licence and none of it was read), where spend follows the business. A quarterly budget follows
sales, spend rises ahead of promotions, television bursts before Christmas, and paid shopping is bid
up after good weeks, while every parameter is known. Its tests hold his reference instance's
reported returns, shares and spend in distribution over 40 histories, and each media effect to
`chc.response`'s `Channel`. The first thing it scores is `chc.lift` (`just track-m2-lift`,
`causaldyn_bench.lift_calibration`): his (2026a) four go-dark tests of paid shopping, with noise of
a percent of mean weekly sales in each group, on a fresh world each history. Over 500 histories the
retention's 95% profile interval covered the truth in **0.954** (Clopper-Pearson 0.932-0.971), the
gate being 0.95 ± 0.02; the scale's in 0.976 and the coefficient's in 0.978, the scale's mostly
open upward, since his tests bend the curve little. The kill is measured, not argued: at three
times the noise the retention still covers, 0.958, but 15% of its intervals close, against 81% at
his; at ten times it under-covers, 0.878, and 4 fits of 500 raise rather than converge. On Meta at
his noise the retention covers 0.918 (0.890-0.941), under the nominal; on television, 0.958. By
construction the world is a reading of his paper (R19). See `results/track_m2_lift.md`.

Its second score is the decision (`just track-m2-budgets`, `causaldyn_bench.budget_regret`,
pre-registered in that module's docstring before any scored world ran): a quarter's budget over his
channels, each world's channels drawn from the hull of his table, and each arm's regret per euro
against the best plan in the box, carryover included. Over 200 drawn worlds CHC, reading each
channel from its own four go-dark tests and planning with `chc.allocation`, loses **0.050** per euro
[0.038, 0.062], PyMC-Marketing 1.2.0 with the same tests as lift measurements and its own optimiser
**0.048** [0.038, 0.058], the status quo 0.209, the equal split 0.218, the fit to the history alone
0.220 and the myopic plan 0.075. CHC beats every arm but PyMC-Marketing, each paired interval under
nought; against PyMC-Marketing it ties, **+0.002** [-0.011, +0.015], lower in 79 worlds and higher
in 78. The pre-registered gate, CHC below PyMC-Marketing, is not met, as predicted. On his reference
instance, whose best plan is a corner, the myopic plan and PyMC-Marketing both beat CHC. In 554 of
the 600 drawn channels the tests left the curve's bend unread. See `results/track_m2_budgets.md`.

Its third score is the check of an observational channel (`just track-m2-check`,
`causaldyn_bench.observational_check`, pre-registered in that module's docstring):
`chc.lift.check_observational` reads the channel his realistic specification fits to the history
against the four go-dark tests of it, an F test of whether the tests' gaps could be its own and the
factor of its predicted lift they read. Over 500 histories on paid shopping the F rejected the truth
in **0.054** [0.036, 0.078], the factor's interval covered 1 in 0.938 and the observational
channel's noise-free factor in 0.939, and the F rejected the observational channel in **0.998**:
the pre-registered gate, the truth rejected in at most 0.07, both intervals covering in at least
0.93 and the observational channel rejected in at least 0.90, is met. The pilot's F looked
conservative, 0.030 of 100; over 500 it holds its level. The tests read a median factor of 0.505 of
the observational channel's lift on paid shopping, a median least Γ of 2.75 at an effect-scale gap
of 1, and of 0.181 on Meta, a Γ of 7.82: the history alone overstates the two about two- and
fivefold. It reads no television in 430 of 500 histories. See `results/track_m2_check.md`.

Its fourth score is where a lift test runs (`just track-m2-geo`, `causaldyn_bench.geo_selection`,
pre-registered in that module's docstring): a panel of 40 geos over his market, whose spend per
head on paid shopping differs, so a set of heavy spenders reads the curve's bend and a set of light
ones little more than its slope. Each arm picks a set holding a tenth of the population for his
four go-dark tests, read against a synthetic control of the other geos, and the quarter is planned
on the channel the tests fit. The regret arm picks the set whose test
`chc.allocation.decision_weight` expects to leave the plan the least regret, its noise read on a
pre-period window apart from the one the expectation reads. Over 500 worlds its plan loses
**0.0050** per euro [0.0037, 0.0062], against **0.0143** for Abadie and Zhao's representative set,
0.0164 for the set the synthetic control fits best, 0.0241 for a random set and 0.0289 for the
heaviest spenders; paired, -0.0094 [-0.0128, -0.0059] against Abadie-Zhao's and -0.0191 [-0.0246,
-0.0136] against random. It took Abadie-Zhao's set in 1% of worlds, and its tests' readings stayed
within the placebo's band as often as a random set's, +0.004 [-0.004, +0.012]: the pre-registered
gate is met and its kill does not fire. The regret it expected its tests to leave, 0.0028 per euro,
under-states the 0.0050 they left. By construction the panel is this track's generator, not
Heusch's (R19). See `results/track_m2_geo.md`.

Its fifth score is the curve's family (`just track-m2-families`, `causaldyn_bench.family_regret`,
pre-registered in that module's docstring): the decision again, on worlds whose channels all follow
one of six curves, tanh, the exponential and Michaelis-Menten, concave from zero, and Hill, Weibull
and the logistic, S-shaped, each family on the same 100 seeds. Each channel's four go-dark tests are
fitted under all six. One arm plans on each channel's least-AIC family; the robust arm keeps every
family the tests cannot tell from the best and plans the split whose worst regret over them is
least, `chc.allocation.minimax_allocate`. Both do worst on the logistic worlds, the robust plan at
**0.389** per euro [0.308, 0.469] and AIC's at **0.370** [0.292, 0.448]; robust less AIC **+0.019**
[-0.009, +0.055], bootstrapped over the seeds. The pre-registered gate, the robust plan's worst below
AIC's, is not met; it was predicted met, narrowly. The robust plan's mean is above AIC's on all six
families, and below the status quo's on all six, each of those intervals under nought. Not
predicted: his tanh, planned whatever the world's curve, has the lowest mean of the arms on five
families of six, the S-shaped Hill (0.126 against AIC's 0.206) and Weibull (0.229 against 0.302)
among them; only on the logistic does AIC's choice do better (0.370 against 0.583). The run's first
launch stopped before any score was written, on a degenerate S-shaped reading the hedge's linear
program could not take; the library now builds that program in units of the largest best return,
and the run was launched again whole. See `results/track_m2_families.md`.

Its sixth score places CHC among the tools (`just track-m2-external`,
`causaldyn_bench.external_arms`, pre-registered in that module's docstring): Google's Meridian
2.1.0 and Meta's Robyn 3.12.1 plan the budgets run's quarter on its worlds, each fitted outside the
bench from the same export with the go-dark tests as its calibration, and each planning with its
own optimiser. Over the 200 drawn worlds Meridian loses **0.120** per euro [0.108, 0.131], and over
the first 100 Robyn **0.243** [0.196, 0.289]. CHC beats both, by -0.070 [-0.086, -0.054] and
-0.182 [-0.229, -0.134], and so does PyMC-Marketing, by -0.072 [-0.085, -0.058] and -0.191 [-0.238,
-0.145]: each as predicted. Meridian beats the status quo, -0.090 [-0.104, -0.075]; Robyn loses to
it, +0.046 [+0.012, +0.080], higher in 55 of 100 worlds, where a tie was predicted. Meridian's plans
keep inside the box where the best plan goes to its edges, 24 of 600 channel-plans on a bound
against the oracle's 347, and 199 of them were moved into the box by at most 0.00032 of a week's
budget. Robyn's own convergence check passed NRMSE in 54 of its fits and DECOMP.RSSD and MAPE in
none. On his reference instance Meridian loses 0.087, above CHC's 0.028 and PyMC-Marketing's
0.012. By construction (R19) two tools count less of a plan's carryover than the oracle scores:
Meridian's optimiser nothing the quarter's spend returns after it, Robyn's plan one steady week
whose carryover is the window's mean. See `results/track_m2_external.md`.

**Track K** (`causaldyn_bench.delay_identification`) is the only track whose payoff is
*discontinuous*. Every other board scores a cost gap; here the closed loop is `x' = -K·x(t − τ)`,
whose exact boundary is `K·τ = π/2`, so getting the delay wrong enough is a Hopf bifurcation rather
than a worse number. It also scores something no other track does — the **argmax** of an effect. The
confounder acts at 0.6 s while the incentive acts at 1.0 s, so an unadjusted impulse response peaks
on the *confounder's* lag: the estimate is wrong about **when**, not about how much.

The three tiers are the result. Ignoring the delay diverges (regret `18953`, gain `3.10` = 1.97× past
`π/2`); estimating it *badly* still stabilises and pays `0.68`, because the stabilising set in delay
space is a **half-line** — under-estimating survives down to `2/(πe) = 0.234`, and `0.6/1.0` is well
inside it; adjusting reaches `0.016`, and sub-grid refinement `0.0067`. Note the two ratios: a 4×
smaller delay error buys a **42×** smaller regret. Cross-correlation and an *unadjusted* local
projection are carried as separate rows and score identically — they are the same argmax, so the
board's gap is adjustment, not the estimator family. This track is not allowed to flatter CHC by
comparing its adjusted estimator against a weaker unadjusted one.

The track also states the limit of its own claim. Aggregated over one observation stride the
incentive splits 2:1 across lags 3 and 4 while the confounder lands wholly in lag 2, so the peak
relocates iff `σ_η² < 1.5·|c·κ|·σ_z²/|b| − κ²·σ_z²` — `σ_η = 1.4697` here, bracketed by measurement
in `[1.45, 1.50]`, and the 2:1 split is visible in the fitted response (`0.168` at lag 3, `0.085` at
lag 4). So **enough exploration in the logging policy recovers the delay with no adjustment at all**.
The failure is a property of a thin log, not of cross-correlation; the shipped `σ_η = 0.5` sits
deliberately below the threshold.

Track I is the odd one out on purpose: it scores a **modelling assumption**, not a method. The
confounder is absent from the log, so nothing can be estimated better; the board carries a
deliberately under-assumed and a deliberately over-assumed `Γ` beside the calibrated one, and the
score is non-monotone in `Γ`. "More pessimism is better" is a claim this track exists to refute.

Track D bundles the CHC oracle-regret tasks (pricing / inventory / support-shift) **and** an
**adaptive-CV-compute** task (`adaptive_cv`): split a shared GPU budget across video streams under
known, bursty arrivals and heterogeneous priorities. A priority-blind, load-proportional myopic split
crowds out critical streams; the constrained CHC-MPC plans over the known dynamics and matches the
oracle. First numbers — CHC-MPC regret `0.0`, myopic `166`, uniform `330`.

**Track D-planner** (`causaldyn_bench.shooting`) exists to let CHC lose. Cross-entropy-method
planning needs no adjoint, no Jacobian and no differentiable model — only rollout evaluations —
so crossing it with the gradient planner over three models (true plant / learned hybrid /
physics-only) separates what the library's adjoint machinery is worth from what *learning the
residual* is worth. It is not close, and not in the flattering direction for the adjoint: regret
on the true plant is `plant/gradient` 0, `hybrid/gradient` 0.0016, `plant/cem` 0.0019,
`hybrid/cem` 0.0030, against `known_only/*` at **1.72**. The model axis is ~1000× the planner
axis, and the sampling planner with full knowledge of the plant is *worse* than the gradient
planner on a learned model. The defensible claim is therefore about identification and the
learned residual, not about the solver.

Design rule: to be honest about the win, **never** claim "best model" — claim the decision under a stated
budget. Track A is expected to go to the trees; the value is Tracks B–D.

## Run

```bash
uv sync --extra trees --extra gym        # tree baselines for A/B, Gymnasium for Tracks J, O and P
uv run python -m causaldyn_bench         # print the leaderboard
uv run python -m causaldyn_bench --save  # also write results/leaderboard.{md,json}
uv run pytest                            # smoke tests
just check                               # the ladder ci.yml runs: format, lint, tests
just track-l                             # Track L against DCBO -> results/track_l.{md,json}
just track-o                             # Track O, OPE coverage -> results/track_o.{md,json}
just track-p                             # Track P, drift false alarms -> results/track_p.{md,json}
just track-q                             # Track Q, graph errors -> results/track_q.{md,json}
just track-m2-lift                       # Track M v2, lift intervals -> results/track_m2_lift.{md,json}
just track-m2-budgets-pilot              # Track M v2, budgets, the pilot -> results/track_m2_budgets_pilot.{md,json}
just track-m2-budgets                    # Track M v2, budgets, pre-registered -> results/track_m2_budgets.{md,json}
just track-m2-check-pilot                # Track M v2, observational check, the pilot -> results/track_m2_check_pilot.{md,json}
just track-m2-check                      # Track M v2, observational check, pre-registered -> results/track_m2_check.{md,json}
just track-m2-geo-pilot                  # Track M v2, geo selection, the pilot -> results/track_m2_geo_pilot.{md,json}
just track-m2-geo                        # Track M v2, geo selection, pre-registered -> results/track_m2_geo.{md,json}
just track-m2-families-pilot             # Track M v2, curve families, the pilot -> results/track_m2_families_pilot.{md,json}
just track-m2-families                   # Track M v2, curve families, pre-registered -> results/track_m2_families.{md,json}
just track-m2-external-pilot genre       # Track M v2, external arms, the pilot -> results/track_m2_external_pilot_genre.{md,json}
just track-m2-external                   # Track M v2, external arms, pre-registered -> results/track_m2_external.{md,json}
```

Every recipe runs on the CPU, the device the committed results came from; a GPU reproduces them to
rounding, not bit for bit. Track M v2's arms fitted outside the bench are tied to their worlds by a
digest of the worlds' exact bits, which were exported on x86 without AVX-512: where numpy runs its
AVX-512 kernels for float64 `exp`, `log` and `tanh`, the last bits differ (CI's faster runners), and
the scoring refuses those records rather than score them on other worlds. `just sync` adds JAX's build for this machine's accelerator, CUDA 13 or
12 as `nvidia-smi` reports the driver, and `just test` runs the tests on it. The accelerator
extras, `cuda13`, `cuda12`, `cuda13-local`, `cuda12-local`, `rocm7-local`, `tpu` and `oneapi`, are
the library's, which are JAX's. Python 3.11–3.15, the free-threaded 3.14t and 3.15t included.

### Paper tables

Every table in papers P1 ("Debias every channel"), P2 ("Fold design for cross-fitting on networks
and panels") and P3 ("Information-exploration duality") comes out of one command, so a number in
the manuscript can be traced to a run rather than to a transcription:

```bash
just paper-1         # -> results/paper1/tables.{md,json}; hours, run it detached
just paper-1-smoke   # 30 seeds and two windows: plumbing only, resolves no ladder
just paper-2         # -> results/paper2/tables.{md,json}; hours, run it detached
just paper-2-smoke   # the same pipeline at 12 draws: plumbing only, quotes nothing
just paper-3         # -> results/paper3/tables.{md,json}; minutes
just paper-3-smoke   # one seed and a short ladder: plumbing only
```

P1's headline numbers are all exponents fitted to log-log sweeps, so every table names the window
it was fitted on. Its organising finding is that a fitted slope is not the exponent: the
order-transfer certificate's `2.05 / 4.01 / 6.00` against a theoretical `2 / 4 / 6` is not
agreement-up-to-noise but the exponent plus a **closed-form window term**, and Table 1 reconstructs
that term from the plant (Maxima-derived, then gated against the certificate) and reports what is
left. Two terms cut the `2.83e-2` miss at `p = 1` to `7.6e-3`. The same window has a lower end that
fails catastrophically rather than gracefully -- push the sweep to `delta in [1e-5, 2e-4]` and the
three-channel full-orthogonality slope reads `nan`, because `delta^4` regret has underflowed to
exactly zero -- so every cell carries the double-precision cancellation floor beside it. The two
`G` ladders come with the only error bar the certificate surface admits -- a chain of *nested
prefixes* in which rung `j` carries a known multiple of the wanted variance -- and each caption
**states whether its walk clears that scale instead of assuming it does**. On the first full run
both refused: reading a single rung gave a scale of `0.0475` at 240 seeds and `0.0665` at 960,
larger after four times the work, because one draw of `|N(0, sigma^2)|` decides nothing. Pooling
three rescaled rungs on the same budget turned the refusals into `4.9x` and `3.1x`.

P2's ratios are mean squared error against the random-unit split **on the same draws**, and the
interval is a **paired** percentile bootstrap: one resampled index set applied to numerator and
denominator. That matters because the headline comparison (designed against random units) is a
near-tie -- an unpaired interval would report draw noise that cancels in the ratio and turn "these
two tie" into "we cannot tell". The quadrature table is relative max-entry error throughout, stated
in its header.

P3 runs the same discipline in the opposite direction: four of its five tables are closed-form or
exact-quadrature functions of the model -- a minimax floor, a digamma sum, the root of a quadratic --
and carry **no** intervals, because a band around an exact number invents uncertainty. The fifth is
a range over seeds with a relative-spread column, which is how the run revealed that its headline
alignment factor moves `2.83 .. 3.64` across seeds while the bracket it illustrates does not. The
plant constants the tables need are not exposed by any certificate, so they are reconstructed and
**gated** against two identities that are; `plant_constants` raises rather than returning a silent
second copy.

This repo depends on `causal-hybrid-control` through a sibling **path**, so it expects the two checked
out next to each other. CI reproduces that layout with two checkouts and runs the full suite.

Without the `trees` extra the tree baselines are simply omitted; the hybrid/causal methods still run.
On Tracks A/B the dynamics competitors are **known-only** (true physics), **dlm** (data-driven linear /
state-space), **tree-surrogate** (LightGBM), and **hybrid-CHC** (physics + learned residual).

A committed snapshot lives in [`results/leaderboard.md`](results/leaderboard.md): hybrid wins B-rollout
~18× over the tree and ~14× over the DLM, causal wins C ~800× over naive, causal-CHC is near-oracle on
D while predictive blows up, and CHC-MPC matches the oracle on the adaptive-CV task (myopic loses).

For the visual version — leaderboard bar charts + the "prediction ≠ decision" figure — see the executed
notebook [`notebooks/leaderboard.ipynb`](notebooks/leaderboard.ipynb) (renders on GitHub), or run it:

```bash
uv sync --extra trees --group notebooks
uv run --group notebooks jupyter lab   # notebooks/leaderboard.ipynb
```

## Running BOPTEST — the real Track-D HVAC target (Fedora / Podman)

Track D can run against a live **BOPTEST-Service** via `causaldyn_bench.boptest`. Current BOPTEST deploys
as a web-service; on Fedora it runs under **Podman** (no Docker needed):

```bash
# one-time tooling (podman ships with Fedora; add the compose front-end, no sudo)
uv tool install podman-compose               # or: sudo dnf install -y podman-compose

# clone + bring up the service (first build is ~4 GB, ~15-30 min)
git clone https://github.com/ibpsa/project1-boptest.git
cd project1-boptest

# SELinux: relabel the bind mount, or the containers cannot read the tree they are handed.
# Fedora mounts with SELinux enforcing and the upstream compose file predates rootless Podman,
# so without the :z suffix the worker fails on permission errors that name no cause.
sed -i 's|- ./:/usr/src/boptest$|- ./:/usr/src/boptest:z|' docker-compose.yml

podman-compose up web worker provision       # REST API at http://127.0.0.1:8000
curl http://127.0.0.1:8000/version           # sanity-check once it is up
```

Then point the client at it (from this repo):

```bash
BOPTEST_URL=http://127.0.0.1:8000 uv run pytest tests/test_boptest.py   # the live episode test runs
BOPTEST_URL=http://127.0.0.1:8000 uv run python -c \
  "from causaldyn_bench.boptest import boptest_track; print(boptest_track())"   # baseline KPIs
```

Shut down with `podman-compose down`. Test cases include `bestest_hydronic_heat_pump` (default),
`bestest_air`, `singlezone_commercial_hydronic`, and others. The CHC hybrid-MPC controller
(RC-thermal + learned residual, MPC under comfort constraints) is wired in via
`causaldyn_bench.boptest_chc` — see the results below.

Two operational notes, both learned the hard way. The `mc` bucket-init container exits non-zero on a
second bring-up (the bucket already exists), which blocks `podman start` of anything that depends on
it — `podman-compose down && podman-compose up -d` is the reliable cycle. And the worker runs **one
test at a time**: a client killed with `SIGTERM` skips its `finally: client.stop(testid)`, leaks the
registration in redis, and every later `select` then blocks until the leak is cleared
(`redis-cli KEYS 'tests:*'`). Run long sweeps detached, not under a timeout that kills them.

### Track D-causal — does de-confounding the control channel pay on a real emulator?

`causaldyn_bench.boptest_causal` asks the question the existing harness cannot. `boptest_chc`
identifies its thermal model from a **randomised** exploration episode: a clean experiment, and the
one thing production HVAC data never is. Real logs come from a controller, and every sensible
controller is weather-compensated, so the logged action is a function of the outdoor temperature —
which is also what drives the zone. Regress the temperature rate on `(1, T, u)` and the outdoor term
lands in the error term.

The experiment is a 2×2 over {outdoor-reset, randomised PRBS} × {adjust for weather, don't}, so both
directions are falsifiable rather than only the flattering one: adjustment must repair the
confounded arm **and** must leave the randomised arm alone. If it "helped" the randomised arm too,
the estimator would be distorting rather than de-confounding.

```bash
# JAX_ENABLE_X64=1 is required, not cosmetic. float32 and float64 agree on the *affine* channel to
# 0.6% -- two orders of magnitude inside the effect -- but 3000 Adam steps compound rounding into
# the derivative of the fitted surface, and the physics-off arm's decay moves by 10x and changes
# sign. An earlier float32 run is what produced the retracted numbers in results/ SS6.
JAX_ENABLE_X64=1 BOPTEST_URL=http://127.0.0.1:8000 uv run python -c \
  "from causaldyn_bench.boptest_causal import track_boptest_causal as t; print(t())"
```

Three things this track deliberately does not claim:

- **No ground-truth channel exists** on an emulator. The reference is identification *by design* —
  the randomised log, fitted without adjustment — not a known number.
- **The steady-state gain `−b/a` is not identified** from a 5–20 day window at 30-minute resolution;
  `b₀` and the pole are collinear over that span. The reported quantity is the finite-horizon
  8-hour step response, which the data does pin down.
- **Closed-loop KPIs cannot rank models at a single operating point.** Which way a channel error
  moves the controller is a property of the cost, not of the error: under BOPTEST's
  comfort-dominated objective an *attenuated* channel makes the controller over-actuate, so the bias
  acts as an unintended safety margin that buys comfort and pays energy. `run_pareto` sweeps the
  requested margin and compares frontiers instead.

Identification also needs **overlap**. A perfectly deterministic reset policy makes the action an
exact function of the covariates, the orthogonal moment has no regressor left, and nothing is
identified at any sample size. `overlap_report` measures the surviving share and
`run_overlap_ablation` drives the exploration noise to zero to show the collapse — the assumption
gets a falsifiable curve, not a sentence.

**Physics-off ablation** (`run_structure_ablation`). Against the structured arms sits a black box: an
MLP for `dT/dt` given `(T, z, u)`, trained on the same log, planning through the *same* MPC — the two
arms differ in the model and in nothing else, because the solver takes any `PlantModel` and reads
only its `rate`. It is not a straw man: the confounder is inside its conditioning set, it can fit
nonlinearities the affine model cannot, and it gets far more fitting compute than a closed-form fit.

The point of it is what identification benchmarks usually miss. On the synthetic fixture the black
box's held-out one-step error is **indistinguishable** from the structured causal fit — 4.20e-4
against 4.13e-4, under half a seed standard deviation, both on the 4.0e-4 noise floor — while its
control authority carries **2.6× the RMSE** (0.047 against 0.018 about a truth of 1.200) and a 3.4%
bias against 0.07%. Two models that agree on every forecast they will ever be scored on disagree
measurably on the one number a controller consumes. *Held-out predictive accuracy does not rank
causal models*, so a dynamics leaderboard reported in rollout error cannot see this failure at all.
The other side is tested too: omitting the confounder entirely — the `naive` affine arm — attenuates
the authority to 14% of truth **and** costs 19× the held-out error, and prediction *does* catch that.

The reported estimand is the **authority** `∂(dT/dt)/∂u` at the operating point, not the affine
`b₀`. `b₀` is that channel extrapolated to 0 °C, some 21 K outside anything a heated building
visits; reading it off a nonlinear model measures the extrapolation, and on the emulator it came back
between −0.045 and +0.254 for a plant whose structured fit says +1.25 — a factor of 31, while the two
arms' *authorities* differ by 1.5×.

On the emulator the black box's failure is blunter than on the fixture. Read at the action the log
sat at, its fitted decay is **positive on three of five seeds**: those models assert a zone that
warms away from its own equilibrium, the stability check refuses them offline, and the arm has no
closed-loop mean left to report. Of the two seeds that do plan, one reaches the comfort floor and the
other spends 2.2× the de-confounded arm's discomfort. Sharper still, *within* the black-box arm
prediction is ordered against plannability: of its five fits the **best** one-step predictor
(held-out 0.0301, against 0.0971 for the worst) is one of the three that gets refused.

**Measured result** (5 seeds, 20-day identification episodes, `results/boptest_causal.md`): logged by
a weather-compensated controller the naive fit understates the heat pump's 8-hour authority by
**54.8%**; de-confounding recovers it to within **0.211 K** of the randomised reference, an 11.5×
reduction, and moves the randomised arm by **0.068 K** — 1.5% of the level — while cutting that arm's
spread by 2.3×. Closed loop, the de-confounded arm has both the best mean discomfort (7.295 K·h
against 7.892) and a **51× tighter** seed-to-seed spread (s.e. 0.011 against 0.539), at 0.2% more
energy and roughly half the actuator saturation.

The confounded arm's failure is sharper than attenuation: on seed 1 its fitted channel
`+2.989 − 0.1321·T` changes **sign at 22.62 °C**, inside the occupied comfort band, so above that
temperature the model believes the heat pump cools the room and pins the command at zero. The
de-confounded channel crosses at 25.82 °C, outside the band.

**The certificate closes the loop, and separates the arms before it acts.** Wrapping every MPC horizon
in a `chc.plan.CausalPlan` and pricing it with `certify_safety` (§9) says the confounded fit is
uncertifiable at **any** sensitivity level on 15.2% and 21.7% of control steps, against 7.4% and 7.7%
for the de-confounded one — a 2–3× separation read off the plan, with nothing executed. Enforcing it
through `robust_safety_filter` then moves **one command in 336**: where the barrier is in deficit a
comfort-dominated MPC is already saturated, so the two agree everywhere except the last occupied
half-hour before the night setback, which the planner's horizon cannot score and the barrier can.

An earlier version of this paragraph reported the opposite closed-loop ordering. That was an artefact
of the planner's constant step size, which sat 75–284× past its stability limit — and because the
limit scales with the authority a model *believes* it has, the shared constant punished exactly the
arms that identified the channel best. Equal compute is not an equal iteration count; it is an
optimiser that does not depend on the scale of the model being compared. See §7 of
`results/boptest_causal.md`.

## Status

Tracks A–E run on the synthetic CHC systems (a damped oscillator with hidden cubic
physics for A/B/E, a confounded linear system for C, the CHC oracle-regret tasks plus the
**adaptive-CV-compute** task for D). A **BOPTEST** (HVAC control) client + control episode ship in
`causaldyn_bench.boptest`, gated on a running BOPTEST service (`BOPTEST_URL`). The CHC identification +
forecast-MPC (`causaldyn_bench.boptest_chc`) is **validated live** on `bestest_hydronic_heat_pump`: it
beats the tuned built-in baseline on discomfort, energy, cost and emissions at once, with a peak
electrical demand 2% higher (see `results/boptest.md`).
**Track D-causal** (`causaldyn_bench.boptest_causal`) adds the falsifiable 2×2 that the randomised-only
harness could not ask, plus a physics-off black-box arm planning through the identical MPC —
`results/boptest_causal.md`.

**Track J** (`causaldyn_bench.pendulum_causal`, needs the `gym` extra) re-asks that 2×2 on a plant
that is **not a building** and whose answer is known exactly: Gymnasium's `Pendulum-v1`, whose
`step` fixes the control channel at `3/(m l²) = 3.0`. Adjustment recovers it to seven digits from a
log where the unadjusted fit returns `−1.053 ± 0.025` — the **sign backwards on all five seeds** —
and closes the loop within 3.1% of an oracle that the confounded controller misses by a factor of
4800 while saturating the actuator on every step. Held-out one-step error ranks all of it backwards.
Reproducing the seven digits needs `JAX_ENABLE_X64=1`; at JAX's default float32 the recovery is exact
to six. See `results/pendulum_causal.md`.

```python
from causaldyn_bench.boptest import BOPTestClient, baseline_controller, run_episode

kpis = run_episode(
    BOPTestClient("http://127.0.0.1:8000"), baseline_controller()
)  # needs a live service
```

## License

Apache-2.0 © Ilia Gradina, with a [`NOTICE`](NOTICE). Releases up to 0.1.0 stay under MIT.
