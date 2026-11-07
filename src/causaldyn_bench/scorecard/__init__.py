"""The head-to-head scorecard: families of worlds whose truth is known, each scored the same way.

A *family* (:mod:`causaldyn_bench.scorecard.family`) draws a world from its own seeds
(:mod:`~causaldyn_bench.scorecard.seeds`), says what an arm may read of it
(:mod:`~causaldyn_bench.scorecard.observe`), with the lift tests a rung of the ladder gives it
(:mod:`~causaldyn_bench.scorecard.ladder`), and holds what the arm's plan is scored against: the
quarter's best plan (:mod:`~causaldyn_bench.scorecard.truth`) and the quarter the world goes on to
run at the status quo (:mod:`~causaldyn_bench.scorecard.continuation`).
:mod:`~causaldyn_bench.scorecard.scoring` scores an arm's record of a world against it, and
:mod:`~causaldyn_bench.scorecard.gate` reads two arms' paired scores against a margin at looks fixed
before the run starts, stopping the comparison where its verdict is plain.
:mod:`~causaldyn_bench.scorecard.mapping` reads an arm's fitted response as :mod:`chc.response`'s
channels, once they reproduce the tool's own decomposition of the history.

Family 0 is Track M v2 itself (:mod:`~causaldyn_bench.scorecard.track_m2`); family 1 lets each
channel's effect drift, or the price of its impressions (:mod:`~causaldyn_bench.scorecard.drift`,
:mod:`~causaldyn_bench.scorecard.cpm`); family 2 sets the spend by the
sentiment no arm observes, or paid shopping's alone beside a query volume every arm reads
(:mod:`~causaldyn_bench.scorecard.confounded`); family 3 moves two
channels' spend together (:mod:`~causaldyn_bench.scorecard.collinear`); family 4 makes every curve
S-shaped (:mod:`~causaldyn_bench.scorecard.s_curves`); family 5 sets the spend by last week's sales
(:mod:`~causaldyn_bench.scorecard.performance`); family 6 changes every curve outside the range the
history shows (:mod:`~causaldyn_bench.scorecard.outside`); family 7 lets an arm read only the last
weeks of a history (:mod:`~causaldyn_bench.scorecard.short`); family 8 is ten geos drawn by
:mod:`chc.geo_world`, planned geo by geo and tested by geo experiments
(:mod:`~causaldyn_bench.scorecard.geo`); family 9 is family 8 with each geo's own media lifting
its ring neighbours too (:mod:`~causaldyn_bench.scorecard.spillover`); family 10 is family 8 at
three geos, at eight and at fifty small, noisy ones (:mod:`~causaldyn_bench.scorecard.few_geos`);
family 12 is Meridian's home
ground, its media in Meridian's own form at its default priors
(:mod:`~causaldyn_bench.scorecard.meridian`); family 13 is PyMC-Marketing's home ground, its media
in PyMC-Marketing's default form at its default adstock and saturation priors
(:mod:`~causaldyn_bench.scorecard.pymc_marketing`); family 14 is Robyn's home ground, its media in
Robyn's own form (:mod:`~causaldyn_bench.scorecard.robyn`); family 15 is
siMMMulator's own worlds, Meta's simulator run as its documentation's demo
(:mod:`~causaldyn_bench.scorecard.simmmulator`); family 17 is a pair of twins one history cannot
tell apart, a drifting line and a fixed S-curve, whose best plans differ
(:mod:`~causaldyn_bench.scorecard.twins`); family 18 exports a brand measure through which nothing
lasts in sales (:mod:`~causaldyn_bench.scorecard.long_term`); family 19 holds one channel's spend
flat (:mod:`~causaldyn_bench.scorecard.flat`); and family 20 sets the promotions by television's
bursts, with the price a lever (:mod:`~causaldyn_bench.scorecard.promotions`). Families 11 and 16
draw no worlds. :mod:`~causaldyn_bench.scorecard.lift_inputs` writes a world's lift tests as the
files a tool calibrates on, and :mod:`~causaldyn_bench.scorecard.variance` measures what common
random numbers and control variates remove of a comparison's variance.
"""
