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
channel's effect drift (:mod:`~causaldyn_bench.scorecard.drift`).
"""
