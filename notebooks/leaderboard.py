# %% [markdown]
# # causaldyn-bench — scoring *decisions*, not one-step error
#
# Most ML benchmarks rank models by one-step accuracy, where gradient-boosted trees are very hard to
# beat. But a good **prediction** is not a good **decision**: decisions play out over a horizon, under
# constraints, and depend on the *interventional* effect of the action — axes where trees, naive causal
# reads, and myopic policies fall down. `causaldyn-bench` scores every competitor on the axis it deserves:
#
# | track | question | metric |
# |---|---|---|
# | **A** one-step | `x_t,u_t → x_{t+1}` | RMSE |
# | **B** rollout | `x_t,u_{t:t+H} → x_{t+1:t+H}` | rollout RMSE |
# | **C** counterfactual | `x_{t+1}(do(u))` under confounding | \|effect − truth\| |
# | **D** control | `u_t = π(x_t)` under constraints | regret vs oracle |
# | **D** adaptive-CV | split a GPU budget across video streams | regret vs oracle |
# | **E** systems | control-solve latency | ms |

# %%
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

from causaldyn_bench import run_all, to_frame

results = run_all()  # trains the dynamics models + runs every track once (~1 min)
frame = to_frame(results)
frame

# %% [markdown]
# ## One figure per track
#
# Each track is sorted on its own metric (lower is better everywhere here). The regret tracks span
# several orders of magnitude, so they use a log scale with a small floor to keep the near-zero bars
# visible.

# %%
COLOR = {
    "hybrid-CHC": "#54A24B", "known-only": "#72B7B2", "dlm": "#F2A900", "tree-surrogate": "#E45756",
    "double-ml-CHC": "#54A24B", "backdoor-CHC": "#88B04B", "naive": "#E45756",
    "CHC-MPC": "#54A24B", "myopic": "#E45756", "uniform": "#B279A2",
}


def plot_track(ax, track, log=False, floor=1e-3):
    sub = frame[frame["track"] == track].sort_values("value")
    vals = sub["value"].clip(lower=floor) if log else sub["value"]
    colors = [COLOR.get(m.split("/")[-1], "#4C78A8") for m in sub["method"]]
    ax.barh(range(len(sub)), vals, color=colors)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([m.replace("/", "\n") for m in sub["method"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"{track}  ({sub['metric'].iloc[0]})", fontsize=10)
    if log:
        ax.set_xscale("log")


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
plot_track(axes[0, 0], "A-onestep")
plot_track(axes[0, 1], "B-rollout")
plot_track(axes[0, 2], "C-effect", log=True)
plot_track(axes[1, 0], "D-control", log=True)
plot_track(axes[1, 1], "D-adaptive-cv", log=True)
plot_track(axes[1, 2], "E-systems")
fig.suptitle("causaldyn-bench leaderboard — each competitor scored on the axis it deserves", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## The headline: a good predictor is not a good model to *plan* with
#
# The tree and the data-driven linear model (DLM) are competitive — even best — on **one-step** error.
# But roll them forward over a horizon and they drift, because neither respects the known physics. The
# **hybrid** model (known dynamics + a small learned residual) is anchored and dominates the rollout —
# which is what a controller actually uses.

# %%
dyn = ["known-only", "dlm", "tree-surrogate", "hybrid-CHC"]
a = {r.method: r.value for r in results if r.track == "A-onestep"}
b = {r.method: r.value for r in results if r.track == "B-rollout"}
x = np.arange(len(dyn))
w = 0.38
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(x - w / 2, [a[m] for m in dyn], w, label="one-step RMSE (Track A)", color="#BAB0AC")
ax.bar(x + w / 2, [b[m] for m in dyn], w, label="rollout RMSE (Track B)", color="#4C78A8")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(dyn)
ax.set_ylabel("RMSE (log scale)")
ax.set_title("Trees/DLM keep pace one-step, then drift; the hybrid wins the horizon")
ax.legend()
for i, m in enumerate(dyn):
    ax.text(i + w / 2, b[m], f" {b[m]:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.show()

print("rollout RMSE — hybrid vs the best non-hybrid:")
best_other = min(b[m] for m in dyn if m != "hybrid-CHC")
print(f"  hybrid {b['hybrid-CHC']:.3f}  vs  {best_other:.3f}  ->  {best_other / b['hybrid-CHC']:.0f}x lower")

# %% [markdown]
# ## The decision tracks: prediction ≠ intervention ≠ myopia
#
# On the control tracks the gap is not subtle. Under confounding, the naive effect is the wrong sign, so
# a predictive controller drives the system away from target (pricing regret ~13,700 vs ~0). And a
# priority-blind, load-proportional GPU split lets cheap streams crowd out critical ones, where the
# constrained CHC-MPC matches the oracle.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
d = frame[frame["track"] == "D-control"].sort_values("value")
ax1.barh(range(len(d)), d["value"].clip(lower=1e-3),
         color=["#54A24B" if "causal" in m or "oracle" in m or "pessim" in m else "#E45756" for m in d["method"]])
ax1.set_yticks(range(len(d)))
ax1.set_yticklabels([m.replace("/", "\n") for m in d["method"]], fontsize=8)
ax1.invert_yaxis()
ax1.set_xscale("log")
ax1.set_xlabel("regret vs oracle (log)")
ax1.set_title("Track D — causal/pessimistic control vs predictive/greedy")

cv = frame[frame["track"] == "D-adaptive-cv"].sort_values("value")
ax2.bar(cv["method"], cv["value"].clip(lower=1e-3), color=[COLOR[m] for m in cv["method"]])
ax2.set_yscale("log")
ax2.set_ylabel("regret vs oracle (log)")
ax2.set_title("Adaptive-CV — constrained CHC-MPC vs myopic / uniform")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## The moat's first real win: BOPTEST HVAC control (`bestest_hydronic_heat_pump`)
#
# Everything above is synthetic. This is not. **BOPTEST** is the standard building-control emulator, and
# the task — hold thermal comfort at least cost on a hydronic heat-pump house — is one the authors do not
# control. The comfort band relaxes to 15 °C while the house is empty, then snaps back to 21 °C for the
# evening return (hour ~20). A controller that tracks only the *current* setpoint lets the house coast
# cold and then can't recover — the heat pump moves the zone only ~0.4 K per 30-min step. The CHC
# **forecast-MPC** sees the ramp coming and **pre-heats**. Trajectories below are replayed from
# `results/` (captured live; reproduce with `boptest_chc.compare_controllers` against a running service).

# %%
import json  # noqa: E402

bo = json.load(open("../results/boptest_trajectory.json"))
cmp = json.load(open("../results/boptest_compare.json"))
traces = bo["traces"]
TCOLOR = {"baseline": "#72B7B2", "chc-naive": "#E45756", "chc-mpc": "#54A24B"}
LABEL = {
    "baseline": "baseline (BOPTEST)",
    "chc-naive": "CHC naive (setpoint-track)",
    "chc-mpc": "CHC forecast-MPC",
}

fig, (axT, axU) = plt.subplots(
    2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
h = traces["chc-mpc"]["hour"]
axT.fill_between(
    h, traces["chc-mpc"]["lower"], traces["chc-mpc"]["upper"],
    step="post", color="#BAB0AC", alpha=0.30, label="comfort band",
)
for name in ("baseline", "chc-naive", "chc-mpc"):
    axT.plot(traces[name]["hour"], traces[name]["tzon"], color=TCOLOR[name], lw=2.2,
             marker="o", ms=3, label=LABEL[name])
axT.set_ylabel("zone temperature (°C)")
axT.set_title("Zone temperature vs the comfort band — naive coasts cold, CHC-MPC pre-heats the ramp")
axT.legend(loc="lower center", ncol=2, fontsize=8)

for name in ("chc-naive", "chc-mpc"):
    axU.step(traces[name]["hour"], traces[name]["action"], where="post",
             color=TCOLOR[name], lw=1.8, label=LABEL[name])
axU.set_ylabel("heat-pump\nmodulation u")
axU.set_xlabel("hour of the day")
axU.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# The forecast-MPC's modulation ramps up *before* the evening comfort return, so its temperature line
# stays in the band while the naive controller — tracking the relaxed daytime setpoint — coasts down and
# is caught cold. On the standard KPIs this is not a trade-off but a **domination**: less discomfort *and*
# less energy at once.

# %%
names = ["baseline", "chc-naive", "chc-mpc"]
tdis = [cmp[n]["tdis_tot"] for n in names]
ener = [cmp[n]["ener_tot"] for n in names]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax2 = ax.twinx()
x = np.arange(len(names))
w = 0.36
ax.bar(x - w / 2, tdis, w, color="#4C78A8")
ax2.bar(x + w / 2, ener, w, color="#F2A900")
ax.set_xticks(x)
ax.set_xticklabels([LABEL[n] for n in names], fontsize=8)
ax.set_ylabel("thermal discomfort (K·h)")
ax2.set_ylabel("energy")
ax.set_title("CHC forecast-MPC beats the tuned baseline on BOTH discomfort and energy")
for i, v in enumerate(tdis):
    ax.text(i - w / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
for i, v in enumerate(ener):
    ax2.text(i + w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#4C78A8", "#F2A900")]
ax.legend(handles, ["discomfort (K·h)", "energy"], loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

print("BOPTEST bestest_hydronic_heat_pump — CHC forecast-MPC vs the tuned baseline:")
for k in ("tdis_tot", "ener_tot", "cost_tot", "emis_tot"):
    b, m = cmp["baseline"][k], cmp["chc-mpc"][k]
    print(f"  {k:9} baseline {b:.3f}  ->  CHC-MPC {m:.3f}  ({100 * (m - b) / b:+.0f}%)")

# %% [markdown]
# ## Scope & honesty (read this before quoting the numbers)
#
# The **synthetic** tracks above exercise each axis, and the CHC-style method wins the decision tracks
# partly *by construction of the data-generating processes* — a strong **demonstration and regression
# harness** and a clean statement of the *framing* (score decisions, not one-step error), but not a moat
# on their own. The **BOPTEST** section is the first step past that: a real, method-agnostic emulator the
# authors do not control, where the CHC forecast-MPC dominates the tuned baseline on every KPI. What is
# still owed for a full moat is **external adoption** and more real tasks; the framing and the
# reproducible, multi-competitor comparison are the defensible contribution today.
