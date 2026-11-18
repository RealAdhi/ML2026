"""Extra EDA figures from the cached candidate tables (training partition only).

Reads data/processed/train.pkl (made by 01_build_dataset.py). Writes PNGs and one CSV to results/eda/.
"""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
eda = os.path.join(ROOT, "results", "eda")
df = joblib.load(os.path.join(ROOT, "data", "processed", "train.pkl"))["df"].reset_index(drop=True)
df["gid"] = pd.factorize(df["trip"] + "_" + df["obs"].astype(str))[0]
reach = df.groupby("gid")["label"].transform("max") > 0          # observation's reference arc is a candidate
first = df[df.d_rank == 0].copy()                                # one row per observation: the nearest arc
first["reach"] = reach[first.index].values

# 1. where does the reference arc sit in the distance ranking, and how good is "nearest arc"?
pos = df[(df.label == 1)]
rank_share = pos.d_rank.value_counts(normalize=True).sort_index()
bins = [0, 2, 4, 6, 8, 10, 15, 20, 30, 60]
first["dbin"] = pd.cut(first.d, bins, include_lowest=True)
acc_d = first[first.reach].groupby("dbin", observed=True).label.agg(["mean", "size"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(rank_share.index, rank_share.values, color="#1565c0")
ax[0].set(title="Distance rank of the reference arc (train, reachable fixes)", xlabel="rank among candidates (0 = nearest arc)",
          ylabel="share of fixes", xticks=range(12))
for r, v in rank_share.items():
    ax[0].text(r, v + .005, f"{v:.1%}", ha="center", fontsize=7)
ax[1].bar(range(len(acc_d)), acc_d["mean"], color="#ef6c00")
ax[1].set_xticks(range(len(acc_d))); ax[1].set_xticklabels([str(i) for i in acc_d.index], rotation=30, fontsize=7)
ax[1].set(title="Is the nearest arc correct? By fix-to-nearest-arc distance", xlabel="distance to nearest arc (m)", ylabel="nearest-arc accuracy", ylim=(0, 1))
for i, (m, n) in enumerate(zip(acc_d["mean"], acc_d["size"])):
    ax[1].text(i, m + .01, f"n={n:,}", ha="center", fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(eda, "nearest_arc_baseline.png"), dpi=130); plt.close(fig)
nearest_acc = first[first.reach].label.mean()

# 2. speed and sampling gaps per observation
sp = first.speed.dropna()
dt = first.dt_prev.dropna()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(sp.clip(upper=45) * 3.6, bins=60, color="#26a69a")
ax[0].set(title="Vehicle speed per fix (train, from consecutive fixes)", xlabel="km/h (clipped at 162)", ylabel="fixes")
ax[1].hist(dt.clip(upper=30), bins=np.arange(0.5, 31.5, 1), color="#8d6e63")
ax[1].set(title="Time since previous fix (train)", xlabel="seconds (clipped at 30)", ylabel="fixes", yscale="log")
fig.tight_layout(); fig.savefig(os.path.join(eda, "speed_and_gaps.png"), dpi=130); plt.close(fig)

# 3. per-trip candidate miss rate and candidates per fix
trip_obs = first.groupby("trip").agg(n=("obs", "size"), miss=("reach", lambda s: 1 - s.mean()))
trip_obs["n_cands_mean"] = df.groupby("trip").n_cands.mean()
trip_obs.sort_values("miss", ascending=False).to_csv(os.path.join(eda, "candidate_miss_by_trip_train.csv"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
top = trip_obs.sort_values("miss", ascending=False).head(12)
ax[0].bar(top.index.str[-3:], top.miss, color="#c62828")
ax[0].set(title="Trips with the highest candidate miss rate (train)", xlabel="trip id", ylabel="share of fixes with no candidate on the reference")
ax[1].hist(first.n_cands, bins=np.arange(0.5, 13.5, 1), color="#5c6bc0")
ax[1].set(title="Candidates per fix (cap = 12)", xlabel="candidate arcs within 60 m", ylabel="fixes")
fig.tight_layout(); fig.savefig(os.path.join(eda, "candidate_miss_by_trip.png"), dpi=130); plt.close(fig)

print("nearest-arc accuracy (reachable fixes):", round(nearest_acc, 4))
print("nearest-arc accuracy (all fixes):", round(first.label.mean(), 4))
print("rank shares:", rank_share.round(4).to_dict())
print("accuracy by distance bin:\n", acc_d.round(4))
print("speed km/h percentiles 5/50/95/99:", (np.percentile(sp, [5, 50, 95, 99]) * 3.6).round(1))
print("share of fixes with dt==1:", round((dt == 1).mean(), 4), " dt>5:", round((dt > 5).mean(), 4))
print("mean candidates per fix:", round(first.n_cands.mean(), 2), " share with 12:", round((first.n_cands == 12).mean(), 3))
print("overall miss:", round(1 - first.reach.mean(), 4))
print(trip_obs.sort_values("miss", ascending=False).head(6))
