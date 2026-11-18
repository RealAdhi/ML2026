"""EDA figures and Folium maps. Partition used is stated in every title/file name.
Outputs: results/eda/*.png, results/maps/eda_train_trip_*.html
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.data import load_trip
from pathsnap.viz import trip_map

eda = os.path.join(ROOT, "results", "eda"); maps = os.path.join(ROOT, "results", "maps")
os.makedirs(maps, exist_ok=True)
inv = pd.read_csv(os.path.join(eda, "dataset_inventory.csv"))

# ---- inventory plots
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for part, c in (("train", "#1565c0"), ("student_test", "#ef6c00")):
    d = inv[inv.partition == part]
    ax[0, 0].hist(d.n_gps, bins=25, alpha=.6, color=c, label=part)
    ax[0, 1].hist(np.log10(d.n_arcs), bins=20, alpha=.6, color=c, label=part)
    ax[0, 2].hist(d.n_components, bins=25, alpha=.6, color=c, label=part)
    ax[1, 0].hist(d.mean_step_m, bins=20, alpha=.6, color=c, label=part)
    ax[1, 1].scatter(d.frac_dt1, d.n_gap_gt5s + 1, alpha=.7, color=c, label=part)
    ax[1, 2].hist(np.log10(d.extent_x_m.clip(lower=1)), bins=20, alpha=.6, color=c, label=part)
ax[0, 0].set(title="GPS points per trip", xlabel="points"); ax[0, 1].set(title="Road arcs per trip", xlabel="log10(arcs)")
ax[0, 2].set(title="Connected components per trip", xlabel="components")
ax[1, 0].set(title="Mean step between fixes", xlabel="metres"); ax[1, 1].set(title="Sampling regularity", xlabel="share of 1-s steps", ylabel="1 + #gaps >5 s", yscale="log")
ax[1, 2].set(title="East-west extent of trajectory", xlabel="log10(metres)")
for a in ax.ravel(): a.legend()
fig.suptitle("Dataset inventory (train: 80 trips, student_test: 10 trips)"); fig.tight_layout()
fig.savefig(os.path.join(eda, "inventory.png"), dpi=130); plt.close(fig)

# ---- missing/invalid summary
miss = inv.groupby("partition").agg(trips=("trip", "count"), nan_values=("n_nan", "sum"),
                                    non_monotonic_time=("monotonic_t", lambda s: int((~s).sum())),
                                    trips_with_gaps_gt5s=("n_gap_gt5s", lambda s: int((s > 0).sum())),
                                    isolated_nodes=("n_isolated_nodes", "sum"),
                                    trips_with_duplicate_arcs=("n_duplicate_arc_pairs", lambda s: int((s > 0).sum())),
                                    max_gap_s=("max_dt", "max"))
miss.to_csv(os.path.join(eda, "missing_value_summary.csv")); print(miss.T)

# ---- trajectory + network + reference plots for a few trips; dist-to-reference diagnostics
tr = joblib.load(os.path.join(ROOT, "data", "processed", "train.pkl"))
df, info = tr["df"], tr["info"]
trips = ["00000001", "00000003", "00000004", "00000002"]
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for a, tid in zip(axes.ravel(), trips):
    t = load_trip(os.path.join(ROOT, "train"), tid)
    near = np.unique(df[df.trip == tid].arc.values)
    for arc in near:
        u, v = t.arcs[arc]; a.plot([t.px[0] * 0 + t.node_x[u], t.node_x[v]], [t.node_y[u], t.node_y[v]], color="#bdbdbd", lw=.6)
    for arc in t.route:
        u, v = t.arcs[arc]; a.plot([t.node_x[u], t.node_x[v]], [t.node_y[u], t.node_y[v]], color="#ff9800", lw=2.5, alpha=.7)
    a.plot(t.px, t.py, ".", ms=2, color="#1565c0")
    a.set_aspect("equal"); a.set_title(f"train trip {tid}: GPS (blue), reference route (orange), nearby arcs (grey)", fontsize=9)
    a.set_xlabel("local x (m)"); a.set_ylabel("local y (m)")
fig.tight_layout(); fig.savefig(os.path.join(eda, "trajectories_train.png"), dpi=130); plt.close(fig)

# distance from each fix to its aligned reference arc, per trip
from pathsnap.candidates import arc_xy, seg_dist
rows = []
for tid in sorted(info):
    t = load_trip(os.path.join(ROOT, "train"), tid)
    ax_, ay_, bx_, by_ = arc_xy(t); ref = info[tid]["ref_arc"]
    d, _ = seg_dist(t.px, t.py, ax_[ref], ay_[ref], bx_[ref], by_[ref])
    rows.append(dict(trip=tid, median_m=np.median(d), p95_m=np.percentile(d, 95), frac_gt60m=float((d > 60).mean())))
q = pd.DataFrame(rows); q.to_csv(os.path.join(eda, "gps_to_reference_distance_train.csv"), index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(q.median_m, bins=25); ax[0].set(title="Median GPS-to-reference distance per trip (train)", xlabel="m")
ax[1].hist(q.frac_gt60m, bins=25); ax[1].set(title="Share of fixes >60 m from reference arc (train)", xlabel="share", yscale="log")
fig.tight_layout(); fig.savefig(os.path.join(eda, "gps_to_reference.png"), dpi=130); plt.close(fig)

# ---- Folium EDA maps (training partition): GPS + network + reference
for tid in ("00000001", "00000004"):
    t = load_trip(os.path.join(ROOT, "train"), tid)
    m = trip_map(t, network_arcs=df[df.trip == tid].arc.values, ref_arcs=t.route, title=f"EDA - train trip {tid}")
    m.save(os.path.join(maps, f"eda_train_trip_{tid}.html"))
print("done")
