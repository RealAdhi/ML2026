"""EDA overview: trip locations, trip length/duration, network connectivity and two-way road share.
Writes everything to results/eda_extra/.  Uses only .track/.arcs files and existing results/eda CSVs."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "eda_extra")
os.makedirs(OUT, exist_ok=True)

EDA = os.path.join(ROOT, "results", "eda")
PARTITIONS = ("train", "student_test")
COLOUR = {"train": "#1565c0", "student_test": "#ef6c00"}
SUMMARY_COLS = ["partition", "trip", "n_gps", "duration_h", "path_km", "mean_speed_kmh",
                "centroid_lon", "centroid_lat", "arc_reverse_share", "n_arcs", "n_components",
                "max_gap_s", "candidate_miss"]


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between consecutive points (arrays)."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def track_stats(track_file):
    """.track = 3 whitespace columns without header: lon, lat, time_seconds."""
    t = np.loadtxt(track_file, ndmin=2)
    lon, lat, ts = t[:, 0], t[:, 1], t[:, 2]
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    dt = np.diff(ts)
    path_km = step.sum() / 1000.0
    dur_h = (ts[-1] - ts[0]) / 3600.0
    return dict(n_gps=len(ts), path_km=path_km, duration_h=dur_h,
                mean_speed_kmh=path_km / dur_h if dur_h > 0 else np.nan,
                centroid_lon=float(np.median(lon)), centroid_lat=float(np.median(lat)), max_gap_s=float(dt.max()))


def reverse_share(arcs_file):
    """Share of arcs (a->b) for which the opposite arc (b->a) also exists. .arcs = 2 int columns: from_node, to_node."""
    a = np.loadtxt(arcs_file, dtype=np.int64, ndmin=2)
    n = a.max() + 1
    keys = a[:, 0] * n + a[:, 1]
    rkeys = a[:, 1] * n + a[:, 0]
    return float(np.isin(rkeys, keys).mean())


def trips(partition):
    folder = os.path.join(ROOT, partition)
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))


# ---- 1. one row per trip -> trip_summary.csv
def build_summary():
    rows = []
    for part in PARTITIONS:
        for tid in trips(part):
            base = os.path.join(ROOT, part, tid, tid)
            row = dict(partition=part, trip=tid)
            row.update(track_stats(base + ".track"))
            row["arc_reverse_share"] = reverse_share(base + ".arcs")
            rows.append(row)
            print(part, tid, row["n_gps"], f"{row['path_km']:.1f} km", flush=True)
    df = pd.DataFrame(rows)

    inv = pd.read_csv(os.path.join(EDA, "dataset_inventory.csv"), dtype={"trip": str})
    df = df.merge(inv[["partition", "trip", "n_arcs", "n_components"]], on=["partition", "trip"], how="left")

    miss = pd.read_csv(os.path.join(EDA, "candidate_miss_by_trip_train.csv"), dtype={"trip": str})
    miss = miss[["trip", "miss"]].rename(columns={"miss": "candidate_miss"})
    df = df.merge(miss, on="trip", how="left")
    # candidate misses were only computed for train; keep the column empty for student_test
    df.loc[df.partition != "train", "candidate_miss"] = np.nan

    df = df[SUMMARY_COLS].sort_values(["partition", "trip"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "trip_summary.csv"), index=False)
    print(f"trip_summary.csv: {len(df)} rows", flush=True)
    return df


# ---- 2. where in the world each trip was recorded -> trip_locations.html
def trip_locations(df):
    import folium

    m = folium.Map(tiles="OpenStreetMap")
    for part in PARTITIONS:
        d = df[df.partition == part]
        layer = folium.FeatureGroup(name=f"{part} ({len(d)} trips)", show=True)
        for r in d.itertuples():
            folium.CircleMarker(
                location=[r.centroid_lat, r.centroid_lon], radius=6,
                color=COLOUR[part], fill=True, fill_color=COLOUR[part], fill_opacity=0.75, weight=1,
                popup=folium.Popup(
                    f"<b>trip {r.trip}</b><br>partition: {part}"
                    f"<br>GPS points: {r.n_gps:,}<br>path length: {r.path_km:.1f} km", max_width=250),
                tooltip=f"{part} {r.trip}",
            ).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    counts = df.partition.value_counts()
    legend = (
        '<div style="position:fixed; bottom:25px; left:25px; z-index:9999; background:white;'
        ' padding:10px 14px; border:2px solid #777; border-radius:5px; font:14px sans-serif;">'
        '<b>Trip centroids</b><br>'
        + "<br>".join(f'<span style="color:{COLOUR[p]};font-size:20px">&#9679;</span> {p}'
                      f' ({counts.get(p, 0)} trips)' for p in PARTITIONS)
        + '</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[df.centroid_lat.min(), df.centroid_lon.min()],
                  [df.centroid_lat.max(), df.centroid_lon.max()]])
    m.save(os.path.join(OUT, "trip_locations.html"))
    print("trip_locations.html written", flush=True)


# ---- 3. the four figures
def figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 3a. how far and how long is each trip
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = df[(df.path_km > 0) & (df.duration_h > 0)]
    for part in PARTITIONS:
        d = pos[pos.partition == part]
        ax.scatter(d.duration_h, d.path_km, s=45, alpha=.75, color=COLOUR[part],
                   edgecolor="k", linewidth=.4, label=f"{part} (n={len(d)})")
    # the three longest trips sit almost on top of each other, so park the labels in the free
    # top-left corner at fixed axes positions and point a leader line at each one
    for i, r in enumerate(pos.nlargest(3, "path_km").itertuples()):
        ax.annotate(f"trip {r.trip} ({r.path_km:.0f} km)",
                    xy=(r.duration_h, r.path_km), xycoords="data",
                    xytext=(0.03, 0.97 - 0.05 * i), textcoords="axes fraction",
                    ha="left", va="top", fontsize=8,
                    arrowprops=dict(arrowstyle="-", lw=.6, color="#555", shrinkB=4))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("trip duration (hours, log scale)")
    ax.set_ylabel("path length along the GPS trace (km, log scale)")
    ax.set_title("Trip length against duration: trips far to the right have long\ntime gaps, not long drives", fontsize=11)
    ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trip_length_vs_duration.png"), dpi=150); plt.close(fig)

    # 3b. network size against fragmentation (straight from the committed inventory)
    inv = pd.read_csv(os.path.join(EDA, "dataset_inventory.csv"), dtype={"trip": str})
    fig, ax = plt.subplots(figsize=(8, 6))
    for part in PARTITIONS:
        d = inv[inv.partition == part]
        ax.scatter(d.n_arcs, d.n_components, s=45, alpha=.75, color=COLOUR[part],
                   edgecolor="k", linewidth=.4, label=f"{part} (n={len(d)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("road arcs in the trip's network (log scale)")
    ax.set_ylabel("connected components in that network (log scale)")
    ax.set_title("Bigger road networks are more fragmented, so no trip\nnetwork can be assumed connected", fontsize=11)
    ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "network_size_vs_components.png"), dpi=150); plt.close(fig)

    # 3c. how much of each network is two-way (one panel per partition)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for axis, part in zip(axes, PARTITIONS):
        d = df[df.partition == part]
        axis.hist(d.arc_reverse_share, bins=np.linspace(0, 1, 21), color=COLOUR[part],
                  alpha=.8, edgecolor="k", linewidth=.4)
        axis.axvline(d.arc_reverse_share.median(), color="k", ls="--", lw=1.2,
                     label=f"median {d.arc_reverse_share.median():.2f}")
        axis.set_title(f"{part} (n={len(d)})")
        axis.set_xlabel("share of a trip's arcs that have a reverse twin")
        axis.set_ylabel("number of trips")
        axis.grid(alpha=.3); axis.legend()
    fig.suptitle("Most arcs come in opposing pairs: distance alone cannot tell the two directions apart", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "two_way_share.png"), dpi=150); plt.close(fig)
    print("figures written", flush=True)


# ---- 4. the hardest training trips -> trip_difficulty_ranking.csv
def difficulty_ranking(df):
    hard = (df[df.partition == "train"]
            .sort_values("candidate_miss", ascending=False)
            .head(15)
            .reset_index(drop=True))
    hard.to_csv(os.path.join(OUT, "trip_difficulty_ranking.csv"), index=False)
    print(f"trip_difficulty_ranking.csv: {len(hard)} hardest training trips", flush=True)


def main():
    df = build_summary()
    trip_locations(df)
    figures(df)
    difficulty_ranking(df)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
