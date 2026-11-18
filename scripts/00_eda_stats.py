"""Dataset inventory and data-quality statistics -> results/eda/*.csv"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from collections import Counter
import networkx as nx
from pathsnap.data import list_trips, load_trip

ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = []
for part in ("train", "student_test"):
    for tid in list_trips(os.path.join(ROOT, part)):
        tr = load_trip(os.path.join(ROOT, part), tid)
        dt = np.diff(tr.t)
        step = np.hypot(np.diff(tr.px), np.diff(tr.py))
        key = Counter(map(tuple, tr.arcs))
        n_nodes = len(tr.node_x)
        # connectivity (undirected, via scipy for speed)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        A = coo_matrix((np.ones(len(tr.arcs)), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(n_nodes, n_nodes))
        ncomp, lab = connected_components(A, directed=False)
        deg = np.bincount(tr.arcs.ravel(), minlength=n_nodes)
        rt = tr.route
        chain = float(np.mean(tr.arcs[rt[:-1], 1] == tr.arcs[rt[1:], 0])) if rt is not None and len(rt) > 1 else np.nan
        rows.append(dict(
            partition=part, trip=tid, n_gps=len(tr.t),
            n_nan=int(np.isnan(tr.lon).sum() + np.isnan(tr.lat).sum() + np.isnan(tr.t).sum()),
            monotonic_t=bool((dt > 0).all()), frac_dt1=float((dt == 1).mean()), max_dt=float(dt.max()),
            n_gap_gt5s=int((dt > 5).sum()), mean_step_m=float(step.mean()), max_step_m=float(step.max()),
            lon_min=tr.lon.min(), lon_max=tr.lon.max(), lat_min=tr.lat.min(), lat_max=tr.lat.max(),
            extent_x_m=float(np.ptp(tr.px)), extent_y_m=float(np.ptp(tr.py)),
            n_nodes=n_nodes, n_arcs=len(tr.arcs), n_duplicate_arc_pairs=int(sum(v - 1 for v in key.values())),
            n_components=int(ncomp), n_isolated_nodes=int((deg == 0).sum()),
            n_route_arcs=0 if rt is None else len(rt), route_chain_ok=chain,
        ))
        print(part, tid, len(tr.t), flush=True)
os.makedirs(os.path.join(ROOT, "results", "eda"), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "eda", "dataset_inventory.csv"), index=False)
print(df.groupby("partition").describe().T.to_string())
