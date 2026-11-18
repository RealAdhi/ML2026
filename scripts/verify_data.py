"""Check that the raw data folders match the committed inventory (results/eda/dataset_inventory.csv).

usage (repo root):  python scripts/verify_data.py
Counts the lines of every .track / .nodes / .arcs / .route file of every trip in train/ and student_test/ and compares
them with the numbers recorded when the dataset was first inspected. Use it after copying the data into place.
"""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
inv = pd.read_csv(os.path.join(ROOT, "results", "eda", "dataset_inventory.csv"), dtype={"trip": str})


def n_lines(path):
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


bad = 0
for r in inv.itertuples():
    base = os.path.join(ROOT, r.partition, r.trip, r.trip)
    want = {"track": r.n_gps, "nodes": r.n_nodes, "arcs": r.n_arcs, "route": r.n_route_arcs}
    for ext, n in want.items():
        p = f"{base}.{ext}"
        if not os.path.exists(p):
            print("MISSING", p); bad += 1
        elif n_lines(p) != n:
            print(f"MISMATCH {p}: {n_lines(p)} lines, inventory says {n}"); bad += 1
for part, n in (("train", 80), ("student_test", 10)):
    folder = os.path.join(ROOT, part)
    got = len([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]) if os.path.isdir(folder) else 0
    print(f"{part}: {got} trip folders (expected {n})")
    bad += int(got != n)
print("DATA OK: every file matches the inventory" if bad == 0 else f"{bad} problem(s) found")
sys.exit(1 if bad else 0)
