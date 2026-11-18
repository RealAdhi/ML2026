"""Folium prediction maps for student-test trips (GPS, road network, reference route, classifier-only, classifier + HMM).

usage: python scripts/08_maps.py [trip ids ...]       default: every student-test trip
Outputs: results/maps/pred_student_<trip>.html
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pathsnap.candidates import generate
from pathsnap.data import list_trips, load_trip
from pathsnap.pipeline import Predictor
from pathsnap.viz import trip_map

root = os.path.join(ROOT, "student_test")
ids = sys.argv[1:] or list_trips(root)
pred = Predictor(os.path.join(ROOT, "models", "final"))
os.makedirs(os.path.join(ROOT, "results", "maps"), exist_ok=True)
for tid in ids:
    trip = load_trip(root, tid, with_route=True)   # reference is used only for drawing the comparison layer
    out = pred.predict(trip)
    cand, _ = generate(trip, with_labels=False)
    m = trip_map(trip, network_arcs=cand.arc.values, ref_arcs=trip.route, clf_arcs=out["arc_clf"], hmm_arcs=out["arc_hmm"],
                 title=f"student-test trip {tid}")
    path = os.path.join(ROOT, "results", "maps", f"pred_student_{tid}.html")
    m.save(path)
    print("wrote", path, flush=True)
