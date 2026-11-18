"""Server-side scoring of predictions against the reference route (never exposed to the UI)."""
from __future__ import annotations

import numpy as np

from .candidates import reference_alignment
from .data import load_trip
from .decode_eval import seq_metrics


def score(root, trip_id: str, pred: dict) -> dict:
    trip = load_trip(root, trip_id, with_route=True)
    if trip.route is None:
        return {"available": False}
    _, ref_arc = reference_alignment(trip)
    ref = ref_arc[pred["obs"]]
    out = {"available": True, "n_obs": int(len(pred["obs"]))}
    for key in ("clf", "hmm"):
        m = seq_metrics(pred[f"arc_{key}"], ref, pred["obs"], trip.arcs, trip.route)
        out[key] = {k: float(v) for k, v in m.items()}
    return out
