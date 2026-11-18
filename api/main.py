"""PathSnap FastAPI backend (runs on Laptop 1).

Environment variables
    EVAL_DATA_DIR  folder with the trips to serve (same layout as train/: <id>/<id>.track|nodes|arcs|route)
                   default: ./student_test
    MODEL_DIR      folder with model.joblib + decoder.json   default: ./models/final

Reference routes are used only inside `scoring.score` after the prediction is made; they are never given to the
prediction pipeline and only aggregate metrics are returned to the UI.
"""
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.data import load_trip
from pathsnap.pipeline import Predictor, predict_trip_dir
from pathsnap.scoring import score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DATA_DIR = Path(os.environ.get("EVAL_DATA_DIR", ROOT / "student_test"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "final"))

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = Predictor(MODEL_DIR)
TRIP_IDS = sorted(p.name for p in EVAL_DATA_DIR.iterdir() if p.is_dir()) if EVAL_DATA_DIR.exists() else []
log.info("model=%s data_dir=%s trips=%d", predictor.model_name, EVAL_DATA_DIR, len(TRIP_IDS))


class PredictRequest(BaseModel):
    trip_id: str


def _arc_polylines(trip, arcs):
    """lon/lat polyline (as [lat, lon] pairs) of each decoded arc, consecutive duplicates collapsed."""
    seq = [int(arcs[0])]
    for a in arcs[1:]:
        if int(a) != seq[-1]:
            seq.append(int(a))
    lines = []
    for a in seq:
        u, v = trip.arcs[a]
        lines.append([[float(trip.node_lat[u]), float(trip.node_lon[u])],
                      [float(trip.node_lat[v]), float(trip.node_lon[v])]])
    return seq, lines


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.model_name, "n_trips": len(TRIP_IDS), "data_dir": str(EVAL_DATA_DIR)}


@app.get("/trips")
def trips():
    out = []
    for t in TRIP_IDS:
        trip = load_trip(EVAL_DATA_DIR, t, with_route=False)
        out.append({"trip_id": t, "n_gps": int(len(trip.t)),
                    "center": [float(np.median(trip.lat)), float(np.median(trip.lon))]})
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in TRIP_IDS:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    log.info("POST /predict trip_id=%s", req.trip_id)
    trip, pred = predict_trip_dir(predictor, EVAL_DATA_DIR, req.trip_id)
    metrics = score(EVAL_DATA_DIR, req.trip_id, pred)          # server-side only
    seq, lines = _arc_polylines(trip, pred["arc_hmm"])
    _, lines_clf = _arc_polylines(trip, pred["arc_clf"])
    gps = [[float(a), float(b)] for a, b in zip(trip.lat, trip.lon)]
    resp = {
        "trip_id": req.trip_id, "model": predictor.model_name, "n_obs": int(len(pred["obs"])),
        "gps": gps, "predicted_route": lines, "classifier_only_route": lines_clf,
        "predicted_arc_ids": seq, "metrics": metrics, "timing": pred["timing"],
    }
    m = metrics.get("hmm", {})
    log.info("trip=%s obs=%d hmm_acc=%.4f clf_acc=%.4f continuity=%.4f time=%.2fs", req.trip_id, resp["n_obs"],
             m.get("acc", float("nan")), metrics.get("clf", {}).get("acc", float("nan")),
             m.get("continuity", float("nan")), time.time() - t0)
    return resp
