"""Label-free inference pipeline: raw trip files -> candidates -> classifier -> HMM/Viterbi route.

This module never reads reference routes. Scoring against references lives in `scoring.py`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from .candidates import FEATURES, generate
from .data import Trip, load_trip
from .decode_eval import top1_arcs
from .hmm import DecoderParams, precompute, viterbi


class Predictor:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "model.joblib")
        cfg = json.loads((model_dir / "decoder.json").read_text())
        self.params = DecoderParams(**cfg["decoder_params"])
        self.model_name = cfg.get("model_name", "model")

    def predict(self, trip: Trip) -> dict:
        t0 = time.time()
        df, _ = generate(trip, with_labels=False)
        t1 = time.time()
        p = self.model.predict_proba(df[FEATURES].values.astype(np.float32))[:, 1]
        t2 = time.time()
        obs, arc_clf = top1_arcs(df, p)
        pre = precompute(trip, df)
        arc_hmm = viterbi(pre, p, self.params)
        t3 = time.time()
        return dict(obs=obs, arc_clf=arc_clf, arc_hmm=arc_hmm,
                    timing=dict(candidates_s=t1 - t0, classifier_s=t2 - t1, hmm_s=t3 - t2, total_s=t3 - t0))


def predict_trip_dir(predictor: Predictor, root, trip_id: str) -> tuple[Trip, dict]:
    trip = load_trip(root, trip_id, with_route=False)   # reference route deliberately not loaded
    return trip, predictor.predict(trip)
