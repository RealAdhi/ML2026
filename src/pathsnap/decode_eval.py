"""Evaluate classifier-only vs classifier+HMM predictions at observation and sequence level."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm import DecoderParams, Precomputed, precompute, viterbi


def seq_metrics(pred_arc: np.ndarray, ref_arc: np.ndarray, obs: np.ndarray, arcs: np.ndarray, route: np.ndarray):
    """Per-trip metrics for a predicted arc per observation."""
    acc = float(np.mean(pred_arc == ref_arc))
    sel = np.diff(obs) == 1
    a0, a1 = pred_arc[:-1][sel], pred_arc[1:][sel]
    cont = float(np.mean((a0 == a1) | (arcs[a0, 1] == arcs[a1, 0]))) if sel.any() else np.nan
    ref, pr = set(route.tolist()), set(pred_arc.tolist())
    tp = len(ref & pr)
    p_ = tp / max(len(pr), 1); r_ = tp / max(len(ref), 1)
    f1 = 0.0 if tp == 0 else 2 * p_ * r_ / (p_ + r_)
    sw = int(np.sum(a0 != a1))  # number of arc switches along the predicted path
    return dict(acc=acc, continuity=cont, route_f1=f1, route_precision=p_, route_recall=r_, n_switch=sw,
                ref_switch=int(np.sum(ref_arc[:-1][sel] != ref_arc[1:][sel])))


def top1_arcs(df_trip: pd.DataFrame, p: np.ndarray):
    d = df_trip.reset_index(drop=True)[["obs", "arc"]].copy(); d["p"] = p
    idx = d.groupby("obs")["p"].idxmax()
    top = d.loc[idx].sort_values("obs")
    return top["obs"].values, top["arc"].values
