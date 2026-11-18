"""Tune the HMM/Viterbi decoder and compare classifier-only vs classifier+HMM.

Decoder parameters are tuned (Optuna TPE) on out-of-fold classifier scores of a sample of TRAINING trips
(maximising pointwise accuracy). The student-test partition is used only to report the result.
Outputs: results/hmm/{decoder_trials.csv,per_trip_train.csv,per_trip_student.csv,summary.csv}
         models/final/decoder.json (decoder_params filled in)
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.data import load_trip
from pathsnap.decode_eval import seq_metrics, top1_arcs
from pathsnap.hmm import DecoderParams, precompute, viterbi

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=50)
ap.add_argument("--model", default="xgboost", help="tuned model whose out-of-fold / test scores feed the decoder")
ap.add_argument("--tune-trips", type=int, default=30)
args = ap.parse_args()
out = os.path.join(ROOT, "results", "hmm"); os.makedirs(out, exist_ok=True)


def prep(part, tid, df_t, p_t):
    trip = load_trip(os.path.join(ROOT, part), tid)
    return tid, trip, precompute(trip, df_t), df_t, p_t


def load_part(part, proba_file, key):
    d = joblib.load(os.path.join(ROOT, "data", "processed", f"{part}.pkl"))
    df = d["df"].reset_index(drop=True); info = d["info"]
    p = joblib.load(proba_file)[key]
    items = []
    for tid, g in df.groupby("trip", sort=True):
        items.append((tid, g.index.values))
    res = Parallel(n_jobs=8)(delayed(prep)(part, tid, df.loc[idx], p[idx]) for tid, idx in items)
    return res, info


def run(res, info, prm):
    rows = []
    for tid, trip, pre, df_t, p_t in res:
        obs, a_clf = top1_arcs(df_t, p_t)
        ref = info[tid]["ref_arc"][obs]
        a_h = viterbi(pre, p_t, prm)
        mc = seq_metrics(a_clf, ref, obs, trip.arcs, trip.route)
        mh = seq_metrics(a_h, ref, obs, trip.arcs, trip.route)
        rows.append(dict(trip=tid, n=len(obs), **{f"clf_{k}": v for k, v in mc.items()}, **{f"hmm_{k}": v for k, v in mh.items()}))
    return pd.DataFrame(rows)


t0 = time.time()
tr_res, tr_info = load_part("train", os.path.join(ROOT, "results", "bayes", args.model, "oof_train_tuned.pkl"), "oof")
te_res, te_info = load_part("student_test", os.path.join(ROOT, "results", "bayes", args.model, "student_test_tuned.pkl"), "proba")
print("precomputed in", round(time.time() - t0), "s", flush=True)
rng = np.random.RandomState(0)
sample = [tr_res[i] for i in sorted(rng.choice(len(tr_res), args.tune_trips, replace=False))]
# drop the two trips that are mostly off the reference route from the tuning objective (label noise)
sample = [s for s in sample if s[0] not in ("00000054", "00000033")]


def objective(trial):
    prm = DecoderParams(
        w_emit=1.0,
        w_dist=trial.suggest_float("w_dist", 0.0, 2.0),
        sigma=trial.suggest_float("sigma", 5.0, 40.0),
        beta=trial.suggest_float("beta", 5.0, 120.0, log=True),
        w_trans=trial.suggest_float("w_trans", 0.1, 3.0, log=True),
        break_pen=trial.suggest_float("break_pen", 1.0, 20.0),
        uturn_pen=trial.suggest_float("uturn_pen", 0.0, 10.0),
        gap_s=trial.suggest_float("gap_s", 5.0, 60.0))
    df = run(sample, tr_info, prm)
    return float((df.hmm_acc * df.n).sum() / df.n.sum())


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7, n_startup_trials=10))
study.optimize(objective, n_trials=args.trials)
study.trials_dataframe().to_csv(os.path.join(out, "decoder_trials.csv"), index=False)
best = DecoderParams(w_emit=1.0, **study.best_params)
print("best decoder", best.as_dict(), study.best_value, flush=True)

summ = []
for name, res, info in (("train_oof", tr_res, tr_info), ("student_test", te_res, te_info)):
    base = run(res, info, DecoderParams())
    tuned = run(res, info, best)
    tuned.to_csv(os.path.join(out, f"per_trip_{'train' if name == 'train_oof' else 'student'}.csv"), index=False)
    for label, df in (("classifier_only", tuned), ("hmm_default_params", base), ("hmm_tuned", tuned)):
        pre = "clf_" if label == "classifier_only" else "hmm_"
        w = df.n / df.n.sum()
        summ.append(dict(partition=name, method=label, n_trips=len(df),
                         point_acc=float((df[pre + "acc"] * w).sum()), mean_trip_acc=float(df[pre + "acc"].mean()),
                         continuity=float(df[pre + "continuity"].mean()), route_f1=float(df[pre + "route_f1"].mean()),
                         route_precision=float(df[pre + "route_precision"].mean()), route_recall=float(df[pre + "route_recall"].mean()),
                         avg_path_switches=float(df[pre + "n_switch"].mean()), ref_switches=float(df["clf_ref_switch"].mean())))
pd.DataFrame(summ).to_csv(os.path.join(out, "summary.csv"), index=False)
print(pd.DataFrame(summ).to_string(), flush=True)

fp = os.path.join(ROOT, "models", "final", "decoder.json")
cfg = json.load(open(fp)); cfg["decoder_params"] = best.as_dict()
json.dump(cfg, open(fp, "w"), indent=2)
