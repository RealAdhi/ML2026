"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model.

Objective  : maximise 5-fold GroupKFold (groups = trips) macro-F1 on the 80 training trips.
Outputs    : results/bayes/{trials.csv,best_params.json,tuned_vs_untuned.csv,oof_train.pkl,history.png}
             models/final/model.joblib  (tuned model refit on all 80 training trips)
The student-test partition is used only AFTER the search to compare tuned vs untuned.
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
from sklearn.model_selection import GroupKFold

from pathsnap.candidates import FEATURES
from pathsnap.modeling import (add_gid, evaluate, fast_macro_f1, fit_model, load_trip_meta, make_model, trainable_mask)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="lightgbm")
ap.add_argument("--trials", type=int, default=30)
ap.add_argument("--parallel", type=int, default=4)
ap.add_argument("--final", action="store_true", help="also write the tuned model to models/final/")
args = ap.parse_args()

out = os.path.join(ROOT, "results", "bayes", args.model); os.makedirs(out, exist_ok=True)
fin = os.path.join(ROOT, "models", "final" if args.final else os.path.join("experiments", "tuned_" + args.model)); os.makedirs(fin, exist_ok=True)

tr = joblib.load(os.path.join(ROOT, "data", "processed", "train.pkl"))["df"].reset_index(drop=True)
te = joblib.load(os.path.join(ROOT, "data", "processed", "student_test.pkl"))["df"].reset_index(drop=True)
add_gid(tr); add_gid(te)
X = tr[FEATURES].values.astype(np.float32); y = tr.label.values
fit_mask = trainable_mask(tr)
folds = list(GroupKFold(n_splits=5).split(X, y, groups=tr.trip.values))


from concurrent.futures import ThreadPoolExecutor


def _fold(params, a, b):
    a = a[fit_mask[a]]
    m, _ = fit_model(args.model, X[a], y[a], groups=tr.trip.values[a], params=params)
    return b, m.predict_proba(X[b])[:, 1]


def cv_score(params, return_oof=False):
    """The five group folds are fitted concurrently (tree libraries release the GIL)."""
    oof = np.zeros(len(tr)); sc = []
    with ThreadPoolExecutor(max_workers=len(folds)) as ex:
        for b, pb in ex.map(lambda ab: _fold(params, *ab), folds):
            oof[b] = pb
            sc.append(fast_macro_f1(tr.gid.values[b], y[b], pb))
    return (np.mean(sc), np.std(sc), oof) if return_oof else float(np.mean(sc))


def space(trial):
    if args.model == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True))
    if args.model == "xgboost":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 12))
    raise SystemExit("search space defined for lightgbm / xgboost")


def objective(trial):
    t0 = time.time()
    s = cv_score(space(trial))
    trial.set_user_attr("cv_time_s", time.time() - t0)
    print(f"trial {trial.number}: macroF1={s:.5f} ({time.time() - t0:.0f}s) {trial.params}", flush=True)
    return s


t_start = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
                            study_name="pathsnap_" + args.model, storage="sqlite:///" + os.path.join(out, "study.db").replace("\\", "/"),
                            load_if_exists=True)   # every finished trial is persisted
from optuna.trial import TrialState
for t in study.get_trials(deepcopy=False):          # trials left RUNNING by an interrupted earlier run
    if t.state == TrialState.RUNNING:
        study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)


def stop_when_enough(st, _trial):
    if sum(t.state == TrialState.COMPLETE for t in st.get_trials(deepcopy=False)) >= args.trials:
        st.stop()


study.optimize(objective, n_trials=args.trials * 2, n_jobs=args.parallel, callbacks=[stop_when_enough])
search_time = time.time() - t_start
study.trials_dataframe().to_csv(os.path.join(out, "trials.csv"), index=False)
best = study.best_params
json.dump({"model": args.model, "best_params": best, "best_cv_macro_f1": study.best_value,
           "n_trials": sum(t.state == TrialState.COMPLETE for t in study.trials), "search_time_s": search_time, "sampler": "TPE (Optuna), seed 42, 8 startup trials",
           "objective": "5-fold GroupKFold mean macro-F1 (top-1 decision rule, reachable observations)"},
          open(os.path.join(out, "best_params.json"), "w"), indent=2)
print("best", best, study.best_value, flush=True)

# ---- tuned vs untuned, identical splits and metrics --------------------------------------------------------
meta_tr = load_trip_meta(os.path.join(ROOT, "train"), tr.trip.unique())
meta_te = load_trip_meta(os.path.join(ROOT, "student_test"), te.trip.unique())
rows = []
for label, prm in (("untuned", {}), ("tuned", best)):
    m_cv, s_cv, oof = cv_score(prm, return_oof=True)
    cvm = evaluate(tr, oof, meta_tr)
    model, tt = fit_model(args.model, X[fit_mask], y[fit_mask], groups=tr.trip.values[fit_mask], params=prm)
    t0 = time.time(); p = model.predict_proba(te[FEATURES].values.astype(np.float32))[:, 1]; inf = time.time() - t0
    mt = evaluate(te, p, meta_te)
    rows.append(dict(variant=label, cv_macro_f1=m_cv, cv_macro_f1_std=s_cv, cv_log_loss=cvm["log_loss"],
                     cv_top1_reachable=cvm["top1_acc_reachable"], test_top1_all=mt["top1_acc_all"],
                     test_top1_reachable=mt["top1_acc_reachable"], test_macro_f1=mt["macro_f1"], test_log_loss=mt["log_loss"],
                     test_route_continuity=mt["route_continuity"], train_time_s=tt,
                     infer_ms_per_1k_obs=inf / te.gid.nunique() * 1e6))
    joblib.dump({"keys": tr[["trip", "obs", "arc", "label"]], "oof": oof}, os.path.join(out, f"oof_train_{label}.pkl"))
    joblib.dump({"keys": te[["trip", "obs", "arc", "label"]], "proba": p}, os.path.join(out, f"student_test_{label}.pkl"))
    if label == "tuned":
        joblib.dump(model, os.path.join(fin, "model.joblib"))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, "tuned_vs_untuned.csv"), index=False)
json.dump({"model_name": args.model, "features": FEATURES, "model_params": best, "decoder_params": {}},
          open(os.path.join(fin, "decoder.json"), "w"), indent=2)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = study.trials_dataframe()
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(df.number, df.value, "o-", alpha=.6, label="trial CV macro-F1")
ax.plot(df.number, df.value.cummax(), "r-", label="best so far")
ax.set_xlabel("trial"); ax.set_ylabel("CV macro-F1"); ax.legend(); ax.set_title(f"Bayesian optimisation ({args.model})")
fig.tight_layout(); fig.savefig(os.path.join(out, "history.png"), dpi=150)
