"""Train and compare the five models.

* 5-fold GroupKFold (groups = trips) on the 80 training trips -> CV metrics (reachable-observation basis)
* refit on all 80 training trips, evaluate on the 10 student-testing trips
Outputs: results/models/{cv_folds.csv,comparison.csv,student_test_predictions.pkl,mlp_history.csv}
         models/experiments/*.joblib (git-ignored, only the final model is committed)
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
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.candidates import FEATURES
from pathsnap.modeling import (MODEL_INFO, SEED, add_gid, evaluate, fit_model, load_trip_meta, trainable_mask)

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--models", default="logreg,random_forest,lightgbm,xgboost,mlp")
ap.add_argument("--tag", default="")
args = ap.parse_args()
TAG = ("_" + args.tag) if args.tag else ""

out = os.path.join(ROOT, "results", "models"); os.makedirs(out, exist_ok=True)
exp = os.path.join(ROOT, "models", "experiments"); os.makedirs(exp, exist_ok=True)

tr = joblib.load(os.path.join(ROOT, "data", "processed", "train.pkl"))["df"].reset_index(drop=True)
te = joblib.load(os.path.join(ROOT, "data", "processed", "student_test.pkl"))["df"].reset_index(drop=True)
if args.quick:
    keep = sorted(tr.trip.unique())[:15]; tr = tr[tr.trip.isin(keep)].reset_index(drop=True)
add_gid(tr); add_gid(te)
meta_tr = load_trip_meta(os.path.join(ROOT, "train"), tr.trip.unique())
meta_te = load_trip_meta(os.path.join(ROOT, "student_test"), te.trip.unique())
Xtr_all = tr[FEATURES].values.astype(np.float32); ytr_all = tr.label.values
Xte = te[FEATURES].values.astype(np.float32)
fit_mask = trainable_mask(tr)  # train only on observations whose reference arc is a candidate
print("train rows", len(tr), "fit rows", fit_mask.sum(), "test rows", len(te), flush=True)

gkf = GroupKFold(n_splits=5)
fold_rows, summary, test_preds, hist_rows = [], [], {}, []
for name in args.models.split(","):
    disp, fam, cfg = MODEL_INFO[name]
    print("=== ", disp, flush=True)
    t_all = time.time()
    for k, (a, b) in enumerate(gkf.split(Xtr_all, ytr_all, groups=tr.trip.values)):
        a = a[fit_mask[a]]
        model, tt = fit_model(name, Xtr_all[a], ytr_all[a], groups=tr.trip.values[a])
        va = tr.iloc[b].reset_index(drop=True)
        t0 = time.time(); p = model.predict_proba(Xtr_all[b])[:, 1]; inf = time.time() - t0
        m = evaluate(va, p, {t: meta_tr[t] for t in va.trip.unique()})
        m.update(model=name, fold=k, train_time_s=tt, infer_time_s=inf)
        fold_rows.append(m)
        print(f"  fold {k}: macroF1={m['macro_f1']:.4f} top1(reach)={m['top1_acc_reachable']:.4f} "
              f"ll={m['log_loss']:.4f} cont={m['route_continuity']:.3f} t={tt:.0f}s", flush=True)
    # final fit on all training trips -> student-test
    model, tt = fit_model(name, Xtr_all[fit_mask], ytr_all[fit_mask], groups=tr.trip.values[fit_mask])
    t0 = time.time(); p = model.predict_proba(Xte)[:, 1]; inf = time.time() - t0
    mt = evaluate(te, p, meta_te)
    n_obs = te.gid.nunique()
    test_preds[name] = p
    joblib.dump(model, os.path.join(exp, f"{name}.joblib"))
    if name == "mlp":
        h = model[-1].history_; hist_rows = pd.DataFrame(h); hist_rows.to_csv(os.path.join(out, f"mlp_history{TAG}.csv"), index=False)
    cv = pd.DataFrame([r for r in fold_rows if r["model"] == name])
    summary.append(dict(
        model=name, display=disp, family=fam, config=cfg,
        cv_macro_f1_mean=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(),
        cv_top1_reach=cv.top1_acc_reachable.mean(), cv_top1_all=cv.top1_acc_all.mean(),
        cv_log_loss=cv.log_loss.mean(), cv_route_continuity=cv.route_continuity.mean(),
        cv_route_set_f1=cv.route_set_f1.mean(),
        test_macro_f1=mt["macro_f1"], test_top1_reach=mt["top1_acc_reachable"], test_top1_all=mt["top1_acc_all"],
        test_precision=mt["precision_pos"], test_recall=mt["recall_pos"],
        test_log_loss=mt["log_loss"], test_route_continuity=mt["route_continuity"], test_route_set_f1=mt["route_set_f1"],
        test_candidate_miss=mt["candidate_miss_rate"],
        train_time_s=tt, infer_ms_per_1k_obs=1000 * inf / n_obs * 1000, total_cv_time_s=time.time() - t_all,
    ))
    print(f"  TEST: macroF1={mt['macro_f1']:.4f} top1={mt['top1_acc_all']:.4f} ll={mt['log_loss']:.4f} "
          f"cont={mt['route_continuity']:.3f}", flush=True)
    pd.DataFrame(fold_rows).to_csv(os.path.join(out, f"cv_folds{TAG}.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(out, f"comparison{TAG}.csv"), index=False)

joblib.dump({"keys": te[["trip", "obs", "arc", "label", "gid"]], "proba": test_preds},
            os.path.join(out, f"student_test_predictions{TAG}.pkl"))
print(pd.DataFrame(summary)[["display", "cv_macro_f1_mean", "cv_top1_reach", "test_macro_f1", "test_top1_all", "test_log_loss",
                             "test_route_continuity", "train_time_s"]].to_string())
