"""Merge the per-model outputs of 02_compare_models.py (run in parallel) into the consolidated files."""
import os
import shutil

import joblib
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out = os.path.join(ROOT, "results", "models")
order = ["logreg", "random_forest", "lightgbm", "xgboost", "mlp"]
pd.concat([pd.read_csv(os.path.join(out, f"comparison_{m}.csv")) for m in order]).to_csv(os.path.join(out, "comparison.csv"), index=False)
pd.concat([pd.read_csv(os.path.join(out, f"cv_folds_{m}.csv")) for m in order]).to_csv(os.path.join(out, "cv_folds.csv"), index=False)
parts = [joblib.load(os.path.join(out, f"student_test_predictions_{m}.pkl")) for m in order]
joblib.dump({"keys": parts[0]["keys"], "proba": {k: v for p in parts for k, v in p["proba"].items()}},
            os.path.join(out, "student_test_predictions.pkl"))
shutil.copy(os.path.join(out, "mlp_history_mlp.csv"), os.path.join(out, "mlp_history.csv"))
print(pd.read_csv(os.path.join(out, "comparison.csv"))[["display", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_reach",
      "test_macro_f1", "test_top1_all", "test_log_loss", "test_route_continuity", "train_time_s", "infer_ms_per_1k_obs"]].to_string())
