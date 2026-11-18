"""Writes docs/error_analysis.md from the outputs of 11_error_analysis.py, the saved predictions and the final model.
Every number in the document is computed here; nothing is typed in by hand.

usage (repo root, after scripts/01 and scripts/11):  python scripts/12_error_analysis_doc.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.candidates import FEATURES

A = os.path.join(ROOT, "results", "analysis")
MODELS = ["logreg", "random_forest", "lightgbm", "xgboost", "mlp"]
NAME = {"logreg": "Logistic Regression", "random_forest": "Random Forest", "lightgbm": "LightGBM", "xgboost": "XGBoost", "mlp": "MLP (64-32)"}

pt = pd.read_csv(os.path.join(A, "per_trip_accuracy.csv"), dtype={"trip": str})
et = {m: pd.read_csv(os.path.join(A, f"error_types_{m}.csv")).set_index("error_type") for m in MODELS}
n_obs = int(pt.n_obs.sum())
acc = {m: et[m].loc["correct", "count"] / n_obs for m in MODELS}
x = et["xgboost"]["count"]
errs = int(x.sum() - x["correct"])
pt["xgb_err"] = (pt.n_obs * (1 - pt.xgboost)).round().astype(int)
t77 = pt[pt.trip == "00000077"].iloc[0]

d = joblib.load(os.path.join(ROOT, "data", "processed", "student_test.pkl"))["df"].reset_index(drop=True)
pr = joblib.load(os.path.join(ROOT, "results", "models", "student_test_predictions.pkl"))["proba"]
mask = (d.groupby(["trip", "obs"])["label"].transform("max") > 0).values
y = d.label.values[mask]
cal = {}
for m in MODELS:
    p = pr[m][mask]
    pt_, pp_ = calibration_curve(y, p, n_bins=10)
    counts = np.histogram(p, bins=np.linspace(0, 1, 11))[0]
    counts = counts[counts > 0]
    cal[m] = dict(ap=average_precision_score(y, p), ece=float(np.sum(np.abs(pt_ - pp_) * counts) / len(p)),
                  gap=float(np.max(np.abs(pt_ - pp_))), at=float(pp_[np.argmax(np.abs(pt_ - pp_))]), signed=float((pp_ - pt_)[(pp_ > 0.15) & (pp_ < 0.7)].mean()))

model = joblib.load(os.path.join(ROOT, "models", "final", "model.joblib"))
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

L = []
L.append("# Error analysis, calibration and failure modes (student-test partition)\n")
L.append(f"This document investigates where the five models fail on the held-out `student_test` partition ({len(pt)} trips, {n_obs:,} GPS observations, {int(mask.sum() + (~mask).sum()):,} candidate rows). "
         "Every number below is computed by `scripts/12_error_analysis_doc.py` from the outputs of `scripts/11_error_analysis.py` (`results/analysis/`), the saved predictions "
         "(`results/models/student_test_predictions.pkl`) and the final model. Observations whose reference arc is not among the candidates count as wrong. "
         "The analysis scripts were written by Saathvik Shenoy; this text was revised to match the repository's reference numbers.\n")
L.append("## 1. Per-trip accuracy\n")
L.append("| Trip | Observations | " + " | ".join(NAME[m] for m in MODELS) + " |")
L.append("|---|---:|" + "---:|" * len(MODELS))
for r in pt.itertuples():
    L.append(f"| `{r.trip}` | {int(r.n_obs):,} | " + " | ".join(f"{getattr(r, m):.4f}" for m in MODELS) + " |")
L.append(f"| **All trips** | **{n_obs:,}** | " + " | ".join(f"**{acc[m]:.4f}**" for m in MODELS) + " |\n")
easy, hard = pt.sort_values("xgboost").iloc[-1], pt.sort_values("xgboost").iloc[0]
L.append(f"Source: `results/analysis/per_trip_accuracy.csv`, heatmap `per_trip_accuracy_heatmap.png`. The overall XGBoost value ({acc['xgboost']:.4f}) is the student-test top-1 accuracy quoted in the README.\n")
L.append(f"* **Easiest trip for XGBoost:** `{easy.trip}` ({easy.xgboost:.4f}, {int(easy.n_obs):,} observations). **Hardest:** `{hard.trip}` ({hard.xgboost:.4f}, {int(hard.n_obs):,} observations).")
L.append(f"* Trip `{hard.trip}` alone produces {int(t77.xgb_err):,} of the {errs:,} XGBoost errors ({t77.xgb_err / errs * 100:.1f}%). "
         "A first look at the map and at the error types (section 2) points to wrong-direction picks on two-way roads, but we have not examined its road types, so we make no further claim about what the roads look like.\n")
L.append("## 2. Error breakdown for the selected model (XGBoost, untuned scores used in the comparison)\n")
L.append("Each wrong observation is assigned to the first matching category: `candidate_miss` (reference arc not a candidate), `duplicate_arc` (same node pair, same direction, different arc id), "
         "`reverse_direction` (same node pair, opposite direction), `adjacent_arc` (shares a node with the reference arc), `other`.\n")
L.append("| Category | Count | % of all observations | % of all errors |")
L.append("|---|---:|---:|---:|")
for k in ["correct", "reverse_direction", "adjacent_arc", "other", "candidate_miss", "duplicate_arc"]:
    c = int(x[k])
    L.append(f"| `{k}` | {c:,} | {c / n_obs * 100:.2f}% | " + ("-" if k == "correct" else f"{c / errs * 100:.1f}%") + " |")
L.append(f"| **Total errors** | **{errs:,}** | **{errs / n_obs * 100:.2f}%** | **100%** |\n")
L.append("Source: `results/analysis/error_types_xgboost.csv`; all five models in `error_types.png`.\n")
L.append(f"`reverse_direction` is the largest category ({x['reverse_direction'] / errs * 100:.0f}% of the errors), followed by `adjacent_arc` ({x['adjacent_arc'] / errs * 100:.0f}%). "
         "This matches what the EDA found: most roads are stored as two arcs with identical geometry, so distance cannot separate the two directions and only the direction features can. "
         "A single-fix classifier has to guess when the movement is slow or the heading is noisy; a sequence model can use the fact that a vehicle does not reverse between consecutive fixes, "
         "which is the kind of error the HMM in README section 8 can correct. We have not yet repeated this breakdown on the HMM output, so we do not know how much of it the HMM removes.\n")
L.append("## 3. Probability calibration and why it matters for the HMM\n")
L.append("Reliability curves and precision-recall curves use the candidate rows of observations whose reference arc is a candidate "
         f"({int(mask.sum()):,} rows), 10 probability bins (`calibration.png`, `pr_curves.png`). ECE is the bin-weighted mean absolute gap between predicted probability and observed frequency.\n")
L.append("| Model | Average precision | ECE | Largest bin gap (at predicted probability) |")
L.append("|---|---:|---:|---:|")
for m in MODELS:
    L.append(f"| {NAME[m]} | {cal[m]['ap']:.4f} | {cal[m]['ece']:.4f} | {cal[m]['gap']:.3f} (at {cal[m]['at']:.2f}) |")
over = [m for m in MODELS if cal[m]["signed"] > 0.04]
L.append("")
L.append(f"* XGBoost has the lowest ECE ({cal['xgboost']['ece']:.4f}); Random Forest, LightGBM and the MLP are also close to the diagonal.")
L.append(f"* {', '.join(NAME[m] for m in over) if over else 'No model'} predicts probabilities that are on average higher than the observed frequency in the 0.15 to 0.70 range, i.e. it is over-confident there "
         f"(Logistic Regression: mean gap {cal['logreg']['signed'] * 100:.1f} points).")
L.append("* The HMM adds the log of these probabilities along the whole trajectory and compares them with a transition penalty. If probabilities are over-confident, the emission term "
         "outweighs the transition term and the decoder accepts jumps between unconnected roads; well-calibrated probabilities let the road network override a weak local choice. "
         "This was one of the reasons for using XGBoost with the HMM, alongside its cross-validation results.\n")
L.append("## 4. Feature importance of the final model\n")
L.append("Importance is `feature_importances_` (gain) of the tuned XGBoost model in `models/final/model.joblib`; figure `xgboost_feature_importance.png`.\n")
L.append("| Rank | Feature | Importance |")
L.append("|---:|---|---:|")
for i, (k, v) in enumerate(imp.head(8).items(), 1):
    L.append(f"| {i} | `{k}` | {v:.4f} |")
L.append(f"\nThe top 5 features carry {imp.head(5).sum() * 100:.0f}% of the total and the top 8 carry {imp.head(8).sum() * 100:.0f}%. "
         "The model mostly asks how much farther an arc is than the closest candidate (`d_rel`, `d_rank`, `d_vs_best_dir`) and whether the movement agrees with the arc direction (`abs_ang`, `cos_h`). "
         "These are model-internal importances: they show what the model relies on, not what causes a correct match.\n")
L.append("## 5. What we would try with more time\n")
L.append("These are hypotheses that we have **not** tested; each is tied to a measured error count.\n")
L.append(f"1. **Longer-window heading** (targets `reverse_direction`, {int(x['reverse_direction']):,} errors). Heading is currently computed from the previous and next fix only; a window of a few fixes might be steadier at low speed. Untested.")
L.append(f"2. **Junction-proximity feature** (targets `adjacent_arc`, {int(x['adjacent_arc']):,} errors). A feature combining the distance to the arc ends with the speed might help the model decide when the vehicle has passed a junction. Untested.")
L.append(f"3. **Road-class information** (targets `other`, {int(x['other']):,} errors). The data has no road classes, so this would need another data source; we have not checked whether such errors concentrate on particular road types.")
L.append(f"4. **Larger candidate radius** (targets `candidate_miss`, {int(x['candidate_miss'])} errors, {x['candidate_miss'] / n_obs * 100:.2f}% of observations). The gain is capped at that small share on this partition. The large training-set miss rate comes from two trips whose reference routes do not match their traces (`docs/data_quality.md`), which a larger radius would not fix.")
L.append(f"5. **Sequence-aware classifier input** (targets trip `{t77.trip}`, {int(t77.xgb_err):,} errors, mostly wrong direction). Giving the classifier scores of neighbouring fixes, a stacked model, could move part of the HMM's work into the classifier. Untested.\n")
open(os.path.join(ROOT, "docs", "error_analysis.md"), "w", encoding="utf8").write("\n".join(L))
print("wrote docs/error_analysis.md;", "xgboost acc", round(acc["xgboost"], 4), "errors", errs)
