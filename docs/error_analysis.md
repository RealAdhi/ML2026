# Error analysis, calibration and failure modes (student-test partition)

This document investigates where the five models fail on the held-out `student_test` partition (10 trips, 19,160 GPS observations, 160,114 candidate rows). Every number below is computed by `scripts/12_error_analysis_doc.py` from the outputs of `scripts/11_error_analysis.py` (`results/analysis/`), the saved predictions (`results/models/student_test_predictions.pkl`) and the final model. Observations whose reference arc is not among the candidates count as wrong. The analysis scripts were written by Saathvik Shenoy; this text was revised to match the repository's reference numbers.

## 1. Per-trip accuracy

| Trip | Observations | Logistic Regression | Random Forest | LightGBM | XGBoost | MLP (64-32) |
|---|---:|---:|---:|---:|---:|---:|
| `00000000` | 2,503 | 0.9349 | 0.9405 | 0.9389 | 0.9441 | 0.9409 |
| `00000007` | 1,354 | 0.9077 | 0.9217 | 0.9077 | 0.9121 | 0.9003 |
| `00000009` | 1,049 | 0.8360 | 0.8551 | 0.8532 | 0.8561 | 0.8627 |
| `00000040` | 2,995 | 0.9055 | 0.9225 | 0.9222 | 0.9319 | 0.9149 |
| `00000052` | 262 | 0.9351 | 0.9427 | 0.9389 | 0.9504 | 0.9542 |
| `00000056` | 3,958 | 0.8699 | 0.8914 | 0.8926 | 0.8951 | 0.8954 |
| `00000065` | 768 | 0.8216 | 0.8281 | 0.8281 | 0.8346 | 0.8438 |
| `00000076` | 1,486 | 0.8304 | 0.8769 | 0.8829 | 0.8910 | 0.8715 |
| `00000077` | 3,911 | 0.6799 | 0.6906 | 0.6829 | 0.6937 | 0.6886 |
| `00000091` | 874 | 0.8375 | 0.8604 | 0.8638 | 0.8741 | 0.8638 |
| **All trips** | **19,160** | **0.8404** | **0.8575** | **0.8554** | **0.8623** | **0.8562** |

Source: `results/analysis/per_trip_accuracy.csv`, heatmap `per_trip_accuracy_heatmap.png`. The overall XGBoost value (0.8623) is the student-test top-1 accuracy quoted in the README.

* **Easiest trip for XGBoost:** `00000052` (0.9504, 262 observations). **Hardest:** `00000077` (0.6937, 3,911 observations).
* Trip `00000077` alone produces 1,198 of the 2,639 XGBoost errors (45.4%). A first look at the map and at the error types (section 2) points to wrong-direction picks on two-way roads, but we have not examined its road types, so we make no further claim about what the roads look like.

## 2. Error breakdown for the selected model (XGBoost, untuned scores used in the comparison)

Each wrong observation is assigned to the first matching category: `candidate_miss` (reference arc not a candidate), `duplicate_arc` (same node pair, same direction, different arc id), `reverse_direction` (same node pair, opposite direction), `adjacent_arc` (shares a node with the reference arc), `other`.

| Category | Count | % of all observations | % of all errors |
|---|---:|---:|---:|
| `correct` | 16,521 | 86.23% | - |
| `reverse_direction` | 1,274 | 6.65% | 48.3% |
| `adjacent_arc` | 868 | 4.53% | 32.9% |
| `other` | 458 | 2.39% | 17.4% |
| `candidate_miss` | 39 | 0.20% | 1.5% |
| `duplicate_arc` | 0 | 0.00% | 0.0% |
| **Total errors** | **2,639** | **13.77%** | **100%** |

Source: `results/analysis/error_types_xgboost.csv`; all five models in `error_types.png`.

`reverse_direction` is the largest category (48% of the errors), followed by `adjacent_arc` (33%). This matches what the EDA found: most roads are stored as two arcs with identical geometry, so distance cannot separate the two directions and only the direction features can. A single-fix classifier has to guess when the movement is slow or the heading is noisy; a sequence model can use the fact that a vehicle does not reverse between consecutive fixes, which is the kind of error the HMM in README section 8 can correct. We have not yet repeated this breakdown on the HMM output, so we do not know how much of it the HMM removes.

## 3. Probability calibration and why it matters for the HMM

Reliability curves and precision-recall curves use the candidate rows of observations whose reference arc is a candidate (159,878 rows), 10 probability bins (`calibration.png`, `pr_curves.png`). ECE is the bin-weighted mean absolute gap between predicted probability and observed frequency.

| Model | Average precision | ECE | Largest bin gap (at predicted probability) |
|---|---:|---:|---:|
| Logistic Regression | 0.9044 | 0.0083 | 0.087 (at 0.45) |
| Random Forest | 0.9439 | 0.0043 | 0.038 (at 0.65) |
| LightGBM | 0.9282 | 0.0029 | 0.078 (at 0.65) |
| XGBoost | 0.9457 | 0.0019 | 0.047 (at 0.65) |
| MLP (64-32) | 0.9396 | 0.0025 | 0.048 (at 0.25) |

* XGBoost has the lowest ECE (0.0019); Random Forest, LightGBM and the MLP are also close to the diagonal.
* Logistic Regression predicts probabilities that are on average higher than the observed frequency in the 0.15 to 0.70 range, i.e. it is over-confident there (Logistic Regression: mean gap 6.6 points).
* The HMM adds the log of these probabilities along the whole trajectory and compares them with a transition penalty. If probabilities are over-confident, the emission term outweighs the transition term and the decoder accepts jumps between unconnected roads; well-calibrated probabilities let the road network override a weak local choice. This was one of the reasons for using XGBoost with the HMM, alongside its cross-validation results.

## 4. Feature importance of the final model

Importance is `feature_importances_` (gain) of the tuned XGBoost model in `models/final/model.joblib`; figure `xgboost_feature_importance.png`.

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `d_rel` | 0.4178 |
| 2 | `abs_ang` | 0.1761 |
| 3 | `d_rank` | 0.1154 |
| 4 | `d_vs_best_dir` | 0.0962 |
| 5 | `cos_h` | 0.0586 |
| 6 | `dprog_ok` | 0.0170 |
| 7 | `has_reverse` | 0.0136 |
| 8 | `d_ratio` | 0.0112 |

The top 5 features carry 86% of the total and the top 8 carry 91%. The model mostly asks how much farther an arc is than the closest candidate (`d_rel`, `d_rank`, `d_vs_best_dir`) and whether the movement agrees with the arc direction (`abs_ang`, `cos_h`). These are model-internal importances: they show what the model relies on, not what causes a correct match.

## 5. What we would try with more time

These are hypotheses that we have **not** tested; each is tied to a measured error count.

1. **Longer-window heading** (targets `reverse_direction`, 1,274 errors). Heading is currently computed from the previous and next fix only; a window of a few fixes might be steadier at low speed. Untested.
2. **Junction-proximity feature** (targets `adjacent_arc`, 868 errors). A feature combining the distance to the arc ends with the speed might help the model decide when the vehicle has passed a junction. Untested.
3. **Road-class information** (targets `other`, 458 errors). The data has no road classes, so this would need another data source; we have not checked whether such errors concentrate on particular road types.
4. **Larger candidate radius** (targets `candidate_miss`, 39 errors, 0.20% of observations). The gain is capped at that small share on this partition. The large training-set miss rate comes from two trips whose reference routes do not match their traces (`docs/data_quality.md`), which a larger radius would not fix.
5. **Sequence-aware classifier input** (targets trip `00000077`, 1,198 errors, mostly wrong direction). Giving the classifier scores of neighbouring fixes, a stacked model, could move part of the HMM's work into the classifier. Untested.
