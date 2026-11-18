"""Error analysis of the five models on the student-test partition. Writes to results/analysis/."""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.candidates import FEATURES  # the 28 feature names, in model order

OUT = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True)
MODELS = ["logreg", "random_forest", "lightgbm", "xgboost", "mlp"]
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "mlp": "MLP Neural Net",
}
ERROR_TYPE_ORDER = [
    "correct",
    "reverse_direction",
    "adjacent_arc",
    "other",
    "candidate_miss",
    "duplicate_arc",
]


def load_inputs():
    d = joblib.load(os.path.join(ROOT, "data", "processed", "student_test.pkl"))
    df, info = d["df"].reset_index(drop=True), d["info"]
    pred = joblib.load(os.path.join(ROOT, "results", "models", "student_test_predictions.pkl"))
    keys = pred["keys"].reset_index(drop=True)
    # safety check: probabilities must line up row by row with the candidate table
    assert (keys["trip"].values == df["trip"].values).all() and (keys["arc"].values == df["arc"].values).all()
    return df, info, pred["proba"]


def top1_per_observation(df, p, info):
    """One row per (trip, obs): the highest-probability candidate and whether the reference arc was a candidate."""
    t = df[["trip", "obs", "arc", "label"]].copy()
    t["p"] = p
    best = t.loc[t.groupby(["trip", "obs"])["p"].idxmax()].rename(columns={"arc": "pred_arc"}).reset_index(drop=True)
    best["ref_arc"] = [int(info[tr]["ref_arc"][o]) for tr, o in zip(best["trip"], best["obs"])]
    reach = t.groupby(["trip", "obs"])["label"].max().rename("reachable").reset_index()
    best = best.merge(reach, on=["trip", "obs"])
    best["correct"] = best["pred_arc"] == best["ref_arc"]
    return best.drop(columns=["label", "p"])


def load_arcs(trip):
    """.arcs = two integer columns without header: from_node, to_node. Row index = arc id."""
    return np.loadtxt(os.path.join(ROOT, "student_test", trip, f"{trip}.arcs"), dtype=np.int64, ndmin=2)


def classify_errors(best):
    """Add column error_type to the output of top1_per_observation."""
    arcs = {tr: load_arcs(tr) for tr in best["trip"].unique()}
    types = []
    for tr, pa, ra, ok, reach in zip(best["trip"], best["pred_arc"], best["ref_arc"], best["correct"], best["reachable"]):
        if ok:
            types.append("correct")
            continue
        if not reach:
            types.append("candidate_miss")
            continue
        (pf, pt), (rf, rt) = arcs[tr][pa], arcs[tr][ra]
        if pf == rf and pt == rt:
            types.append("duplicate_arc")
        elif pf == rt and pt == rf:
            types.append("reverse_direction")
        elif len({pf, pt} & {rf, rt}) > 0:
            types.append("adjacent_arc")
        else:
            types.append("other")
    out = best.copy()
    out["error_type"] = types
    return out


def main():
    print("Loading candidate data and saved predictions...")
    df, info, proba = load_inputs()

    trips = sorted(df["trip"].unique())
    print(f"Loaded {len(df)} candidate rows across {len(trips)} student-test trips.")

    # 1. Classify errors and predictions for each model
    bests = {}
    for m in MODELS:
        print(f"Evaluating model predictions: {m}...")
        b = top1_per_observation(df, proba[m], info)
        bests[m] = classify_errors(b)
        acc_all = bests[m]["correct"].mean()
        acc_reach = bests[m][bests[m]["reachable"] == 1]["correct"].mean()
        print(f"  {m:15s} -> Top-1 (all): {acc_all:.4f} | Top-1 (reachable): {acc_reach:.4f}")

    # 2. Per-trip accuracy table and heatmap
    print("Generating per-trip accuracy table and heatmap...")
    per_trip = pd.DataFrame(index=trips)
    per_trip["n_obs"] = [len(info[t]["ref_arc"]) for t in trips]
    for m in MODELS:
        per_trip[m] = bests[m].groupby("trip")["correct"].mean()

    csv_path = os.path.join(OUT, "per_trip_accuracy.csv")
    per_trip.to_csv(csv_path, index_label="trip")
    print(f"  Wrote {csv_path}")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    data = per_trip[MODELS].values
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0.65, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Top-1 Accuracy", rotation=90)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(trips)))
    ax.set_yticklabels([f"Trip {t} (n={per_trip.loc[t, 'n_obs']})" for t in trips], fontsize=9)
    ax.set_title("Student-Test Per-Trip Top-1 Accuracy by Model", fontsize=11, fontweight="bold", pad=12)

    # Annotate numbers inside cells
    for i in range(len(trips)):
        for j in range(len(MODELS)):
            val = data[i, j]
            text_color = "white" if val > 0.85 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=8.5, fontweight="semibold")

    plt.tight_layout()
    heatmap_path = os.path.join(OUT, "per_trip_accuracy_heatmap.png")
    fig.savefig(heatmap_path)
    plt.close(fig)
    print(f"  Wrote {heatmap_path}")

    # 3. Error types per model
    print("Exporting error type breakdowns and comparison chart...")
    err_counts_dict = {}
    for m in MODELS:
        counts = bests[m]["error_type"].value_counts()
        total_obs = len(bests[m])
        # Ensure all types exist in order
        err_df = pd.DataFrame(index=ERROR_TYPE_ORDER)
        err_df["count"] = [counts.get(t, 0) for t in ERROR_TYPE_ORDER]
        err_df["percentage"] = (err_df["count"] / total_obs) * 100.0
        err_df.index.name = "error_type"

        m_csv = os.path.join(OUT, f"error_types_{m}.csv")
        err_df.to_csv(m_csv)
        err_counts_dict[m] = err_df["count"]
        print(f"  Wrote {m_csv}")

    # Grouped bar chart comparing errors (excluding 'correct' to focus on error distribution)
    error_only_types = [t for t in ERROR_TYPE_ORDER if t != "correct"]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    x = np.arange(len(error_only_types))
    width = 0.16

    colors = ["#4e79a7", "#59a14f", "#edc948", "#e15759", "#76b7b2"]
    for idx, m in enumerate(MODELS):
        vals = [err_counts_dict[m].get(t, 0) for t in error_only_types]
        ax.bar(x + (idx - 2) * width, vals, width, label=MODEL_LABELS[m], color=colors[idx])

    ax.set_ylabel("Count of Mistakes (out of 19,160 observations)", fontsize=10)
    ax.set_title("Distribution of Prediction Errors by Category (Student-Test Partition)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    clean_labels = [
        "Reverse Direction\n(2-way street opposite arc)",
        "Adjacent Arc\n(neighbouring segment)",
        "Other\n(distant arc / mismatch)",
        "Candidate Miss\n(ref arc not in top-12)",
        "Duplicate Arc\n(redundant parallel arc)",
    ]
    ax.set_xticklabels(clean_labels, fontsize=8.5)
    ax.legend(title="Model", frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    err_plot_path = os.path.join(OUT, "error_types.png")
    fig.savefig(err_plot_path)
    plt.close(fig)
    print(f"  Wrote {err_plot_path}")

    # 4. Calibration curve and PR curve
    print("Generating calibration and precision-recall curves...")
    reach_mask = df.groupby(["trip", "obs"])["label"].transform("max") > 0
    y_true = df.loc[reach_mask, "label"].values

    # 4a. Calibration curve
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")

    for idx, m in enumerate(MODELS):
        y_prob = proba[m][reach_mask]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", linewidth=1.5, markersize=4, label=MODEL_LABELS[m], color=colors[idx])

    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Observed Fraction of Positives", fontsize=10)
    ax.set_title("Reliability Diagram (Probability Calibration on Reachable Candidates)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    calib_path = os.path.join(OUT, "calibration.png")
    fig.savefig(calib_path)
    plt.close(fig)
    print(f"  Wrote {calib_path}")

    # 4b. PR curve
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for idx, m in enumerate(MODELS):
        y_prob = proba[m][reach_mask]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, linewidth=1.5, label=f"{MODEL_LABELS[m]} (AP = {ap:.4f})", color=colors[idx])

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall Curve (Reachable Observations)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    pr_path = os.path.join(OUT, "pr_curves.png")
    fig.savefig(pr_path)
    plt.close(fig)
    print(f"  Wrote {pr_path}")

    # 5. XGBoost feature importance (conditional)
    final_model_path = os.path.join(ROOT, "models", "final", "model.joblib")
    if os.path.exists(final_model_path):
        print("Final model detected at models/final/model.joblib. Generating feature importances...")
        model = joblib.load(final_model_path)
        importances = model.feature_importances_
        indices = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.barh(range(len(importances)), importances[indices], align="center", color="#e15759")
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([FEATURES[i] for i in indices], fontsize=8.5)
        ax.set_xlabel("Gain / Importance", fontsize=10)
        ax.set_title("XGBoost Final Model Feature Importances (28 Candidate Features)", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fi_path = os.path.join(OUT, "xgboost_feature_importance.png")
        fig.savefig(fi_path)
        plt.close(fig)
        print(f"  Wrote {fi_path}")
    else:
        print("models/final/model.joblib does not exist yet; skipping xgboost_feature_importance.png as instructed.")

    print("All error analysis tasks completed successfully!")


if __name__ == "__main__":
    main()
