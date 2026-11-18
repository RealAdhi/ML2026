"""Model-comparison, tuning and HMM figures drawn from the recorded experiment outputs only."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

R = os.path.join(ROOT, "results"); fig_dir = os.path.join(R, "figures"); os.makedirs(fig_dir, exist_ok=True)
comp = pd.read_csv(os.path.join(R, "models", "comparison.csv"))
folds = pd.read_csv(os.path.join(R, "models", "cv_folds.csv"))
names = comp.display.tolist()
col = ["#5c6bc0", "#26a69a", "#ef6c00", "#8d6e63", "#ab47bc"][: len(names)]

# CV macro-F1 (mean +- std over the 5 folds)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(names, comp.cv_macro_f1_mean, yerr=comp.cv_macro_f1_std, color=col, capsize=4)
for i, v in enumerate(comp.cv_macro_f1_mean): ax.text(i, v + .005, f"{v:.3f}", ha="center", fontsize=8)
ax.set(ylabel="Macro F1 (5-fold GroupKFold)", title="Cross-validation macro-F1 (80 training trips)", ylim=(0.8, 1.0))
plt.setp(ax.get_xticklabels(), rotation=15); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "cv_macro_f1.png"), dpi=140); plt.close(fig)

# fold-level spread
fig, ax = plt.subplots(figsize=(8, 4))
for k, (m, g) in enumerate(folds.groupby("model", sort=False)):
    ax.plot(g.fold, g.macro_f1, "o-", label=comp.set_index("model").loc[m, "display"], color=col[k])
ax.set(xlabel="fold", ylabel="macro-F1", title="Per-fold macro-F1 (stability across folds)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "cv_fold_stability.png"), dpi=140); plt.close(fig)

# student-test top-1 and macro F1
fig, ax = plt.subplots(figsize=(9, 4)); w = .38; x = np.arange(len(names))
ax.bar(x - w / 2, comp.test_top1_all, w, label="Top-1 accuracy (all obs)", color="#42a5f5")
ax.bar(x + w / 2, comp.test_macro_f1, w, label="Macro F1", color="#66bb6a")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15); ax.set_ylim(0.75, 1.0); ax.legend()
ax.set_title("Student-test (10 trips): Top-1 accuracy and macro-F1")
for i in range(len(names)):
    ax.text(i - w / 2, comp.test_top1_all[i] + .003, f"{comp.test_top1_all[i]:.3f}", ha="center", fontsize=7)
    ax.text(i + w / 2, comp.test_macro_f1[i] + .003, f"{comp.test_macro_f1[i]:.3f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "test_top1_macro_f1.png"), dpi=140); plt.close(fig)

# log loss
fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(names))
ax.bar(x - w / 2, comp.cv_log_loss, w, label="CV", color="#ffa726"); ax.bar(x + w / 2, comp.test_log_loss, w, label="Student-test", color="#ef5350")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15); ax.set_ylabel("log loss (lower is better)"); ax.legend(); ax.set_title("Log loss")
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "log_loss.png"), dpi=140); plt.close(fig)

# route continuity + inference time
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(names, comp.test_route_continuity, color=col); ax[0].set(title="Route continuity (classifier-only, student-test)", ylim=(0.85, 1))
ax[1].bar(names, comp.infer_ms_per_1k_obs, color=col); ax[1].set(title="Inference time per 1000 observations (ms)", yscale="log")
for a in ax: plt.setp(a.get_xticklabels(), rotation=20, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "continuity_inference_time.png"), dpi=140); plt.close(fig)

# confusion matrices (candidate-level, top-1 decision rule, student-test, reachable observations)
sp = joblib.load(os.path.join(R, "models", "student_test_predictions.pkl"))
keys = sp["keys"].reset_index(drop=True)
reach = keys.groupby("gid").label.transform("max").values > 0
fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4))
for a, (m, disp) in zip(np.atleast_1d(axes), zip(comp.model, comp.display)):
    p = sp["proba"][m]
    t = keys.assign(p=p); first = t.groupby("gid").p.idxmax().values
    pred = np.zeros(len(t), int); pred[first] = 1
    cm = confusion_matrix(t.label[reach], pred[reach], labels=[0, 1])
    a.imshow(cm, cmap="Blues"); a.set_title(disp, fontsize=9)
    for i in range(2):
        for j in range(2): a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=8)
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("predicted (selected)"); a.set_ylabel("true (reference arc)")
fig.suptitle("Confusion matrices, student-test candidates (top-1 decision per observation)"); fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "confusion_matrices.png"), dpi=140); plt.close(fig)

# MLP training history (the only model with an epoch-wise history)
hp = os.path.join(R, "models", "mlp_history.csv")
if os.path.exists(hp):
    h = pd.read_csv(hp); fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h.epoch, h.train_loss, "o-", label="train loss"); ax.plot(h.epoch, h.val_loss, "s-", label="validation loss (held-out trips)")
    ax.set(xlabel="epoch", ylabel="log loss", title="MLP training / validation loss"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "mlp_loss_curve.png"), dpi=140); plt.close(fig)

# tuned vs untuned
tp = os.path.join(R, "bayes", "xgboost", "tuned_vs_untuned.csv")
if os.path.exists(tp):
    t = pd.read_csv(tp); fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, c, ttl in zip(ax, ("cv_macro_f1", "test_top1_all", "test_log_loss"), ("CV macro-F1", "Student-test top-1", "Student-test log loss")):
        a.bar(t.variant, t[c], color=["#90a4ae", "#2e7d32"]); a.set_title(ttl)
        for i, v in enumerate(t[c]): a.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        lo, hi = t[c].min(), t[c].max(); a.set_ylim(lo - (hi - lo) * 3 - 1e-3, hi + (hi - lo) * 2 + 1e-3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "tuned_vs_untuned.png"), dpi=140); plt.close(fig)

# HMM vs classifier-only
hp = os.path.join(R, "hmm", "per_trip_student.csv")
if os.path.exists(hp):
    h = pd.read_csv(hp, dtype={"trip": str}); x = np.arange(len(h)); fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(x - .2, h.clf_acc, .4, label="classifier only", color="#ef5350"); ax[0].bar(x + .2, h.hmm_acc, .4, label="classifier + HMM", color="#2e7d32")
    ax[0].set_xticks(x); ax[0].set_xticklabels(h.trip.str[-3:]); ax[0].set(title="Point accuracy per student-test trip", ylim=(.7, 1)); ax[0].legend()
    ax[1].bar(x - .2, h.clf_continuity, .4, color="#ef5350"); ax[1].bar(x + .2, h.hmm_continuity, .4, color="#2e7d32")
    ax[1].set_xticks(x); ax[1].set_xticklabels(h.trip.str[-3:]); ax[1].set(title="Route continuity per student-test trip", ylim=(.8, 1.01))
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "hmm_vs_classifier.png"), dpi=140); plt.close(fig)
print("figures written to", fig_dir)
