"""Model zoo, group-aware cross-validation and route-aware evaluation metrics."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .candidates import FEATURES

SEED = 42
NJOBS = int(os.environ.get("PATHSNAP_JOBS", "8"))


# --------------------------------------------------------------------------- data helpers
def add_gid(df: pd.DataFrame) -> pd.DataFrame:
    """Integer id for each (trip, observation) group."""
    df["gid"] = pd.factorize(df["trip"] + "_" + df["obs"].astype(str))[0]
    return df


def trainable_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows of observations whose reference arc is among the candidates (label noise otherwise)."""
    return (df.groupby("gid")["label"].transform("max") > 0).values


# --------------------------------------------------------------------------- models
class EpochMLP(BaseEstimator, ClassifierMixin):
    """sklearn MLP trained epoch by epoch so train/validation loss curves can be recorded."""

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, batch=2048, epochs=25, patience=4, seed=SEED):
        self.hidden, self.alpha, self.lr, self.batch = hidden, alpha, lr, batch
        self.epochs, self.patience, self.seed = epochs, patience, seed

    def fit(self, X, y, groups=None):
        rng = np.random.RandomState(self.seed)
        n = len(y)
        if groups is not None:  # hold out ~10% of groups (trips) as the early-stopping monitor
            ug = np.unique(groups); rng.shuffle(ug)
            val_g = set(ug[: max(1, len(ug) // 10)])
            vm = np.isin(groups, list(val_g))
        else:
            vm = rng.rand(n) < 0.1
        Xtr, ytr, Xv, yv = X[~vm], y[~vm], X[vm], y[vm]
        self.net_ = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha, learning_rate_init=self.lr,
                                  batch_size=self.batch, random_state=self.seed)
        classes = np.array([0, 1])
        self.history_ = {"epoch": [], "train_loss": [], "val_loss": []}
        best, best_state, bad = np.inf, None, 0
        import copy
        for ep in range(1, self.epochs + 1):
            perm = rng.permutation(len(ytr))
            for s in range(0, len(perm), self.batch):
                b = perm[s:s + self.batch]
                self.net_.partial_fit(Xtr[b], ytr[b], classes=classes)
            tl = log_loss(ytr[::5], self.net_.predict_proba(Xtr[::5])[:, 1], labels=[0, 1])
            vl = log_loss(yv, self.net_.predict_proba(Xv)[:, 1], labels=[0, 1])
            print(f"epoch {ep}: train_loss={tl:.4f} val_loss={vl:.4f}", flush=True)
            self.history_["epoch"].append(ep); self.history_["train_loss"].append(tl); self.history_["val_loss"].append(vl)
            if vl < best - 1e-5:
                best, bad, best_state = vl, 0, copy.deepcopy(self.net_)
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_ = best_state
        self.classes_ = classes
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(X)


def make_model(name: str, params: dict | None = None):
    p = dict(params or {})
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=p.get("C", 1.0), max_iter=300, random_state=SEED))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 150), max_depth=p.get("max_depth", 14),
                                      min_samples_leaf=p.get("min_samples_leaf", 10), max_samples=0.4,
                                      n_jobs=NJOBS, random_state=SEED)
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05),
                                  num_leaves=p.get("num_leaves", 31), min_child_samples=p.get("min_child_samples", 20),
                                  subsample=p.get("subsample", 0.8), subsample_freq=1,
                                  colsample_bytree=p.get("colsample_bytree", 0.8),
                                  reg_lambda=p.get("reg_lambda", 0.0), reg_alpha=p.get("reg_alpha", 0.0),
                                  random_state=SEED, n_jobs=NJOBS, verbose=-1)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.1),
                                 max_depth=p.get("max_depth", 6), subsample=0.8, colsample_bytree=0.8,
                                 tree_method="hist", n_jobs=NJOBS, random_state=SEED, eval_metric="logloss")
    if name == "mlp":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             EpochMLP(hidden=p.get("hidden", (64, 32)), alpha=p.get("alpha", 1e-4),
                                      lr=p.get("lr", 1e-3), epochs=p.get("epochs", 25)))
    raise ValueError(name)


MODEL_INFO = {
    "logreg": ("Logistic Regression", "Linear / probabilistic", "C=1.0, L2, median-impute + standardise, lbfgs"),
    "random_forest": ("Random Forest", "Bagging ensemble", "150 trees, depth 14, min_leaf 10, 40% bootstrap"),
    "lightgbm": ("LightGBM", "Gradient boosting (histogram)", "300 trees, lr 0.05, 31 leaves, subsample 0.8"),
    "xgboost": ("XGBoost", "Gradient boosting (histogram)", "300 trees, lr 0.1, depth 6, subsample 0.8"),
    "mlp": ("MLP (64-32)", "Neural network", "2 hidden layers 64/32, ReLU, Adam lr 1e-3, early stop on val loss"),
}


def fit_model(name, X, y, groups=None, params=None):
    m = make_model(name, params)
    t0 = time.time()
    if name == "mlp":
        # pipeline: pass groups to the last step after preprocessing
        pre = m[:-1].fit(X)
        Xp = pre.transform(X)
        m[-1].fit(Xp, y, groups=groups)
    else:
        m.fit(X, y)
    return m, time.time() - t0


# --------------------------------------------------------------------------- metrics
def load_trip_meta(root, trip_ids):
    """arcs + reference route per trip (cheap files only) for continuity / route-level metrics."""
    meta = {}
    for t in trip_ids:
        base = Path(root) / t / t
        arcs = np.loadtxt(f"{base}.arcs", dtype=np.int64, ndmin=2)
        rt = np.loadtxt(f"{base}.route", dtype=np.int64, ndmin=1) if Path(f"{base}.route").exists() else None
        meta[t] = (arcs, rt)
    return meta


def top1_table(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """One row per observation: predicted arc (highest score), hit flag, reachable flag."""
    d = df[["trip", "obs", "arc", "label", "gid"]].copy()
    d["p"] = p
    idx = d.groupby("gid")["p"].idxmax()
    top = d.loc[idx].sort_values(["trip", "obs"]).reset_index(drop=True)
    reach = d.groupby("gid")["label"].max()
    top["reachable"] = top["gid"].map(reach).astype(bool)
    return top


def continuity(top: pd.DataFrame, meta) -> float:
    """Fraction of consecutive observations whose predicted arcs are equal or head-to-tail adjacent."""
    ok = tot = 0
    for t, g in top.groupby("trip"):
        arcs = meta[t][0]
        a = g["arc"].values; o = g["obs"].values
        sel = np.diff(o) == 1
        a0, a1 = a[:-1][sel], a[1:][sel]
        good = (a0 == a1) | (arcs[a0, 1] == arcs[a1, 0])
        ok += good.sum(); tot += len(good)
    return ok / max(tot, 1)


def route_set_f1(top: pd.DataFrame, meta) -> float:
    """Sequence-level score: mean per-trip F1 between the predicted arc set and the reference route arc set."""
    f = []
    for t, g in top.groupby("trip"):
        ref = set(meta[t][1].tolist()); pred = set(g["arc"].tolist())
        tp = len(ref & pred)
        pr = tp / max(len(pred), 1); rc = tp / max(len(ref), 1)
        f.append(0 if tp == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f))


def evaluate(df: pd.DataFrame, p: np.ndarray, meta) -> dict:
    """Metrics for candidate scores p on candidate table df (all trips given)."""
    top = top1_table(df, p)
    reach = top["reachable"].values
    hit = top["label"].values == 1
    n_obs = len(top)
    out = {
        "n_obs": n_obs,
        "candidate_miss_rate": float(1 - reach.mean()),
        "top1_acc_all": float(hit.mean()),
        "top1_acc_reachable": float(hit[reach].mean()) if reach.any() else np.nan,
    }
    # candidate-level classification quality, over observations whose reference arc is a candidate
    rm = trainable_mask(df)
    y = df["label"].values[rm]
    pr = p[rm]
    pred_top = np.zeros(len(df), dtype=np.int8)
    tmp = df[["gid"]].copy(); tmp["p"] = p
    first = tmp.groupby("gid")["p"].idxmax().values
    pred_top[first] = 1
    yp = pred_top[rm]
    out["macro_f1"] = float(f1_score(y, yp, average="macro"))
    out["precision_pos"] = float(precision_score(y, yp, zero_division=0))
    out["recall_pos"] = float(recall_score(y, yp, zero_division=0))
    out["f1_pos"] = float(f1_score(y, yp))
    out["log_loss"] = float(log_loss(y, np.clip(pr, 1e-7, 1 - 1e-7), labels=[0, 1]))
    out["route_continuity"] = float(continuity(top, meta))
    out["route_set_f1"] = route_set_f1(top, meta)
    return out


def fast_macro_f1(gid: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Macro-F1 (top-1-per-observation decision rule) over reachable observations; gid must be contiguous."""
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    gmax = np.maximum.reduceat(p, starts)
    sizes = np.diff(np.r_[starts, len(p)])
    pred = (p >= np.repeat(gmax, sizes)).astype(np.int8)
    reach = np.repeat(np.maximum.reduceat(y, starts) > 0, sizes)
    return float(f1_score(y[reach], pred[reach], average="macro"))
