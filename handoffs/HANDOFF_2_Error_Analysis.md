# HANDOFF 2: Where do the models go wrong? Error analysis, calibration and feature importance

**Repo:** https://github.com/RealAdhi/ML2026
**Your branch name:** `handoff-errors`
**Time needed:** about 3 hours
**Difficulty:** medium. You read saved predictions, count mistakes and plot. You do not train anything.
**Mission-critical?** No. This gives the report a "why did the model fail where it failed" section, which evaluators like. Nothing else depends on you, and you will not edit any existing file.

---

## STEP 0. Read first, touch nothing until you finish this

1. Open the repo link above and read `README.md` **from the first line to the last line**. All 11 sections. Sections 4 (candidates and features), 5 (the five models) and 6 (comparison) are the ones you will build on. Section 8 (HMM) is still being filled in by the lead; read what is there.
2. Read every file in the `context/` folder. These are the event's own documents:
   * `context/Main Guidelines.pdf`
   * `context/Dataset Exploration.pdf`
   * `context/Models and Training Guidelines.pdf` (most important for you: section 5 defines the metrics)
   * `context/Deliverables.pdf` (report format, section 3 is the model section you will feed)
   * `context/Assessment_and_Academic_Integrity_Policy.md`
3. Answer these six questions **in your own words** in the description of your pull request (Step 5).
   1. What are the three data partitions, and which one may we never train or tune on?
   2. Why is the nearest road not always the correct road?
   3. What is a "candidate", and what does the classifier predict?
   4. Which five models did we compare and which one did we select?
   5. What does the HMM/Viterbi step add on top of the classifier?
   6. Which rule in the integrity policy changes how you must write your part?

---

## STEP 1. Setup (15 minutes)

You need the raw data folders `train/` and `student_test/` in the repo root (not on GitHub; ask the team lead if you do not have them).

```bash
git clone https://github.com/RealAdhi/ML2026.git
cd ML2026
git config user.name  "YOUR NAME"
git config user.email "YOUR EMAIL USED ON GITHUB"
git checkout -b handoff-errors

python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      Git Bash / macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt

# place train/ and student_test/ in the repo root, then build the candidate tables (about 15 seconds):
python scripts/01_build_dataset.py
```

You should now have `data/processed/student_test.pkl`. The five models' saved predictions are already in the repo at `results/models/student_test_predictions.pkl`.

**What those two files contain**

* `data/processed/student_test.pkl` is `{"df": <DataFrame>, "info": <dict>}`.
  * `df` has one row per (GPS observation, candidate road arc). Useful columns: `trip` (string id), `obs` (observation index inside the trip), `arc` (candidate arc id), `d` (distance in metres from the GPS point to the arc), `d_rank` (0 = nearest candidate), `label` (1 if this candidate is the reference arc, else 0).
  * `info[trip]["ref_arc"]` is a numpy array with the reference arc id of every observation of that trip, so `info[trip]["ref_arc"][obs]` is the correct arc for observation `obs`.
* `results/models/student_test_predictions.pkl` is `{"keys": <DataFrame>, "proba": {model_name: numpy array}}`. `proba[model]` has one probability per row of `df`, in the same order. Model names: `logreg`, `random_forest`, `lightgbm`, `xgboost`, `mlp`. The model's prediction for an observation is the candidate with the highest probability.

---

## STEP 2. Rules (read once, they matter for grading)

* **Only create new files.** Do not edit any existing file (no changes to `README.md`, `src/`, `scripts/00..08`, `api/`, `ui/`). The lead merges your branch.
* Write all explanations **in your own words**. The event runs plagiarism and AI-text checks, and you may be asked to explain your part at the viva.
* Every number in `docs/error_analysis.md` must come from a CSV you produced. No guessing.
* Never look for the evaluator test data. You only use `student_test/` (and `train/` to build the files).
* Commit under your own name and email. No co-author lines.

---

## STEP 3. What to build

### Deliverable A: `scripts/11_error_analysis.py`

Running `python scripts/11_error_analysis.py` from the repo root must create these files in `results/analysis/` (create the folder in the script). It must finish in under 3 minutes.

| Output file | What it contains |
|---|---|
| `per_trip_accuracy.csv` | Rows = the 10 student-test trips, columns = the 5 models, value = top-1 accuracy of that model on that trip (share of observations whose highest-probability candidate is the reference arc; observations whose reference arc is not a candidate count as wrong). Add a column `n_obs`. |
| `per_trip_accuracy_heatmap.png` | Heatmap of the table above (matplotlib `imshow`, numbers written inside the cells, colour bar). |
| `error_types_<model>.csv` and `error_types.png` | Counts of each error type for **all five models**, plus one grouped or stacked bar chart comparing the models. Error types are defined below. |
| `calibration.png` | Reliability diagram: 5 lines (one per model) of mean predicted probability (x) against observed share of positives (y) using 10 bins, plus the diagonal. Use `sklearn.calibration.calibration_curve`. Use only rows of observations whose reference arc is a candidate. |
| `pr_curves.png` | Precision-recall curve of each model (`sklearn.metrics.precision_recall_curve`), same rows as above. Put the average precision (`average_precision_score`) in the legend. |
| `xgboost_feature_importance.png` | Horizontal bar chart of the 28 feature importances of the selected XGBoost model. **Only after the lead tells you the final model is pushed** (it will be at `models/final/model.joblib`, pull `main` again). Until then the script must skip this figure without crashing (check `os.path.exists`). |

**The five error types** (apply in this order to every observation whose prediction is wrong):

| Type | Meaning |
|---|---|
| `candidate_miss` | the reference arc is not among the candidates of that observation, so no model could get it right |
| `duplicate_arc` | the predicted arc connects the same two nodes as the reference arc but has a different arc id (the data contains duplicate arcs) |
| `reverse_direction` | the predicted arc runs between the same two nodes in the opposite direction (two-way road, wrong direction) |
| `adjacent_arc` | the predicted arc shares at least one node with the reference arc (the model picked the neighbouring segment) |
| `other` | none of the above |

**Starter code (tested helpers, copy these, write the rest yourself).** Create `scripts/11_error_analysis.py` starting like this:

```python
"""Error analysis of the five models on the student-test partition. Writes to results/analysis/."""
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.candidates import FEATURES  # the 28 feature names, in model order

OUT = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True)
MODELS = ["logreg", "random_forest", "lightgbm", "xgboost", "mlp"]


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
            types.append("correct"); continue
        if not reach:
            types.append("candidate_miss"); continue
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


# TODO(you) 1: df, info, proba = load_inputs();  loop over MODELS, build best -> classify_errors for each model.
# TODO(you) 2: per_trip_accuracy.csv (+ n_obs column) and per_trip_accuracy_heatmap.png.
# TODO(you) 3: error_types_<model>.csv for each model and the comparison chart error_types.png.
# TODO(you) 4: calibration.png and pr_curves.png (use rows of observations with reachable == True).
#              Hint: row-level labels are df["label"], row-level probabilities are proba[model];
#              keep only rows whose observation is reachable: build a boolean mask with
#              df.groupby(["trip","obs"])["label"].transform("max") > 0
# TODO(you) 5: xgboost_feature_importance.png, only if models/final/model.joblib exists.
#              model = joblib.load(path); importances = model.feature_importances_; names = FEATURES
if __name__ == "__main__":
    pass  # replace with your code
```

Sanity checks that tell you your code is right:
* For the `xgboost` model, `correct.mean()` over all observations should be **about 0.862** (the README table says student-test top-1 = 0.862). If you get something very different, your join is wrong.
* The error counts of one model must add up to the number of observations (19,160).
* Calibration lines for XGBoost and Random Forest should hug the diagonal; the MLP and Logistic Regression may bend away. Say what you see.

### Deliverable B: `docs/error_analysis.md` (your own words, with numbers)

1. **Per-trip table.** Paste the table from `per_trip_accuracy.csv` and name the easiest and the hardest student-test trip. Open the Folium map of the hardest trip if the lead has pushed maps (`results/maps/pred_student_<trip>.html`; ask if missing) and write one sentence on what makes it hard.
2. **Error types.** A table for the selected model (XGBoost): count and percentage of each error type. Then answer in 3 to 4 sentences: which error type dominates, and why does that match what the README says about two-way roads and duplicate arcs?
3. **Calibration.** Two sentences per model that is clearly over- or under-confident. Why does calibration matter for the HMM? (Hint: the HMM multiplies these probabilities over time; README section 8 and `docs/how_it_works.md` explain.)
4. **Feature importance.** The top 8 features with one sentence each on why that feature plausibly matters. (Only after the final model is pushed; leave a clearly marked "TO ADD" if it is not there yet.)
5. **What we would improve with more time.** Five concrete ideas, each tied to an error you measured above. Example format: "Because N% of errors are `reverse_direction`, add a feature that ...".

---

## STEP 4. Acceptance checklist (tick every box before you open the pull request)

- [ ] `python scripts/11_error_analysis.py` runs from a clean checkout in under 3 minutes without errors (the feature-importance part is skipped cleanly if the final model file does not exist yet).
- [ ] `results/analysis/` contains every file from the table.
- [ ] The XGBoost overall accuracy you compute matches the README (about 0.862).
- [ ] Figures have titles, axis labels and legends; the heatmap shows the numbers.
- [ ] `docs/error_analysis.md` has all five sections and every number is traceable to a CSV.
- [ ] You changed **no** existing file (`git status` shows only new files).
- [ ] Your PR description contains the answers to the six questions from Step 0.

---

## STEP 5. How to hand in

```bash
git add scripts/11_error_analysis.py docs/error_analysis.md results/analysis
git commit -m "Add error analysis, calibration and per-trip accuracy for the five models"
git push -u origin handoff-errors
```

Then on GitHub: **Pull requests, New pull request, base: `main`, compare: `handoff-errors`**. Paste your answers to the six questions in the description and tag the team lead. Do not merge it yourself.

If anything is unclear, ask the team lead before guessing.
