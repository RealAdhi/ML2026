# HANDOFF 3: Automated tests and the two-laptop demo runbook

**Repo:** https://github.com/RealAdhi/ML2026
**Your branch name:** `handoff-tests-runbook`
**Time needed:** about 3 hours (part 1: tests, part 2: runbook, part 3: dry-run after the lead tells you the system is ready)
**Difficulty:** easy to medium. You mostly copy tested starter code, add a few tests of your own, and write clear instructions.
**Mission-critical?** Not for the models, but the demo day depends on a clean runbook. You test and document the system; you do not change it. You will not edit any existing file.

---

## STEP 0. Read first, touch nothing until you finish this

1. Open the repo link above and read `README.md` **from the first line to the last line**. All 11 sections. Section 9 (API, UI) and section 11 (How to run everything) are the ones you will test and document. Sections 7, 8 and 9 are still being filled in by the lead; read what is there.
2. Read every file in the `context/` folder. These are the event's own documents:
   * `context/Main Guidelines.pdf` (section 3 describes the two-laptop architecture you will document)
   * `context/Dataset Exploration.pdf`
   * `context/Models and Training Guidelines.pdf` (section 6 describes exactly what happens at the final evaluation)
   * `context/Deliverables.pdf` (section 7 lists the screenshots the report needs)
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
git checkout -b handoff-tests-runbook

python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      Git Bash / macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install pytest==9.2.0 httpx==0.28.1
```

---

## STEP 2. Rules (read once, they matter for grading)

* **Only create new files.** Do not edit any existing file (no changes to `README.md`, `src/`, `scripts/`, `api/`, `ui/`). If a test fails because you think the real code is wrong, **do not fix the code**: write the failing test name and what you expected in `docs/test_findings.md` and tell the lead.
* Write all explanations **in your own words**. The event runs plagiarism and AI-text checks, and you may be asked to explain your part at the viva (for example "how do you start the API and how do you know it works?").
* Never look for the evaluator test data. You only use `train/` and `student_test/`.
* The ground truth must never be shown in the UI. If during your dry-run you ever see the reference route in the browser, stop and tell the lead immediately.
* Commit under your own name and email. No co-author lines.

---

## PART 1. Automated tests (about 1.5 hours)

### Files to create

| File | Purpose |
|---|---|
| `requirements-dev.txt` | exactly two lines: `pytest==9.2.0` and `httpx==0.28.1` |
| `tests/conftest.py` | makes `src` and the repo root importable (starter below) |
| `tests/test_geometry.py` | tests for `pathsnap.candidates.seg_dist` (starter below) |
| `tests/test_metrics.py` | tests for `pathsnap.decode_eval.seq_metrics` (starter below) |
| `tests/test_hmm_toy.py` | tests that the Viterbi decoder uses road connectivity (starter below) |
| `tests/test_api.py` | tests for the FastAPI backend (starter below). They are skipped automatically until `models/final/model.joblib` exists; the lead will tell you when it is pushed, then `git pull origin main` and run them. |
| `tests/test_data_and_candidates.py` | **you write this one** (spec below) |
| `docs/test_findings.md` | short log: how to run the tests, the result, anything suspicious (spec below) |

The starter files below are **already tested by the lead: 9 tests pass and 4 API tests are skipped.** Copy them exactly, run them, then write your own tests.

**`tests/conftest.py`**
```python
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
```

**`tests/test_geometry.py`**
```python
import numpy as np

from pathsnap.candidates import seg_dist


def one(px, py, ax, ay, bx, by):
    d, t = seg_dist(np.array([px]), np.array([py]), np.array([ax]), np.array([ay]), np.array([bx]), np.array([by]))
    return float(d[0]), float(t[0])


def test_point_above_middle_of_segment():
    d, t = one(0, 1, -1, 0, 1, 0)          # segment on the x axis from -1 to 1, point at (0, 1)
    assert d == 1.0
    assert t == 0.5


def test_point_beyond_end_is_clamped_to_endpoint():
    d, t = one(3, 0, -1, 0, 1, 0)          # closest point is the end (1, 0), 2 metres away
    assert d == 2.0
    assert t == 1.0


def test_point_on_segment_has_zero_distance():
    d, _ = one(0.5, 0, 0, 0, 1, 0)
    assert d == 0.0


def test_zero_length_segment_does_not_crash():
    d, t = one(3, 4, 0, 0, 0, 0)           # degenerate arc: distance to the single point is 5
    assert d == 5.0
    assert t == 0.0
```

**`tests/test_metrics.py`**
```python
import numpy as np

from pathsnap.decode_eval import seq_metrics

# toy road: node chain 0 -> 1 -> 2 -> 3, arc i goes from node i to node i+1 (arc ids 0, 1, 2)
ARCS = np.array([[0, 1], [1, 2], [2, 3]])
ROUTE = np.array([0, 1, 2])


def test_perfect_prediction():
    obs = np.array([0, 1, 2])
    m = seq_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]), obs, ARCS, ROUTE)
    assert m["acc"] == 1.0
    assert m["continuity"] == 1.0
    assert m["route_f1"] == 1.0


def test_wrong_middle_arc_breaks_accuracy_and_continuity():
    obs = np.array([0, 1, 2])
    m = seq_metrics(np.array([0, 2, 2]), np.array([0, 1, 2]), obs, ARCS, ROUTE)   # arc 0 followed by arc 2 is not connected
    assert abs(m["acc"] - 2 / 3) < 1e-9
    assert m["continuity"] == 0.5          # pair (0,2) not connected, pair (2,2) same arc counts as continuous


def test_staying_on_the_same_arc_is_continuous():
    obs = np.array([0, 1, 2, 3])
    m = seq_metrics(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), obs, ARCS, ROUTE)
    assert m["continuity"] == 1.0
    assert m["n_switch"] == 0
```

**`tests/test_hmm_toy.py`**
```python
import numpy as np

from pathsnap.hmm import DecoderParams, Precomputed, viterbi


def toy():
    """Two GPS steps, two candidate arcs each.
    step 0 candidates: arcs 10, 11     step 1 candidates: arcs 20, 21
    road network: 10 -> 21 is connected (route distance 10 m), 11 -> 20 is connected (10 m),
                  10 -> 20 and 11 -> 21 are NOT connected (inf).
    """
    inf = np.inf
    rd = [np.array([[inf, 10.0],
                    [10.0, inf]])]
    return Precomputed(obs=np.array([0, 1]), row_start=np.array([0, 2, 4]), rd=rd,
                       gc=np.array([10.0]), dt=np.array([1.0]), uturn=[np.zeros((2, 2), dtype=bool)],
                       arc=np.array([10, 11, 20, 21]), d=np.zeros(4))


def test_decoder_uses_road_connectivity_to_fix_a_weak_classifier_choice():
    pre = toy()
    p = np.array([0.90, 0.10, 0.55, 0.45])          # classifier alone would pick arcs 10 then 20 (not connected)
    classifier_only = [10, 20]
    decoded = viterbi(pre, p, DecoderParams()).tolist()
    assert decoded != classifier_only
    assert decoded == [10, 21]                      # the connected pair wins


def test_confident_classifier_is_not_overruled_when_connectivity_agrees():
    pre = toy()
    p = np.array([0.05, 0.95, 0.95, 0.05])          # classifier picks 11 then 20, which are connected
    assert viterbi(pre, p, DecoderParams()).tolist() == [11, 20]
```

**`tests/test_api.py`**
```python
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.path.join(ROOT, "models", "final", "model.joblib")
DATA = os.path.join(ROOT, "student_test")

pytestmark = pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.isdir(DATA)),
                                reason="needs models/final/model.joblib and the student_test folder")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_trips_lists_ten_trips(client):
    trips = client.get("/trips").json()["trips"]
    assert len(trips) == 10
    assert all({"trip_id", "n_gps"} <= set(t) for t in trips)


def test_unknown_trip_returns_404(client):
    assert client.post("/predict", json={"trip_id": "does-not-exist"}).status_code == 404


def test_predict_returns_route_and_metrics_but_no_ground_truth(client):
    trip_id = client.get("/trips").json()["trips"][0]["trip_id"]
    r = client.post("/predict", json={"trip_id": trip_id})
    assert r.status_code == 200
    body = r.json()
    assert body["trip_id"] == trip_id and len(body["gps"]) == body["n_obs"]
    assert len(body["predicted_route"]) > 0
    for key in ("acc", "continuity", "route_f1"):
        assert 0.0 <= body["metrics"]["hmm"][key] <= 1.0
    # the reference route must never be sent to the UI
    assert not any("reference" in k or k == "ref_arc" for k in body)
```

Run all tests from the repo root:
```bash
python -m pytest tests -q
```
Expected right now: `9 passed, 4 skipped`.

### The test file you write yourself: `tests/test_data_and_candidates.py`

Write **at least 6 tests**. Skip them cleanly when the data folder is missing (use the same `pytest.mark.skipif(not os.path.isdir(DATA), ...)` pattern as above). Use the first trip of `student_test/`. Look at `src/pathsnap/data.py` (`load_trip`, `list_trips`) and `src/pathsnap/candidates.py` (`generate`, `FEATURES`) to see how they work.

1. `load_trip(root, trip_id)` returns arrays with `len(lon) == len(lat) == len(t) == len(px) == len(py)`, and timestamps are strictly increasing.
2. `load_trip(root, trip_id, with_route=False).route is None` (this is how the API keeps ground truth out of the prediction path).
3. Every arc endpoint is a valid node: `arcs.max() < len(node_x)` and `arcs.min() >= 0`.
4. `generate(trip, with_labels=False)` returns a DataFrame that contains every column listed in `FEATURES`, and **has no `label` column**.
5. Every GPS observation has at least one candidate (`df.obs.nunique() == len(trip.t)`) and `d_rank` starts at 0 for each observation.
6. Candidate distances are non-negative and no observation has more than `MAX_CANDS` candidates (import `MAX_CANDS` from `pathsnap.candidates`).

Bonus (optional): a test that two calls to `generate` on the same trip return identical frames (determinism).

### `docs/test_findings.md` (your own words)

* How to run the tests (two lines).
* The final line of pytest output.
* Any test that failed or looked suspicious, with what you expected. If none, write "none found".

---

## PART 2. The demo runbook `docs/DEMO_RUNBOOK.md` (about 1 hour)

Write a step-by-step document that someone who has never seen the project can follow on demo day. Use the facts below; they are exact.

**The final evaluation (from `context/Models and Training Guidelines.pdf`, section 6):** the API runs on Laptop 1, the UI runs on Laptop 2. The evaluator opens the UI, sees the ten reserved trips, picks any one, the UI sends the trip id to the API, the API runs the whole pipeline and calculates accuracy from its internal reference data, and the UI shows the predicted route and results. It must work for all ten trips. We must show the evaluator the **API logs on the Laptop 1 command window**.

**Exact commands (run from the repo root on each laptop after `pip install -r requirements.txt`):**

Laptop 1 (API). `EVAL_DATA_DIR` is the folder that holds the trips to serve, in the same layout as `student_test/` (`<id>/<id>.track`, `.nodes`, `.arcs`, `.route`). By default it serves `student_test/`.
```powershell
# PowerShell
$env:EVAL_DATA_DIR = "C:\path\to\evaluator_trips"      # optional, default is ./student_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
```bat
:: cmd.exe
set EVAL_DATA_DIR=C:\path\to\evaluator_trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Laptop 2 (UI). The UI is static files in `ui/`:
```bash
python -m http.server 5500 --directory ui
```
then open `http://localhost:5500/?api=http://<LAPTOP-1-IP>:8000` in a browser (the `?api=` part pre-fills the API address box; you can also type it in the box and click Connect).

**Things your runbook must cover** (use headings, numbered steps, and a table where noted):

1. **Before demo day checklist** (checkbox list): both laptops charged, same Wi-Fi or one laptop's hotspot, repo cloned and `pip install` done on both, `models/final/model.joblib` present on Laptop 1, evaluator folder path known, browser ready, screenshots folder created.
2. **Finding Laptop 1's IP address** (Windows: `ipconfig`, which line to read, `IPv4 Address`).
3. **Opening the Windows firewall for the API** (run PowerShell **as Administrator** on Laptop 1):
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```
   and how to remove it after the demo: `Remove-NetFirewallRule -DisplayName "PathSnap API 8000"`.
4. **Check from Laptop 2 that Laptop 1 is reachable** before opening the UI: open `http://<LAPTOP-1-IP>:8000/health` in a browser; expected result is JSON containing `"status":"ok"`.
5. **The demo script**: what to say and click, in order (open UI, point out the 10 trips, click a trip, show the map layers (GPS points, predicted route, classifier-only), show the accuracy table, then show the API log window on Laptop 1). Write it as 8 to 10 short steps.
6. **What the evaluator should see in the API window**: the log lines of the form `POST /predict trip_id=...` followed by `trip=... obs=... hmm_acc=... clf_acc=... continuity=... time=...s`. Include one real example after your dry-run (Part 3).
7. **Troubleshooting table** with at least 8 rows, columns `Symptom | Likely cause | Fix`. Cover: UI says "connection failed", `/health` times out on Laptop 2, `ModuleNotFoundError`, port 8000 already in use, empty trip list (wrong `EVAL_DATA_DIR`), blank map (no internet for map tiles), very slow first request, CORS error in the browser console.
8. **Backup plan**: run both parts on one laptop (`http://localhost:8000`) if the network fails, and say clearly that the two-laptop version is what the rules require.
9. **After the demo**: stop the server (Ctrl+C), remove the firewall rule, do not commit any evaluator data to GitHub.

---

## PART 3. Dry-run and screenshots (about 30 minutes, **only after the lead tells you the system is ready**)

The lead will message you when `models/final/model.joblib` and the API are on `main`. Then:

1. `git pull origin main`, run `python -m pytest tests -q`, all 13 tests of the starter files must now pass with nothing skipped (4 skipped before). Write the result in `docs/test_findings.md`.
2. Do the full two-laptop run following **your own runbook**. Fix any unclear step in the runbook as you go.
3. Save screenshots in `docs/screenshots/` with exactly these names (the report needs them, see `context/Deliverables.pdf` section 7):
   * `ui_trip_list.png`: UI after Connect, showing the ten trips
   * `ui_trip_result.png`: UI after clicking one trip, showing map and metrics
   * `api_log_laptop1.png`: the Laptop 1 command window showing the `POST /predict` lines for at least three different trips
   * `health_check.png`: the `/health` JSON opened from Laptop 2
4. Copy one real log line from your dry-run into runbook item 6.
5. In the runbook, add a table `Trip | Worked? (yes/no) | Time (s)` and fill it for all ten student-test trips by clicking through each one. Any "no" goes to the lead immediately.

---

## STEP 4. Acceptance checklist (tick every box before you open the pull request)

- [ ] `python -m pytest tests -q` passes (with the API tests skipped only if the final model is not on `main` yet).
- [ ] `tests/test_data_and_candidates.py` has at least 6 tests of your own.
- [ ] `requirements-dev.txt`, `docs/test_findings.md` and `docs/DEMO_RUNBOOK.md` exist; the runbook covers all 9 items of Part 2.
- [ ] If Part 3 is possible: 4 screenshots with the exact names and the ten-trip table filled.
- [ ] You changed **no** existing file (`git status` shows only new files).
- [ ] Your PR description contains the answers to the six questions from Step 0.

---

## STEP 5. How to hand in

```bash
git add requirements-dev.txt tests docs/test_findings.md docs/DEMO_RUNBOOK.md docs/screenshots
git commit -m "Add automated tests and the two-laptop demo runbook"
git push -u origin handoff-tests-runbook
```

Then on GitHub: **Pull requests, New pull request, base: `main`, compare: `handoff-tests-runbook`**. Paste your answers to the six questions in the description and tag the team lead. Do not merge it yourself. If Part 3 is not possible yet, open the PR with Parts 1 and 2 and push Part 3 to the same branch later.

If anything is unclear, ask the team lead before guessing.
