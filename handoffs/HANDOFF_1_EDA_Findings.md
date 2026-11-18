# HANDOFF 1: EDA overview maps, extra EDA figures and the EDA presentation script

**Repo:** https://github.com/RealAdhi/ML2026
**Your branch name:** `handoff-eda`
**Time needed:** about 2 to 3 hours
**Difficulty:** easy to medium. No machine learning needed, only reading data files and plotting.
**Mission-critical?** No. This is extra EDA depth for the report and for the EDA part of the viva. Nothing else depends on you, and you will not edit any existing file.

---

## STEP 0. Read first, touch nothing until you finish this

1. Open the repo link above and read `README.md` **from the first line to the last line**. All 11 sections. The EDA part (section 3) is the part you will extend, but sections 1, 4, 5 and 6 explain the rest of the project and you must understand them to present the EDA.
2. Read every file in the `context/` folder. These are the event's own documents:
   * `context/Main Guidelines.pdf`
   * `context/Dataset Exploration.pdf` (most important for you)
   * `context/Models and Training Guidelines.pdf`
   * `context/Deliverables.pdf` (the report format we must follow, section 2 is the EDA section)
   * `context/Assessment_and_Academic_Integrity_Policy.md`
3. Answer these six questions **in your own words** in the description of your pull request (Step 5). Short answers are fine. This is how we know you read everything.
   1. What are the three data partitions, and which one may we never train or tune on?
   2. Why is the nearest road not always the correct road?
   3. What is a "candidate", and what does the classifier predict?
   4. Which five models did we compare and which one did we select?
   5. What does the HMM/Viterbi step add on top of the classifier?
   6. Which rule in the integrity policy changes how you must write your part?

---

## STEP 1. Setup (15 minutes)

You need the raw data folders `train/` and `student_test/` in the repo root (they are not on GitHub because they are the event's data; ask the team lead if you do not have them).

```bash
git clone https://github.com/RealAdhi/ML2026.git
cd ML2026
git config user.name  "YOUR NAME"
git config user.email "YOUR EMAIL USED ON GITHUB"
git checkout -b handoff-eda

python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      Git Bash / macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt

# put the data folders here so the layout is:  ML2026/train/00000001/00000001.track ...  and  ML2026/student_test/...
python scripts/00_eda_stats.py        # about 1 minute, rewrites results/eda/dataset_inventory.csv (same numbers as before)
```

If `python scripts/00_eda_stats.py` prints one line per trip and ends without an error, your setup is correct.

---

## STEP 2. Rules (read once, they matter for grading)

* **Only create new files.** Do not edit any file that already exists (no changes to `README.md`, `src/`, `scripts/00..08`, `api/`, `ui/`). The lead merges your branch, so nothing can clash.
* Put your code in the files named below and write all explanations **in your own words**. The event runs plagiarism and AI-text checks on the report and the repo, and you may be asked to explain your part at the viva.
* Every number you write in `docs/eda_findings.md` must come from a CSV or a figure you produced. No guessing, no rounding to look nicer.
* Never open or look for the evaluator test data. You only use `train/` and `student_test/`.
* Commit under your own name and email. No co-author lines.

---

## STEP 3. What to build

### Deliverable A: `scripts/10_eda_overview.py`

One script. Running `python scripts/10_eda_overview.py` from the repo root must create everything below in the folder `results/eda_extra/` (create the folder in the script). It must finish in under 3 minutes and use only the `.track`, `.arcs` files and the existing CSVs.

| Output file | What it contains |
|---|---|
| `trip_summary.csv` | One row per trip (90 rows: 80 train + 10 student_test). Columns: `partition, trip, n_gps, duration_h, path_km, mean_speed_kmh, centroid_lon, centroid_lat, arc_reverse_share, n_arcs, n_components, max_gap_s, candidate_miss`. `candidate_miss` is filled only for train trips (from `results/eda/candidate_miss_by_trip_train.csv`), empty for student_test. |
| `trip_locations.html` | Folium map. One circle marker per trip at its centroid, blue for train, orange for student_test. Popup shows trip id, number of GPS points, path length in km. Add a legend and layer control (one layer per partition). Zoom the map to fit all markers. |
| `trip_length_vs_duration.png` | Scatter plot of path length (km) against duration (hours), log scale on both axes, colour by partition, one annotated label for the 3 longest trips. |
| `network_size_vs_components.png` | Scatter of number of road arcs (x, log) against number of connected components (y, log) per trip, using `results/eda/dataset_inventory.csv`. Add a one-line title that states what you conclude. |
| `two_way_share.png` | Histogram of `arc_reverse_share` (the share of a trip's arcs that have an arc running the opposite way between the same two nodes), one histogram per partition. |
| `trip_difficulty_ranking.csv` | The 15 hardest training trips, ranked by `candidate_miss` (highest first), with the other `trip_summary.csv` columns next to it. |

**Starter code (tested helpers, copy these, write the rest yourself).** Create `scripts/10_eda_overview.py` starting like this:

```python
"""EDA overview: trip locations, trip length/duration, network connectivity and two-way road share.
Writes everything to results/eda_extra/.  Uses only .track/.arcs files and existing results/eda CSVs."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "eda_extra")
os.makedirs(OUT, exist_ok=True)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between consecutive points (arrays)."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def track_stats(track_file):
    """.track = 3 whitespace columns without header: lon, lat, time_seconds."""
    t = np.loadtxt(track_file, ndmin=2)
    lon, lat, ts = t[:, 0], t[:, 1], t[:, 2]
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    dt = np.diff(ts)
    path_km = step.sum() / 1000.0
    dur_h = (ts[-1] - ts[0]) / 3600.0
    return dict(n_gps=len(ts), path_km=path_km, duration_h=dur_h,
                mean_speed_kmh=path_km / dur_h if dur_h > 0 else np.nan,
                centroid_lon=float(np.median(lon)), centroid_lat=float(np.median(lat)), max_gap_s=float(dt.max()))


def reverse_share(arcs_file):
    """Share of arcs (a->b) for which the opposite arc (b->a) also exists. .arcs = 2 int columns: from_node, to_node."""
    a = np.loadtxt(arcs_file, dtype=np.int64, ndmin=2)
    n = a.max() + 1
    keys = a[:, 0] * n + a[:, 1]
    rkeys = a[:, 1] * n + a[:, 0]
    return float(np.isin(rkeys, keys).mean())


def trips(partition):
    folder = os.path.join(ROOT, partition)
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))


# TODO(you) 1: loop over both partitions, call track_stats and reverse_share for every trip, and merge with
#              results/eda/dataset_inventory.csv (columns n_arcs, n_components) and
#              results/eda/candidate_miss_by_trip_train.csv (column miss -> candidate_miss). Save trip_summary.csv.
# TODO(you) 2: build the Folium map (import folium; folium.CircleMarker, folium.FeatureGroup, folium.LayerControl).
# TODO(you) 3: make the four PNG figures with matplotlib (use matplotlib.use("Agg") before importing pyplot).
# TODO(you) 4: write trip_difficulty_ranking.csv.
if __name__ == "__main__":
    pass  # replace with your code
```

Sanity check you can do in 10 seconds: trip `00000001` should come out at roughly 50 km and about 1315 GPS points, and `arc_reverse_share` should be between 0 and 1.

### Deliverable B: `docs/eda_findings.md` (this is the part you will present)

A document written by you, in your own words, with these four sections:

1. **Ten findings.** Each finding is one bullet: one plain sentence, one number from your CSVs, and the name of the figure that shows it. Example format: "Most trips are short: the median trip has N GPS points (see `results/eda/inventory.png`)." Cover at least: trip count and sizes, the range of locations (how many distinct regions, from your map), time gaps, speed range, how many trips have duplicate or reverse arcs, network fragmentation, the hardest trips.
2. **How to present it in 3 minutes.** A speaking script, about 8 to 10 sentences, that walks through your map and two figures in the order you would show them.
3. **Ten EDA viva questions with your answers.** Write the question an evaluator might ask about the dataset (for example "Why did you not use Euclidean distance on longitude and latitude?", "What is an arc and how is it different from a node?", "How did you handle the trips with huge time gaps?") and answer each in 2 to 3 sentences. The README section 10 has examples to start from; do not copy them, write your own.
4. **Fact check of README section 3.** Read README section 3 (Data and EDA) and compare every number in it with `results/eda/*.csv` and your new `trip_summary.csv`. List each number you checked and write "OK" or write the correct value if something is wrong. If you find a mistake, **do not edit the README**; list it here and tell the lead.

---

## STEP 4. Acceptance checklist (tick every box before you open the pull request)

- [ ] `python scripts/10_eda_overview.py` runs from a clean checkout in under 3 minutes without errors.
- [ ] `results/eda_extra/` contains the 6 outputs from the table, and `trip_summary.csv` has 90 rows.
- [ ] `trip_locations.html` opens in a browser, shows all trips, has a legend and layer control.
- [ ] Figures have titles, axis labels with units, and a readable legend.
- [ ] `docs/eda_findings.md` has all four sections and every number is traceable to a file.
- [ ] You changed **no** existing file (`git status` shows only new files).
- [ ] Your PR description contains the answers to the six questions from Step 0.

---

## STEP 5. How to hand in

```bash
git add scripts/10_eda_overview.py docs/eda_findings.md results/eda_extra
git commit -m "Add EDA overview script, extra figures and EDA presentation notes"
git push -u origin handoff-eda
```

Then on GitHub: **Pull requests, New pull request, base: `main`, compare: `handoff-eda`**. Paste your answers to the six questions in the description and tag the team lead. Do not merge it yourself.

If anything is unclear, ask the team lead before guessing.
