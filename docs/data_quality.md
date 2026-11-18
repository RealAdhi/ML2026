# Data quality notes and modelling assumptions

All numbers come from `results/eda/` (scripts `00_eda_stats.py`, `05_eda_plots.py`) and `scripts/01_build_dataset.py`.

## What the raw data looks like
* 80 training trips, 10 student-testing trips. Every trip has its own road sub-network (299 to 155,003 nodes, 649 to 307,392 arcs).
* The trips are not from one city: longitudes run from about -100 to 137 degrees and latitudes from about -35 to 61 degrees. A single UTM zone would be wrong, so each trip is projected with its own local azimuthal-equidistant projection (metres).
* GPS is nominally 1 Hz, but 45 of 80 training trips contain gaps above 5 s (largest gap 153,064 s). Student-test trips: 2 of 10, largest gap 210 s.
* No NaN coordinates or timestamps, timestamps are monotonic, no isolated nodes. Road networks are fragmented (several connected components per trip) and 53 of 80 training trips contain duplicate arcs (same from/to nodes, different arc id).
* `.route` is an ordered arc list. 99.9 % of consecutive route arcs are head-to-tail connected.

## The reference route is not per-observation ground truth
`.route` lists arcs, not which arc each GPS fix belongs to. We align fixes to route arcs with a monotone dynamic programme (cost = distance to the arc, backward moves heavily penalised); see `reference_alignment` in `src/pathsnap/candidates.py`. Labels are therefore derived, not given.

## Trips where the reference route does not match the GPS track
For most trips the GPS-to-reference distance is a few metres. For a few it is not:

| trip | share of fixes with no candidate arc within 60 m of the reference |
|---|---|
| 54 | 88.7 % (19,313 fixes) |
| 33 | 66.8 % |
| 72 | 13.2 % |
| 2  | 11.6 % |

Overall candidate miss rate: 10.9 % on the training partition, 0.2 % on the student-test partition (`results/eda/`, `scripts/01_build_dataset.py` output).

What we did, and why:
* Observations whose reference arc is not among the candidates carry no usable positive label. They are **not used to fit models** (they can only teach "everything is negative").
* They are **not hidden from evaluation**: `top1_acc_all` counts them as errors and the candidate miss rate is reported next to every result. Cross-validation macro-F1, precision and recall are computed on observations whose reference arc is a candidate, and every table says which basis it uses.
* The decoder-tuning objective leaves out trips 54 and 33 because their labels are unreliable for most fixes.
* No whole trip was removed from the training partition for model fitting; only the unlabelled observations were.

This is a choice we made because of label noise, not a standard procedure we are citing.

## Other assumptions
* Candidates: up to 12 nearest directed arcs within 60 m of each fix (fallback: the single nearest arc when none is within 60 m).
* Kinematic features (heading, speed) are set to missing across gaps longer than 5 s and handled by median imputation (linear and neural models) or natively (tree models).
* Duplicate arcs with identical geometry cannot be told apart by geometry; the classifier can only pick between them using network features, so some errors there are unavoidable.
