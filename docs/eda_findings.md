# EDA findings, presentation notes and a fact check of README section 3

Scope: the `train` (80 trips) and `student_test` (10 trips) partitions only. The evaluator test data was never opened.

Sources used for every number below:

* `results/eda/dataset_inventory.csv` (one row per trip, 90 rows)
* `results/eda/candidate_miss_by_trip_train.csv`
* `results/eda/gps_to_reference_distance_train.csv`
* `results/eda/missing_value_summary.csv`
* `results/eda_extra/trip_summary.csv` and `results/eda_extra/trip_difficulty_ranking.csv` (produced by `scripts/10_eda_overview.py`)

Figures referenced live in `results/eda/` and `results/eda_extra/`.

---

## 1. Ten findings

1. **The dataset is small in trips but large in points.** 90 trips hold 227,577 GPS observations, 208,417 in `train` and 19,160 in `student_test` (`results/eda/dataset_inventory.csv`; see `results/eda/inventory.png`). This is why validation has to be grouped by trip: 90 groups is few, but 227,577 rows looks like plenty if you split carelessly.

2. **Trip sizes differ by a factor of 50.** Training trips run from 361 to 19,313 GPS points, median 1,529 (`results/eda/dataset_inventory.csv`; see `results/eda/inventory.png`). One trip alone (54, 19,313 points) is 9.3 % of all training observations, so a single bad trip can move a global average.

3. **The trips are spread across the world, not one city.** Longitudes run from -100.42 to 137.24 and latitudes from -34.64 to 61.70, and grouping the 90 trip centroids so that any two within 3 degrees join the same group gives **37 separate regions**, the largest holding only 17 trips (`results/eda_extra/trip_summary.csv`; see `results/eda_extra/trip_locations.html`). No single map projection can serve 37 regions spanning both hemispheres, which is why each trip is projected to metres around its own centre.

4. **Sampling is close to 1 Hz, with a thin tail of gaps.** 96.50 % of training steps are exactly 1 s and only 0.58 % exceed 5 s — 1,216 of 208,337 steps (`results/eda/dataset_inventory.csv`; see `results/eda/speed_and_gaps.png`). Gaps are rare but 45 of 80 training trips contain at least one.

5. **A few gaps are enormous and distort any duration-based number.** Trip 90 spans 42.79 hours end to end but covers only 8.58 km, because a single 153,064 s gap sits inside it; computing speed as distance over elapsed time gives 0.20 km/h for a vehicle that was plainly driving normally (`results/eda_extra/trip_summary.csv`; see `results/eda_extra/trip_length_vs_duration.png`, where trip 90 is the isolated point at the far right). Only 2 trips have a gap longer than an hour and 11 have one longer than 5 minutes, so this affects few trips but ruins any duration-based average that includes them.

6. **The data is clean in the ordinary sense.** Zero NaN values, zero non-monotonic timestamps and zero isolated nodes across all 90 trips (`results/eda/missing_value_summary.csv`). Every data-quality problem in this project is structural, not missing-value cleaning.

7. **Road networks are large and badly fragmented.** Training networks hold 649 to 307,392 arcs with a median of 20 connected components and a maximum of 421 (`results/eda/dataset_inventory.csv`; see `results/eda_extra/network_size_vs_components.png`). Fragmentation rises with network size, so no trip's network may be assumed to be one connected graph — the HMM has to tolerate transitions it cannot route.

8. **Most roads exist twice, once per direction.** The median share of a trip's arcs that have a reverse twin between the same two nodes is **0.95 in `train` and 0.93 in `student_test`**, and 40 of the 80 training trips sit above 0.95 (`results/eda_extra/trip_summary.csv`; see `results/eda_extra/two_way_share.png`); 53 of 80 training trips also contain outright duplicate arc pairs (`results/eda/missing_value_summary.csv`). Two arcs with identical geometry sit at identical distance from a GPS point, so distance cannot separate them and direction features must.

9. **The GPS trace normally sits within a few metres of its reference route.** The median of the per-trip median fix-to-reference distances is 4.64 m, and 77 of 80 training trips have a median under 10 m (`results/eda/gps_to_reference_distance_train.csv`; see `results/eda/gps_to_reference.png`). So for the large majority of trips the reference route is a fair target to learn against.

10. **Two trips are the whole data-quality problem.** Trip 54 misses its reference arc for 88.66 % of fixes and trip 33 for 66.77 %, against 13.21 % for the next worst (trip 72); those two trips produce 92.57 % of all 22,712 candidate misses, and the training miss rate falls from 10.90 % to 0.92 % without them (`results/eda/candidate_miss_by_trip_train.csv`, `results/eda_extra/trip_difficulty_ranking.csv`; see `results/eda/candidate_miss_by_trip.png`). Their median fix-to-reference distances are 138.63 m and 122.42 m, far beyond GPS noise, so their reference routes describe a different path from their traces.

---

## 2. How to present it in 3 minutes

Open with the shape of the data: 90 trips, 227,577 GPS points, and every trip carrying its own road network of up to 307,392 arcs. Show `results/eda_extra/trip_locations.html` first and point out that the trip centroids land in several separate parts of the world, which is the reason each trip is converted to metres around its own centre rather than through one shared map projection. Say that the data is clean in the usual sense — no missing values, no time reversals, no isolated nodes — so the interesting problems are structural ones. Move to `results/eda_extra/network_size_vs_components.png` and show that the bigger a trip's road network is, the more disconnected pieces it breaks into, median 20 components and up to 421, so connectivity can never be assumed. Then show `results/eda_extra/two_way_share.png` and make the central point: in the median training trip 95 % of arcs have a reverse twin with identical geometry, so the two directions of a street sit at exactly the same distance from a GPS point and the nearest road is a coin flip between them — this is why we engineer direction features instead of just snapping to the closest arc. Add the supporting number from `results/eda/nearest_arc_baseline.png`: the nearest arc is correct only 55.9 % of the time. Finish on data quality with `results/eda/candidate_miss_by_trip.png`: the overall training candidate miss rate looks alarming at 10.90 %, but two trips cause 92.57 % of it, and without them the rate is 0.92 %. State what we did about it — those fixes are not used for training because they carry no correct answer, but they are still counted as errors in `top1_acc_all`, so nothing is hidden from the score. Close by noting that `student_test` is a much cleaner partition than `train`, which is why later test scores sit above cross-validation scores; that is a property of the data, not evidence of a better model.

---

## 3. Ten EDA viva questions and answers

**Q1. Why can you not use plain Euclidean distance on longitude and latitude?**
Degrees are not metres, and one degree of longitude shrinks as you move away from the equator. Our trips span latitudes from -34.6 to 61.7, so the same longitude difference means very different ground distances in different trips. We project each trip into local metres around its own centre first, and only then measure distances.

**Q2. What is an arc and how does it differ from a node?**
A node is a junction, stored as one coordinate pair; its node id is its row number in the `.nodes` file. An arc is a directed road segment, stored as a pair of node ids (from, to); its arc id is its row number in the `.arcs` file. Nodes are points, arcs are the one-way connections between them, and a two-way street is normally stored as two arcs pointing opposite ways.

**Q3. Why does each trip have its own road network instead of one shared map?**
The trips come from different parts of the world, so there is no single network that covers them. Each trip ships the road data around its own trace. It also means network statistics such as arc count or component count are per-trip properties, and a model must not learn anything that depends on a specific network's node numbering.

**Q4. How did you handle the trips with huge time gaps?**
We did not delete them. A gap breaks the assumption that the next fix is near the previous one, so the kinematic features that depend on consecutive fixes — speed and heading — are left missing across a gap rather than computed from a 42-hour interval. The tree models take those missing values natively; the linear model and the neural network get median imputation with a missing indicator. We also avoid reporting mean speed as distance over elapsed time for those trips, because the elapsed time is mostly waiting.

**Q5. Is the `.route` file a per-point ground truth?**
No. It is an ordered list of arc ids for the whole trip, and it never says which GPS point sits on which arc. We derive the per-point label ourselves with a monotone dynamic programme that walks the fixes along the route in order, paying the distance to each arc and penalising any backwards move. So our labels are derived, and any error in that alignment is our error, not the data's.

**Q6. What is a candidate, and why cap it at 12?**
A candidate is a directed arc within 60 m of a GPS fix, and we keep at most the 12 nearest. The cap exists because a trip network can hold 307,392 arcs and a classifier cannot be asked to choose one of those directly; reducing it to a shortlist turns the problem into a small binary classification per row. The cost of the cap is that the true arc occasionally falls outside the shortlist, which is what the candidate miss rate measures.

**Q7. Why is the nearest road not simply the answer?**
Because most roads are stored as two arcs with identical geometry, so both directions lie at exactly the same distance from the fix. Distance cannot break that tie at all. The nearest-arc baseline is correct only 55.9 % of the time, which is close to a coin flip between the two directions, and that is the single strongest argument for using a classifier with direction features.

**Q8. Your training candidate miss rate is 10.9 %. Does that not mean your candidate generator is broken?**
No, and the per-trip breakdown is the evidence. Trips 54 and 33 account for 92.57 % of the misses, and the rate across the other 78 trips is 0.92 %. Those two trips have median fix-to-reference distances of 138.63 m and 122.42 m, which is far outside GPS noise, so their reference route describes a different journey from their trace. The generator is correctly failing to find a road that is not there.

**Q9. You excluded data — how is that not manipulating your score?**
What is excluded from model fitting is the fixes that have no correct answer among their candidates, because there is no positive label to learn from. Nothing is excluded from evaluation: `top1_acc_all` still counts every one of those 22,712 fixes as an error, and the candidate miss rate is reported next to every table. No whole trip was dropped from fitting. The decoder tuning objective does leave out trips 54 and 33, because tuning against a reference route we believe is wrong would tune towards the wrong answer.

**Q10. Why do the `student_test` scores come out higher than the cross-validation scores?**
Because `student_test` is an easier partition, not because the models improved. Its candidate miss rate is 0.20 % against 10.90 % in training, its longest time gap is 210 s against 153,064 s, and it contains none of the trips whose reference routes disagree with their traces. Cross-validation is the honest estimate of difficulty; the test number is flattered by the partition.

---

## 4. Fact check of README section 3

Method: every number in README section 3 was recomputed from the committed CSVs in `results/eda/`. "OK" means the recomputed value matches the README to the precision the README quotes.

### 4.1 Section 3.2, the inventory table

| README claim | Recomputed | Verdict |
|---|---|---|
| Trips 80 / 10 | 80 / 10 | OK |
| GPS observations 208,417 / 19,160 | 208,417 / 19,160 | OK |
| GPS per trip min/median/max 361 / 1,529 / 19,313 | 361 / 1,529 / 19,313 | OK |
| GPS per trip (student_test) 262 / 1,420 / 3,958 | 262 / 1,420 / 3,958 | OK |
| Nodes per trip 299 / 155,003 and 1,770 / 26,635 | identical | OK |
| Arcs per trip 649 / 307,392 and 3,577 / 56,027 | identical | OK |
| Components median/max 20 / 421 and 11 / 100 | identical | OK |
| Reference route arcs 59 / 2,266 and 95 / 1,323 | identical | OK |
| Mean step between fixes 16.6 m / 15.7 m | 16.61 m / 15.68 m | OK |
| Trips with a gap longer than 5 s: 45 of 80 / 2 of 10 | 45 / 2 | OK |
| Longest gap 153,064 s / 210 s | identical | OK |
| Trips containing duplicate arcs 53 of 80 / 7 of 10 | 53 / 7 | OK |
| NaN / non-monotonic timestamps / isolated nodes all 0 | all 0 in both partitions | OK |

### 4.2 Section 3.2, the bullet points

| README claim | Recomputed | Verdict |
|---|---|---|
| 96.5 % of training fixes arrive exactly 1 s after the previous one | 96.50 % (weighted by steps per trip) | OK |
| only 0.6 % arrive more than 5 s after | 0.58 % — 1,216 of 208,337 steps | OK |
| 227,577 GPS observations over 90 trips | 227,577 over 90 | OK |
| Longitudes about -100 to 137 | -100.42 to 137.24 | OK |
| Latitudes about -35 to 62 | -34.64 to 61.70 | OK |
| 98.1 % to 100 % of route arcs connect head to tail | minimum 98.13 % (train), 99.73 % (student_test) | OK |

### 4.3 Section 3.4, label quality

| README claim | Recomputed | Verdict |
|---|---|---|
| Median of the per-trip medians 4.6 m | 4.64 m | OK |
| "95th percentile about 14 m" | see note below | **Wording is ambiguous** |
| 77 of 80 trips sit in the first bar | 77 trips have a per-trip median below 10 m | OK |
| Trips 54 and 33 medians 138.6 m / 122.4 m | 138.63 m / 122.42 m | OK |
| Trips 54 and 33 reach 86 % and 52 % of fixes beyond 60 m | 86.01 % / 52.40 % | OK |
| Trip 54 misses 88.7 %, trip 33 misses 66.8 % | 88.66 % / 66.77 % | OK |
| trip 72: 13.2 %, trip 2: 11.6 %, trip 61: 6.5 % | 13.21 % / 11.57 % / 6.50 % | OK |
| Overall training candidate miss rate 10.9 %, 22,712 of 208,417 | 10.897 %, 22,712 of 208,417 | OK |
| Trips 54 and 33 account for 92.6 % of misses (17,122 and 3,902) | 92.57 %, 17,122 and 3,902 | OK |
| Without them the miss rate is 0.9 % | 0.921 % | OK |
| Trip 54 alone is 9.3 % of all training fixes | 9.27 % | OK |

**Note on the 95th percentile.** The README sentence reads "median of the per-trip medians 4.6 m, 95th percentile about 14 m", which can be read two ways, and only one of them gives 14 m:

* the 95th percentile **of the per-trip medians** is **7.83 m**;
* the **median of the per-trip 95th percentiles** is **14.29 m**.

So the 14 m figure is correct under the second reading only. This is a wording problem rather than a wrong number. Suggested rewording for the lead: "median of the per-trip medians 4.6 m; the typical trip's 95th-percentile distance is about 14 m". **The README was not edited** — this is reported here for the lead to action, per the handoff rules.

### 4.4 Claims that could not be checked against a committed CSV

These numbers appear in README section 3 but the CSV behind them is not committed, only the figure. They are not disputed; they simply could not be independently recomputed from this branch.

| README claim | Why it could not be checked |
|---|---|
| student_test candidate miss rate 0.20 % | no `candidate_miss_by_trip_student_test.csv` is committed |
| 34 % of fixes hit the 12-candidate cap | `candidate_miss_by_trip_train.csv` records `n_cands_mean` only (weighted mean 9.00 candidates per fix, which is consistent with heavy capping) |
| Nearest arc correct 55.9 %, second nearest 36.5 %, rest 7.6 % | computed inside `scripts/05b_extra_eda.py`, figure only, no CSV written |
| 89.3 % of second-nearest cases are exact distance ties; 65.1 % of fixes have the two nearest tied | same, figure only |
| Median speed 39.9 km/h, 95th percentile 119.9 km/h, 99th 167 km/h | same, figure only |

**Suggestion for the lead:** having `scripts/05b_extra_eda.py` also write the numbers behind `nearest_arc_baseline.png` and `speed_and_gaps.png` to CSV would make every number in README section 3 reproducible from committed files, which the assessment policy asks for under reproducibility.

### 4.5 Summary

30 of the 30 checkable numbers in README section 3 match the committed CSVs. One sentence has ambiguous wording (the 95th percentile in section 3.4) and five numbers rest on figures whose underlying CSVs are not committed. No numeric error was found.
