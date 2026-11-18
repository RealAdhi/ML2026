# How PathSnap works (short guide for the viva)

## Pipeline
```
raw trip files -> candidate arcs per GPS fix -> features -> classifier score per (fix, arc)
                                                         -> HMM / Viterbi over the candidate arcs -> one arc per fix
```

1. **Candidates** (`src/pathsnap/candidates.py`, `generate`). The road network is a set of directed arcs (node to node). For each GPS fix we take the nearest arcs within 60 m (max 12). The correct arc is only sometimes the nearest one (about 50 % on training data), so we need a model to choose.
2. **Labels.** The `.route` file gives the ordered arcs of the trip but not which arc each fix is on. `reference_alignment` assigns each fix to a route arc with a monotone dynamic programme. A candidate row is positive if its arc equals the aligned arc.
3. **Features** (28, listed in `FEATURES`): distance to the arc and its rank among candidates, position along the arc, arc length, direction agreement between the GPS movement and the arc (cosine and angle), speed and step lengths, time gaps, node degrees, whether the reverse arc exists, how many candidates agree with the direction, and how well the previous and next fixes also fit the same arc. Direction matters because two-way roads appear as two arcs with the same geometry.
4. **Classifier.** Binary: is this candidate the reference arc? Output is a probability. The arc with the highest probability per fix is the classifier-only prediction.
5. **Model comparison** (`scripts/02_compare_models.py`). Five models, 5-fold `GroupKFold` with trips as groups (a trip never appears in both train and validation), then a refit on all 80 trips and evaluation on the 10 student-test trips.
6. **Bayesian optimisation** (`scripts/03_bayes_opt.py`). Optuna TPE on the selected model, objective = group-CV macro-F1, search record in `results/bayes/<model>/`.
7. **HMM / Viterbi** (`src/pathsnap/hmm.py`).
   * hidden states: the candidate arcs of each fix (arc ids are preserved)
   * emission: log of the classifier probability (normalised across the fix's candidates) plus an optional Gaussian distance term
   * transition: `-|straight-line distance between fixes - network distance between the two projected positions| / beta`; the network distance is a bounded Dijkstra over the arc graph; unreachable pairs get a finite break penalty, reversing onto the opposite arc gets a U-turn penalty, and time gaps above `gap_s` use an uninformative transition (restart)
   * Viterbi: standard max-product recursion with back-pointers
   * decoder parameters are tuned (Optuna) on out-of-fold classifier scores of training trips only
8. **API** (`api/main.py`). `POST /predict {trip_id}` runs the label-free pipeline (`src/pathsnap/pipeline.py`), then scores the result server-side (`src/pathsnap/scoring.py`) against the reference route. The UI receives GPS points, predicted route and metrics, never the reference route.

## Metrics you may be asked about
* **Top-1 accuracy (observation level):** share of fixes whose highest-scored candidate is the reference arc. `top1_acc_all` counts fixes whose reference arc is not a candidate as wrong; `top1_acc_reachable` ignores them.
* **Candidate miss rate:** share of fixes whose reference arc is not among the candidates.
* **Macro-F1:** F1 averaged over the two classes (reference / not reference) using the "pick the top candidate per fix" decision rule, over fixes whose reference arc is a candidate.
* **Route continuity:** share of consecutive fixes whose predicted arcs are the same arc or head-to-tail adjacent.
* **Route F1:** F1 between the set of predicted arcs and the set of reference route arcs, averaged over trips.

## Honest limitations
* Reference labels are derived by alignment and are wrong for a few trips (see `docs/data_quality.md`).
* Duplicate arcs with identical geometry cannot be separated by geometry.
* Decoder and classifier were validated on 10 student-test trips; the evaluator set has not been seen.
