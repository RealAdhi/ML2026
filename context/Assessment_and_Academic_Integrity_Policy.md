# PathSnap: Assessment and Academic Integrity Policy (supporting document)

_Text of the event's supporting document, kept here so every team member reads the same rules._

## 1. Purpose
This policy explains how PathSnap submissions will be assessed and how organisers may review the originality and authenticity of submitted work.

It applies to: source code, model implementations, model artifacts, configurations, experimental results, technical reports, demonstrations.

Assessment considers correctness, completeness, technical understanding, experimental rigour, reproducibility and the team's ability to explain its work. A high score on one metric does not automatically compensate for missing or invalid parts of the system.

## 3. Evidence of Meaningful Experimentation
Teams must:
* Train and compare at least five models before selecting a final model.
* Support model selection with recorded experimental results.
* Explain the strengths and limitations of the chosen approach.
* Demonstrate Bayesian hyperparameter optimisation.
* Retain the optimisation search record.
* Evaluate models and the decoder using permitted development data.

Selecting a model without meaningful experimentation may result in deductions.

## 4. Independent Work and Model Similarity Checks
**STRICT INDEPENDENT-WORK REQUIREMENT.** Each team must independently implement, train, evaluate and select its own models. Sharing model files, trained weights, parameter configurations, implementation code, derived datasets or experimental results between teams is prohibited.

Submitted architectures, source code, hyperparameter configurations, trained weights, model artifacts and reported results will be compared across teams, across sections. Similarities may be flagged for further review.

## Viva and Demonstration
During evaluation, organisers may ask team members to explain: dataset structure, data preprocessing, feature-engineering decisions, model comparisons, model selection, Bayesian hyperparameter optimisation, HMM implementation, Viterbi decoding, API implementation, UI implementation, experimental results.

Teams should be able to demonstrate that they understand and can reproduce their submitted work.

## 7. Data Leakage and Evaluation Integrity
The following rules are mandatory:
* Evaluator-testing data must remain untouched during development.
* Evaluator ground-truth labels must not enter the student-side prediction pipeline or UI.
* Hardcoded predictions are prohibited; the repositories will be examined for any hard coded process.

## 5. Similarity Review and Investigation
Organisers will compare: source code, repository history, model architectures, model weights, hyperparameter configurations, experimental results, technical reports, graphs and tables.

Similarity may trigger a review or a request for further explanation. Because standard algorithms and common approaches can naturally produce similarities, no single similarity measure should automatically be treated as proof of plagiarism.
