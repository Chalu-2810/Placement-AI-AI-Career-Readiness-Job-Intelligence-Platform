# PlacementAI — ML Methodology & Limitations

This document explains, honestly, how the placement-readiness model is
built, what it can and can't tell you, and the correctness issues found
and fixed during an ML/data audit.

Model artifacts are versioned in `models/model_metadata.json`, generated
fresh every time `python model_training.py` runs. The numbers below were
current as of that file's `trained_at_utc` timestamp — check the JSON
directly for the latest run rather than trusting this document's numbers
to stay current forever.

---

## 1. What the dataset actually is

`dataset/placement_data.csv` is **100% synthetic**. It is not real student
outcome data from any institution. It was produced by `generate_dataset.py`,
which:

1. Samples random-but-plausible values for 10th %, 12th %, CGPA, backlogs,
   aptitude/technical/communication scores, resume score, and domain.
2. Computes a **hand-written linear formula**:

   ```
   score = 0.15*(tenth/100) + 0.15*(twelfth/100) + 0.25*(cgpa/10)
          + 0.15*(aptitude/100) + 0.18*(technical/100)
          + 0.07*(communication/100) + 0.05*(resume/100)
          - 0.10*backlogs + noise(mean=0, std=0.04)
   ```
3. Labels the row `placement_status = 1` if `score >= 0.55`, else `0`.

The dataset contains **1,200 rows** across all 8 domains the app supports,
with a **74.8% / 25.2%** class split (placed / not placed).

### Target leakage: quantified, not hand-waved

Because `placement_status` is *itself* a threshold on a near-linear
function of the same raw columns the model is trained on, the label is
largely "given away" by the features before any learning happens. We
measured this directly: thresholding the **noise-free** version of the
same formula against the actual saved labels agrees **92.33%** of the
time, and the point-biserial correlation between that deterministic score
and the label is **0.7476**.

**What this means in practice:** every model in this project scores
90%+ accuracy not because it has discovered a real relationship between
student profiles and employment outcomes, but because it is recovering a
mostly-linear formula that was used to construct its own training labels.
This is also *why* plain Logistic Regression — the simplest, most linear
model in the comparison — was selected as the best performer (see §4):
that's exactly what you'd expect if the underlying signal really is
close to linear.

**This is disclosed in the UI.** Result and dashboard screens now read
"Model-Based Placement Readiness Estimate" rather than presenting output
as a real-world probability of employment, and both templates link back
to this document.

> If real placement outcome data becomes available in the future, this
> dataset should be replaced (not blended) — mixing synthetic,
> formula-labeled rows with real outcome rows would make the same
> leakage invisible while still contaminating the model.

---

## 2. Feature engineering — single source of truth

**Bug found and fixed:** prior to this audit, `model_training.py` (fit
time) and `prediction.py` (inference time) each hard-coded their own copy
of the `skill_score` formula, and the two had drifted apart:

| | `academic_score` weights | `skill_score` weights |
|---|---|---|
| `model_training.py` (training) | tenth 0.20 / twelfth 0.20 / cgpa×10 0.60 | aptitude 0.35 / technical 0.45 / communication 0.20 (**no resume term**) |
| `prediction.py` (inference, old) | tenth 0.20 / twelfth 0.20 / cgpa×10 0.60 | aptitude 0.30 / technical 0.40 / communication 0.15 / resume 0.15 |
| `app.py` `/predict` (DB display value, old) | tenth 0.20 / twelfth **0.30** / cgpa×10 **0.50** | plain average of aptitude/technical/communication |

Three different formulas for two "features," none of them fully agreeing.
The model was trained on one `skill_score` definition and scored live
users with another — every prediction's `skill_score` input was
systematically wrong relative to what the model actually learned.
`academic_score` had a third, disconnected definition used only for the
value shown/stored on the student's dashboard.

**Fix:** all three call sites now import from a single new module,
`feature_engineering.py`:

```python
FEATURE_NAMES = [
    'tenth_percentage', 'twelfth_percentage', 'cgpa', 'backlogs',
    'aptitude_score', 'technical_score', 'communication_score',
    'resume_score', 'domain_encoded', 'academic_score', 'skill_score',
    'backlog_penalty',
]

academic_score = tenth*0.20 + twelfth*0.20 + (cgpa*10)*0.60
skill_score    = aptitude*0.30 + technical*0.40 + communication*0.15 + resume_score*0.15
backlog_penalty = 1 if backlogs == 0 else 0
```

We kept the weights that were already driving live predictions
(`prediction.py`'s old formula), on the reasoning that changing
user-facing scoring behavior silently is worse than fixing training to
match what users were already being evaluated against. `model_training.py`
now retrains against this same formula, so the drift is closed at the
source rather than patched at the edges.

`tests/test_feature_consistency.py` asserts, for a battery of sample
students, that the training-style (vectorized/pandas) and inference-style
(scalar/per-request) code paths produce bit-identical feature rows. This
test exists specifically to catch a regression of this exact bug.

---

## 3. Scaler leakage — fixed

**Bug found and fixed:** the previous `preprocess()` called
`StandardScaler().fit_transform(X)` on the **entire** dataset — train and
test rows combined — *before* `train_test_split()` ran. That means the
scaler's mean/std were computed using information from rows that were
later treated as a "held-out" test set, a classic form of data leakage
that inflates reported test metrics.

**Fix:** the split now happens first. `StandardScaler` is fit **only** on
the training split; the test split (and, at inference time, real user
input via `prediction.py`) is transformed using those training-only
statistics. Cross-validation used for model comparison goes further:
each of the 5 folds fits its **own** scaler on that fold's training
portion via an `sklearn.pipeline.Pipeline(StandardScaler, model)`, so
there's no leakage even between CV folds, not just at the final
train/test boundary.

---

## 4. Training methodology

1. **Stratified train/test split** — 80/20, `random_state=42`, stratified
   on `placement_status` because the classes are imbalanced (~75/25).
   The test split (240 rows) is touched exactly **once**, at the end, for
   final reporting — never for model selection.
2. **Stratified 5-fold cross-validation**, run only on the training split
   (960 rows), used to *compare and select* between five candidate model
   families: Logistic Regression, Decision Tree, Random Forest, Gradient
   Boosting, SVM (RBF kernel).
3. **Model selection metric:** mean of cross-validated `balanced_accuracy`
   and `roc_auc` on the training split. We deliberately did **not** select
   by raw accuracy or F1 alone — with a 75/25 class split, a model that
   always predicts "Placed" scores ~75% accuracy for free, and F1 alone
   still rewards recall on the majority class. Balanced accuracy and
   ROC-AUC do not.
4. **Final fit + held-out evaluation:** the scaler and every candidate
   model are refit on the full training split, then evaluated once on the
   untouched test split, reporting accuracy, precision, recall, F1,
   balanced accuracy, ROC-AUC, and the confusion matrix.

### Results (see `models/model_metadata.json` / `models/evaluation_results.json` for the live numbers)

Cross-validated (train split, 5-fold):

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV Balanced Acc. | CV ROC-AUC |
|---|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 92.29% | 94.92% | 94.84% | 94.84% | 89.76% | 97.44% |
| Decision Tree | 86.04% | 90.11% | 91.36% | 90.72% | 80.80% | 84.35% |
| Random Forest | 91.67% | 92.99% | 96.09% | 94.51% | 87.30% | 95.54% |
| Gradient Boosting | 90.42% | 92.32% | 95.12% | 93.68% | 85.79% | 95.42% |
| SVM (RBF) | 91.04% | 93.11% | 95.12% | 94.07% | 87.01% | 96.39% |

Held-out test split (240 rows, touched once):

| Model | Accuracy | Precision | Recall | F1 | Balanced Acc. | ROC-AUC |
|---|--:|--:|--:|--:|--:|--:|
| **Logistic Regression (selected)** | 92.92% | 93.58% | 97.22% | 95.37% | 88.61% | 97.89% |
| Decision Tree | 90.83% | 93.89% | 93.89% | 93.89% | 87.78% | 90.38% |
| Random Forest | 93.33% | 93.62% | 97.78% | 95.65% | 88.89% | 97.58% |
| Gradient Boosting | 93.75% | 94.12% | 97.78% | 95.91% | 89.72% | 97.81% |
| SVM (RBF) | 92.50% | 94.02% | 96.11% | 95.05% | 88.89% | 97.28% |

**Selected model: Logistic Regression** — highest mean of CV balanced
accuracy and CV ROC-AUC (93.60) among all five candidates. Note that
several tree-based models score marginally higher on the held-out test
set alone; we selected on the training-split CV metric specifically to
avoid picking a model because it happened to fit the one-time test split
best. That Logistic Regression — the simplest, most linear model — is
essentially tied with or ahead of every non-linear alternative is
itself consistent with §1's finding: the target really is close to a
linear function of the inputs.

**Read these numbers as:** "how well each model recovers a known,
formula-generated synthetic label," not as validated evidence about
real-world employability.

---

## 5. Model artifacts & metadata

Every training run overwrites:

- `models/trained_model.pkl` — the selected production model
- `models/all_models.pkl` — all 5 trained candidates (used for the
  in-app model-comparison view)
- `models/scaler.pkl` — `StandardScaler` fit on the training split only
- `models/label_encoder.pkl` — domain name → integer encoder
- `models/feature_names.pkl` — canonical column order (must match
  `feature_engineering.FEATURE_NAMES`; enforced by
  `tests/test_feature_consistency.py`)
- `models/evaluation_results.json` — full CV + held-out-test metrics
- `models/model_metadata.json` — model version/timestamp, selected model
  name, feature names, dataset stats, the target-leakage audit numbers,
  split configuration, and library/environment versions

`prediction.py` reads `model_metadata.json` at import time so the
`model_used` value saved to the database always reflects whichever model
was **actually** selected by the last training run — previously this was
hardcoded to `'Random Forest'` in `app.py` regardless of what
`model_training.py` picked.

---

## 6. Explainability — factors influencing a specific estimate

Added in Phase 4. Result and dashboard screens now show "Factors
influencing this model estimate" — a per-prediction breakdown of what
the model's own math weighted for that specific input. Implemented in
`explainability.py`, called from `prediction.py::predict_placement()`.

### Method for the currently-selected model (Logistic Regression)

`LogisticRegression`'s decision function is exactly:

```
decision_score = sum_i( coef_i * X_scaled_i ) + intercept
```

Each feature's **contribution** is exactly one term of that sum:

```
contribution_i = coef_i * X_scaled_i
```

This is **not** an approximation and **not SHAP** — it is the model's
own linear decision function, decomposed term by term, for the exact
scaled feature vector `prediction.py` already computed for that request
(`explainability.py` takes that same `X_scaled` row as input; it does
not re-scale or recompute any feature). Because it's an exact algebraic
identity, `sum(contributions) + intercept` reproduces
`model.decision_function(X_scaled)` to within floating-point precision —
verified directly in `tests/test_explainability.py` (both against
synthetic fixtures with known coefficients, and against the real trained
model artifact when present).

Factors are ranked by `|contribution|` (strongest influence first,
regardless of sign), and each carries a `direction`: `'up'` (pushes the
estimate higher), `'down'` (pushes it lower), or `'flat'` (contribution
== 0). The UI groups the top 5 by magnitude into "Supports the estimate"
/ "Pulls the estimate down" purely from these real, computed values —
nothing about which features appear in which group is hand-picked.

### Fallback for a tree-based model

If a future retrain selects a tree-based model (Random Forest, Gradient
Boosting, Decision Tree) instead, there is no per-prediction linear
decomposition available without a library like SHAP, which this project
deliberately does not depend on (see project scope). Instead,
`explainability.py` falls back to the model's own global
`feature_importances_` — clearly labeled in the UI as "what the model
weighs most **across all predictions in general**," explicitly not a
per-prediction explanation. This is a materially different, weaker claim
than the linear method's exact contributions, and the two are never
presented with the same language.

### Fallback for any other model type

If the selected model has neither `coef_` nor `feature_importances_`
(e.g. an SVM with a non-linear kernel), `compute_feature_contributions()`
returns an explicit `'unavailable'` result — an empty factor list, no
fabricated ranking. The UI simply omits the section rather than showing
placeholder or guessed values. This path, and a malformed/mismatched
`coef_` shape, are both covered by dedicated tests.

### Model contribution is not real-world causation

This is worth repeating in its own paragraph: these numbers describe
what the **algorithm** weighted internally, for a model trained on the
100% synthetic, formula-generated dataset described in §1. A large
positive "Technical assessment" contribution means the model's own math
leaned on that feature for this input — it is not evidence that
technical scores *cause* real-world placement outcomes, and given the
target-leakage finding in §1, the model is substantially recovering a
known linear formula rather than a validated real-world relationship in
the first place. The UI's copy ("Factors influencing this model
estimate," never "Why you will get placed") and the disclosure text
directly beside the breakdown both exist to keep this distinction
visible, not just documented here.

## 7. Known limitations

- **Synthetic data.** See §1. Results should not be interpreted as
  real-world placement probabilities under any circumstances.
- **Domain augmentation is currently dead code.** `model_training.py`
  (pre-audit) shipped an `augment_dataset()` step meant to add synthetic
  rows for any of the 8 app domains missing from the CSV. In practice all
  8 domains are already present in `placement_data.csv`, so this step
  never actually triggers — it was preserved conceptually but is not part
  of the active pipeline described above. Worth removing or actively
  exercising in a future pass rather than leaving it as unreachable code.
- **No real-world validation set.** There is no external, non-synthetic
  dataset to validate the model against, so we cannot report how the
  model would perform on genuine student outcomes.
- **Small held-out test set.** 240 rows is enough to compute stable
  aggregate metrics but not enough to trust per-domain or per-subgroup
  breakdowns.
- **`sklearn` version drift risk.** Artifacts are tied to the `sklearn`
  version recorded in `models/model_metadata.json` (`environment.
  sklearn_version`). Loading `.pkl` files with a different installed
  `sklearn` version can raise `InconsistentVersionWarning` or fail
  outright — retrain locally with `python model_training.py` if you hit
  this.
- **Explainability is only exact for a linear model.** See §6. The
  current selected model (Logistic Regression) gets an exact,
  per-prediction breakdown; a future retrain that selects a tree-based
  model would only get global (not per-prediction) importances; any other
  model type gets no explainability section at all rather than a
  fabricated one.

## 8. What was intentionally NOT changed in this pass

Per the scope of this audit, the following were left untouched and are
tracked for a future pass: the underlying dataset generation formula
itself (not regenerated — the existing CSV was reused for reproducibility
of this audit), the full UI/UX redesign, the job-market intelligence
layer, and the README rewrite.
