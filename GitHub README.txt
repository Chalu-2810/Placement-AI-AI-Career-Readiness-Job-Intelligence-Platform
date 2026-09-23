# PlacementAI

### AI Career Readiness & Job Intelligence Platform

PlacementAI helps students answer four questions honestly: *Where am I
now? Am I placement-ready? What should I improve? What jobs should I
target?* — combining a full assessment pipeline, an explainable ML
readiness estimate, resume intelligence, and a real external job-market
layer, built end-to-end on Flask/MySQL with a from-scratch design
system ("Waypoint").

**A note on scope, upfront:** this is a portfolio/demonstration project.
The ML model is trained on a **100% synthetic dataset** (disclosed in
detail below and in [`docs/ml-methodology.md`](docs/ml-methodology.md))
and the job-market data comes from a real external API but with the
limitations of a free, remote-jobs-focused source. Every claim in this
README is something the codebase actually does — nothing here is
aspirational.

---

## Problem

Final-year students get generic placement-prep advice ("improve your
resume," "practice DSA") with no way to see where *they specifically*
stand, what a model actually weighs about their profile, or how their
skills compare against real job postings. Most student ML projects that
attempt this either (a) present a black-box percentage with no
explanation, or (b) quietly train and evaluate on the same leaked data
and never say so.

## Solution

PlacementAI runs a student through academic details, a domain-specific
skill assessment, and resume analysis, then produces a **Model-Based
Placement Readiness Estimate** — deliberately not called a "probability
of getting a job" — with a genuine, math-verified explanation of what
drove that specific number. Separately, it pulls real job postings from
an external API to show actual in-demand skills for the student's target
domain and a transparent, keyword-based match against their own profile.
These two systems (synthetic-trained ML estimate vs. real job-market
data) are kept structurally separate everywhere in the UI and docs —
never blended into one number.

---

## Key Features

- **Assessment pipeline** — academic profile, a 30-question domain-specific quiz (aptitude/technical/communication), and PDF resume upload with real ATS scoring
- **Placement readiness estimate** — 5-model comparison (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM), selected by cross-validated balanced accuracy + ROC-AUC, not raw accuracy
- **Per-prediction explainability** — exact `coefficient × scaled_value` decomposition of the model's own decision function for *this specific* prediction, verified against `sklearn`'s `decision_function()` in tests
- **Skill gap analysis** — quiz/resume scores vs. domain benchmarks, visualized as a skill landscape (Strong / Developing / Priority Gap); per-role **Skill Gap Engine** ranks specific missing skills by `Gap × Importance`, grounded in real job-posting demand data, not invented proficiency scores
- **Job Market Intelligence** — real postings fetched from RemoteOK, cached as timestamped snapshots, skill-demand ranked, matched against the student's own extracted skills via transparent set overlap — no embeddings, no LLM
- **Job Role Recommendation Engine** — every role in a snapshot ranked by real skill match %, with Strong/Missing skill previews, not a single generic list
- **"Am I Qualified?"** — pick a specific role, toggle which skills you have, and see match % and gap recompute instantly (client-side, from real per-role posting data)
- **Resume Intelligence** — real ATS breakdown (Skills / Sections / Length / Contact / Action Verbs, each out of 20), missing-keyword detection, improvement suggestions, and a target-role keyword comparison (Strong/Missing vs. a real role's requirements)
- **Career roadmap** — 30/60/90-day plan with explicit Focus/Deliverables per stage, generated from the student's own weak areas, with persisted per-task completion tracking
- **Admin console** — summary stats (total students, assessments completed, average readiness, average ATS score, placement rate), student table, and 6 charts: placement distribution, average scores, domain distribution, readiness distribution, skill-gap frequency, registration trend
- **Security-hardened** — CSRF protection on every state-changing form, rate limiting on auth endpoints, timing-safe login, hardened resume upload (magic-byte validation, size limits), sanitized error responses (full detail: [`docs/security.md`](docs/security.md))

---

## System Architecture

```mermaid
graph TD
    Student[Student] --> Flask[Flask Application]
    Flask --> Assessment[Assessment Engine<br/>academic + quiz]
    Flask --> ResumeIntel[Resume Intelligence<br/>utils/resume.py]
    Assessment --> FeatureEng[feature_engineering.py<br/>single source of truth]
    ResumeIntel --> FeatureEng
    FeatureEng --> MLPredict[prediction.py<br/>ML Prediction]
    MLPredict --> Explain[explainability.py<br/>per-prediction factors]
    MLPredict --> CareerIntel[Career Intelligence<br/>skill gap, roadmap, courses]

    JobSource[(RemoteOK API)] -->|offline refresh only| RefreshScript[scripts/refresh_job_market.py]
    RefreshScript --> Snapshot[(job_market_snapshots)]
    Flask -->|reads snapshot only, never live| JobMarket[job_market.py<br/>Job Market Intelligence]
    Snapshot --> JobMarket

    Explain --> UI[Waypoint UI<br/>Jinja2 + Chart.js]
    CareerIntel --> UI
    JobMarket --> UI
    UI --> Student

    Flask --> DB[(MySQL)]
```

**Deliberately excluded, on purpose:** the assessment/ML side and the
job-market side never call into each other's data. `predict.py` has no
`job_market` import; `job_market.py` has no ML/prediction import. The
only connection is presentational — cross-links in the UI, never a
merged score.

---

## ML Pipeline

```
dataset/placement_data.csv (synthetic, formula-generated)
    → feature_engineering.py (canonical features — ONE implementation,
      shared by training and inference, no drift possible)
    → stratified train/test split, scaler fit on train split only
    → 5-fold cross-validated comparison of 5 model families
    → model selected by mean(cv_balanced_accuracy, cv_roc_auc)
    → held-out test evaluation (touched once)
    → models/*.pkl + model_metadata.json + evaluation_results.json
```

Retrain with `python model_training.py`. Full methodology, the
target-leakage investigation, and the train/inference-consistency bug
that was found and fixed are documented in
[`docs/ml-methodology.md`](docs/ml-methodology.md) — read that before
trusting any number below.

### Model Evaluation (held-out test split, from `models/evaluation_results.json`)

| Model | Accuracy | Precision | Recall | F1 | Balanced Acc. | ROC-AUC |
|---|--:|--:|--:|--:|--:|--:|
| **Logistic Regression (selected)** | 92.92% | 93.58% | 97.22% | 95.37% | 88.61% | 97.89% |
| Random Forest | 93.33% | 93.62% | 97.78% | 95.65% | 88.89% | 97.58% |
| Gradient Boosting | 93.75% | 94.12% | 97.78% | 95.91% | 89.72% | 97.81% |
| SVM (RBF) | 92.50% | 94.02% | 96.11% | 95.05% | 88.89% | 97.28% |
| Decision Tree | 90.83% | 93.89% | 93.89% | 93.89% | 87.78% | 90.38% |

**Read this table honestly, not optimistically:** the dataset is
100% synthetic, and a noise-free version of its own label-generating
formula agrees with the actual labels 92.33% of the time — these numbers
largely reflect the model recovering a known formula, not a validated
real-world relationship. Logistic Regression — the simplest, most linear
candidate — being selected is itself consistent with that finding. Full
detail, including why Logistic Regression was selected over models that
score marginally higher on this one test split, in
[`docs/ml-methodology.md`](docs/ml-methodology.md#4-training-methodology).

### Explainability

The results page shows **"Factors influencing this model estimate"** —
an exact decomposition of `Σ(coefficient × scaled_feature_value) +
intercept` for that specific prediction, not an approximation and not
SHAP. `tests/test_explainability.py` verifies the sum reproduces
`model.decision_function()` to floating-point precision, against both
synthetic fixtures and the real trained model. Full methodology,
including the tree-model and unsupported-model fallback behavior, in
[`docs/ml-methodology.md`](docs/ml-methodology.md#6-explainability--factors-influencing-a-specific-estimate).

---

## ATS Engine

`utils/resume.py` extracts text (PyMuPDF, falling back to pdfminer),
detects skills against a fixed taxonomy, and scores five categories out
of 20 points each: Skills Found, Key Sections, Length, Contact Info,
Action Verbs. The same skill-detection function is reused — not
duplicated — by the Job Market Intelligence pipeline, so a student's
resume skills and job-posting skills are extracted with one consistent
vocabulary.

## Job Market Intelligence

```
RemoteOK API (real postings, one call per refresh)
    → scripts/refresh_job_market.py (offline only — NOT a Flask route)
    → classify into the app's 8 domains, extract skills, aggregate
    → job_market_snapshots (MySQL, timestamped)
    → app reads the latest snapshot only — never calls RemoteOK live
```

Every number on the Job Market Intelligence page traces to either a
real fetched posting, a deterministic aggregation of fetched postings,
or a deterministic calculation against the student's own profile — see
the verification table in
[`docs/job-market-methodology.md`](docs/job-market-methodology.md#13-whats-real-vs-derived-vs-matched-vs-estimated--summary-table).
RemoteOK's real limitations (remote-jobs-focused, sparse salary data, no
India-city location data) are shown as honest empty states, never
estimated. Adzuna is documented as a possible future source, not
implemented, pending a separate terms-of-service decision — see that doc
for why.

---

## Limitations

- **The placement dataset is 100% synthetic** — see
  [`docs/ml-methodology.md`](docs/ml-methodology.md) for the full
  target-leakage audit. Do not interpret the readiness estimate as a
  real-world employment probability.
- **Job market data is RemoteOK-only, free-tier** — remote/global-tilted,
  not representative of India-local or on-campus hiring, small sample
  sizes per domain.
- **Skill matching is keyword-based** — no synonym/semantic matching by
  design (deliberately avoiding embeddings/LLM dependencies).
- **`/predict` is a GET route with a side effect** (it writes a
  prediction row) — a known, documented residual risk, not silently
  fixed; see [`docs/security.md`](docs/security.md) for the reasoning
  and impact assessment.
- **Rate limiting uses in-memory storage** — fine for a single-worker
  deployment, needs a shared backend (e.g. Redis) for multi-worker
  production.

## Future Improvements

- Adzuna integration (pending ToS resolution — see `docs/job-market-methodology.md`)
- Account lockout beyond IP-based rate limiting
- Persisted historical explainability snapshots for trend comparison
- Shared-backend rate limiting for multi-worker deployment

---

## Technology Stack

`Flask 3` · `MySQL 8` · `scikit-learn` · `pandas` / `numpy` · `PyMuPDF` / `pdfminer.six` · `Flask-WTF` (CSRF) · `Flask-Limiter` (rate limiting) · `requests` (RemoteOK) · `Bootstrap 5` (grid/utilities only) · `Chart.js 4` · `Font Awesome 6` · Fraunces / IBM Plex Sans / IBM Plex Mono (Waypoint design system, see [`docs/design-system.md`](docs/design-system.md))

No React/Next.js, no vector database, no LLM API, no SHAP — deliberately, per the project's own stated architecture constraints. See `docs/*.md` for the reasoning behind each "we didn't add X" decision.

---

## Screenshots

Screenshots aren't included in this repository yet — add them here once available (e.g. `docs/screenshots/dashboard.png`, `docs/screenshots/results.png`), captured from a locally running instance.

In the meantime, [`docs/design-system.md`](docs/design-system.md) documents the visual language in detail — color tokens, typography, component patterns, and the reasoning behind the Waypoint design system — for anyone who wants to understand the interface without a live screenshot.

---

## Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up MySQL
mysql -u root -p < schema.sql
# Creates all tables (including job_market_snapshots and roadmap_progress)
# and a default admin account with a placeholder password hash. Change
# it before any real deployment — see the comment directly above the
# INSERT in schema.sql for the exact generate_password_hash() command to run.

# 3. Configure environment
cp .env.example .env
# Fill in SECRET_KEY and your MySQL credentials.
# RemoteOK needs no key. Adzuna vars are future-only placeholders — leave blank.

# 4. Train the ML model (skip if models/*.pkl already exist)
python generate_dataset.py
python model_training.py

# 5. (Optional) Populate job market data
python scripts/refresh_job_market.py

# 6. Run
python app.py
# Visit http://localhost:5000
```

### Tests

```bash
pytest              # 166 tests: ML consistency, security, explainability, job market,
                    #   role matching, skill gap engine, roadmap, admin analytics, core workflows
```

### Deployment

`Procfile` runs `gunicorn app:app`. `SECRET_KEY` is required from the
environment with no fallback (the app fails fast at startup if it's
missing, rather than running with a guessable default). See
`docs/security.md` for the full production-configuration checklist
(session cookies, `MAX_CONTENT_LENGTH`, rate-limit storage, etc.).

---

## Project Structure

```
PlacementAI/
├── app.py                      # Flask routes
├── database.py                 # MySQL access layer (parameterized queries only)
├── prediction.py                # ML inference, skill gap, roadmap, courses
├── explainability.py           # Per-prediction factor decomposition
├── feature_engineering.py      # Canonical feature formulas (single source of truth)
├── job_market.py               # RemoteOK ingestion, skill extraction, matching
├── model_training.py           # Training pipeline (run offline)
├── generate_dataset.py         # Synthetic dataset generator
├── schema.sql                  # MySQL schema
├── requirements.txt
├── Procfile
│
├── scripts/
│   └── refresh_job_market.py   # Offline job-market snapshot refresh (not a route)
│
├── utils/
│   ├── resume.py                # Skill taxonomy + ATS scoring core
│   ├── resume_parser.py         # PDF text extraction
│   └── ats_score.py             # ATS scoring wrapper
│
├── models/                     # Trained artifacts (tracked — see docs/security.md for why)
│   ├── trained_model.pkl / all_models.pkl / scaler.pkl / label_encoder.pkl
│   ├── model_metadata.json      # Version, features, dataset stats, leakage audit
│   └── evaluation_results.json  # Full CV + held-out-test metrics
│
├── dataset/                    # Generated CSV (regenerable, not tracked)
├── questions/                  # Quiz question bank (aptitude/technical/communication)
├── data/domains.json
│
├── templates/                  # Jinja2, Waypoint design system
│   ├── base.html, error.html, _gauge.html
│   ├── login.html, registration.html
│   ├── dashboard.html, academic.html, quiz_setup.html, quiz.html
│   ├── resume.html, result.html, opportunities.html, qualified.html
│   └── admin_login.html, admin_dashboard.html
├── static/css/style.css
│
├── tests/                      # 166 tests, fixture/mock-based (no live API/DB calls)
│   ├── test_feature_consistency.py    # ML train/inference consistency
│   ├── test_security.py               # Auth, CSRF, rate limiting, upload validation
│   ├── test_explainability.py         # Per-prediction factor decomposition
│   ├── test_job_market.py             # RemoteOK ingestion, aggregation, matching
│   ├── test_admin_analytics.py        # Admin dashboard queries and rendering
│   ├── test_qualified.py              # "Am I Qualified?" role-vs-skills matching
│   ├── test_resume_intelligence.py    # Resume-to-role keyword comparison
│   ├── test_roadmap_progress.py       # Roadmap Focus/Deliverables + completion tracking
│   ├── test_role_recommendations.py   # Job Role Recommendation Engine ranking
│   ├── test_skill_gap_engine.py       # Gap x Importance skill prioritization
│   ├── test_core_workflows.py         # Registration, quiz load/score, ATS bounds
│   └── fixtures/
│
└── docs/
    ├── ml-methodology.md            # Training, leakage audit, explainability
    ├── security.md                  # Threat model, fixes, residual risks
    ├── design-system.md             # Waypoint design tokens/components
    ├── job-market-methodology.md    # RemoteOK pipeline, role matching, skill gap engine, real-vs-derived table
    └── frontend-assets.md           # External CDN assets used
```
