"""
feature_engineering.py
-----------------------
SINGLE SOURCE OF TRUTH for turning raw student inputs into the exact
feature vector consumed by the placement-readiness model.

Why this file exists
---------------------
Before this fix, `model_training.py` (fit time) and `prediction.py`
(inference time) each hard-coded their OWN copy of the `academic_score`
and `skill_score` formulas, and the two copies had drifted out of sync:

    model_training.py (preprocess):
        skill_score = aptitude*0.35 + technical*0.45 + communication*0.20
        (no resume term at all)

    prediction.py (_build_feature_vector):
        skill_score = aptitude*0.30 + technical*0.40 + communication*0.15
                       + resume_score*0.15

    app.py (/predict route, saved to prediction_results.academic_score):
        academic_score = tenth*0.2 + twelfth*0.3 + cgpa*10*0.5

    model_training.py / prediction.py (model FEATURE academic_score):
        academic_score = tenth*0.2 + twelfth*0.2 + cgpa*10*0.6

The model was therefore trained on one definition of `skill_score` and
served predictions using a different one — every live prediction's
`skill_score` feature was computed with the wrong weights relative to
what the model actually learned. `academic_score` had a THIRD, separate
definition used only for the value stored in the database, disconnected
from both of the above.

Both training and inference now import the functions below instead of
recomputing these formulas locally. There is exactly one implementation
of each formula, so training and inference can never drift apart again.
(See tests/test_feature_consistency.py, which asserts this directly.)

Canonical weight choice
------------------------
Where the historical implementations disagreed, we kept the formula that
was already being used for live inference (i.e. the one users were
actually being scored against), on the theory that changing user-facing
behavior silently is worse than fixing training to match it. Both the
old and new numbers are documented above and in docs/ml-methodology.md.
"""

from __future__ import annotations

# Canonical, exported feature order. The trained model, scaler, and every
# caller in the codebase must agree on this order.
FEATURE_NAMES = [
    "tenth_percentage",     # 0
    "twelfth_percentage",   # 1
    "cgpa",                 # 2
    "backlogs",             # 3
    "aptitude_score",       # 4
    "technical_score",      # 5
    "communication_score",  # 6
    "resume_score",         # 7
    "domain_encoded",       # 8
    "academic_score",       # 9
    "skill_score",          # 10
    "backlog_penalty",      # 11
]

# Human-readable labels for each canonical feature, in the same order.
# Single source of truth for display names -- used by explainability.py
# so the "Why this estimate?" panel never invents its own copy of these
# names. Keep in sync with FEATURE_NAMES (tests/test_explainability.py
# asserts the key sets match).
FEATURE_LABELS = {
    "tenth_percentage"    : "10th percentage",
    "twelfth_percentage"  : "12th percentage",
    "cgpa"                : "CGPA",
    "backlogs"            : "Backlog history",
    "aptitude_score"      : "Aptitude assessment",
    "technical_score"     : "Technical assessment",
    "communication_score" : "Communication assessment",
    "resume_score"        : "Resume skill coverage",
    "domain_encoded"      : "Selected domain",
    "academic_score"      : "Academic performance",
    "skill_score"         : "Overall skill score",
    "backlog_penalty"     : "Zero-backlog status",
}

# --- Canonical formula weights (documented, not magic numbers) ---------
# academic_score: cgpa is weighted most heavily because it is the most
# recent and most comprehensive academic signal; 10th/12th marks are
# older and less predictive by the time a student is job-hunting.
ACADEMIC_WEIGHTS = {"tenth": 0.20, "twelfth": 0.20, "cgpa": 0.60}

# skill_score: technical score weighted highest (most direct signal for
# technical-hiring domains), aptitude second, communication and resume
# quality contribute smaller, complementary signals.
SKILL_WEIGHTS = {
    "aptitude": 0.30,
    "technical": 0.40,
    "communication": 0.15,
    "resume": 0.15,
}


def compute_academic_score(tenth, twelfth, cgpa):
    """
    academic_score = tenth*0.20 + twelfth*0.20 + (cgpa*10)*0.60

    cgpa is assumed to be on a 0-10 scale and is rescaled to 0-100 before
    weighting so all three inputs share the same scale. Works on plain
    floats or pandas Series (vectorized) since it only uses arithmetic.
    """
    return (
        tenth * ACADEMIC_WEIGHTS["tenth"]
        + twelfth * ACADEMIC_WEIGHTS["twelfth"]
        + (cgpa * 10) * ACADEMIC_WEIGHTS["cgpa"]
    )


def compute_skill_score(aptitude, technical, communication, resume_score):
    """
    skill_score = aptitude*0.30 + technical*0.40 + communication*0.15
                  + resume_score*0.15

    Works on plain floats or pandas Series (vectorized).
    """
    return (
        aptitude * SKILL_WEIGHTS["aptitude"]
        + technical * SKILL_WEIGHTS["technical"]
        + communication * SKILL_WEIGHTS["communication"]
        + resume_score * SKILL_WEIGHTS["resume"]
    )


def compute_backlog_penalty(backlogs):
    """1 if the student has zero active backlogs, else 0. Vectorized-safe."""
    try:
        # Scalar path
        return 1 if int(backlogs) == 0 else 0
    except TypeError:
        # pandas Series path
        return (backlogs == 0).astype(int)


def encode_domain(domain, label_encoder):
    """
    Encode a domain name using an already-fit LabelEncoder.
    Falls back to class 0 for an unseen/invalid domain rather than raising,
    since a bad domain value should degrade gracefully at inference time.
    """
    try:
        return int(label_encoder.transform([domain])[0])
    except Exception:
        return 0


def build_feature_row(
    *,
    tenth,
    twelfth,
    cgpa,
    backlogs,
    aptitude,
    technical,
    communication,
    resume_score,
    domain,
    label_encoder,
):
    """
    Build a single inference-time feature row (dict keyed by FEATURE_NAMES)
    from raw student inputs. Used by prediction.py. Training builds the
    same columns in bulk (vectorized) using the same compute_* functions
    directly on DataFrame columns — see model_training.py::preprocess().
    """
    academic_score = compute_academic_score(float(tenth), float(twelfth), float(cgpa))
    skill_score = compute_skill_score(
        float(aptitude), float(technical), float(communication), float(resume_score)
    )
    backlog_penalty = compute_backlog_penalty(backlogs)
    domain_encoded = encode_domain(domain, label_encoder)

    row = {
        "tenth_percentage": float(tenth),
        "twelfth_percentage": float(twelfth),
        "cgpa": float(cgpa),
        "backlogs": int(backlogs),
        "aptitude_score": float(aptitude),
        "technical_score": float(technical),
        "communication_score": float(communication),
        "resume_score": float(resume_score),
        "domain_encoded": domain_encoded,
        "academic_score": academic_score,
        "skill_score": skill_score,
        "backlog_penalty": backlog_penalty,
    }
    # Return values in canonical order as a plain list too, for callers
    # that need a positional vector rather than a dict.
    return row


def row_as_vector(row: dict):
    """Return a row dict's values as a list in canonical FEATURE_NAMES order."""
    return [row[name] for name in FEATURE_NAMES]
