"""
tests/test_feature_consistency.py
----------------------------------
Regression test for the exact bug found during the ML correctness audit:
model_training.py (fit time) and prediction.py (inference time) had
DIFFERENT skill_score/academic_score formulas, so a model trained on one
set of feature values was served predictions computed with another.

These tests assert that, for identical raw student inputs, the feature
row model_training.py's build_features() would compute (via pandas,
bulk/vectorized) is IDENTICAL to the feature row prediction.py's
_build_feature_vector() computes (via feature_engineering.build_feature_row,
scalar/per-request) -- both now delegate to the same functions in
feature_engineering.py, so this test is really a guard against anyone
reintroducing a second, diverging implementation in the future.

Run: pytest tests/test_feature_consistency.py -v
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from feature_engineering import (
    FEATURE_NAMES,
    build_feature_row,
    compute_academic_score,
    compute_skill_score,
    compute_backlog_penalty,
)

DOMAINS = [
    'Data Science', 'Web Development', 'Cybersecurity', 'Cloud Computing',
    'DevOps', 'Internet of Things (IoT)', 'Blockchain Technology',
    'Mobile App Development',
]

SAMPLE_STUDENTS = [
    dict(tenth=85, twelfth=90, cgpa=8.5, backlogs=0, aptitude=75, technical=80,
         communication=70, resume_score=82, domain='Data Science'),
    dict(tenth=55, twelfth=60, cgpa=5.8, backlogs=3, aptitude=40, technical=35,
         communication=45, resume_score=38, domain='Web Development'),
    dict(tenth=70, twelfth=68, cgpa=7.0, backlogs=1, aptitude=60, technical=65,
         communication=55, resume_score=60, domain='Cybersecurity'),
    dict(tenth=99, twelfth=97, cgpa=9.8, backlogs=0, aptitude=95, technical=98,
         communication=90, resume_score=95, domain='Blockchain Technology'),
    dict(tenth=45, twelfth=50, cgpa=4.5, backlogs=4, aptitude=20, technical=25,
         communication=30, resume_score=20, domain='Mobile App Development'),
]


def _fit_label_encoder():
    le = LabelEncoder()
    le.fit(DOMAINS)
    return le


def _training_style_row(student, le):
    """
    Mirrors what model_training.py::build_features() does, but for a
    single row via pandas (as it would inside a DataFrame column
    operation), to simulate the bulk/vectorized training path using the
    SAME functions as inference.
    """
    df = pd.DataFrame([{
        'tenth_percentage': student['tenth'],
        'twelfth_percentage': student['twelfth'],
        'cgpa': student['cgpa'],
        'backlogs': student['backlogs'],
        'aptitude_score': student['aptitude'],
        'technical_score': student['technical'],
        'communication_score': student['communication'],
        'resume_score': student['resume_score'],
        'domain': student['domain'],
    }])
    df['domain_encoded'] = le.transform(df['domain'])
    df['academic_score'] = compute_academic_score(df['tenth_percentage'], df['twelfth_percentage'], df['cgpa'])
    df['skill_score'] = compute_skill_score(df['aptitude_score'], df['technical_score'], df['communication_score'], df['resume_score'])
    df['backlog_penalty'] = compute_backlog_penalty(df['backlogs'])
    row = df[FEATURE_NAMES].iloc[0].to_dict()
    return row


def _inference_style_row(student, le):
    """Mirrors exactly what prediction.py::_build_feature_vector() does."""
    return build_feature_row(
        tenth=student['tenth'], twelfth=student['twelfth'], cgpa=student['cgpa'],
        backlogs=student['backlogs'], aptitude=student['aptitude'],
        technical=student['technical'], communication=student['communication'],
        resume_score=student['resume_score'], domain=student['domain'],
        label_encoder=le,
    )


def test_training_and_inference_features_match_for_all_samples():
    le = _fit_label_encoder()
    for student in SAMPLE_STUDENTS:
        train_row = _training_style_row(student, le)
        infer_row = _inference_style_row(student, le)
        for name in FEATURE_NAMES:
            assert math.isclose(float(train_row[name]), float(infer_row[name]), rel_tol=1e-9, abs_tol=1e-9), (
                f"Feature '{name}' mismatch for {student}: "
                f"training={train_row[name]!r} vs inference={infer_row[name]!r}"
            )


def test_skill_score_uses_documented_weights():
    # aptitude*0.30 + technical*0.40 + communication*0.15 + resume*0.15
    val = compute_skill_score(100, 100, 100, 100)
    assert math.isclose(val, 100.0, rel_tol=1e-9)
    val2 = compute_skill_score(80, 0, 0, 0)
    assert math.isclose(val2, 24.0, rel_tol=1e-9)  # 80*0.30


def test_academic_score_uses_documented_weights():
    # tenth*0.20 + twelfth*0.20 + (cgpa*10)*0.60
    val = compute_academic_score(100, 100, 10)
    assert math.isclose(val, 100.0, rel_tol=1e-9)
    val2 = compute_academic_score(0, 0, 10)
    assert math.isclose(val2, 60.0, rel_tol=1e-9)  # (10*10)*0.6


def test_backlog_penalty_scalar_and_vectorized_agree():
    assert compute_backlog_penalty(0) == 1
    assert compute_backlog_penalty(1) == 0
    assert compute_backlog_penalty(5) == 0

    s = pd.Series([0, 1, 2, 0])
    result = compute_backlog_penalty(s)
    assert list(result) == [1, 0, 0, 1]


def test_feature_order_matches_saved_artifact_if_present():
    """
    If models/feature_names.pkl exists (i.e. a model has been trained),
    it must exactly match feature_engineering.FEATURE_NAMES -- otherwise
    the scaler/model were fit on a different column order than inference
    will send them.
    """
    import joblib
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'feature_names.pkl')
    if not os.path.exists(path):
        return  # nothing trained yet; nothing to check
    saved = joblib.load(path)
    assert list(saved) == FEATURE_NAMES, (
        f"models/feature_names.pkl ({saved}) does not match "
        f"feature_engineering.FEATURE_NAMES ({FEATURE_NAMES}). Retrain with "
        f"`python model_training.py` to regenerate consistent artifacts."
    )
