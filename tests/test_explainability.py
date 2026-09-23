"""
tests/test_explainability.py
-----------------------------
Deterministic tests for explainability.py (Phase 4) and its integration
into prediction.py::predict_placement().

Covers:
  1. Logistic Regression contribution calculation
  2. contribution signs
  3. contribution ranking
  4. contribution sum + intercept ~= decision score
  5. tree-model fallback
  6. unsupported-model fallback
  7. predict_placement() regression compatibility (existing keys untouched)
  8. empty/unavailable explainability rendering (no crash, template-safe shape)

Run: pytest tests/test_explainability.py -v
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from feature_engineering import FEATURE_NAMES, FEATURE_LABELS
from explainability import (
    compute_feature_contributions,
    LINEAR_METHOD, TREE_METHOD, UNAVAILABLE_METHOD,
)


# ------------------------------------------------------------------
# Fakes -- deterministic, hand-built "models" so contribution math can
# be checked against known-by-construction values rather than whatever
# the real trained model happens to output.
# ------------------------------------------------------------------
class FakeLinearModel:
    """Mimics sklearn's LogisticRegression coef_/intercept_ shape."""
    def __init__(self, coef, intercept):
        self.coef_ = np.array([coef])       # shape (1, n_features), like sklearn binary LR
        self.intercept_ = np.array([intercept])  # shape (1,)


class FakeTreeModel:
    """Mimics sklearn's RandomForest/GradientBoosting feature_importances_."""
    def __init__(self, importances):
        self.feature_importances_ = np.array(importances)


class FakeUnsupportedModel:
    """Neither coef_ nor feature_importances_ -- e.g. a non-linear-kernel SVM."""
    pass


N = len(FEATURE_NAMES)


def _uniform_coef(value=0.1):
    return [value] * N


# ------------------------------------------------------------------
# 1 + 2 + 3: linear contribution calculation, signs, ranking
# ------------------------------------------------------------------
def test_linear_contribution_calculation_is_exact():
    # coef_i * x_i by construction, per feature -- deliberately varied
    # magnitudes/signs so ranking is unambiguous.
    coef = [0.5, -0.3, 0.0, 1.2, -2.0, 0.1, 0.05, -0.05, 0.0, 0.8, -1.5, 0.02]
    assert len(coef) == N
    x = [1.0] * N  # scaled feature value of 1.0 for every feature -> contribution == coef
    intercept = 0.75

    model = FakeLinearModel(coef, intercept)
    result = compute_feature_contributions(model, x, feature_names=FEATURE_NAMES)

    assert result['method'] == LINEAR_METHOD
    by_feature = {f['feature']: f for f in result['factors']}

    for name, expected_contrib in zip(FEATURE_NAMES, coef):
        assert math.isclose(by_feature[name]['contribution'], expected_contrib, rel_tol=1e-9, abs_tol=1e-9)


def test_contribution_signs_match_coefficient_and_value_sign():
    coef = [1.0] * N
    x = [2.0, -3.0, 0.0] + [1.0] * (N - 3)
    model = FakeLinearModel(coef, intercept=0.0)
    result = compute_feature_contributions(model, x, feature_names=FEATURE_NAMES)

    by_feature = {f['feature']: f for f in result['factors']}
    assert by_feature[FEATURE_NAMES[0]]['direction'] == 'up'     # coef 1.0 * x 2.0  = +2.0
    assert by_feature[FEATURE_NAMES[1]]['direction'] == 'down'   # coef 1.0 * x -3.0 = -3.0
    assert by_feature[FEATURE_NAMES[2]]['direction'] == 'flat'   # coef 1.0 * x 0.0  = 0.0


def test_factors_are_ranked_by_absolute_contribution_strongest_first():
    coef = [0.1, -5.0, 2.0, 0.0, 0.3, -0.2, 0.05, 0.01, 0.0, 0.02, 0.5, -0.1]
    x = [1.0] * N
    model = FakeLinearModel(coef, intercept=0.0)
    result = compute_feature_contributions(model, x, feature_names=FEATURE_NAMES)

    magnitudes = [abs(f['contribution']) for f in result['factors']]
    assert magnitudes == sorted(magnitudes, reverse=True), "factors must be sorted strongest-first by |contribution|"
    # The largest-magnitude coefficient (-5.0, on FEATURE_NAMES[1]) must rank first.
    assert result['factors'][0]['feature'] == FEATURE_NAMES[1]


# ------------------------------------------------------------------
# 4: sum(contributions) + intercept ~= decision score
# ------------------------------------------------------------------
def test_contribution_sum_plus_intercept_equals_decision_score():
    rng = np.random.default_rng(42)
    coef = rng.normal(size=N).tolist()
    x = rng.normal(size=N).tolist()
    intercept = 1.2345

    model = FakeLinearModel(coef, intercept)
    result = compute_feature_contributions(model, x, feature_names=FEATURE_NAMES)

    manual_decision_score = float(np.dot(coef, x) + intercept)
    assert math.isclose(result['decision_score'], manual_decision_score, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(result['contribution_sum_check'], manual_decision_score, rel_tol=1e-9, abs_tol=1e-9)

    contrib_sum = sum(f['contribution'] for f in result['factors'])
    assert math.isclose(contrib_sum + result['intercept'], result['decision_score'], rel_tol=1e-9, abs_tol=1e-9)


def test_against_real_trained_model_if_artifacts_exist():
    """
    If a real trained model is present (models/trained_model.pkl), verify
    our contribution_sum_check reproduces sklearn's own decision_function()
    output exactly for a real input -- not just our own fake models.
    """
    import joblib
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'trained_model.pkl')
    if not os.path.exists(model_path):
        return  # nothing trained yet; nothing to check
    model = joblib.load(model_path)
    if not hasattr(model, 'coef_'):
        return  # currently-selected model isn't linear; covered by the tree/fallback tests instead

    import prediction as pred
    sample = dict(tenth=85, twelfth=90, cgpa=8.5, backlogs=0, aptitude=75, technical=80,
                  communication=60, resume_score=82, domain='Data Science')
    X = pred._build_feature_vector(sample)
    X_scaled = pred.scaler.transform(X)
    real_decision = float(pred.model.decision_function(X_scaled)[0])

    result = compute_feature_contributions(pred.model, X_scaled[0], feature_names=(pred.feature_names or FEATURE_NAMES))
    assert result['method'] == LINEAR_METHOD
    assert math.isclose(result['contribution_sum_check'], real_decision, rel_tol=1e-9, abs_tol=1e-9)


# ------------------------------------------------------------------
# 5: tree-model fallback
# ------------------------------------------------------------------
def test_tree_model_uses_global_importance_fallback():
    importances = [0.05, 0.30, 0.02, 0.0, 0.10, 0.08, 0.03, 0.02, 0.0, 0.15, 0.20, 0.05]
    assert len(importances) == N
    model = FakeTreeModel(importances)

    result = compute_feature_contributions(model, [0.0] * N, feature_names=FEATURE_NAMES)

    assert result['method'] == TREE_METHOD
    assert result['decision_score'] is None
    assert result['intercept'] is None
    # Unsigned: importances have no direction, must be reported as neutral.
    assert all(f['direction'] == 'neutral' for f in result['factors'])
    # Ranked strongest importance first.
    assert result['factors'][0]['contribution'] == max(importances)
    contribs = [f['contribution'] for f in result['factors']]
    assert contribs == sorted(contribs, reverse=True)


# ------------------------------------------------------------------
# 6: unsupported-model fallback
# ------------------------------------------------------------------
def test_unsupported_model_returns_unavailable_not_a_crash():
    model = FakeUnsupportedModel()
    result = compute_feature_contributions(model, [0.0] * N, feature_names=FEATURE_NAMES)

    assert result['method'] == UNAVAILABLE_METHOD
    assert result['factors'] == []
    assert result['decision_score'] is None
    assert result['contribution_sum_check'] is None


def test_malformed_linear_model_shape_mismatch_falls_back_safely():
    """A coef_ vector with the wrong length must not raise -- degrade to 'unavailable'."""
    model = FakeLinearModel(coef=[1.0, 2.0, 3.0], intercept=0.0)  # wrong length vs N
    result = compute_feature_contributions(model, [0.0] * N, feature_names=FEATURE_NAMES)
    assert result['method'] == UNAVAILABLE_METHOD


def test_feature_labels_cover_every_canonical_feature():
    """feature_engineering.FEATURE_LABELS must have an entry for every FEATURE_NAMES key."""
    assert set(FEATURE_LABELS.keys()) == set(FEATURE_NAMES)


# ------------------------------------------------------------------
# 7: predict_placement() regression compatibility
# ------------------------------------------------------------------
def test_predict_placement_still_returns_all_prior_keys_plus_explainability():
    import prediction as pred
    sample = dict(tenth=85, twelfth=90, cgpa=8.5, backlogs=0, aptitude=75, technical=80,
                  communication=60, resume_score=82, domain='Data Science')
    result = pred.predict_placement(sample)

    prior_keys = {
        'prediction', 'confidence', 'readiness_score', 'readiness_level',
        'all_model_preds', 'skill_gap', 'domain_recommendation', 'job_roles',
        'roadmap', 'courses', 'model_used', 'result_label',
    }
    assert prior_keys.issubset(result.keys()), "Phase 4 must not remove/rename any pre-existing predict_placement() key"
    assert 'explainability' in result
    assert result['explainability']['method'] in (LINEAR_METHOD, TREE_METHOD, UNAVAILABLE_METHOD)


# ------------------------------------------------------------------
# 8: empty/unavailable explainability rendering (template-safety)
# ------------------------------------------------------------------
def test_unavailable_result_shape_is_safe_for_template_iteration():
    """
    result.html iterates result.explainability.factors and checks
    result.explainability.method. An 'unavailable' result must still have
    both keys, with factors as an (empty) list Jinja can safely loop over
    with a {% for %} and {% if factors %} without raising.
    """
    result = compute_feature_contributions(FakeUnsupportedModel(), [0.0] * N, feature_names=FEATURE_NAMES)
    assert isinstance(result['factors'], list)
    assert len(result['factors']) == 0
    assert isinstance(result['method'], str)
