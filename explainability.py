"""
explainability.py
------------------
Per-prediction "Factors influencing this model estimate" breakdown for
the placement readiness model. See docs/ml-methodology.md for the full
write-up (methodology, verification, and honesty caveats).

Method
------
For a LINEAR model (LogisticRegression -- the currently selected model,
see models/model_metadata.json), the model's own decision function is:

    decision_score = sum_i( coef_i * X_scaled_i ) + intercept

Each feature's CONTRIBUTION to that score is exactly:

    contribution_i = coef_i * X_scaled_i

This is an EXACT decomposition of the model's own linear decision
function for THIS specific (already-scaled) input -- not an approximation,
and not SHAP. tests/test_explainability.py verifies, for real data, that
sum(contributions) + intercept reproduces the model's own
decision_function() output.

For a TREE-BASED model (RandomForest/GradientBoosting/DecisionTree), no
per-prediction linear decomposition exists without a library like SHAP,
which is explicitly out of scope for this project (see project brief).
We fall back to the model's GLOBAL feature_importances_ instead --
clearly labeled as "what the model generally weighs across all
predictions," not a per-prediction explanation. This is a materially
weaker, different claim, and callers/templates must not blur the two.

For any other model type (e.g. an SVM with a non-linear kernel), neither
is available; compute_feature_contributions() returns an explicit
'unavailable' result rather than guessing or fabricating a ranking.

IMPORTANT -- model contribution is not real-world causation
--------------------------------------------------------------
These numbers describe what the ALGORITHM weighted internally, for a
model trained on a 100% synthetic, formula-generated dataset (see
docs/ml-methodology.md's target-leakage audit). A large positive
"Technical assessment" contribution means the model's own math leaned on
that feature for this input -- it is not evidence that technical scores
*cause* real-world placement outcomes. Callers must present this as
"factors influencing this model's estimate," never as "why you will get
placed."
"""

import numpy as np

from feature_engineering import FEATURE_NAMES, FEATURE_LABELS

LINEAR_METHOD = 'linear_coefficient'
TREE_METHOD = 'tree_importance_global'
UNAVAILABLE_METHOD = 'unavailable'


def compute_feature_contributions(model, X_scaled_row, feature_names=None):
    """
    model: the fitted sklearn estimator actually used for this prediction
           (the same `model` object prediction.py already calls
           .predict()/.predict_proba() on -- no separate model is loaded
           or retrained here).
    X_scaled_row: 1D array-like, the SAME already-scaled feature vector
                  prediction.py already computed via `scaler.transform(X)`
                  for this request. This function does not re-scale
                  anything -- there remains exactly one place in the
                  codebase (prediction.py) that scales a feature vector,
                  per the Phase 1 train/inference-consistency fix.
    feature_names: canonical feature name order; defaults to
                   feature_engineering.FEATURE_NAMES.

    Returns a dict:
        {
          'method': 'linear_coefficient' | 'tree_importance_global' | 'unavailable',
          'factors': [ {feature, label, contribution, direction}, ... ],  # strongest first
          'intercept': float | None,
          'decision_score': float | None,          # linear method only
          'contribution_sum_check': float | None,   # sum(contributions)+intercept
        }

    Never raises: any unexpected model shape/type falls through to the
    'unavailable' result, so an explainability failure can never break
    the prediction page itself.
    """
    names = list(feature_names) if feature_names else list(FEATURE_NAMES)
    x = np.asarray(X_scaled_row, dtype=float).reshape(-1)

    try:
        if hasattr(model, 'coef_'):
            return _linear_contributions(model, x, names)
        elif hasattr(model, 'feature_importances_'):
            return _tree_importance_fallback(model, names)
        else:
            return _unavailable()
    except Exception:
        return _unavailable()


def _linear_contributions(model, x, names):
    coefs = np.asarray(model.coef_).reshape(-1)
    intercept_arr = np.asarray(model.intercept_).reshape(-1)

    if len(coefs) != len(x) or len(coefs) != len(names) or len(intercept_arr) == 0:
        return _unavailable()

    intercept = float(intercept_arr[0])
    contributions = coefs * x  # exact term-by-term decomposition: coef_i * scaled_value_i
    decision_score = float(contributions.sum() + intercept)

    factors = []
    for name, contrib in zip(names, contributions):
        contrib = float(contrib)
        factors.append({
            'feature'     : name,
            'label'       : FEATURE_LABELS.get(name, name),
            'contribution': contrib,
            'direction'   : 'up' if contrib > 0 else ('down' if contrib < 0 else 'flat'),
        })

    factors.sort(key=lambda f: abs(f['contribution']), reverse=True)

    return {
        'method'                : LINEAR_METHOD,
        'factors'               : factors,
        'intercept'             : intercept,
        'decision_score'        : decision_score,
        # Identical to decision_score by construction (sum(contributions) + intercept)
        # -- kept as a separate, explicitly-named key so tests assert the
        # invariant rather than just trusting the label.
        'contribution_sum_check': decision_score,
    }


def _tree_importance_fallback(model, names):
    importances = np.asarray(model.feature_importances_).reshape(-1)
    if len(importances) != len(names):
        return _unavailable()

    factors = []
    for name, imp in zip(names, importances):
        factors.append({
            'feature'     : name,
            'label'       : FEATURE_LABELS.get(name, name),
            'contribution': float(imp),  # unsigned global importance, not a per-prediction direction
            'direction'   : 'neutral',
        })
    factors.sort(key=lambda f: f['contribution'], reverse=True)

    return {
        'method'                : TREE_METHOD,
        'factors'               : factors,
        'intercept'             : None,
        'decision_score'        : None,
        'contribution_sum_check': None,
    }


def _unavailable():
    return {
        'method'                : UNAVAILABLE_METHOD,
        'factors'               : [],
        'intercept'             : None,
        'decision_score'        : None,
        'contribution_sum_check': None,
    }
