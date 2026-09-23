"""
model_training.py
-----------------
Trains and evaluates placement-readiness models on the placement dataset.
Run: python model_training.py

CHANGE LOG (ML correctness pass)
---------------------------------
1. Feature engineering (academic_score, skill_score, backlog_penalty) now
   comes from feature_engineering.py -- the SAME functions prediction.py
   uses at inference time. Previously this file computed skill_score with
   different weights than prediction.py used, silently corrupting every
   live prediction. See feature_engineering.py's module docstring.

2. StandardScaler leakage fix. Previously the scaler was fit on the FULL
   dataset (train+test combined) before the train/test split, so scaling
   statistics (mean/std) were computed using data the model was later
   "tested" on. The scaler is now fit ONLY on the training split; the
   test split (and, at inference time, real user input) is transformed
   using those training-only statistics.

3. Reproducible, stratified train/test split (test held out completely,
   never touched during model selection) plus stratified 5-fold cross-
   validation WITHIN the training split for model comparison. Both the
   split and the CV folds are stratified by placement_status because the
   classes are imbalanced (~75/25).

4. Full metrics: accuracy, precision, recall, F1, balanced accuracy, and
   ROC-AUC (plus confusion matrix), reported separately for cross-
   validation (train-only) and the held-out test set. Model selection is
   based on cross-validated ROC-AUC + F1 on the training split only --
   the test set is touched exactly once, at the very end, to report
   final numbers.

5. Saves models/model_metadata.json documenting model version, feature
   names/order, dataset info, training configuration, and evaluation
   metrics -- see docs/ml-methodology.md for the full write-up, including
   the target-leakage investigation.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import platform
from datetime import datetime, timezone

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
)
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import (
    FEATURE_NAMES, compute_academic_score, compute_skill_score,
    compute_backlog_penalty,
)

RANDOM_STATE = 42


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
def load_data(path='dataset/placement_data.csv'):
    df = pd.read_csv(path)
    print(f"[INFO] Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"[INFO] Domains  : {sorted(df['domain'].unique())} ({df['domain'].nunique()} domains)")
    print(f"[INFO] Class distribution:\n{df['placement_status'].value_counts()}")
    print(f"[INFO] Positive class rate: {df['placement_status'].mean()*100:.2f}%\n")
    return df


# ============================================================
# STEP 2: TARGET-LEAKAGE AUDIT
# ============================================================
def audit_target_leakage(df):
    """
    dataset/placement_data.csv is 100% synthetic (see generate_dataset.py):
    every row's placement_status label is produced by thresholding a hand-
    written LINEAR function of the same raw columns used as model inputs,
    plus a small amount of Gaussian noise:

        score = 0.15*(tenth/100) + 0.15*(twelfth/100) + 0.25*(cgpa/10)
               + 0.15*(aptitude/100) + 0.18*(technical/100)
               + 0.07*(communication/100) + 0.05*(resume/100)
               - 0.10*backlogs + noise(0, 0.04)
        label = 1 if score >= 0.55 else 0

    This function quantifies exactly how much of the label is recoverable
    from a NOISE-FREE version of that same formula, i.e. how "given away"
    the label already is by the raw features before any model training
    happens. This is not train/test leakage (that's fixed separately) --
    it's a property of how the dataset itself was constructed, and it is
    the reason every model scores 90%+ accuracy: they are approximating a
    near-linear function of their own inputs, not learning a real-world
    relationship between student profiles and employment outcomes.
    """
    score_no_noise = (
        0.15 * (df.tenth_percentage / 100) +
        0.15 * (df.twelfth_percentage / 100) +
        0.25 * (df.cgpa / 10) +
        0.15 * (df.aptitude_score / 100) +
        0.18 * (df.technical_score / 100) +
        0.07 * (df.communication_score / 100) +
        0.05 * (df.resume_score / 100) -
        0.10 * df.backlogs
    )
    pred_from_formula = (score_no_noise >= 0.55).astype(int)
    agreement = float((pred_from_formula == df.placement_status).mean())
    corr = float(np.corrcoef(score_no_noise, df.placement_status)[0, 1])

    print("=" * 60)
    print("[AUDIT] TARGET LEAKAGE CHECK (synthetic dataset)")
    print("=" * 60)
    print(f"  Noise-free formula vs actual label agreement : {agreement*100:.2f}%")
    print(f"  Correlation(deterministic score, label)       : {corr:.4f}")
    print("  -> The label is a near-deterministic linear function of the")
    print("     model's own input features. High model accuracy on this")
    print("     dataset reflects recovering a known formula, NOT a")
    print("     validated real-world placement relationship.\n")

    return {
        'is_synthetic': True,
        'formula_agreement_pct': round(agreement * 100, 2),
        'formula_label_correlation': round(corr, 4),
        'note': (
            'placement_status is generated by thresholding a linear '
            'combination of the same raw features used as model inputs, '
            'plus small Gaussian noise. Model accuracy on this dataset '
            'measures recovery of a known synthetic formula, not a '
            'validated real-world employment relationship.'
        ),
    }


# ============================================================
# STEP 3: BUILD CANONICAL FEATURES (no scaling yet -- that happens after split)
# ============================================================
def build_features(df):
    df = df.copy()

    le = LabelEncoder()
    df['domain_encoded'] = le.fit_transform(df['domain'])

    # Same functions prediction.py uses at inference time -- see
    # feature_engineering.py. Vectorized: these operate on full columns.
    df['academic_score']  = compute_academic_score(df['tenth_percentage'], df['twelfth_percentage'], df['cgpa'])
    df['skill_score']     = compute_skill_score(df['aptitude_score'], df['technical_score'], df['communication_score'], df['resume_score'])
    df['backlog_penalty'] = compute_backlog_penalty(df['backlogs'])

    X = df[FEATURE_NAMES].copy()
    y = df['placement_status'].copy()

    print(f"[INFO] Canonical features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    print(f"[INFO] Label encoder classes: {list(le.classes_)}")
    return X, y, le


# ============================================================
# STEP 4: CROSS-VALIDATED MODEL COMPARISON (train split only)
# ============================================================
def compare_models_cv(X_train, y_train):
    """
    Compares candidate models using stratified 5-fold CV on the TRAINING
    split only. Each fold fits its own StandardScaler on that fold's
    training portion (via Pipeline) so there is no leakage even within
    cross-validation, not just at the final train/test split.
    """
    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree'      : DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        'Random Forest'      : RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
        'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM'                : SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    }

    scoring = {
        'accuracy'         : 'accuracy',
        'precision'        : 'precision',
        'recall'           : 'recall',
        'f1'               : 'f1',
        'balanced_accuracy': 'balanced_accuracy',
        'roc_auc'          : 'roc_auc',
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    print("=" * 60)
    print("CROSS-VALIDATED MODEL COMPARISON (train split only, 5-fold)")
    print("=" * 60)

    for name, model in candidates.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None)
        summary = {
            metric: round(float(np.mean(scores[f'test_{metric}'])) * 100, 2)
            for metric in scoring
        }
        cv_results[name] = summary
        print(f"\n{name}")
        for metric, val in summary.items():
            print(f"  cv_{metric:<18}: {val:.2f}%")

    return cv_results, candidates


# ============================================================
# STEP 5: FIT FINAL MODELS ON FULL TRAIN SPLIT, EVALUATE ON HELD-OUT TEST
# ============================================================
def fit_and_evaluate_final(candidates, X_train, X_test, y_train, y_test):
    """
    Fits a StandardScaler on X_train ONLY, applies it to both splits, then
    fits every candidate model on the (scaled) training split and reports
    metrics on the untouched test split. The test split is used here
    exactly once, for final reporting -- not for model selection.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    test_results = {}
    trained = {}

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET EVALUATION (scaler fit on train split only)")
    print("=" * 60)

    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        bacc  = balanced_accuracy_score(y_test, y_pred)
        cm    = confusion_matrix(y_test, y_pred)

        try:
            proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = None

        test_results[name] = {
            'accuracy'         : round(acc  * 100, 2),
            'precision'        : round(prec * 100, 2),
            'recall'           : round(rec  * 100, 2),
            'f1_score'         : round(f1   * 100, 2),
            'balanced_accuracy': round(bacc * 100, 2),
            'roc_auc'          : round(auc  * 100, 2) if auc is not None else None,
            'confusion_matrix' : cm.tolist(),
        }
        trained[name] = model

        print(f"\n{name}")
        print(f"  Accuracy          : {acc*100:.2f}%")
        print(f"  Precision         : {prec*100:.2f}%")
        print(f"  Recall            : {rec*100:.2f}%")
        print(f"  F1-Score          : {f1*100:.2f}%")
        print(f"  Balanced Accuracy : {bacc*100:.2f}%")
        print(f"  ROC-AUC           : {auc*100:.2f}%" if auc is not None else "  ROC-AUC           : n/a")
        print(f"  Confusion Matrix  :\n{cm}")

    return test_results, trained, scaler


# ============================================================
# STEP 6: SELECT PRODUCTION MODEL
# ============================================================
def select_best_model(cv_results, test_results):
    """
    Selection is based on cross-validated performance on the TRAINING
    split (not the test set, to avoid picking a model because it happens
    to fit the test set best). Because classes are imbalanced (~75/25),
    we rank by the mean of balanced_accuracy and roc_auc rather than raw
    accuracy or F1 alone -- raw accuracy rewards a model for defaulting to
    the majority class, balanced_accuracy and ROC-AUC do not.
    """
    def rank_key(name):
        r = cv_results[name]
        return (r['balanced_accuracy'] + r['roc_auc']) / 2

    best_name = max(cv_results, key=rank_key)
    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    for name in cv_results:
        print(f"  {name:<22} cv(balanced_acc+roc_auc)/2 = {rank_key(name):.2f}")
    print(f"\n[SELECTED] {best_name}")
    print(f"  Held-out test metrics: {test_results[best_name]}")
    return best_name


# ============================================================
# STEP 7: SAVE ARTIFACTS + METADATA
# ============================================================
def save_artifacts(best_name, trained, scaler, label_encoder, cv_results, test_results,
                    leakage_audit, X_train, X_test, df):
    os.makedirs('models', exist_ok=True)

    best_model = trained[best_name]
    joblib.dump(best_model, 'models/trained_model.pkl')
    joblib.dump(trained, 'models/all_models.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(FEATURE_NAMES, 'models/feature_names.pkl')

    evaluation_results = {
        'cross_validation_train_split': cv_results,
        'held_out_test_split': test_results,
    }
    with open('models/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    metadata = {
        'model_version': datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'),
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'selected_model': best_name,
        'selection_criterion': 'mean(cv_balanced_accuracy, cv_roc_auc) on training split (5-fold stratified CV)',
        'feature_names_in_order': FEATURE_NAMES,
        'n_features': len(FEATURE_NAMES),
        'label_classes_domain_encoder': list(label_encoder.classes_),
        'dataset': {
            'path': 'dataset/placement_data.csv',
            'n_rows': int(df.shape[0]),
            'n_domains': int(df['domain'].nunique()),
            'positive_class_rate_pct': round(float(df['placement_status'].mean()) * 100, 2),
            'is_synthetic': True,
            'synthetic_disclosure': (
                'This entire dataset is generated by generate_dataset.py using a '
                'hand-written linear formula plus Gaussian noise to assign '
                'placement_status. It is NOT real student outcome data. See '
                'docs/ml-methodology.md and target_leakage_audit below.'
            ),
        },
        'target_leakage_audit': leakage_audit,
        'split_config': {
            'method': 'stratified train/test split, stratified 5-fold CV within train split',
            'test_size': 0.2,
            'random_state': RANDOM_STATE,
            'n_train_rows': int(X_train.shape[0]),
            'n_test_rows': int(X_test.shape[0]),
            'scaler_fit_on': 'train split only (leakage-free)',
        },
        'environment': {
            'python_version': sys.version.split()[0],
            'sklearn_version': sklearn.__version__,
            'platform': platform.platform(),
        },
        'interpretation_guidance': (
            'Because the training target is formula-generated from the same '
            'features the model observes (see target_leakage_audit), reported '
            'accuracy/F1/ROC-AUC reflect how well each model recovers a known '
            'synthetic formula, not validated real-world placement probability. '
            'Do not present model output as a real-world "chance of getting a '
            'job." Use "model-based placement readiness estimate" language in '
            'the UI. See docs/ml-methodology.md.'
        ),
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[SAVED] models/trained_model.pkl        (selected: {best_name})")
    print("[SAVED] models/all_models.pkl")
    print("[SAVED] models/scaler.pkl                (fit on train split only)")
    print("[SAVED] models/label_encoder.pkl")
    print("[SAVED] models/feature_names.pkl")
    print("[SAVED] models/evaluation_results.json")
    print("[SAVED] models/model_metadata.json")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    df = load_data()
    leakage_audit = audit_target_leakage(df)
    X, y, label_encoder = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[SPLIT] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows (stratified, held out)\n")

    cv_results, candidates = compare_models_cv(X_train, y_train)
    test_results, trained, scaler = fit_and_evaluate_final(candidates, X_train, X_test, y_train, y_test)
    best_name = select_best_model(cv_results, test_results)
    save_artifacts(best_name, trained, scaler, label_encoder, cv_results, test_results,
                    leakage_audit, X_train, X_test, df)

    print("\n[TRAINING COMPLETE]")
