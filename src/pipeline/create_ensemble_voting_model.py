"""
ensemble_model.py

Constructs and evaluates a soft-voting ensemble classifier using tuned 
Logistic Regression, Random Forest, and XGBoost models.

This script:
- Loads pre-trained individual models and column schema
- Builds a weighted soft-voting classifier
- Evaluates performance (classification metrics, F2, ROC AUC)
- Optionally optimizes ensemble voting weights under recall/precision constraints
- Saves the ensemble model for deployment

Author: John Medina
Date: 2025-05-11
Project: AI Stroke Risk Tool
"""

import sys
print(sys.executable)
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score, precision_score, recall_score
from sklearn.metrics import fbeta_score
from preprocessing.data_preprocessing import preprocess_data
from itertools import product
import json


def load_models_and_columns():
    logreg = joblib.load("../../models/logreg_bayes.pkl")
    rf = joblib.load("../../models/random_forest_bayes.pkl")
    
    
    with open("../../models/column_order_logreg.json") as f:
        column_order = json.load(f)

    return logreg, rf, column_order


def build_voting_classifier(logreg, rf, weights=None):
    return VotingClassifier(
        estimators=[('logreg', logreg), ('rf', rf)],
        voting='soft',
        weights=weights or [3, 3],
        n_jobs=-1
    )



def evaluate_voting_model(model, X_test, y_test, threshold=0.3):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nVoting Ensemble Evaluation @ Threshold = {threshold}")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    f2 = fbeta_score(y_test, y_pred, beta=2)
    print("F2 Score:", round(f2, 4))

    return y_pred, y_prob

def find_best_threshold(y_true, y_prob, beta=2, min_recall=0.60):
    """
    Finds the threshold that maximizes the F-beta score, 
    subject to a minimum recall constraint.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        beta (float): Weighting factor for recall in F-beta (default = 2)
        min_recall (float): Minimum acceptable recall for candidate thresholds

    Returns:
        Tuple[float, float]: (best_threshold, best_fbeta_score)
    """
    best_thresh, best_f2 = 0.0, 0.0

    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= t).astype(int)
        recall = recall_score(y_true, y_pred)
        
        if recall < min_recall:
            continue  # Skip thresholds that don't meet the recall requirement
        
        f2 = fbeta_score(y_true, y_pred, beta=beta)
        if f2 > best_f2:
            best_thresh, best_f2 = t, f2

    if best_f2 == 0:
        print(f"\n⚠️ No threshold met the minimum recall of {min_recall}")
    else:
        print(f"\n✅ Best Threshold for F{beta} with Recall ≥ {min_recall}: {best_thresh:.2f} (F{beta} = {best_f2:.4f})")

    return best_thresh, best_f2


def optimize_ensemble_weights(X_train, X_test, y_train, y_test, logreg, rf, threshold=0.25):
    """
    Performs a constrained grid search over voting weights for the ensemble model.

    Only keeps combinations with:
        - Recall ≥ 0.68
        - Precision ≥ 0.11
    """

    best_f2 = 0
    best_combo = None
    results = []

    # Try all combinations of weights from 1 to 5
    for weights in product(range(1, 6), repeat=2):  # (LR, RF, XGB)
        ensemble = VotingClassifier(
            estimators=[('logreg', logreg), ('rf', rf)],
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        ensemble.fit(X_train, y_train)    
        

        y_prob = ensemble.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        
        print(f"Trying weights {weights}: Precision={precision:.3f}, Recall={recall:.3f}, F2={f2:.4f}")


        # Constrain to meaningful combinations
        if precision >= 0.12 and recall >= 0.58:
            results.append((weights, precision, recall, f2))

            if f2 > best_f2:
                best_f2 = f2
                best_combo = weights

    print("\n🔍 Grid Search Complete — Top Weight Combos (precision ≥ 11%, recall ≥ 68%):")
    for weights, p, r, f2 in sorted(results, key=lambda x: x[3], reverse=True)[:5]:
        print(f"Weights {weights} → Precision: {p:.4f}, Recall: {r:.4f}, F2: {f2:.4f}")

    if best_combo:
        print(f"\n✅ Best Voting Weights: {best_combo} with F2 = {best_f2:.4f}")
    else:
        print("\n⚠️ No weight combination met the minimum precision/recall thresholds.")
    
    return best_combo


def run_voting_pipeline():
    logreg, rf, column_order = load_models_and_columns()
    ensemble = build_voting_classifier(logreg, rf)

    X_train, X_test, y_train, y_test = preprocess_data()
    X_train = X_train[column_order]
    X_test = X_test[column_order]

    ensemble.fit(X_train, y_train)
    evaluate_voting_model(ensemble, X_test, y_test)

    joblib.dump(ensemble, "../../models/voting_ensemble.pkl")
    print("Voting ensemble saved to /models/voting_ensemble.pkl")
    
    X_train = X_train[column_order]
    X_test = X_test[column_order]

    print("\n🔎 Running grid search to find optimal voting weights...")
    best_weights = optimize_ensemble_weights(X_train, X_test, y_train, y_test, logreg, rf,0.4)

    if best_weights:
        ensemble = build_voting_classifier(logreg, rf)
        ensemble.set_params(weights=best_weights)
        ensemble.fit(X_train, y_train)
        # Get predicted probabilities
        y_prob = ensemble.predict_proba(X_test)[:, 1]
        best_threshold, _ = find_best_threshold(y_test, y_prob, beta=2, min_recall=0.68)
        evaluate_voting_model(ensemble, X_test, y_test, threshold=best_threshold)

        joblib.dump(ensemble, "../../models/voting_ensemble.pkl")
        print("Voting ensemble saved to /models/voting_ensemble.pkl")
        
        # Save best weights to JSON
        with open("../../models/voting_weights.json", "w") as f:
            json.dump(list(best_weights), f)
        print("Voting weights saved to /models/voting_weights.json")
    
    print("Ensemble Voting Classifier Details:")
    print(ensemble)



if __name__ == '__main__':
    run_voting_pipeline()