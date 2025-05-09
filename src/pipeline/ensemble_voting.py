"""
ensemble_model.py

Constructs and evaluates a voting ensemble using Logistic Regression,
Random Forest, and XGBoost models optimized via BayesSearchCV.

The ensemble aggregates predictions and evaluates final performance
on the imbalanced test set. This module assumes all individual models
and the column order are already saved and available.
"""
import sys
print(sys.executable)


import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score, precision_score, recall_score
from preprocessing.data_preprocessing import preprocess_data
import json


def load_models_and_columns():
    logreg = joblib.load("../../models/logistic_regression_bayes.pkl")
    rf = joblib.load("../../models/random_forest_bayes.pkl")
    xgb = joblib.load("../../models/xgboost_bayes.pkl")
    
    with open("../../models/column_order_logreg.json") as f:
        column_order = json.load(f)

    return logreg, rf, xgb, column_order


def build_voting_classifier(logreg, rf, xgb):
    ensemble = VotingClassifier(
        estimators=[
            ('logreg', logreg),
            ('rf', rf),
            ('xgb', xgb)
        ],
        voting='soft',
        weights=[3, 1, 3],  # You can adjust this depending on model reliability
        n_jobs=-1
    )
    return ensemble


def evaluate_voting_model(model, X_test, y_test, threshold=0.3):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n📊 Voting Ensemble Evaluation @ Threshold = {threshold}")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    f2 = fbeta_score(y_test, y_pred, beta=2)
    print("F2 Score:", round(f2, 4))

    return y_pred, y_prob


def run_voting_pipeline():
    logreg, rf, xgb, column_order = load_models_and_columns()
    ensemble = build_voting_classifier(logreg, rf, xgb)

    X_train, X_test, y_train, y_test = preprocess_data()
    X_train = X_train[column_order]
    X_test = X_test[column_order]

    ensemble.fit(X_train, y_train)
    evaluate_voting_model(ensemble, X_test, y_test)

    joblib.dump(ensemble, "../../models/voting_ensemble.pkl")
    print("Voting ensemble saved to /models/voting_ensemble.pkl")


if __name__ == '__main__':
    run_voting_pipeline()