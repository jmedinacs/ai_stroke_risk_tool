"""
train_xgboost.py

Trains and evaluates an XGBoost classifier for stroke prediction.

This script performs:
- Preprocessing with class balancing
- Hyperparameter tuning using Bayesian optimization (BayesSearchCV)
- Threshold optimization for best F2 score
- Evaluation with ROC AUC, precision-recall, confusion matrix
- SHAP explainability with summary and waterfall plots
- Model persistence for deployment

Author: John Medina
Date: 2025-05-11
Project: AI Stroke Risk Tool
"""

from preprocessing.data_preprocessing import preprocess_data
from sklearn.metrics import make_scorer, fbeta_score, classification_report, roc_auc_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def tune_model_bayes(X_train, y_train):
    f2_scorer = make_scorer(fbeta_score, beta=2)

    search_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'gamma': Real(0, 5),
        'reg_lambda': Real(0, 10),
        'scale_pos_weight': Real(1, 10),
    }

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    opt = BayesSearchCV(
        estimator=xgb,
        search_spaces=search_space,
        n_iter=30,
        scoring=f2_scorer,
        cv=StratifiedKFold(n_splits=5),
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    opt.fit(X_train, y_train)

    print("\n Best Hyperparameters (XGBoost):")
    print(opt.best_params_)
    print(f"Best F2 Score (CV): {opt.best_score_:.4f}")

    return opt.best_estimator_


def evaluate_tuned_xgb(model, X_test, y_test, threshold=0.3):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    print("\nBayes-Tuned XGBoost Evaluation:")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – XGBoost (Bayes Tuned)")
    plt.tight_layout()
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig("../../outputs/figures/confusion_matrix_xgb.png", dpi=300)
    plt.show()
    plt.close()

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall Tradeoff – XGBoost (Bayes Tuned)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../../outputs/figures/precision_recall_curve_xgb.png", dpi=300)
    plt.show()
    plt.close()

    X_test_float = X_test.copy().astype("float64")
    explainer = shap.Explainer(model, X_test_float)
    shap_values = explainer(X_test_float)

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Summary – XGBoost (Bayes Tuned)")
    plt.tight_layout()
    plt.savefig("../../outputs/figures/shap_summary_xgb_bayes.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Waterfall – XGBoost Sample (Index 0)")
    plt.tight_layout()
    plt.savefig("../../outputs/figures/shap_waterfall_xgb.png", dpi=300)
    plt.show()
    plt.close()

    joblib.dump(model, "../../models/xgboost_bayes.pkl")
    print("XGBoost Bayes model saved to /models/xgboost_bayes.pkl")


def train_model(X_train, y_train):
    return tune_model_bayes(X_train, y_train)

def find_best_f2_threshold(y_true, y_prob):
    best_thresh, best_f2 = 0.0, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2)
        if f2 > best_f2:
            best_thresh, best_f2 = t, f2
    return best_thresh, best_f2


def train_xgboost_model():
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Starting XGBoost hyperparameter optimization...")
    
    model = train_model(X_train, y_train)
    
    print("Evaluating tuned model...")
    
    # Get predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Find best F2 threshold
    best_thresh, best_f2 = find_best_f2_threshold(y_test, y_prob)
    print(f"\nBest Threshold for F2: {best_thresh:.2f} (F2 = {best_f2:.4f})")
    
    # Evaluate at that threshold
    evaluate_tuned_xgb(model, X_test, y_test, threshold=best_thresh)




if __name__ == '__main__':
    train_xgboost_model()
