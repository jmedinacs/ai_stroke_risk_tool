"""
compare_models.py

Loads trained models (LogReg, RF, XGB), evaluates them on the same test set,
and prints a side-by-side comparison of performance metrics.
"""

import joblib
import numpy as np
import pandas as pd
from preprocessing.data_preprocessing import preprocess_data
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import make_scorer, fbeta_score, f1_score, precision_score, recall_score

def load_model_and_threshold(model_path, threshold):
    model = joblib.load(model_path)
    return model, threshold


def evaluate_model(model, X_test, y_test, threshold):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    auc = roc_auc_score(y_test, y_prob)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "roc_auc": auc
    }


def summarize_results():
    X_train, X_test, y_train, y_test = preprocess_data()

    models_info = [
        ("Logistic Regression (Bayes)", "../../models/logreg_bayes.pkl", 0.3),
        ("Random Forest (Bayes)", "../../models/random_forest_bayes.pkl", 0.3),
        ("XGBoost (Bayes)", "../../models/xgboost_bayes.pkl", 0.24)
    ]

    summary = []
    for label, path, threshold in models_info:
        model, t = load_model_and_threshold(path, threshold)
        metrics = evaluate_model(model, X_test, y_test, t)
        summary.append({"model": label, **metrics})

    df = pd.DataFrame(summary)
    print("\n🔍 Model Performance Summary:\n")
    print(df.to_string(index=False, float_format="{:.3f}".format))


if __name__ == '__main__':
    summarize_results()