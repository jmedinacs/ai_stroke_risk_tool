"""
ensemble_diagnostic.py

This module runs diagnostics on the final voting ensemble model, including:
- Learning curve visualization
- Model agreement analysis
- Voting breakdown by prediction agreement level
- Export of summaries for reporting and presentation

Author: John Medina
Date: 2025-05-11
Project: ai_stroke_risk_tool
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, fbeta_score

from preprocessing.data_preprocessing import preprocess_data


def plot_learning_curve(model, X_train, y_train, tag="ensemble"):
    """
    Plots and saves the learning curve for a model using F2 score.

    Parameters:
    - model: Trained classifier
    - X_train: Training features
    - y_train: Training labels
    - tag (str): Model identifier used in the plot title and filename
    """
    scorer = make_scorer(fbeta_score, beta=2)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        scoring=scorer,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training F2 Score')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation F2 Score')
    plt.title(f'Learning Curve – {tag}')
    plt.xlabel('Training Set Size')
    plt.ylabel('F2 Score')
    plt.legend()
    plt.grid(True)
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig(f"../../outputs/figures/ensemble_learning_curve_{tag}.png", dpi=300)
    plt.show()
    print(f"Saved learning curve: ensemble_learning_curve_{tag}.png")


def plot_voting_agreement_pie(summary, tag="ensemble"):
    """
    Plots and saves a pie chart summarizing model prediction agreement levels.

    Parameters:
    - summary (Series): Value counts of agreement levels
    - tag (str): Used in filename
    """
    plt.figure(figsize=(6, 6))
    summary.plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        label='',
        ylabel=''
    )
    plt.title(f"Voting Agreement – {tag}")
    plt.tight_layout()
    plt.savefig(f"../../outputs/figures/voting_agreement_pie_{tag}.png", dpi=300)
    plt.show()
    print(f"Saved voting agreement pie chart: voting_agreement_pie_{tag}.png")


def analyze_voting_agreement(ensemble, X_test, y_test, threshold=0.3):
    """
    Analyzes agreement between individual models and ensemble predictions.

    Parameters:
    - ensemble: Trained VotingClassifier
    - X_test (DataFrame): Test features
    - y_test (Series): True test labels
    - threshold (float): Classification threshold for stroke prediction

    Returns:
    - df (DataFrame): Detailed prediction comparison table
    - summary (Series): Agreement level counts
    - google_summary (DataFrame): Aggregated Google Sheets-style table
    """
    estimators = dict(ensemble.named_estimators_)
    preds = {
        "logreg": (estimators['logreg'].predict_proba(X_test)[:, 1] >= threshold).astype(int),
        "rf":     (estimators['rf'].predict_proba(X_test)[:, 1] >= threshold).astype(int),
        "xgb":    (estimators['xgb'].predict_proba(X_test)[:, 1] >= threshold).astype(int),
        "ensemble": (ensemble.predict_proba(X_test)[:, 1] >= threshold).astype(int),
        "actual": y_test.values
    }

    df = pd.DataFrame(preds)
    df['votes_for_stroke'] = df[['logreg', 'rf', 'xgb']].sum(axis=1)
    df['agreement_level'] = df['votes_for_stroke'].map({
        0: "All predict no-stroke",
        1: "1 predicts stroke",
        2: "2 predict stroke",
        3: "All predict stroke"
    })

    summary = df['agreement_level'].value_counts().sort_index()
    print("\n🗳️ Voting Agreement Summary:")
    for level, count in summary.items():
        pct = count / len(df) * 100
        print(f"{level}: {count} cases ({pct:.1f}%)")

    df['correct'] = (df['ensemble'] == df['actual'])

    grouped = df.groupby(['agreement_level', 'actual']).size().unstack(fill_value=0)
    grouped.columns = ['True Negative (0)', 'True Positive (1)']
    grouped['Total'] = grouped.sum(axis=1)
    grouped['TP Rate (%)'] = (grouped['True Positive (1)'] / grouped['Total']) * 100

    print("\nTrue Label Breakdown by Agreement Level:")
    print(grouped.round(1).to_string())

    # Google Sheets-style table
    google_summary = pd.DataFrame(columns=[
        "Agreement Level", "Instances", "True Positive", "Percent of All Strokes"
    ])

    total_strokes = (df["actual"] == 1).sum()
    label_map = {
        "1 predicts stroke": "1 Model Predict Stroke",
        "2 predict stroke": "2 Models Predict Stroke",
        "All predict stroke": "All Models Predict Stroke",
        "All predict no-stroke": "No Model Predicts Stroke"
    }

    for level, row in grouped.iterrows():
        label = label_map.get(level, level)
        instances = int(row["Total"])
        true_positives = int(row["True Positive (1)"])
        pct_of_all = f"{(true_positives / total_strokes) * 100:.2f}%"

        google_summary = pd.concat([
            google_summary,
            pd.DataFrame([{
                "Agreement Level": label,
                "Instances": instances,
                "True Positive": true_positives,
                "Percent of All Strokes": pct_of_all
            }])
        ], ignore_index=True)

    print("\nSummary:")
    print(google_summary.to_string(index=False))
    google_summary.to_csv("../../outputs/voting_agreement_table.csv", index=False)

    return df, summary, google_summary


def load_ensemble_and_columns():
    """
    Loads the final ensemble model and column order used during training.

    Returns:
    - ensemble: Trained VotingClassifier
    - column_order (list): Ordered list of feature names
    """
    ensemble = joblib.load("../../models/voting_ensemble.pkl")
    with open("../../models/column_order_logreg.json") as f:
        column_order = json.load(f)
    return ensemble, column_order


def run_ensemble_diagnostics():
    """
    Runs ensemble diagnostics including:
    - Learning curve visualization
    - Agreement analysis
    - Export of pie chart, agreement table, and summary
    """
    ensemble, column_order = load_ensemble_and_columns()
    X_train, X_test, y_train, y_test = preprocess_data()
    X_train = X_train[column_order]
    X_test = X_test[column_order]

    tag = 'ensemble_v3'
    print(f"\nRunning diagnostics for: {tag}")
    plot_learning_curve(ensemble, X_train, y_train, tag)

    df_votes, agreement_summary, google_summary = analyze_voting_agreement(ensemble, X_test, y_test)
    plot_voting_agreement_pie(agreement_summary, tag)
    df_votes.to_csv("../../outputs/voting_agreement_analysis.csv", index=False)
    print("Saved voting agreement analysis to /outputs/voting_agreement_analysis.csv")


if __name__ == '__main__':
    run_ensemble_diagnostics()
