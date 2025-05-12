import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, fbeta_score
from preprocessing.data_preprocessing import preprocess_data
import matplotlib.pyplot as plt
import json 
import joblib
import os


def plot_learning_curve(model, X_train, y_train, tag="ensemble"):
    """
    Plots and saves the learning curve for the model.
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
    plt.savefig(f"../../outputs/figures/learning_curve_{tag}.png", dpi=300)
    plt.show()
    print(f"Saved learning curve: learning_curve_{tag}.png")
    
def plot_voting_agreement_pie(summary, tag="ensemble"):
    """
    Plots and saves a pie chart from the agreement summary.
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
    Analyzes agreement between individual models and the ensemble decision.

    Outputs a summary of:
        - 3-model agreement
        - 2-model agreement
        - 1-model agreement
        - Who disagreed
    """
    estimators = dict(ensemble.named_estimators_)
    logreg = estimators['logreg']
    rf = estimators['rf']
    xgb = estimators['xgb']

    # Predict with each model
    preds = {
        "logreg": (logreg.predict_proba(X_test)[:, 1] >= threshold).astype(int),
        "rf":     (rf.predict_proba(X_test)[:, 1] >= threshold).astype(int),
        "xgb":    (xgb.predict_proba(X_test)[:, 1] >= threshold).astype(int),
        "ensemble": (ensemble.predict_proba(X_test)[:, 1] >= threshold).astype(int),
        "actual": y_test.values
    }

    df = pd.DataFrame(preds)

    # Count number of models predicting stroke per row
    df['votes_for_stroke'] = df[['logreg', 'rf', 'xgb']].sum(axis=1)
    df['agreement_level'] = df['votes_for_stroke'].map({
        0: "All predict no-stroke",
        1: "1 predicts stroke",
        2: "2 predict stroke",
        3: "All predict stroke"
    })

    # Tally results
    summary = df['agreement_level'].value_counts().sort_index()
    print("\n🗳️ Voting Agreement Summary:")
    for level, count in summary.items():
        pct = count / len(df) * 100
        print(f"{level}: {count} cases ({pct:.1f}%)")
        
    # Determine if ensemble prediction was correct
    df['correct'] = (df['ensemble'] == df['actual'])
    
    # Grouped accuracy summary by agreement level and actual label
    grouped = df.groupby(['agreement_level', 'actual']).size().unstack(fill_value=0)
    grouped.columns = ['True Negative (0)', 'True Positive (1)']
    grouped['Total'] = grouped.sum(axis=1)
    grouped['TP Rate (%)'] = (grouped['True Positive (1)'] / grouped['Total']) * 100
    
    print("\n📊 True Label Breakdown by Agreement Level:")
    print(grouped.round(1).to_string())

    return df, summary

def load_ensemble_and_columns():
    """
    Loads the final ensemble model and column order for diagnostics.
    """
    ensemble = joblib.load("../../models/voting_ensemble.pkl")
    
    with open("../../models/column_order_logreg.json") as f:
        column_order = json.load(f)

    return ensemble, column_order




def run_ensemble_diagnostics():
    """
    Runs calibration and learning curve diagnostics for a trained ensemble model.
    
    Args:
        model: Trained VotingClassifier or other classifier with `predict_proba`.
        X_train, X_test, y_train, y_test: Dataset splits.
        tag (str): Used in filenames for saving plots.
    """
    ensemble, column_order = load_ensemble_and_columns()
    X_train, X_test, y_train, y_test = preprocess_data()
    X_train = X_train[column_order]
    X_test = X_test[column_order]

    tag ='ensemble_v2'
    print(f"\n📈 Running diagnostics for: {tag}")
    plot_learning_curve(ensemble, X_train, y_train, tag)
    
    df_votes, agreement_summary = analyze_voting_agreement(ensemble, X_test, y_test)
    plot_voting_agreement_pie(agreement_summary, tag="ensemble_v2")
    df_votes.to_csv("../../outputs/voting_agreement_analysis.csv", index=False)
    print("Saved voting agreement analysis to /outputs/voting_agreement_analysis.csv")


if __name__ == '__main__':
    run_ensemble_diagnostics()