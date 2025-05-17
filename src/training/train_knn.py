"""
train_knn.py

Trains and evaluates a K-Nearest Neighbors (KNN) model for stroke prediction.

This script performs:
- Preprocessing with normalization
- Bayesian hyperparameter optimization using F2-score
- Evaluation using classification metrics and ROC AUC
- Precision-recall curve and confusion matrix visualization
- Model performance logging and figure export
- Optional persistence of trained model and scaler (for deployment)

Author: John Medina
Date: 2025-05-09
Project: AI Stroke Risk Tool
"""

from preprocessing.data_preprocessing import preprocess_data_knn
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import os


def evaluate_knn_model(model, X_test, y_test, threshold=0.5, tag="KNN"):
    """
    Evaluates the KNN model using classification metrics and a custom threshold.

    Parameters:
        model: Trained KNN model.
        X_test (np.ndarray): Scaled test features.
        y_test (np.ndarray or pd.Series): True labels.
        threshold (float): Probability threshold for positive prediction.

    Saves:
        Confusion matrix plot to /outputs/figures/.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    print(f"\n📊 KNN Evaluation @ Threshold = {threshold}")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"F2-Score: {f2:.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – KNN (Threshold = {threshold})")
    plt.tight_layout()

    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig(f"../../outputs/figures/confusion_matrix_knn_thresh_{tag}.png", dpi=300)
    plt.show()
    plt.close()


def tune_model_bayes(X_train, y_train):
    """
    Optimizes KNN hyperparameters using Bayesian optimization with F2 as the scoring metric.

    Parameters:
        X_train (np.ndarray): Normalized training features.
        y_train (np.ndarray or pd.Series): Training target.

    Returns:
        Trained BayesSearchCV model.
    """
    f2_scorer = make_scorer(fbeta_score, beta=2)

    search_space = {
        'n_neighbors': Integer(3, 25),
        'weights': Categorical(['uniform', 'distance']),
        'metric': Categorical(['euclidean', 'manhattan']),
    }

    opt = BayesSearchCV(
        estimator=KNeighborsClassifier(),
        search_spaces=search_space,
        scoring=f2_scorer,
        cv=StratifiedKFold(n_splits=5),
        n_iter=30,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    tuned_model = opt.fit(X_train, y_train)

    print("\n✅ Best Hyperparameters (KNN):")
    print(opt.best_params_)
    print(f"Best F2 Score (CV): {opt.best_score_:.4f}")

    return tuned_model


def plot_precision_recall_curve(y_test, y_prob, tag="KNN"):
    """
    Plots and saves a precision-recall curve for a given model.

    Parameters:
        y_test (array): True binary labels.
        y_prob (array): Predicted probabilities.
        tag (str): Model name used in titles and filenames.

    Saves:
        Precision-recall curve to /outputs/figures/.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
    plt.plot(thresholds, recall[:-1], label="Recall", color="orange")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision-Recall Tradeoff – {tag}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig(f"../../outputs/figures/precision_recall_curve_tag{tag}.png", dpi=300)
    plt.show()
    plt.close()


def train_knn_model(no_age=False):
    """
    Driver function for training and evaluating a KNN model:
    - Loads and preprocesses data (optionally excluding 'age')
    - Tunes KNN with BayesSearchCV
    - Evaluates model with classification metrics and visualizations
    - Plots precision-recall curve

    Args:
        no_age (bool): Whether to drop 'age' from features.
    """
    X_train, X_test, y_train, y_test, scaler = preprocess_data_knn(no_age)

    # Train
    model = tune_model_bayes(X_train, y_train)

    # Set tag for filenames
    tag = "knn_no_age" if no_age else "knn"

    # Evaluate
    evaluate_knn_model(model, X_test, y_test, threshold=0.3, tag=tag)

    # PR Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    plot_precision_recall_curve(y_test, y_prob, tag)

    # Optional: Save model + scaler
    import joblib
    os.makedirs("../../models", exist_ok=True)
    joblib.dump(model, f"../../models/{tag}_bayes.pkl")
    joblib.dump(scaler, f"../../models/scaler_{tag}.pkl")
    print(f"Model and scaler saved for {tag}.")

    
    


if __name__ == '__main__':
    train_knn_model()
