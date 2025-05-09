"""
train_knn.py

Trains and evaluates a K-Nearest Neighbors (KNN) model for stroke prediction.

This script performs preprocessing, normalization, model training,
evaluation using classification metrics and ROC AUC, and visualizes
and saves the confusion matrix. It also persists the trained KNN model
and the fitted scaler for reproducibility and deployment.
"""

from preprocessing.data_preprocessing import preprocess_data_knn
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import os
import numpy as np


def evaluate_knn_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates a trained KNN model on the test set using a specified threshold.
    
    Args:
        model: Trained KNN model.
        X_test (pd.DataFrame or np.ndarray): Scaled test features.
        y_test (pd.Series or np.ndarray): True test labels.
        threshold (float): Classification threshold for predicting stroke.
    
    Saves:
        Confusion matrix plot to /outputs/figures/
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n📊 KNN Evaluation @ Threshold = {threshold}")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – KNN (Threshold = {threshold})")
    plt.tight_layout()
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig(f"../../outputs/figures/confusion_matrix_knn_thresh_{threshold}.png", dpi=300)
    plt.show()
    plt.close()
    
def tune_model_bayes(X_train, y_train):
    """
    Tunes KNN using Bayesian Optimization (BayesSearchCV) with F2 as the scoring metric.
    """
    f2_scorer = make_scorer(fbeta_score, beta=2)

    search_space = {
        'n_neighbors': Integer(3, 25),
        'weights': Categorical(['uniform', 'distance']),
        'metric': Categorical(['euclidean', 'manhattan']),
    }

    knn = KNeighborsClassifier()

    opt = BayesSearchCV(
        estimator=knn,
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

def plot_precision_recall_curve(y_test, y_prob, model_name="KNN"):
    """
    Plots and saves the precision-recall curve for a given model.
    
    Args:
        y_test (array): True labels
        y_prob (array): Predicted probabilities for class 1
        model_name (str): Model name to use in title and filename
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
    plt.plot(thresholds, recall[:-1], label="Recall", color="orange")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision-Recall Tradeoff – {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig(f"../../outputs/figures/precision_recall_curve_{model_name.lower()}.png", dpi=300)
    plt.show()
    plt.close()

def train_knn_model():
    """ """
    X_train, X_test, y_train, y_test, scaler = preprocess_data_knn()
    model = tune_model_bayes(X_train, y_train)
    evaluate_knn_model(model, X_test, y_test, threshold=0.3)
    # Get predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    plot_precision_recall_curve(y_test, y_prob, model_name="KNN")


if __name__ == '__main__':
    train_knn_model()