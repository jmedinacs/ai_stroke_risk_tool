"""
train_logistic_regression.py

Trains and evaluates a Logistic Regression model for stroke prediction.

This module loads preprocessed data, fits a Logistic Regression model using the 
balanced training data, evaluates performance on the original imbalanced test set, 
displays and saves the confusion matrix, and persists the trained model to disk.

Returns visual and numerical metrics for model performance evaluation.
"""

from preprocessing.data_preprocessing import preprocess_data
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score
import matplotlib.pyplot as plt 
import pandas as pd 
import joblib 
import os 
import json
import shap 

def train_model(X_train, y_train, X_test, y_test):
    """
    Trains a Logistic Regression model and evaluates it on the test set.

    Args:
        X_train (pd.DataFrame): Training feature matrix (SMOTE-applied).
        y_train (pd.Series): Balanced target vector for training.
        X_test (pd.DataFrame): Original feature matrix for testing.
        y_test (pd.Series): Original target vector for testing.

    Returns:
        Tuple[LogisticRegression, np.ndarray]: Trained model and predicted labels on test set.
    """
    # Initialize and train the logistic regression model
    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train, y_train)
    
    y_pred = log_reg_model.predict(X_test)
    y_prob = log_reg_model.predict_proba(X_test)[:,1]
    
    # Apply custom threshold
    threshold = 0.3
    y_pred = (y_prob > threshold).astype(int)

    # Evaluation metrics
    print("\n Classification Report (Logistic Regression): ")
    print(classification_report(y_test, y_pred, digits=3))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc: 3f}")
    
    return log_reg_model, y_pred, y_prob

def create_confusion_matrix(y_test, y_pred):
    """
    Generates, displays, and saves the confusion matrix.

    Args:
        y_test (pd.Series): True stroke labels from test set.
        y_pred (np.ndarray): Predicted stroke labels from the model.

    Saves:
        Confusion matrix plot to /outputs/figures/
    """

    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig("../../outputs/figures/confusion_matrix_logreg.png", dpi=300)
    plt.show()
    print("Confusion matrix saved to /outputs/figures/")    

def plot_precision_recall_curve(y_test, y_prob):
    """
    Plots the precision-recall curve to visualize trade-offs at different thresholds.

    Args:
        y_test (pd.Series): True stroke labels.
        y_prob (np.ndarray): Predicted probabilities for class 1 (stroke).
    """    
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall Tradeoff (Logistic Regression)")
    plt.legend()
    plt.grid(True)    
    
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig("../../outputs/figures/precision_recall_curve.png", dpi=300)
    plt.show()
    print("Precision-recall curve saved to /outputs/figures/")
    
def explain_model_with_shap(model, X_test):
    """
    Generates SHAP values and plots for Logistic Regression.

    Args:
        model (LogisticRegression): Trained logistic regression model.
        X_test (pd.DataFrame): Test features used for SHAP explanations.

    Saves:
        SHAP summary plot and waterfall plot for one example.
    """
    # Force all features to float64
    X_test_float = X_test.copy()
    for col in X_test_float.columns:
        try:
            X_test_float[col] = pd.to_numeric(X_test_float[col], errors="raise").astype("float64")
        except Exception as e:
            print(f"Failed to convert column '{col}': {e}")

    # Initialize SHAP explainer and compute values
    explainer = shap.Explainer(model, X_test_float)
    shap_values = explainer(X_test_float)

    # Create directory
    os.makedirs("../../outputs/figures", exist_ok=True)

    # Summary plot
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Summary – Logistic Regression")
    plt.tight_layout()
    plt.savefig("../../outputs/figures/shap_summary_logreg.png", dpi=300)
    plt.show()
    plt.close()

    # Optional: Waterfall plot for one test row
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Waterfall – Logistic Regression Sample")
    plt.tight_layout()
    plt.savefig("../../outputs/figures/shap_waterfall_logreg.png", dpi=300)
    plt.show()
    plt.close()

    print("SHAP summary and waterfall plots saved.")
 
def tune_logistic_regression_bayes(X_train, y_train):
    """
    Runs BayesSearchCV to find the best hyperparameters for Logistic Regression.
    Returns the tuned model.
    """
    f2_scorer = make_scorer(fbeta_score, beta=2)

    search_space = {
        'C': Real(1e-3, 100.0, prior='log-uniform'),
        'penalty': Categorical(['l2']),
        'solver': Categorical(['lbfgs', 'saga']),
        'max_iter': [1000]
    }

    base_model = LogisticRegression(class_weight='balanced', random_state=42)

    opt = BayesSearchCV(
        estimator=base_model,
        search_spaces=search_space,
        scoring=f2_scorer,
        cv=StratifiedKFold(n_splits=5),
        n_iter=30,
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    opt.fit(X_train, y_train)

    print("\n✅ Best Hyperparameters (LogReg):")
    print(opt.best_params_)
    print(f"Best F2 Score (CV): {opt.best_score_:.4f}")

    return opt.best_estimator_


def train_logistic_regression_model():
    """
    Orchestrates preprocessing, manual training, and Bayesian tuning 
    of the Logistic Regression model for stroke classification.

    Saves evaluation metrics, visualizations, and model artifacts for both versions.
    """
    X_train, X_test, y_train, y_test = preprocess_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"y_test distribution:\n{y_test.value_counts(normalize=True)}")    

    # Step 1 – Manual Logistic Regression (LRv2)
    model, y_pred, y_prob = train_model(X_train, y_train, X_test, y_test)
    create_confusion_matrix(y_test, y_pred)
    plot_precision_recall_curve(y_test, y_prob)
    explain_model_with_shap(model, X_test)
    joblib.dump(model, "../../models/logistic_regression_v2_model.pkl")
    print("Manual Logistic Regression model saved to /models/logistic_regression_v2_model.pkl")

    # Step 2 – Bayes-Optimized Logistic Regression (LRv5)
    print("\n🔄 Beginning Bayesian Optimization...")
    model = tune_logistic_regression_bayes(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred = (y_prob > threshold).astype(int)

    print("\n📊 Bayes-Tuned Logistic Regression Evaluation:")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    create_confusion_matrix(y_test, y_pred)
    plot_precision_recall_curve(y_test, y_prob)
    explain_model_with_shap(model, X_test)

    joblib.dump(model, "../../models/logistic_regression_bayes.pkl")
    print("Bayes-Tuned Logistic Regression model saved to /models/logistic_regression_bayes.pkl")

    with open("../../models/column_order_logreg.json", "w") as f:
        json.dump(list(X_train.columns), f)
    
    
if __name__ == '__main__':
    train_logistic_regression_model()
    