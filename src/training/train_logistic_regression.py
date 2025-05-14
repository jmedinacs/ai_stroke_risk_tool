"""
train_logistic_regression.py

Trains and evaluates a Logistic Regression model for stroke prediction.

This script performs:
- Preprocessing and class balancing (via SMOTE)
- Manual model training
- Bayesian hyperparameter optimization (F2-score)
- SHAP explainability visualizations
- Comparison of L1, L2, and ElasticNet penalties
- Model and evaluation export (confusion matrix, PR curve, CSV summary)

Author: John Medina
Date: 2025-05-11
Project: AI Stroke Risk Tool
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

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)



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
    
    f2 = fbeta_score(y_test, y_pred, beta=2)
    print(f"f2: {f2:.4f}")
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc: 3f}")
    
    return log_reg_model, y_pred, y_prob

def create_confusion_matrix(y_test, y_pred, tag):
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
    plt.title(f"Confusion Matrix – {tag}")
    plt.tight_layout()
    
    output_path = f"../../outputs/figures/confusion_matrix_{tag}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Confusion matrix saved to {output_path}") 

def plot_precision_recall_curve(y_test, y_prob, tag):
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
    plt.title(f"Precision-Recall Tradeoff – {tag}")
    plt.legend()
    plt.grid(True)    
    
    output_path = f"../../outputs/figures/precision_recall_curve_{tag}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Precision-recall curve saved to {output_path}")
    
def explain_model_with_shap(model, X_test, tag):
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
    plt.title(f"SHAP Summary – {tag}")
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f"../../outputs/figures/shap_summary_{tag}.png", dpi=300)
    plt.show()
    plt.close()

    print(f"SHAP summary saved to /outputs/figures/shap_summary_{tag}.png")

    # Optional: Waterfall plot for one test row
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"SHAP Waterfall – {tag}n Sample")
    plt.tight_layout()
    plt.savefig(f"../../outputs/figures/shap_waterfall_{tag}.png", dpi=300)
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

    print("\nBest Hyperparameters (LogReg):")
    print(opt.best_params_)
    print(f"Best F2 Score (CV): {opt.best_score_:.4f}")

    return opt.best_estimator_

def tune_logistic_regression_all_penalties(X_train, y_train, X_test, y_test, threshold=0.3):
    """
    Trains and evaluates Logistic Regression models with L1, L2, and ElasticNet penalties
    using Bayesian optimization. Saves outputs and logs performance metrics.

    Args:
        X_train, y_train: Balanced training data (e.g., SMOTE applied)
        X_test, y_test: Original imbalanced test data
        threshold (float): Probability cutoff for classification
    """
    penalties = ['l1','l2','elasticnet']
    results = []
    
    for penalty in penalties:
        print(f"\n Tuning Logirstic Regression with penalty = '{penalty}'")
        
        search_space = {
            'C': Real(1e-3, 100.0, prior='log-uniform'),
            'solver': Categorical(['saga']), # NOTE: saga supports all 3 penalties
            'penalty': Categorical([penalty]),
            'max_iter':[1000]
        }
    
        # Only ElasticNet uses l1_ratio
        if penalty == 'elasticnet':
            search_space['l1_ratio'] = Real(0.0, 1.0)

        f2_scorer = make_scorer(fbeta_score, beta=2)    
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            scoring=f2_scorer,
            cv=StratifiedKFold(n_splits=5),
            n_iter=30,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        opt.fit(X_train, y_train)
        best_model = opt.best_estimator_

        print(f"Best Params for {penalty}:\n{opt.best_params_}")
        print(f"Best f2 (cv): {opt.best_score_:.4f}")
        
        # Inference
        y_prob = best_model.predict_proba(X_test)[:,1]
        y_pred = (y_prob > threshold).astype(int)
        
        # Evaluation
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        auc = roc_auc_score(y_test, y_prob)
        
        tag = f"logreg_{penalty}"
        create_confusion_matrix(y_test, y_pred, tag)
        plot_precision_recall_curve(y_test, y_prob, tag)
        explain_model_with_shap(best_model, X_test, tag)
        joblib.dump(best_model, f"../../models/{tag}.pkl")
        
        print(f"Model and visuals saved for {penalty}.")
        
        results.append({
            'Penalty': penalty,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F2': round(f2, 4),
            'ROC AUC': round(auc, 4),
            'Best Params': opt.best_params_
        })

    # Show summary
    results_df = pd.DataFrame(results)
    print("\nLogistic Regression Comparison:")
    # Create a copy for clean display without the long params
    summary_df = results_df.drop(columns=["Best Params"])
    
    print("\nLogistic Regression Comparison (Summary):")
    print(summary_df.to_markdown(index=False))
    
    print("\nBest Hyperparameters:")
    for row in results_df.itertuples(index=False):
        print(f"• {row.Penalty.upper()}: {row._5}")


    # Optional: save as CSV for reporting
    results_df.to_csv("../../outputs/logreg_regularization_comparison.csv", index=False)
    print("Saved regularization comparison to /outputs/logreg_regularization_comparison.csv")

def train_logistic_regression_model(train_manual=False):
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

    if train_manual:
        # Step 1 – Manual Logistic Regression (LRv2)
        tag_manual = "logreg_v2"
        model, y_pred, y_prob = train_model(X_train, y_train, X_test, y_test)
        create_confusion_matrix(y_test, y_pred, tag_manual)
        plot_precision_recall_curve(y_test, y_prob, tag_manual)
        explain_model_with_shap(model, X_test, tag_manual)
        joblib.dump(model, f"../../models/{tag_manual}_model.pkl")
        print(f"Manual Logistic Regression model saved to /models/{tag_manual}_model.pkl")

    # Step 2 – Bayes-Optimized Logistic Regression (LRv5)
    print("\nBeginning Bayesian Optimization...")
    tag_bayes = "logreg_bayes"
    model = tune_logistic_regression_bayes(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred = (y_prob > threshold).astype(int)

    print("\nBayes-Tuned Logistic Regression Evaluation:")
    print(classification_report(y_test, y_pred, digits=3))
    f2 = fbeta_score(y_test, y_pred, beta=2)
    print(f"f2: {f2:.4f}")
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    create_confusion_matrix(y_test, y_pred, tag_bayes)
    plot_precision_recall_curve(y_test, y_prob, tag_bayes)
    explain_model_with_shap(model, X_test, tag_bayes)
    joblib.dump(model, f"../../models/{tag_bayes}.pkl")
    print(f"Bayes-Tuned Logistic Regression model saved to /models/{tag_bayes}.pkl")

    with open("../../models/column_order_logreg.json", "w") as f:
        json.dump(list(X_train.columns), f)
        
    tune_logistic_regression_all_penalties(X_train, y_train, X_test, y_test)


    
    
if __name__ == '__main__':
    train_logistic_regression_model()
    