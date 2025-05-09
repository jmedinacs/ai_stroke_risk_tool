"""
train_random_forest.py

Trains and evaluates a Random Forest model for stroke prediction.

This module loads preprocessed data, fits a Random Forest classifier using 
the balanced training set, evaluates the model on the original test set, 
displays and saves the confusion matrix, and stores the trained model to disk.

Returns visual and numerical evaluation metrics for downstream reporting.
"""

from preprocessing.data_preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import StratifiedKFold
import numpy as np 
import pandas as pd 
import shap 
import matplotlib.pyplot as plt 
import joblib 
import os 
from shap.explainers._deep.deep_utils import _check_additivity


def train_model(X_train, y_train, X_test, y_test):
    """
    Trains a Random Forest classifier and evaluates it on the test set.

    Args:
        X_train (pd.DataFrame): SMOTE-balanced training features.
        y_train (pd.Series): Balanced training labels.
        X_test (pd.DataFrame): Original test features (imbalanced).
        y_test (pd.Series): Original test labels.

    Returns:
        Tuple[RandomForestClassifier, np.ndarray]: Trained model and test set predictions.
    """
    rf_model = RandomForestClassifier(
        n_estimators = 200,
        random_state=42,
        max_depth = 5,
        min_samples_split = 10,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:,1] 
    
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred, digits=3))
    
    rf_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {rf_auc:.3f}")
    
    return rf_model, y_pred, y_prob


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
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke","Stroke"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
        
    # Save the figure
    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig("../../outputs/figures/confusion_matrix_rf.png", dpi=300)
    plt.show()
    plt.close()
    print("Confusion matrix saved to /outputs/figures/")
    
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

    # Use new SHAP API for tree-based model
    explainer = shap.Explainer(model, X_test_float)
    shap_values = explainer(X_test_float, check_additivity=False)

    shap_values.feature_names = X_test_float.columns.tolist()
    shap_values.data = X_test_float.values  # ensure consistency

    shap_values_array = shap_values.values[:, :, 1]  # class 1 (stroke)
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    
    shap_df = pd.DataFrame({
        "feature": X_test_float.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)

    
    print("\nSHAP Feature Importance – Random Forest (Top 10):\n")
    print(shap_df.head(10).to_string(index=False))
    
    # Top 10 bar plot
    top_n = 10
    shap_df_top = shap_df.head(top_n)
    
    plt.figure(figsize=(8, 6))
    bars = plt.barh(shap_df_top["feature"][::-1], shap_df_top["mean_abs_shap"][::-1])
    
    # Add SHAP values to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{width:.4f}", va='center')
    plt.xlabel("Mean |SHAP Value|")
    plt.title("SHAP Feature Importance – Random Forest")
    plt.tight_layout()
    
    # Save figure
    plt.savefig("../../outputs/figures/shap_summary_rf_manual.png", dpi=300)
    plt.show()

def plot_precision_recall_curve(y_test, y_prob):
    """
    Plots the precision-recall curve to visualize trade-offs at different thresholds.
    """
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    import os

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall Tradeoff – Random Forest")
    plt.legend()
    plt.grid(True)

    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig("../../outputs/figures/precision_recall_curve_rf.png", dpi=300)
    plt.show()
    plt.close()

    print("Precision-recall curve saved to /outputs/figures/")

def tune_model_bayes(X_train, X_test, y_train, y_test):
    """
    """
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    # Define the search space
    search_space = {
    'n_estimators': Integer(100, 300),
    'max_depth': Integer(3, 10),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2']),
    'bootstrap': Categorical([True, False])
    }
    
    # Initialize model
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    # Initialize search
    opt = BayesSearchCV(
        estimator=rf,
        search_spaces=search_space,
        n_iter=30,  # You can increase this for more thorough search
        scoring=f2_scorer,
        cv=StratifiedKFold(n_splits=5),
        n_jobs=-1,
        random_state=42,
        verbose=2
    )
    
    # Run the optimization
    opt.fit(X_train, y_train)
    
    # Best model summary
    print("\nBest Hyperparameters:")
    print(opt.best_params_)
    print(f"Best F2 Score (CV): {opt.best_score_:.4f}")
    return opt.best_estimator_

def train_bayesian_rf_model():
    """
    Trains and evaluates a BayesSearchCV-tuned Random Forest model.
    """
    X_train, X_test, y_train, y_test = preprocess_data()

    model = tune_model_bayes(X_train, X_test, y_train, y_test)

    # Predict and evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred = (y_prob > threshold).astype(int)

    print("\n📊 Bayes-Tuned RF Evaluation:")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    create_confusion_matrix(y_test, y_pred)
    plot_precision_recall_curve(y_test, y_prob)
    explain_model_with_shap(model, X_test)

    joblib.dump(model, "../../models/random_forest_bayes.pkl")
    print("Bayes-tuned RF model saved to /models/random_forest_bayes.pkl")





def train_random_forest_model():
    """
    Orchestrates preprocessing, training, evaluation, and saving of 
    the Random Forest model for stroke classification.

    Executes the training pipeline using preprocessed data, evaluates 
    the model, saves the confusion matrix plot, and stores the trained model.
    """
    # Step 1 – Manual training
    X_train, X_test, y_train, y_test = preprocess_data()
    model, y_pred, y_prob = train_model(X_train, y_train, X_test, y_test)

    create_confusion_matrix(y_test, y_pred)
    plot_precision_recall_curve(y_test, y_prob)
    explain_model_with_shap(model, X_test)

    os.makedirs("../../models", exist_ok=True)
    joblib.dump(model, "../../models/random_forest_model.pkl")
    print("RFv3 saved to /models/random_forest_model.pkl")

    # Step 2 – Bayesian optimization and retraining
    print("\n🔄 Beginning Bayesian Optimization...")
    train_bayesian_rf_model()
    

if __name__ == '__main__':
    train_random_forest_model()