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
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt 
import joblib 
import os 

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
        n_estimators = 200, # increased to 200 from 100
        max_depth = 5,
        min_samples_split = 10,
        random_state = 42,
        class_weight = 'balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:,1]
    
    # Uncomment to explore threshold tuning
    #threshold = 0.25
    #y_pred = (y_prob > threshold).astype(int)
    
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred, digits=3))
    
    rf_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {rf_auc:.3f}")
    
    return rf_model, y_pred, y_prob

def plot_precision_recall_curve(y_test, y_prob, model_name="Random Forest (Tuned)"):
    """
    Plots precision-recall vs. threshold curve.

    Args:
        y_test (pd.Series): True stroke labels
        y_prob (np.ndarray): Predicted probabilities for class 1
        model_name (str): Title for the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision-Recall Curve – {model_name}")
    plt.legend()
    plt.grid(True)

    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig(f"../../outputs/figures/pr_curve_rf_tuned.png", dpi=300)
    plt.show()
    print("Precision-recall curve saved to /outputs/figures/")

def save_confusion_matrix(y_test, y_pred, model_name="Random Forest (v3)"):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()

    os.makedirs("../../outputs/figures", exist_ok=True)
    plt.savefig(f"../../outputs/figures/confusion_matrix_rf_v3.png", dpi=300)
    plt.show()
    print(f"Confusion matrix saved to /outputs/figures/confusion_matrix_rf_v3.png")


def train_random_forest_model():
    """
    Orchestrates preprocessing, training, evaluation, and saving of 
    the Random Forest model for stroke classification.

    Executes the training pipeline using preprocessed data, evaluates 
    the model, saves the confusion matrix plot, and stores the trained model.
    """    
    # Retrieve the training and test data from preprocessing
    X_train, X_test, y_train, y_test = preprocess_data()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"y_test distribution:\n{y_test.value_counts(normalize=True)}") 
    
    model, y_pred, y_prob = train_model(X_train, y_train, X_test, y_test)

    plot_precision_recall_curve(y_test, y_prob, model_name="Random Forest (Tuned)")
    
    save_confusion_matrix(y_test, y_pred, model_name="Random Forest (v3)")
    
    # Save best RF model (v3)
    os.makedirs("../../models", exist_ok=True)
    joblib.dump(model, "../../models/random_forest_v3_best.pkl")
    print("Random Forest (v3) model saved to /models")


if __name__ == '__main__':
    train_random_forest_model()