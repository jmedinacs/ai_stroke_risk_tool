

from preprocessing.clean_the_data import clean_data
from preprocessing.data_preprocessing import preprocess_data 
from pipeline.eda_driver import run_eda
from training.train_logistic_regression import train_logistic_regression_model
from training.train_rf import train_random_forest_model
from training.train_xgboost import train_xgboost_model 
from training.train_knn import train_knn_model
from pipeline.compare_models import summarize_results
from pipeline.create_ensemble_voting_model import run_voting_pipeline 
from pipeline.check_vif import preprocess_data_for_LR
from evaluation.ensemble_diagnostic import run_ensemble_diagnostics

def main():
    """
    Orchestrates the full end-to-end stroke risk prediction pipeline.

    Steps:
    1. Clean raw data and handle missing values
    2. Preprocess features for machine learning (encoding, balancing)
    3. Run exploratory data analysis (EDA)
    4. Check multicollinearity using VIF
    5. Train and evaluate:
       - Logistic Regression (Bayes optimized)
       - Random Forest (Bayes optimized)
       - XGBoost (Bayes optimized)
       - K-Nearest Neighbors (Bayes optimized)
    6. Summarize model performance across all trained models
    7. Build and tune ensemble soft-voting classifier
    8. Run diagnostics and interpret ensemble predictions

    Returns:
        None
    """
    # 1. Clean raw data
    df = clean_data()

    # 2. Preprocess dataset (encoding, scaling, SMOTE, etc.)
    df = preprocess_data()

    # 3. Perform exploratory data analysis
    run_eda()

    # 4. Examine VIF for multicollinearity (specific to logistic regression)
    preprocess_data_for_LR()    

    # 5. Train models (Bayes-optimized)
    train_logistic_regression_model()
    train_random_forest_model()
    train_xgboost_model()
    train_knn_model()

    # 6. Summarize model comparison and evaluation
    summarize_results()

    # 7. Build and optimize soft-voting ensemble
    run_voting_pipeline()

    # 8. Evaluate ensemble agreement and SHAP diagnostics
    run_ensemble_diagnostics()
   

if __name__ == '__main__':
    main()