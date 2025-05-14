

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
    """ """
    # Initiate raw data cleaning steps
    #df = clean_data()
    # Initiate data preprocessing steps 
    #df = preprocess_data()
    # Conduct EDA process
    #run_eda()
    # Examine VIF for multicollinearity
    preprocess_data_for_LR()    
    # Train Bayes Optimized Logistic Regression Model
    #train_logistic_regression_model()
    # Train Bayes Optimized Random Forest
    #train_random_forest_model()
    # Train Bayes Optimized XGBoost Model
    #train_xgboost_model() 
    # Train Bayes Optimized K-Nearest Neighbors Model
    #train_knn_model()
    # Compare the current acceptable models
    summarize_results()
    # Initiate the ensemble soft voting model and tuning
    run_voting_pipeline() 
    # Test the ensemble and explain
    run_ensemble_diagnostics()
    
    
    

if __name__ == '__main__':
    main()