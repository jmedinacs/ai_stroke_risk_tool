"""
check_vif.py

This module computes the Variance Inflation Factor (VIF) for all encoded features 
in the training set, to detect potential multicollinearity among predictors.

Used during preprocessing and model validation for Logistic Regression.

Author: John Medina
Date: 2025-05-11
Project: ai_stroke_risk_tool
"""

import pandas as pd
import joblib

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from utils.data_io import load_clean_data
import preprocessing.data_preprocessing as process


def compute_VIF(X_encoded):
    """
    Computes VIF for each feature to assess multicollinearity.

    Parameters:
    - X_encoded (DataFrame): One-hot encoded feature matrix

    Returns:
    - DataFrame: Feature names and corresponding VIF scores
    """
    X_encoded = X_encoded.astype(float)
    X_with_const = add_constant(X_encoded)
    
    vif_data = pd.DataFrame({
        "feature": X_encoded.columns,
        "VIF": [
            variance_inflation_factor(X_with_const.values, i + 1)
            for i in range(X_encoded.shape[1])
        ]
    })
    
    return vif_data


def preprocess_data_for_LR():
    """
    Loads the cleaned dataset, applies encoding, splits the data,
    and computes VIF on the training feature matrix.
    """
    data = load_clean_data()
    data = process.drop_insignificant_features(data)
    X, y = process.split_features_target(data)
    X = process.encode_categoricals(X)
    X_train, _, _, _ = process.train_test_stratified_split(X, y)
    
    print("🔍 Computing VIFs...")
    vif_df = compute_VIF(X_train)
    print(vif_df.sort_values("VIF", ascending=False))


if __name__ == '__main__':
    preprocess_data_for_LR()
