"""
data_preprocessing.py

Preprocessing pipeline for stroke risk classification.

This module loads the cleaned dataset and performs the following steps:
- Drops statistically insignificant features (based on EDA/chi-square tests)
- Splits features and target
- One-hot encodes categorical variables
- Performs stratified train/test split
- Applies SMOTE oversampling to the training set only

Returns preprocessed and ready-to-train datasets for machine learning models.
"""

from utils.data_io import load_clean_data
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
import joblib

def drop_insignificant_features(df, no_age=False):
    """
    Drops statistically insignificant features (based on EDA and chi-square tests)


    Args:
        df (pd.DataFrame): Cleaned input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with dropped columns (id, gender, Residence_type).
    """
    if no_age:
        df = df.drop(columns=["age","id","gender","Residence_type"])
    else:
        df = df.drop(columns=["id","gender","Residence_type"])
    return df

def split_features_target(df):
    """
    Splits the DataFrame into features (X) and target (y).

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with all features.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    return X, y
     
def encode_categoricals(X): 
    """
    Performs one-hot encoding on categorical features.

    Args:
        X (pd.DataFrame): Feature matrix before encoding.

    Returns:
        pd.DataFrame: One-hot encoded feature matrix.
    """
    X["hypertension"] = X["hypertension"].astype("category")
    X["heart_disease"] = X["heart_disease"].astype("category")
    X = pd.get_dummies(X, drop_first=True)
    return X  

def train_test_stratified_split(X, y):
    """
    Performs stratified train/test split to preserve class distribution.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """
    Applies SMOTE oversampling to balance the training dataset.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled X_train and y_train with balanced classes.
    """
    # Initialize SMOTE
    #smote = SMOTE(random_state=42)
    
    smote = SMOTETomek(random_state=42)
    # Apply SMOTE to the TRAINING SET ONLY
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled      



def preprocess_data(no_age=False):
    """
    Full preprocessing pipeline for stroke risk classification.

    Steps:
    - Loads cleaned dataset using `load_clean_data()`
    - Drops features found statistically insignificant (id, gender, Residence_type)
    - Splits data into features (X) and target (y)
    - One-hot encodes categorical features
    - Performs stratified train/test split (80/20)
    - Applies SMOTE oversampling to training data only to address class imbalance

    Returns:
        X_train_resampled (pd.DataFrame): Feature matrix for training (balanced)
        X_test (pd.DataFrame): Feature matrix for testing (original distribution)
        y_train_resampled (pd.Series): Balanced target vector for training
        y_test (pd.Series): Target vector for testing (original distribution)
    """
    # Load cleaned version of the dataset
    df = load_clean_data()
    
    # Drop identified insignificant figures
    df = drop_insignificant_features(df,no_age)
    
    # Separate the features from the target
    X, y = split_features_target(df)
    
    # One-hot encode categorical data
    X = encode_categoricals(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_stratified_split(X, y)

    # SMOTE on training set only
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test

def preprocess_data_knn(no_age=False):
    """
    Preprocessing pipeline for KNN model:
    - Loads cleaned dataset
    - Drops insignificant features
    - One-hot encodes categoricals
    - Splits into train/test sets (stratified)
    - Normalizes features using StandardScaler (fit on train)
    - Applies SMOTE on normalized training set only

    Returns:
        X_train_resampled (np.array): Normalized + resampled training features
        X_test_scaled (np.array): Normalized test features
        y_train_resampled (pd.Series): Resampled training target
        y_test (pd.Series): Test target
        scaler (StandardScaler): Fitted scaler for future use
    """
    df = load_clean_data()
    df = drop_insignificant_features(df,no_age)
    X, y = split_features_target(df)
    X = encode_categoricals(X)
    X_train, X_test, y_train, y_test = train_test_stratified_split(X, y)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler    




if __name__ == '__main__':
    # Run both pipelines for testing purposes
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Base pipeline ready")

    X_train_knn, X_test_knn, y_train_knn, y_test_knn, scaler = preprocess_data_knn()
    print("KNN pipeline ready")