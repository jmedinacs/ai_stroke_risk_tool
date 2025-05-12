"""
data_io.py

This module provides functions for loading and saving the cleaned stroke dataset.
It handles path validation and ensures directories exist before saving.

Author: John Medina
Date: 2025-05-5
Project: ai_stroke_risk_tool
"""

import pandas as pd 
import os 

def load_clean_data(filepath="../../data/processed/stroke_cleaned.csv"):
    """
    Load the cleaned stroke dataset from a CSV file.

    Parameters:
    - filepath (str): Full or relative path to the CSV file.

    Returns:
    - DataFrame: Loaded dataset.
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df 

def save_clean_data(df, filepath="../../data/processed/stroke_cleaned.csv"):
    """
    Save the cleaned stroke dataset to a CSV file.

    Parameters:
    - df (DataFrame): The cleaned dataset to save.
    - filepath (str): Full or relative output path for saving the CSV file.

    Returns:
    - None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Cleaned data saved to: {filepath}")
    
def load_raw_data(filepath="../../data/raw/stroke_data.csv"):
    """
    Load the raw stroke dataset from a CSV file.

    Parameters:
    - filepath (str): Path to the raw data file.

    Returns:
    - DataFrame: Loaded raw dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df