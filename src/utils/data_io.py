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