"""
eda_continuous_vs_stroke.py

This module runs point-biserial correlation between continuous features
and the binary target variable `stroke`. It identifies which continuous features
are statistically associated with stroke occurrence.

Author: John Medina
Date: 2025-04-30
Project: Stroke Risk ML Addendum
"""

import pandas as pd 
from scipy.stats import pointbiserialr

def run_point_biserial(df):
    """
    Compute point-biserial correlations between continuous features and `stroke`.

    Parameters:
    - df (DataFrame): Cleaned stroke dataset.

    Returns:
    - DataFrame: Table with:
        - feature name
        - correlation coefficient (r)
        - p-value
        - significance flag (True if p < 0.05)
    """
    continuous_features = [
        'age',
        'bmi',
        'avg_glucose_level'
    ]
    
    results = []
    
    for feature in continuous_features:
        try:
            r, p = pointbiserialr(df['stroke'], df[feature])
            results.append({
                'feature' : feature,
                'correlation' : r,
                'p_value' : p,
                'significant' : p < 0.05
            })
        except Exception as e:
            print(f"Error with '{feature}': {e}")
            
    biserial_df = pd.DataFrame(results).sort_values('p_value')
    return biserial_df