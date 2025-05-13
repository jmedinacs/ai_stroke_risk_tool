"""
eda_bivariate_and_chi_square.py

This module performs chi-squared tests of independence between categorical features 
and the binary target variable `stroke`. It identifies which features are statistically 
associated with stroke occurrence and returns a ranked summary table.

Author: John Medina
Date: 2025-04-30
Project: AI Stroke Risk Tool
"""

import pandas as pd
from scipy.stats import chi2_contingency


def run_chi_square_test(df):
    """
    Runs chi-squared tests between selected categorical features and the binary target (`stroke`).

    Parameters:
        df (pd.DataFrame): Cleaned stroke dataset.

    Returns:
        pd.DataFrame: Summary DataFrame with columns:
            - feature (str)
            - chi2_statistic (float)
            - p_value (float)
            - degrees_of_freedom (int)
            - significant (bool): True if p < 0.05
    """
    features_to_test = [
        'gender',
        'hypertension',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status',
        'heart_disease'
    ]

    chi_square_results = []

    for feature in features_to_test:
        contingency = pd.crosstab(df[feature], df['stroke'])

        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
            chi_square_results.append({
                'feature': feature,
                'chi2_statistic': chi2,
                'p_value': p,
                'degrees_of_freedom': dof
            })
        except Exception as e:
            print(f"⚠️ Error with feature '{feature}': {e}")

    result_df = pd.DataFrame(chi_square_results).sort_values('p_value')
    result_df['significant'] = result_df['p_value'] < 0.05
    return result_df
