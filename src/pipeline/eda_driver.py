"""
eda_driver.py

This script runs the full Exploratory Data Analysis (EDA) pipeline for the cleaned Stroke Risk dataset.
It sequentially performs univariate and bivariate analysis on both categorical and continuous features, 
including statistical tests such as chi-squared and point-biserial correlation.

Results are printed to the console and saved to the `outputs/` folder for downstream reporting or modeling.

Author: John Medina
Date: May 12, 2025 (original ML project eda completed 4/30/2025)
Project: Stroke Risk ML Addendum
"""


import utils.data_io as util
import eda.eda_categorical as categ 
import eda.eda_bivariate_and_chi_square as biv 
import eda.eda_point_biserial as pb 
import eda.eda_distribution as distr 
import eda.eda_target_distribution as targ 



def run_eda():
    """
    Run the full EDA workflow:
    - Load cleaned dataset
    - Visualize numeric and categorical feature distributions
    - Display stroke class distribution
    - Perform chi-squared tests on categorical variables
    - Perform point-biserial correlation on numeric variables

    Saves:
    - visualizations of feature comparisons
    - chi_square_summary.csv
    - point_biserial_summary.csv
    """
    print("\nEDA Process started.\n")
    print("\nLoading cleaned data.")
    df = util.load_clean_data()
    print("\n Data loaded successfully.\n")
    
    print("\nExploring continuous data distribution using bar graphs.")
    distr.explore_numerics(df, None, True)
    
    print("\nExploring categorical data distribution using bar graphs.")
    categ.explore_categoricals(df, None, True)
    
    print("\nExploring target feature (stroke) distribution.")
    targ.explore_target_feature(df, None, True)
    
    print("\nRun chi-squared tests on categorical features vs. stroke")
    chi_results = biv.run_chi_square_test(df)    
    print("\nChi-Square Test Results:")
    print(chi_results[['feature','p_value','significant']])
    chi_results.to_csv("../../outputs/chi_square_summary.csv", index=False)

    print("\nRun point-biserial correlations on continuous features vs. stroke")
    biserial_results = pb.run_point_biserial(df)
    print("\nPoint-Biserial Correlation Results:")
    print(biserial_results)
    biserial_results.to_csv("../../outputs/point_biserial_summary.csv", index=False)

if __name__ == '__main__':
    run_eda()