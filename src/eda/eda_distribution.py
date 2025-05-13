"""
eda_distribution.py

This module provides functions to visualize the distribution of numerical variables
in the stroke risk dataset, including age, BMI, and average glucose level.

Author: John Medina
Date: 2025-04-29
Project: AI Stroke Risk Tool
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_numerics(df, output_dir= None, show_plot=False):
    """
    Generates and saves histograms with KDE overlays for key numeric features:
    - age
    - bmi
    - avg_glucose_level

    Parameters:
        df (pd.DataFrame): Cleaned stroke dataset.
        output_dir (str or None): Directory to save plots. Default: '../../outputs/figures/'.
        show_plot (bool): Whether to display plots interactively.

    Returns:
        None
    """
        
    plot_histogram(df, 'age', output_dir, show_plot)
    plot_histogram(df, 'bmi', output_dir, show_plot)
    plot_histogram(df, 'avg_glucose_level', output_dir, show_plot)
    
def plot_histogram(df, column, output_dir, show_plot=False):
    """
    Creates and saves a histogram with KDE for a specified numeric column.

    Parameters:
        df (pd.DataFrame): Cleaned stroke dataset.
        column (str): Column to plot.
        output_dir (str or None): Directory to save the output plot.
        show_plot (bool): Whether to display the plot interactively.

    Returns:
        None
    """
    if output_dir is None:
        output_dir="../../outputs/figures"
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x=column, kde=True, bins=30, color="steelblue", edgecolor='white')
    plt.title(f"Distribution of {column.title()}")
    plt.xlabel("Frequency")
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"{column}_distribution.png")
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    