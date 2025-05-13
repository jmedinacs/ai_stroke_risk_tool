"""
eda_categorical.py

This module contains functions to visualize the distribution of categorical features
in the stroke risk dataset. Each chart is saved as a labeled PNG and can optionally
be displayed interactively.

Author: John Medina
Date: 2025-04-29
Project: Stroke Risk ML Addendum
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_categoricals(df, output_dir=None, show_plot=False):
    """
    Generates and saves bar charts for key categorical feature distributions.

    Parameters:
        df (pd.DataFrame): Cleaned stroke dataset.
        output_dir (str or None): Directory to save plots. Default: '../../outputs/figures/'.
        show_plot (bool): Whether to display plots interactively.

    Returns:
        None
    """
    
    if output_dir is None:
        output_dir = "../../outputs/figures/"
    os.makedirs(output_dir, exist_ok=True)
    
    cat_columns = [
        'gender',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status',
        'heart_disease'
    ]        
    
    for col in cat_columns:
        plot_bar_chart(df, col, output_dir, show_plot)
        
def plot_bar_chart(df, column, output_dir, show_plot=False):
    """
    Creates and saves a bar chart for a given categorical feature.

    Parameters:
        df (pd.DataFrame): Cleaned stroke dataset.
        column (str): Categorical column to visualize.
        output_dir (str): Directory to save the output PNG.
        show_plot (bool): Whether to display the chart interactively.

    Returns:
        None
    """
    plt.figure(figsize=(8,5))
    ax = sns.countplot(data=df, x=column, 
                       order=df[column].value_counts().index, color ='steelblue', legend=False)
    
    total = len(df)
    
    for p in ax.patches:
        count = p.get_height()
        percentage = 100 * count/total
        ax.annotate(f'{percentage:.1f}%',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    
    plt.title(f"Distribution of {column.title().replace('_',' ')}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"{column}_distribution.png")
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    
