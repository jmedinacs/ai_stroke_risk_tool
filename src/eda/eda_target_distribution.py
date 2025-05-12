"""
eda_target_distribution.py

This module visualizes and summarizes the distribution of the target variable (`stroke`)
in the stroke risk dataset. It prints counts and percentages, and saves a labeled bar chart.

Author: John Medina
Date: 2025-04-30
Project: Stroke Risk ML Addendum
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os 

def explore_target_feature(df, output_dir= None, show_plot=False):
    """
    Visualizes and summarizes the distribution of the target feature (stroke).
    
    Parameters:
    - df (DataFrame): Cleaned stroke dataset
    - output_dir (str): Folder path to save plots
    - show_plot (bool): Whether to display the chart interactively
    """
    
    if output_dir is None:
        output_dir = "../../outputs/figures"
    os.makedirs(output_dir, exist_ok = True)
    
    # Frequency and percentage summary
    stroke_counts = df['stroke'].value_counts()
    stroke_percent = df['stroke'].value_counts(normalize = True) * 100
    
    print("Stroke Counts:")
    print (stroke_counts)
    print("\nStroke Percentages:")
    print(stroke_percent.round(2))
    
    # Plot bar chart
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='stroke', data=df, color=sns.color_palette('Set2')[0])

    for p in ax.patches:
        height = p.get_height()
        pct = height / len(df) * 100
        ax.annotate(f'{pct:.2f}%', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10)

    plt.title('Distribution of Stroke (Target Variable)')
    plt.xlabel('Stroke')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No Stroke', 'Stroke'])
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'stroke_distribution.png')
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
        
     
    
    