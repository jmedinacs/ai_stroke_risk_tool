"""
clean_the_data.py

This script handles the full data cleaning process for the Stroke Risk dataset. It loads the raw data,
performs validation and preprocessing steps, and saves the cleaned version for use in EDA and modeling.

Author: John Medina
Date: 2025-04-28
Project: Stroke Risk ML Addendum

Cleaning steps:
- Impute missing 'bmi' values using the median (due to right skew).
- Remove rare 'Other' category in gender (1 instance).
- Standardize all text fields (lowercase, trimmed).
- Drop duplicates (excluding the unique 'id' column).
- Validate dataset at each major step with summary output.
"""
import utils.data_io as util
import pandas as pd


def inspect_data(df):
    """
    Display dataset info, summary statistics, and missing values.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - None
    """
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())


def impute_missing_bmi(df):
    """
    Fill missing 'bmi' values using the median of the column.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - DataFrame: Dataset with missing BMI values imputed.
    """
    median_bmi = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(median_bmi)
    print(f"Missing BMI values filled with median: {median_bmi:.2f}")
    return df


def inspect_categorical_distribution(df):
    """
    Print value counts and percentage breakdown for all categorical columns.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - None
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    
     # Include int columns with few unique values (likely binary/categorical)
    low_card_ints = [
        col for col in df.select_dtypes(include=['int64', 'int32']).columns
        if df[col].nunique() <= 10 and col != 'id'  # Exclude 'id'
    ]
    
    for col in list(cat_cols) + low_card_ints:
        print(f"Value counts for '{col}':")
        print(df[col].value_counts())
        print("\nPercentage breakdown:")
        print((df[col].value_counts(normalize=True) * 100).round(2))
        print("-" * 50)


def remove_rare_gender(df):
    """
    Remove the single row where 'gender' is 'Other', due to extreme rarity.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - DataFrame: Dataset with rare category removed.
    """
    before_rows = df.shape[0]
    df = df[df['gender'] != 'Other'].copy()
    after_rows = df.shape[0]
    print(f"Removed {before_rows - after_rows} rows with rare 'Other' gender value.")
    return df


def standardize_text_fields(df):
    """
    Convert all object columns to lowercase and strip whitespace.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - DataFrame: Dataset with cleaned text columns.
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower()
    print(f"Standardized {len(cat_cols)} text columns: lowercased and trimmed.")
    return df


def remove_duplicates(df):
    """
    Drop duplicate rows from the dataset based on all columns *except 'id'*.

    Parameters:
    - df (DataFrame): Input dataset.

    Returns:
    - DataFrame: Dataset with duplicates removed.
    """
    before_rows = df.shape[0]
    
    # Drop duplicates ignoring 'id' column
    df = df[df.drop(columns=['id']).duplicated(keep='first') == False].copy()

    after_rows = df.shape[0]
    print(f"Removed {before_rows - after_rows} duplicate rows (excluding 'id').")
    return df

def encode_binary(df):
    """ """
    df["hypertension"] = df["hypertension"].map({0: "no", 1: "yes"})
    df["heart_disease"] = df["heart_disease"].map({0: "no", 1: "yes"})
        
    return df
    
def clean_data():
    """
    Run all data cleaning steps in sequence.

    Parameters:
    - df (DataFrame): Raw stroke dataset.

    Returns:
    - DataFrame: Cleaned dataset ready for EDA and modeling.
    """
    
    print("\nStarting cleaning process...")
    
    print("\nStep 1: Loading raw data.")
    df = util.load_raw_data()
    
    print("\nStep 2:Inspecting data info and description.")
    inspect_data(df)
    
    print("\nStep 3:Imputing missing BMI values with the median value")
    df = impute_missing_bmi(df)    
    print("\nVerify that the BMI values have been imputed.")
    inspect_data(df)
    
    print("\nStep 4: Convert numerical binary to text")
    df = encode_binary(df)
    
    print("\nStep 5: Inspect categorical features.")
    inspect_categorical_distribution(df)
    
    print("\nStep 6: Removing the single instance of 'other' gender")
    df = remove_rare_gender(df)
    
    print("\nStep 7: Standardizing text fields.")
    df = standardize_text_fields(df)
    
    print("\nStep 8: Remove duplicates (ignore id)")
    df = remove_duplicates(df)
    
    print("\nStep 9: Final check of data before saving as 'cleaned'.")
    inspect_data(df)
    
    print(f"Saving cleaned data")
    util.save_clean_data(df)
   
    return df

      
if __name__ == '__main__':
    clean_data()