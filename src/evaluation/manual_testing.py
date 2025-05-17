from pathlib import Path
import joblib
import pandas as pd

# === Utility Function ===
def encode_input(df_raw, column_order):
    """
    One-hot encodes input and aligns it with training-time column order.
    Missing columns are added as 0. Extra columns are discarded.
    
    Args:
        df_raw (pd.DataFrame): Raw input data
        column_order (list): Expected feature order from training
    
    Returns:
        pd.DataFrame: Encoded and aligned input
    """
    df_encoded = pd.get_dummies(df_raw)

    # Add any missing columns from training
    for col in column_order:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder and remove extras
    df_encoded = df_encoded[column_order].astype(float)
    return df_encoded

# === Load Model and Column Order ===
base_dir = Path(__file__).resolve().parents[2]  # Adjust if needed
model_path = base_dir / "models" / "voting_ensemble.pkl"
columns_path = base_dir / "models" / "column_order_logreg.json"

model = joblib.load(model_path)
column_order = pd.read_json(columns_path, typ='series').tolist()

for col in column_order:
    print(col)

# === Define Test Input ===
input_1 = {
    "age": 70,
    "hypertension_yes": 1,
    "heart_disease_yes": 1,
    "ever_married": "yes",
    "work_type": "self-employed",
    "avg_glucose_level": 110.0,
    "bmi": 60.0,
    "smoking_status": "smokes"
}

input_2 = input_1.copy()
input_2["hypertension_yes"] = 0
input_2["heart_disease_yes"] = 0

# === Predict for Input 1 ===
df1 = pd.DataFrame([input_1])
X1 = encode_input(df1, column_order)
prob1 = model.predict_proba(X1)[0, 1]
print(f"With hypertension and heart disease: {prob1 * 100:.2f}%")

# === Predict for Input 2 ===
df2 = pd.DataFrame([input_2])
X2 = encode_input(df2, column_order)
prob2 = model.predict_proba(X2)[0, 1]
print(f"Without hypertension and heart disease: {prob2 * 100:.2f}%")
