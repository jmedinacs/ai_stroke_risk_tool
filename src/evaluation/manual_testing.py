from pathlib import Path
import joblib
import pandas as pd

# Go two levels up from current file
base_dir = Path(__file__).resolve().parents[2]  # <- THIS goes from /src/pipeline to project root
model_path = base_dir / "models" / "voting_ensemble.pkl"
columns_path = base_dir / "models" / "column_order_logreg.json"

# Load model and column order
model = joblib.load(model_path)
column_order = pd.read_json(columns_path, typ='series').tolist()

# Define input (extreme case)
data = {
    "age": 42,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "yes",
    "work_type": "self-employed",
    "avg_glucose_level": 110.0,
    "bmi": 60.0,
    "smoking_status": "smokes"
}

# One-hot encode
df = pd.DataFrame([data])
df_encoded = pd.get_dummies(df)

# Add missing columns
for col in column_order:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Reorder and cast
df_encoded = df_encoded[column_order].astype(float)

# Predict
prob = model.predict_proba(df_encoded)[0, 1]
print(f"Predicted stroke risk: {prob * 100:.2f}%")
