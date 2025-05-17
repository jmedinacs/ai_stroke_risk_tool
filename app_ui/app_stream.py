"""
app_stream.py

Streamlit app for stroke risk prediction using a soft voting ensemble model.

Users input patient profile information via sidebar controls. The app 
one-hot encodes the data, aligns it with the model’s expected column format, 
and returns a stroke-like profile probability based on trained model outputs.

Author: John Medina
Date: 2025-05-12
Project: ai_stroke_risk_tool
"""

import streamlit as st
import joblib
import pandas as pd
import time
from pathlib import Path


def load_model():
    """
    Load the trained ensemble model and column order used during training.

    Returns:
        model: The loaded voting ensemble model.
        column_order (list): Ordered list of feature names used during training.
    """
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "voting_ensemble.pkl"
    columns_path = base_dir / "models" / "column_order_logreg.json"

    model = joblib.load(model_path)
    column_order = pd.read_json(columns_path, typ='series').tolist()

    return model, column_order


def user_input_features():
    """
    Collect user input for patient features via Streamlit sidebar controls.

    Returns:
        pd.DataFrame: A single-row DataFrame with the raw user input.
    """
    st.sidebar.header("Patient Information")

    data = {
        "age": st.sidebar.slider("Age", 0, 100, 50, key="age"),
        "hypertension": 1 if st.sidebar.selectbox("Hypertension", ["yes", "no"], key="hypertension") == "yes" else 0,
        "heart_disease": 1 if st.sidebar.selectbox("Heart Disease", ["yes", "no"], key="heart_disease") == "yes" else 0,
        "ever_married": st.sidebar.selectbox("Ever Married", ["yes", "no"], key="ever_married"),
        "work_type": st.sidebar.selectbox("Work Type", [
            "private", "self-employed", "govt_job", "children", "never_worked"
        ], key="work_type"),
        "avg_glucose_level": st.sidebar.slider("Avg Glucose Level", 50.0, 300.0, 100.0, key="avg_glucose_level"),
        "bmi": st.sidebar.slider("BMI", 10.0, 60.0, 25.0, key="bmi"),
        "smoking_status": st.sidebar.selectbox("Smoking Status", [
            "never smoked", "formerly smoked", "smokes", "unknown"
        ], key="smoking_status")
    }

    return pd.DataFrame([data])


def make_prediction(model, input_df, column_order):
    """
    Prepare encoded input features and return the predicted stroke probability.

    Args:
        model: The trained voting ensemble model.
        input_df (pd.DataFrame): Raw user input in DataFrame format.
        column_order (list): Ordered list of feature columns used during training.

    Returns:
        float: Probability of a stroke-like profile (between 0 and 1).
    """
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(input_df)

    # Ensure all training-time columns are present
    for col in column_order:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder and ensure consistent data types
    df_encoded = df_encoded[column_order].astype(float)

    # Predict stroke risk
    prob = model.predict_proba(df_encoded)[0, 1]
    return prob


def display_result(prob):
    """
    Display the model prediction with visual feedback and interpretation.

    Args:
        prob (float): Probability of a stroke-like profile.
    """
    st.subheader("Prediction Result")
    st.metric(label="Stroke-like Profile Probability", value=f"{prob * 100:.1f}%")
    
    # New explanation line
    st.caption(f"Interpretation: Based on the profile provided, the model estimates a {prob * 100:.1f}% similarity to profiles of patients who previously experienced a stroke in the training data.")
    # Progress bar
    st.progress(min(max(prob, 0.0), 1.0))
    
    # Risk-level messaging
    if prob > 0.7:
        st.error("Very high stroke-like pattern. Recommend immediate medical consultation.")
    elif prob > 0.5:
        st.warning("Elevated stroke-like pattern. Recommend medical screening.")
    else:
        st.success("Pattern appears low-risk based on historical data. Please see age progression below.")

def display_age_progression(model, base_input, column_order):
    """
    Show stroke probability progression across different ages 
    (keeping other features constant).
    """
    st.subheader("Age Progression Forecast")

    age_range = [45, 55, 65, 75, 85, 95]
    results = []

    for age in age_range:
        modified_input = base_input.copy()
        modified_input["age"] = age
        prob = make_prediction(model, modified_input, column_order)
        results.append({"age": age, "probability": prob})

    df_progression = pd.DataFrame(results)

    st.line_chart(df_progression.set_index("age"))
    st.caption("Prediction curve showing how stroke-like profile probability changes with age.")
    st.caption(
        "Note: This simulation assumes your health profile does not change with age, "
        "which may not reflect real-world conditions. For educational use only."
    )



def main():
    """
    Main Streamlit app logic. Loads model, collects user input, makes predictions,
    and handles reset functionality.
    """
    st.title("🧠 Stroke Risk Prediction Tool")
    st.write(
        """
        This tool estimates the probability that a patient's profile resembles historical stroke cases 
        using a machine learning ensemble. **This is not a medical diagnosis**. Use for educational 
        or exploratory purposes only.
        """
    )

    model, column_order = load_model()
    input_df = user_input_features()

    col1, col2 = st.sidebar.columns([1, 1])

    if col1.button("Predict Stroke Risk"):
        prob = make_prediction(model, input_df, column_order)
        display_result(prob)
        display_age_progression(model, input_df, column_order)


    if col2.button("Reset"):
        # Clear all widget states and rerun the app
        for key in st.session_state.keys():
            del st.session_state[key]
        time.sleep(0.1)
        st.rerun()


if __name__ == '__main__':
    main()
