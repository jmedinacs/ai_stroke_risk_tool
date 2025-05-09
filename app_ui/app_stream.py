import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from pandas.api.types import CategoricalDtype

# Load model and column order
def load_model():
    base_dir = Path(__file__).resolve().parent.parent
    model = joblib.load(base_dir / "models" / "voting_ensemble.pkl")
    column_order = pd.read_json(base_dir / "models" / "column_order_logreg.json", typ='series').tolist()
    return model, column_order

# Collect user input via Streamlit sidebar
def user_input_features():
    st.sidebar.header("Patient Information")

    data = {
        'age': st.sidebar.slider("Age", 0, 100, 50),
        'hypertension': 1 if st.sidebar.selectbox("Hypertension", ["yes", "no"]) == 'yes' else 0,
        'heart_disease': 1 if st.sidebar.selectbox("Heart Disease", ["yes", "no"]) == 'yes' else 0,
        'ever_married': st.sidebar.selectbox("Ever Married", ["yes", "no"]),
        'work_type': st.sidebar.selectbox("Work Type", ["private", "self-employed", "govt_job", "children", "never_worked"]),
        'avg_glucose_level': st.sidebar.slider("Avg Glucose Level", 50.0, 300.0, 100.0),
        'bmi': st.sidebar.slider("BMI", 10.0, 60.0, 25.0),
        'smoking_status': st.sidebar.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "unknown"])
    }

    return pd.DataFrame([data])

# One-hot encode input and align with model expectations
def make_prediction(model, input_df, column_order):
    df_encoded = pd.get_dummies(input_df)
    df_encoded = df_encoded.reindex(columns=column_order, fill_value=0)

    prob = model.predict_proba(df_encoded)[0, 1]
    return prob

# Interpret and display the result
def display_result(prob):
    st.subheader("Prediction Result")
    st.metric(label="Stroke-like Profile Probability", value=f"{prob*100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob > 0.7:
        st.error("⚠️ Very high stroke-like pattern. Recommend immediate medical consultation.")
    elif prob > 0.5:
        st.warning("⚠️ Elevated stroke-like pattern. Recommend medical screening.")
    else:
        st.success("✅ Pattern appears low-risk based on historical data.")

# Main app execution
def main():
    st.title("Stroke Risk Predictor")
    st.write("This tool uses historical data patterns to estimate whether a patient profile resembles those who experienced a stroke. **This is not a diagnosis.**")

    model, column_order = load_model()
    input_df = user_input_features()

    if st.sidebar.button("Predict Stroke Risk"):
        prediction = make_prediction(model, input_df, column_order)
        display_result(prediction)

if __name__ == '__main__':
    main()