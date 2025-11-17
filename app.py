import streamlit as st
import pandas as pd
import joblib

model = joblib.load("nb_model.pkl")

st.set_page_config(page_title="Naive Bayes Drug Prediction", layout="centered")
st.title("💊 Drug Prediction Using Naive Bayes")

age = st.number_input("Age", min_value=1, max_value=100)
sex = st.selectbox("Sex", ["F", "M"])
bp = st.selectbox("Blood Pressure (BP)", ["LOW", "NORMAL", "HIGH"])
cholesterol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
na_to_k = st.number_input("Na to K Ratio", min_value=0.0)

sex_map = {"F": 0, "M": 1}
bp_map = {"LOW": 1, "NORMAL": 2, "HIGH": 0}
chol_map = {"NORMAL": 1, "HIGH": 0}

input_data = pd.DataFrame([{
    "Age": age,
    "Sex": sex_map[sex],
    "BP": bp_map[bp],
    "Cholesterol": chol_map[cholesterol],
    "Na_to_K": na_to_k
}])

if st.button("Predict Drug"):
    prediction = model.predict(input_data)[0]
    st.success(f"Recommended Drug: {prediction}")
st.caption("Developed by Supraja ✨ | Powered by Streamlit & Scikit-learn")

