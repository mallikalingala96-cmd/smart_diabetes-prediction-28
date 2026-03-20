import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset (for visualization)
data = pd.read_csv("diabetes_cleaned.csv")

st.title("🩺 Diabetes Prediction System")

st.write("### Enter Patient Details")

# Inputs
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=18.5,
    step=0.1,
    format="%.1f"
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5,
    step=0.01,
    format="%.2f"
)
age = st.number_input("Age", 1, 120)

# Prediction
if st.button("Predict"):

    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)

    st.write("### Result")

    if prediction[0] == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")

# ---------------- VISUALIZATIONS ---------------- #

st.write("---")
st.write("## 📊 Visualizations")

# Feature Importance
if st.button("Show Feature Importance"):

    importance = model.feature_importances_
    features = data.drop("Outcome", axis=1).columns

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# Correlation Heatmap
if st.button("Show Heatmap"):

    corr = data.corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    st.pyplot(fig)

# Bar Chart (Outcome count)
if st.button("Show Diabetes Count"):

    count = data["Outcome"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(["Not Diabetic", "Diabetic"], count)
    ax.set_title("Diabetes Distribution")
    st.pyplot(fig)

# Info section
st.write("---")
st.write("### 📌 About")
st.write("""
- This model uses Decision Tree Algorithm
- Feature importance shows which factor affects prediction most
- Heatmap shows relationship between features
""")